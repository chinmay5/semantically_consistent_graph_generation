import math

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from src.ConStruct import utils
from src.ConStruct.diffusion import diffusion_utils
from src.ConStruct.models.layers import EtoX, Xtoy, Etoy, masked_softmax


def compute_distances(points, sigma=1.0):
    batch_size, num_points, _ = points.shape  # B, N, 3

    points_src = points.unsqueeze(2)  # B, N, 1, 3
    points_dest = points.unsqueeze(1)  # B, 1, N, 3

    coord_diff = points_src - points_dest  # B, N, N, 3
    # Compute the pairwise distances between all points
    # square_distances = coord_diff.pow(2).sum(-1)  # B, N, N

    distance_matrix = torch.linalg.norm(coord_diff, dim=-1)  # B, N, N
    # rbf_matrix = torch.exp(-square_distances / (2 * sigma ** 2))

    return distance_matrix.unsqueeze(-1)  # B, N, N, 1


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(self, dx, de, dy, n_head, predicts_y=False):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        # self.x_e_add = Linear(dx, de)
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_norm_pos1 = Linear(1, de)
        self.lin_norm_pos2 = Linear(1, de)

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)

        # Attention

        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        # Incorporate e to x
        # self.e_att_add = Linear(de, n_head)
        self.e_att_mul = Linear(de, n_head)

        self.pos_att_mul = Linear(de, n_head)

        self.e_x_mul = EtoX(de, dx)
        self.pos_x_mul = EtoX(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, de)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)  # Unused, but needed to load old checkpoints

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.predicts_y = predicts_y
        if self.predicts_y:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)

        # Process_pos
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)  # For EGNN v3: map to pi, pj

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if self.predicts_y:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, pos, node_mask):
        """ :param X: bs, n, d        node features
            :param E: bs, n, n, d     edge features
            :param y: bs, dz           global features
            :param pos: bs, n, 3
            :param node_mask: bs, n
            :return: newX, newE, new_y with the same shape. """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 0. Create a distance matrix that can be used later
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        normalized_pos = pos / (norm_pos + 1e-7)  # bs, n, 3

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1).float()
        cosines = torch.sum(normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2), dim=-1, keepdim=True)
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)

        norm1 = self.lin_norm_pos1(norm_pos)  # bs, n, de
        norm2 = self.lin_norm_pos2(norm_pos)  # bs, n, de
        dist1 = F.relu(self.lin_dist1(pos_info) + norm1.unsqueeze(2) + norm2.unsqueeze(1)) * e_mask1 * e_mask2

        # 1. Process E
        # Remember that Y is the edge features and not the "labels".
        # Often times the symbol Y is used for the latter. Not here though.
        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2

        # 1.2. Incorporate distances
        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2  # bs, n, n, dx

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(E) * e_mask1 * e_mask2  # bs, n, n, de

        diffusion_utils.assert_correctly_masked(Eout, e_mask1 * e_mask2)

        Q = (self.q(X) * x_mask).unsqueeze(2)  # bs, n, 1, dx
        K = (self.k(X) * x_mask).unsqueeze(1)  # bs, 1, n, dx
        prod = Q * K / math.sqrt(Y.size(-1))  # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2  # bs, n, n, n_head

        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a

        # 2.2 Incorporate position features
        pos_x_mul = self.pos_att_mul(dist1)
        a = a + pos_x_mul * a
        a = a * e_mask1 * e_mask2

        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head, 1
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)  # bs, 1, n, 1, dx
        weighted_V = alpha * V  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)  # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, n_head x dx
        weighted_V = self.out(weighted_V) * x_mask  # bs, n, dx

        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V

        pos_x_mul = self.pos_x_mul(dist1, e_mask2)
        weighted_V = weighted_V + pos_x_mul * weighted_V

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)  # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(Xout, x_mask)

        # Process y based on X, E and pos
        if self.predicts_y:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            new_y = y + x_y + e_y + dist_y
            y_out = self.y_out(new_y)  # bs, dy
        else:
            y_out = None

        return Xout, Eout, y_out


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, predicts_y=True) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.predicts_y = predicts_y
        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, predicts_y=self.predicts_y)
        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        if self.predicts_y:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, features: utils.PlaceHolder):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        X = features.X
        E = features.E
        y = features.y
        pos = features.pos
        node_mask = features.node_mask

        # pos should never change
        newX, newE, new_y = self.self_attn(X, E, y, pos, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)

        E = self.normE2(E + ff_outputE)
        # Making the result undirected
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if self.predicts_y:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)
        else:
            y = None

        out = utils.PlaceHolder(X=X, E=E, y=y, pos=pos, charges=None, node_mask=node_mask).mask()

        return out


# Using the Furior feature encoding
def fourier_features(point_cloud, num_frequencies=5):
    """
    Apply Fourier feature mapping to a batch of 3D coordinates.

    Args:
        point_cloud (torch.Tensor): Tensor of shape (B, N, 3), where
                                    B = batch size, N = number of points, 3 = (x, y, z)
        num_frequencies (int): Number of frequency bands to use.

    Returns:
        torch.Tensor: Fourier transformed features of shape (B, N, 6 * num_frequencies)
    """
    B, N, C = point_cloud.shape  # Expecting C = 3 (x, y, z)
    assert C == 3, "Input point cloud should have shape (B, N, 3)"

    # Define frequency bands: Exponentially spaced frequencies
    freqs = torch.tensor([2.0 ** i for i in range(num_frequencies)],
                         device=point_cloud.device)  # Shape: (num_frequencies,)

    # Compute Fourier features
    point_cloud_expanded = point_cloud.unsqueeze(-1) * freqs  # Shape: (B, N, 3, num_frequencies)
    sin_feats = torch.sin(2 * math.pi * point_cloud_expanded)  # Shape: (B, N, 3, num_frequencies)
    cos_feats = torch.cos(2 * math.pi * point_cloud_expanded)  # Shape: (B, N, 3, num_frequencies)

    # Concatenate sin and cos features along the last dimension
    fourier_embedded = torch.cat([sin_feats, cos_feats], dim=-1)  # Shape: (B, N, 3, 2 * num_frequencies)

    # Reshape to (B, N, 6 * num_frequencies)
    return fourier_embedded.view(B, N, -1)


class EquivariantGraph3DTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(self,
                 n_layers: int,
                 input_dims: dict,
                 hidden_mlp_dims: dict,
                 hidden_dims: dict,
                 output_dims: dict,
                 dropout: float = 0.1,
                 dropout_in_and_out: bool = False
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_radius = 1  # Since the radius information is a scalar
        self.out_dim_charges = output_dims.charges
        # To avoide unused parameters in the model (raises issues with strategy ddp)
        self.predicts_final_y = self.out_dim_y > 0

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims['X']),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity())

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims.E, hidden_mlp_dims['E']),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity()
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims.y, hidden_mlp_dims['y']),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']),
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            act_fn_in)

        # Projecting the coordinates to a high-dimensional space.
        self.mlp_in_pos_high_dim = nn.Sequential(
            nn.Linear(3, hidden_mlp_dims['X']),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']),
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            predicts_y=(layer_idx != n_layers - 1) or self.predicts_final_y)
                                        for layer_idx in range(n_layers)])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']),
            act_fn_out,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['X'], output_dims.X + output_dims.charges))

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']),
            act_fn_out,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['E'], output_dims.E))

        # For simplicity, we implement the radius prediction as an extra prediction head on the last layer
        # We will reuse most of the features that are obtained from the transformer blocks

        self.mlp_out_radius = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']),
            act_fn_out,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims['E'], self.out_dim_radius))

        # We do not train for y. The variable is used to provide global feature such as the time
        # and few other graph level feature terms. However, we will not update it and compute CE for it.

    def forward(self, model_input: utils.PlaceHolder):
        X = model_input.X
        E = model_input.E
        y = model_input.y
        pos = model_input.pos
        radius = model_input.radius
        bs, n = X.shape[0], X.shape[1]
        node_mask = model_input.node_mask

        diag_mask = ~torch.eye(n, device=X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]
        E_to_out = E[..., :self.out_dim_E]
        # We are copying over the values of y.
        # Hence, the global condition is not changed.
        # We do not want to update it.
        y_to_out = y[..., :self.out_dim_y]
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        # if no_node_denoising:
        # We are only denoising the edge connectivity
        pos = pos.detach()
        # We look for high dimensional projections of the node positions
        # Applying the Fourier feature encoding to the node positions.
        # The hope is that it leads to better generalizability.
        features = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y), charges=None,
                                     # Passing pos coordinates directly
                                     pos=pos, node_mask=node_mask).mask()

        for layer in self.tf_layers:
            features = layer(features)
        X = self.mlp_out_X(features.X)
        E = self.mlp_out_E(features.E)
        # Predicting the radius information
        radius = self.mlp_out_radius(features.E) if radius is not None else None
        # skip connections
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        # Doing the same with the radius information
        radius = radius * diag_mask if radius is not None else None

        # Copy over the values of y since we do not want to update it.
        # It is a global condition that should not change.
        if self.predicts_final_y:
            y = self.mlp_out_y(features.y)
            y = y + y_to_out
        else:
            y = None

        E = 1 / 2 * (E + torch.transpose(E, 1, 2))
        # Same with the radius
        radius = 1 / 2 * (radius + torch.transpose(radius, 1, 2)) if radius is not None else None

        final_X = X[..., :self.out_dim_X]
        final_charges = X[..., self.out_dim_X: ]
        # We do not touch the node coordinates at all
        out = utils.PlaceHolder(pos=pos, X=final_X, charges=final_charges, E=E, y=y, node_mask=node_mask, radius=radius).mask()
        return out
