import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F

from src.ConStruct import utils
from src.ConStruct.diffusion import diffusion_utils
from src.ConStruct.models.layers import Xtoy, Etoy, masked_softmax


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        predicts_y: bool = True,  # To avoide unused parameters in the model (raises issues with strategy ddp)
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.predicts_y = predicts_y
        self.self_attn = NodeEdgeBlock(
            dx, de, dy, n_head, predicts_y=self.predicts_y, **kw
        )

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
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """
        X, E, y, node_mask = features.X, features.E, features.y, features.node_mask

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if self.predicts_y:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)
        else:
            y = None

        return utils.PlaceHolder(
            X=X, E=E, y=y, charges=None, node_mask=node_mask
        ).mask()


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(
        self,
        dx,
        de,
        dy,
        n_head,
        predicts_y=True,  # To avoide unused parameters in the model (raises issues with strategy ddp)
        **kwargs,
    ):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head
        self.predicts_y = predicts_y

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        if self.predicts_y:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        if self.predicts_y:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask  # (bs, n, dx)
        K = self.k(X) * x_mask  # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask  # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        if self.predicts_y:
            y = self.y_y(y)
            e_y = self.e_y(E, e_mask1, e_mask2)
            x_y = self.x_y(X, x_mask)
            new_y = y + x_y + e_y
            new_y = self.y_out(new_y)  # bs, dy
        else:
            new_y = None

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        dropout: float = 0.1,
        dropout_in_and_out: bool = False,
    ):
        super().__init__()
        # Layer architecture
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges
        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        # To avoide unused parameters in the model (raises issues with strategy ddp)
        self.predicts_final_y = self.out_dim_y > 0

        # Layers
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims.E, hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims.y, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
        )

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    predicts_y=(layer_idx != n_layers - 1) or self.predicts_final_y,
                )
                for layer_idx in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims["X"], output_dims.X + output_dims.charges),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
            nn.Linear(hidden_mlp_dims["E"], output_dims.E),
        )

        if self.predicts_final_y:
            self.mlp_out_y = nn.Sequential(
                nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
                act_fn_out,
                nn.Dropout(dropout) if dropout_in_and_out else nn.Identity(),
                nn.Linear(hidden_mlp_dims["y"], output_dims.y),
            )

    def forward(self, model_input):
        X = model_input.X
        E = model_input.E
        y = model_input.y
        node_mask = model_input.node_mask
        bs, n = X.shape[0], X.shape[1]

        diag_mask = ~torch.eye(n, device=model_input.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X + self.out_dim_charges]
        E_to_out = model_input.E[..., : self.out_dim_E]
        y_to_out = model_input.y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(model_input.E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        features = utils.PlaceHolder(
            X=self.mlp_in_X(X),
            E=new_E,
            y=self.mlp_in_y(model_input.y),
            charges=None,
            node_mask=node_mask,
        ).mask()

        for layer in self.tf_layers:
            features = layer(features)

        X = self.mlp_out_X(features.X)
        E = self.mlp_out_E(features.E)

        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        E = 1 / 2 * (E + torch.transpose(E, 1, 2))  # symmetrize E

        if self.predicts_final_y:
            y = self.mlp_out_y(features.y)
            y = y + y_to_out
        else:
            y = None

        final_X = X[..., : self.out_dim_X]
        final_charges = X[..., self.out_dim_X :]

        return utils.PlaceHolder(
            X=final_X, charges=final_charges, E=E, y=y, node_mask=node_mask
        ).mask()
