import torch
import torch.nn.functional as F
import src.ConStruct.utils as utils


def batch_trace(X):
    """Expect a matrix of shape B N N, returns the trace in shape B."""
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    return diag.sum(dim=-1)


def batch_diagonal(X):
    """Extracts the diagonal from the last two dims of a tensor."""
    return torch.diagonal(X, dim1=-2, dim2=-1)


class DummyExtraFeatures:
    """This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        device = noisy_data.X.device
        empty_x = torch.zeros((*noisy_data.X.shape[:-1], 0), device=device)
        empty_e = torch.zeros((*noisy_data.E.shape[:-1], 0), device=device)
        empty_y = torch.zeros((noisy_data.y.shape[0], 0), device=device)
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, cfg, dataset_infos):
        self.eigenfeatures = (cfg.model.eigenfeatures,)
        self.max_n_nodes = dataset_infos.max_n_nodes
        self.adj_features = AdjacencyFeatures(
            num_degree=cfg.model.num_degree,
            max_degree=cfg.model.max_degree,
            cycle_features=cfg.model.cycle_features,
        )
        if self.eigenfeatures:
            self.eigenfeatures = EigenFeatures(
                num_eigenvectors=cfg.model.num_eigenvectors,
                num_eigenvalues=cfg.model.num_eigenvalues,
            )

    def update_input_dims(self, input_dims):
        if self.eigenfeatures:
            input_dims.y += (
                self.eigenfeatures.num_eigenvalues + 1
            )  # num components + num eigenvalues
            input_dims.y += self.adj_features.num_degree + 2
            input_dims.y += input_dims.X
            input_dims.y += input_dims.E
            input_dims.X += self.eigenfeatures.num_eigenvectors + 1

        input_dims.X += 3
        input_dims.y += 5
        input_dims.E += self.adj_features.max_degree
        input_dims.E += 1

        return input_dims

    def __call__(self, noisy_data):
        # make data dense in the beginning to avoid doing this twice for both cycles and eigenvalues
        n = noisy_data.node_mask.sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles, extra_edge_attr = self.adj_features(
            noisy_data
        )  # (bs, n_cycles)

        if self.eigenfeatures:
            eval_feat, evec_feat = self.eigenfeatures.compute_features(noisy_data)
            return utils.PlaceHolder(
                X=torch.cat((x_cycles, evec_feat), dim=-1),
                E=extra_edge_attr,
                y=torch.hstack((n, y_cycles, eval_feat)),
            )
        else:
            return utils.PlaceHolder(
                X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles))
            )


class EigenFeatures:
    """Some code is taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py."""

    def __init__(self, num_eigenvectors, num_eigenvalues):
        self.num_eigenvectors = num_eigenvectors
        self.num_eigenvalues = num_eigenvalues

    def compute_features(self, noisy_data):
        E_t = noisy_data.E
        mask = noisy_data.node_mask
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = self.compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        eigvals, eigvectors = torch.linalg.eigh(L)
        eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
        eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
        # Retrieve eigenvalues features
        n_connected_comp, batch_eigenvalues = self.eigenvalues_features(
            eigenvalues=eigvals, num_eigenvalues=self.num_eigenvalues
        )
        # Retrieve eigenvectors features
        evector_feat = self.eigenvector_features(
            vectors=eigvectors,
            node_mask=noisy_data.node_mask,
            n_connected=n_connected_comp,
            num_eigenvectors=self.num_eigenvectors,
        )

        evalue_feat = torch.hstack((n_connected_comp, batch_eigenvalues))
        return evalue_feat, evector_feat

    def compute_laplacian(self, adjacency, normalize: bool):
        """
        adjacency : batched adjacency matrix (bs, n, n)
        normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
        Return:
            L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
        """
        diag = torch.sum(adjacency, dim=-1)  # (bs, n)
        n = diag.shape[-1]
        D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
        combinatorial = D - adjacency  # (bs, n, n)

        if not normalize:
            return (combinatorial + combinatorial.transpose(1, 2)) / 2

        diag0 = diag.clone()
        diag[diag == 0] = 1e-12

        diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
        D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
        L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
        L[diag0 == 0] = 0
        return (L + L.transpose(1, 2)) / 2

    def eigenvalues_features(self, eigenvalues, num_eigenvalues):
        """
        values : eigenvalues -- (bs, n)
        node_mask: (bs, n)
        k: num of non zero eigenvalues to keep
        """
        ev = eigenvalues
        bs, n = ev.shape
        n_connected_components = (ev < 1e-5).sum(dim=-1)
        assert (n_connected_components > 0).all(), (n_connected_components, ev)

        to_extend = max(n_connected_components) + num_eigenvalues - n
        if to_extend > 0:
            ev = torch.hstack((ev, 2 * torch.ones(bs, to_extend, device=ev.device)))
        indices = torch.arange(num_eigenvalues, device=ev.device).unsqueeze(
            0
        ) + n_connected_components.unsqueeze(1)
        first_k_ev = torch.gather(ev, dim=1, index=indices)

        return n_connected_components.unsqueeze(-1), first_k_ev

    def eigenvector_features(self, vectors, node_mask, n_connected, num_eigenvectors):
        """
        vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
        returns:
            not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
            k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
        """
        bs, n = vectors.size(0), vectors.size(1)

        # Create an indicator for the nodes outside the largest connected components
        first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
        # Add random value to the mask to prevent 0 from becoming the mode
        random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
        first_ev = first_ev + random
        most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
        mask = ~(first_ev == most_common.unsqueeze(1))
        not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

        # Get the eigenvectors corresponding to the first nonzero eigenvalues
        to_extend = max(n_connected) + num_eigenvectors - n
        if to_extend > 0:
            vectors = torch.cat(
                (vectors, torch.zeros(bs, n, to_extend, device=vectors.device)), dim=2
            )  # bs, n , n + to_extend

        indices = (
            torch.arange(num_eigenvectors, device=vectors.device)
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        indices = indices + n_connected.unsqueeze(2)  # bs, 1, k
        indices = indices.expand(-1, n, -1)  # bs, n, k
        first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
        first_k_ev = first_k_ev * node_mask.unsqueeze(2)

        return torch.cat((not_lcc_indicator, first_k_ev), dim=-1)


class AdjacencyFeatures:
    """Builds cycle counts for each node in a graph."""

    def __init__(self, num_degree, cycle_features, max_degree=10):
        self.num_degree = num_degree
        self.max_degree = max_degree
        self.cycle_features = cycle_features

    def __call__(self, noisy_data):
        adj_matrix = noisy_data.E[..., 1:].sum(dim=-1).float()
        num_nodes = noisy_data.node_mask.sum(dim=1)

        self.calculate_kpowers(adj_matrix)

        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()

        # Node features
        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)

        # Edge features
        dist = self.path_features()
        local_ngbs = self.local_neighbors(num_nodes)

        # Graph features
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        kcyclesy = torch.clamp(kcyclesy, 0, 5) / 5
        degree_dist = self.get_degree_dist(adj_matrix, noisy_data.node_mask)
        node_dist = self.get_node_dist(noisy_data.X)
        edge_dist = self.get_edge_dist(noisy_data.E)

        if not self.cycle_features:
            # zero entries that provide explicit cycle information
            kcyclesx = torch.zeros_like(kcyclesx)
            kcyclesy = torch.zeros_like(kcyclesy)
            dist = torch.zeros_like(dist)
            # TODO: delete below?
            # hide self loops in dist
            # bs, n = dist.shape[:2]
            # deg = dist.shape[-1]
            # diagonal_mask = (
            #     torch.eye(n, dtype=torch.bool)
            #     .unsqueeze(0)
            #     .unsqueeze(3)
            #     .expand(bs, n, n, deg)
            # )
            # dist.masked_fill_(diagonal_mask, 0)

        # Build features
        x_feats = torch.clamp(kcyclesx, 0, 5) / 5 * noisy_data.node_mask.unsqueeze(-1)
        y_feats = torch.cat([kcyclesy, degree_dist, node_dist, edge_dist], dim=-1)
        edge_feats = torch.cat([dist, local_ngbs], dim=-1)
        edge_feats = torch.clamp(edge_feats, 0, 5) / 5

        return x_feats, y_feats, edge_feats

    def calculate_kpowers(self, adj):
        """adj: bs, n, n"""
        shape = (self.max_degree, *adj.shape)
        self.k = torch.zeros(shape, device=adj.device)
        self.k[0] = adj
        self.d = adj.sum(dim=-1)
        for i in range(1, self.max_degree):
            self.k[i] = self.k[i - 1] @ adj

        # Warning: index changes by 1 (count from 1 and not 0)
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = [
            self.k[i] for i in range(6)
        ]

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.k1 @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5)
        triangles = batch_diagonal(self.k3)

        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.k1 @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6)
        term_2_t = batch_trace(self.k3**2)
        term3_t = torch.sum(self.k1 * self.k2.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2)
        a_4_t = batch_diagonal(self.k4)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4)
        term_6_t = batch_trace(self.k3)
        term_7_t = batch_diagonal(self.k2).pow(3).sum(-1)
        term8_t = torch.sum(self.k3, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2).pow(2).sum(-1)
        term10_t = batch_trace(self.k2)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def path_features(self):
        path_features = self.k.bool().float()  # max power, bs, n, n
        # put things in order
        path_features = path_features.permute(1, 2, 3, 0)  # bs, n, n, max power
        return path_features

    def local_neighbors(self, num_nodes):
        """Adamic-Adar index for each pair of nodes.
        this function captures the local neighborhood information, commonly used in social network analysis
        [i, j], sum of 1/log(degree(u)), u is a common neighbor of i and j.
        """
        normed_adj = self.k1 / self.k1.sum(-1).unsqueeze(
            1
        )  # divide each column by its degree

        normed_adj = torch.sqrt(torch.log(normed_adj).abs())
        normed_adj = torch.nan_to_num(1 / normed_adj, posinf=0)
        normed_adj = torch.matmul(normed_adj, normed_adj.transpose(-2, -1))

        # mask self-loops to 0
        mask = torch.eye(normed_adj.shape[-1]).repeat(normed_adj.shape[0], 1, 1).bool()
        normed_adj[mask] = 0

        # normalization
        normed_adj = (
            normed_adj * num_nodes.log()[:, None, None] / num_nodes[:, None, None]
        )
        return normed_adj.unsqueeze(-1)

    def get_degree_dist(self, adj_matrix, node_mask):
        # bs, n = noisy_data.node_mask.shape
        degree = adj_matrix.sum(dim=-1).long()  # (bs, n)
        degree[degree > self.num_degree] = self.num_degree + 1  # bs, n
        one_hot_degree = F.one_hot(
            degree, num_classes=self.num_degree + 2
        ).float()  # bs, n, num_degree + 2
        one_hot_degree[~node_mask] = 0
        degree_dist = one_hot_degree.sum(dim=1)  # bs, num_degree + 2
        s = degree_dist.sum(dim=-1, keepdim=True)
        s[s == 0] = 1
        degree_dist = degree_dist / s

        return degree_dist

    def get_node_dist(self, X):
        node_dist = X.sum(dim=1)  # bs, dx
        s = node_dist.sum(-1)  # bs
        s[s == 0] = 1
        node_dist = node_dist / s.unsqueeze(-1)  # bs, dx
        return node_dist

    def get_edge_dist(self, E):
        edge_dist = E.sum(dim=[1, 2])  # bs, de
        s = edge_dist.sum(-1)  # bs
        s[s == 0] = 1
        edge_dist = edge_dist / s.unsqueeze(-1)  # bs, de
        return edge_dist
