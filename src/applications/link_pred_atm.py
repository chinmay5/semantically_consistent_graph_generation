import glob
import itertools
import math
import os
import os.path as osp
import random
import shutil
import time

import numpy as np
import pyvista
import torch
import torch_geometric
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import BCEWithLogitsLoss, MaxPool1d, Conv1d, ModuleList
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SortAggregation, MLP
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, remove_self_loops

from src.ConStruct.utils import save_graph_as_vtp_file
from src.environment_setup import PROJECT_ROOT_DIR

test_loc = '/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/test_data'
train_loc = '/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/train_data'


def load_vtp(filename: str) -> torch_geometric.data.Data:
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    if 'edge_type' in vtk_data.cell_data:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['edge_type']), dtype=torch.float)
    elif 'radius' in vtk_data.cell_data:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)  # Should always be float
    else:
        raise ValueError("No edge attribute found")
    edges = edges.T
    graph_data = Data(x=torch.ones(nodes.size(0), dtype=torch.float), edge_index=edges, edge_attr=radius,
                      pos=nodes)
    # Now, we convert it to dense adjacency matrix and load the information as a placeholder object

    if not torch_geometric.utils.is_undirected(graph_data.edge_index):
        # Making the graph undirected
        new_edges, new_edge_attr = torch_geometric.utils.to_undirected(edge_index=graph_data.edge_index,
                                                                       edge_attr=graph_data.edge_attr,
                                                                       num_nodes=graph_data.x.size(0),
                                                                       reduce='min')
    else:
        new_edges, new_edge_attr = graph_data.edge_index, graph_data.edge_attr

    graph_data.edge_index, graph_data.edge_attr = new_edges, new_edge_attr
    # We need to normalize the coordinates since they can be very large.
    graph_data.pos = graph_data.pos - torch.mean(graph_data.pos, dim=0, keepdim=True)
    max_scale = torch.max(torch.linalg.norm(graph_data.pos, dim=1))
    graph_data.pos = graph_data.pos / max_scale
    # Just to make our lives easier.
    graph_data.x = graph_data.pos
    graph_data.edge_attr = graph_data.edge_attr.to(torch.long)
    return graph_data


class ATMSEALDataset(InMemoryDataset):
    def __init__(self, data_loc, num_hops, split='train'):
        self.data = [load_vtp(osp.join(data_loc, os.path.basename(file))) for file in os.listdir(data_loc)]
        self.num_hops = num_hops
        super().__init__(os.path.join(PROJECT_ROOT_DIR, "applications"))
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        return ['SEAL_train_data.pt', 'SEAL_val_data.pt', 'SEAL_test_data.pt']

    def save(self, data_list, path):
        # We just save the list of Data objects.
        torch.save(self.collate(data_list), path)

    def process(self):
        transform = RandomLinkSplit(num_val=0.05, num_test=0.1,
                                    is_undirected=True, split_labels=True, key='edge_attr')
        train_data_list, val_data_list, test_data_list = [], [], []
        for idx, data in enumerate(self.data):
            train_data, val_data, test_data = transform(data)

            self._max_z = 0

            # Collect a list of subgraphs for training, validation and testing:
            # NOTE: Get the edge attributes. This line is unfortunately broken.
            # It has to be fixed.

            train_pos_data_list = self.extract_enclosing_subgraphs(
                train_data.edge_index, train_data.pos_edge_attr_index, train_data.pos_edge_attr, idx, train_data.x.size(0), train_data.edge_attr)
            train_neg_data_list = self.extract_enclosing_subgraphs(
                train_data.edge_index, train_data.neg_edge_attr_index, train_data.neg_edge_attr, idx, train_data.x.size(0), train_data.edge_attr)

            val_pos_data_list = self.extract_enclosing_subgraphs(
                val_data.edge_index, val_data.pos_edge_attr_index, val_data.pos_edge_attr, idx, val_data.x.size(0), val_data.edge_attr)
            val_neg_data_list = self.extract_enclosing_subgraphs(
                val_data.edge_index, val_data.neg_edge_attr_index, val_data.neg_edge_attr, idx, val_data.x.size(0), val_data.edge_attr)

            test_pos_data_list = self.extract_enclosing_subgraphs(
                test_data.edge_index, test_data.pos_edge_attr_index, test_data.pos_edge_attr, idx, test_data.x.size(0), test_data.edge_attr)
            test_neg_data_list = self.extract_enclosing_subgraphs(
                test_data.edge_index, test_data.neg_edge_attr_index, test_data.neg_edge_attr, idx, test_data.x.size(0), test_data.edge_attr)

            train_list = train_pos_data_list + train_neg_data_list
            train_data_list.extend(train_list)
            val_list = val_pos_data_list + val_neg_data_list
            val_data_list.extend(val_list)
            test_list = test_pos_data_list + test_neg_data_list
            test_data_list.extend(test_list)
        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])
        self.save(test_data_list, self.processed_paths[2])

    def extract_enclosing_subgraphs(self, edge_index, edge_attr_index, y, idx, num_nodes, edge_attr):
        data_list = []
        for src, dst in edge_attr_index.t().tolist():

            # Hack
            _, _, _, mask = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=False, num_nodes=num_nodes)
            y_label = edge_attr[mask]

            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst,
                                        num_nodes=sub_nodes.size(0))

            data = Data(x=self.data[idx].x[sub_nodes], z=z,
                        # edge_index=sub_edge_index, y=y)
                        edge_index=sub_edge_index, y=y_label)
            data_list.append(data)

        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        self._max_z = max(int(z.max()), self._max_z)

        return z.to(torch.long)


train_dataset = ATMSEALDataset(data_loc=train_loc, num_hops=2, split='train')
val_dataset = ATMSEALDataset(data_loc=test_loc, num_hops=2, split='val')
test_dataset = ATMSEALDataset(data_loc=test_loc, num_hops=2, split='test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, GNN=GCNConv, k=0.6, num_classes=5):
        super().__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = int(max(10, k))

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset.num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.pool = SortAggregation(k)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, 128, num_classes], dropout=0.5, norm=None)

    def forward(self, x, edge_index, batch):
        if self.training:
            x = x + 0.02 * torch.randn_like(x)
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = self.pool(x, batch)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN(hidden_channels=32, num_layers=3).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


def create_and_train_link_predictor():
    print("Training the model")
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            # loss = criterion(out.view(-1), data.y.to(torch.float))
            loss = criterion(out.view(-1), data.y.to(torch.long))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs

        return total_loss / len(train_dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()

        y_pred, y_true = [], []
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        return roc_auc_score(torch.cat(y_true), torch.cat(y_pred), multi_class='ovo')

    times = []
    best_val_auc = test_auc = 0
    for epoch in range(1, 100):
    # for epoch in range(1, 2):
        start = time.time()
        loss = train()
        val_auc = test(val_loader)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = test(test_loader)
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT_DIR, "applications", 'best_model.pth'))
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


# Now we load .vtp files that have missing links and try to predict them
# We will use the trained model to predict the missing links

# We load the best model



def drop_edges(data, drop_ratio=0.2):
    """
    Randomly drops a fraction of edges from the graph.

    Args:
        data (torch_geometric.data.Data): Input graph.
        drop_ratio (float): Fraction of edges to remove.

    Returns:
        data (torch_geometric.data.Data): Graph with edges removed.
        dropped_edges (torch.Tensor): The removed edges.
    """
    edge_index, edge_attr = data.edge_index, data.edge_attr
    num_edges = edge_index.shape[1]
    num_drop = int(drop_ratio * num_edges)

    # Randomly select edges to drop
    drop_indices = random.sample(range(num_edges), num_drop)
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[drop_indices] = False

    # Keep only non-dropped edges
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask]
    dropped_edges = edge_index[:, drop_indices]

    # Ensure graph remains valid
    new_edge_index, _ = remove_self_loops(new_edge_index)

    data.edge_index = new_edge_index
    data.edge_attr = new_edge_attr
    return data, dropped_edges

def predict_new_links_and_save(filename):
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT_DIR, "applications", 'best_model.pth')))
    # We set it to eval
    model.eval()

    # Cleaning away the existing folder
    save_dir = '/home/chinmayp/workspace/ConStruct/outputs/wadwauwau'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    whole_data = load_vtp(filename)
    # We delete some of the edges to simulate the missing links
    whole_data, dropped_edges = drop_edges(whole_data, drop_ratio=0.3)

    # --- Function: Predict links on candidate edges ---
    def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.
        return z.to(torch.long)

    @torch.inference_mode()
    def predict_links(model, graph_data, candidate_edges, num_hops=2):
        model.eval()
        predicted_edges, data_list = [], []
        for src_orig, dst_orig in candidate_edges.t().tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src_orig, dst_orig], num_hops=num_hops, edge_index=graph_data.edge_index, relabel_nodes=True,
                num_nodes=graph_data.x.size(0))
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            # mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            # mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            # sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # Calculate node labeling.
            z = drnl_node_labeling(sub_edge_index, src, dst,
                                   num_nodes=sub_nodes.size(0))

            data = Data(x=graph_data.x[sub_nodes], z=z,
                        edge_index=sub_edge_index, src=src_orig, dst=dst_orig)
            data_list.append(data)
        # Let us collate the data
        # SEAL expects a 'batch' attribute; for a single subgraph, all nodes belong to batch 0.

        batch = Batch.from_data_list(data_list).to(device)

        adjacency = torch.zeros((graph_data.x.size(0), graph_data.x.size(0)), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
            src, dst = batch.src, batch.dst
            preds = torch.sigmoid(out) > 0.5 #out.argmax(dim=1)  # 0 or 1
            adjacency[src, dst] = preds.squeeze().to(torch.long)

        return adjacency.cpu()
        # Collect candidate edges predicted as positive (class 1).
        # print("Evaluating the model")
        # predicted_edges = []
        # for i, pred in enumerate(preds):
        #     if pred.item() == 1:
        #         predicted_edges.append(tuple(candidate_edges[:, i].tolist()))
        # return predicted_edges

    adjacency = predict_links(model, whole_data, dropped_edges, num_hops=2)
    # We add the new edges to the graph and save it

    # if len(new_edges) == 0:
    #     print("No new edges predicted")
    #     shutil.copy2(filename, save_dir)
    #     return
    new_edges = torch.nonzero(adjacency, as_tuple=False).t().contiguous()
    new_edge_attr = torch.ones(new_edges.size(1), dtype=torch.float)
    whole_data.edge_index = torch.cat([whole_data.edge_index, new_edges], dim=1)
    whole_data.edge_attr = torch.cat([whole_data.edge_attr, new_edge_attr], dim=0)
    # Saving the result
    attr_dict = {"edge_type": whole_data.edge_attr.numpy()}
    save_graph_as_vtp_file(
        nodes=whole_data.x.numpy(),
        edges=whole_data.edge_index.T,
        attr_dict=attr_dict,
        filename=os.path.basename(filename),
        save_dir=save_dir,
    )


if __name__ == '__main__':
    import sys
    create_and_train_link_predictor()
    # for filename in glob.glob(os.path.join('/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/test_data', "*.vtp")):
    #     predict_new_links_and_save(filename)
        # sys.exit(0)
