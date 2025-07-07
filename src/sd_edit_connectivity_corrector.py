import glob
import os
import shutil

import hydra
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import pyvista
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj

from ConStruct.diffusion_model_discrete_3d import Discrete3dDenoisingDiffusion
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from ConStruct.projector.projector_3d_utils import ATMTreeProjector, ATMLineGraphCheckAndFixModule, get_multi_class_adj_matrix, \
    ATMVanillaTreeProjector, ATMVanillaLineGraphCheck, TreeProjector, get_adj_matrix
from ConStruct.utils import PlaceHolder, save_graph_as_vtp_file


def load_vtp(dataset_info, filename):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    edge_attr = torch.tensor(np.asarray(vtk_data.cell_data['edge_type']), dtype=torch.float)  # Should always be float
    edges = edges.T
    graph_data = Data(x=torch.ones(nodes.size(0), dtype=torch.float), edge_index=edges, edge_attr=edge_attr, pos=nodes)
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

    # Adding this attribute for compatibility with the model
    graph_data.y = torch.zeros((1, 0), dtype=torch.float)

    X, node_mask = to_dense_batch(x=graph_data.x, batch=graph_data.batch)
    X = X.unsqueeze(-1)
    E = to_dense_adj(
        edge_index=graph_data.edge_index,
        batch=graph_data.batch,
        edge_attr=graph_data.edge_attr,
        max_num_nodes=X.size(1),
    )

    pos, node_mask2 = to_dense_batch(x=graph_data.pos, batch=graph_data.batch)
    assert torch.all(node_mask == node_mask2), "Node mask should be the same"
    # pos, node_mask = to_dense_batch(x=nodes, batch=None)
    # pos = pos.float()

    dense_data = PlaceHolder(X=X, charges=None, E=E.to(torch.long), node_mask=node_mask, y=graph_data.y, pos=pos)
    dense_data = dataset_info.to_one_hot(dense_data)
    dense_data.y = dense_data.X.new_zeros((X.shape[0], 0))
    dense_data = dense_data.device_as(X)
    return dense_data.mask()


def save_vtp(graph_info, cfg, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    for graph in graph_info.split():
        # nodes, edges, pos, num_node_types
        edges, pos, radius = graph.E, graph.pos, graph.radius
        # Let us get the data onto the cpu
        edges, pos = edges.cpu(), pos.cpu()
        edge_indices = torch.nonzero(edges, as_tuple=False).t()
        edge_attr = edges[edge_indices[0], edge_indices[1]]
        attr_dict = {"edge_type": edge_attr.numpy()}
        if len(edge_attr) == 0:
            print("Skipping since no edges in graph")
            continue
        # Unfortunately, we will always get some value for the radius.
        # However, if the network has not been supervised on the radius, it will produce random values.
        if cfg.dataset.use_radius:
            radius = radius.cpu()  # Shape of radius is bs x n x n
            rad = radius[edge_indices[0], edge_indices[1]]
            attr_dict["radius"] = rad.numpy()
        save_graph_as_vtp_file(
            nodes=pos.numpy(),
            edges=edge_indices.numpy().T,
            attr_dict=attr_dict,
            filename=filename,
            save_dir=save_dir,
        )


def convert_adj_to_graph(graph_info):
    assert graph_info.X.size(0) == 1, "Currently, we start with batch size of 1"
    for graph in graph_info.split():
        edges = graph.E.cpu()
        # Let us get the data onto the cpu
        edge_indices = torch.nonzero(edges, as_tuple=False).t()
        return edge_indices


def laplacian_matrix(adjacency_matrix):
    """
    Compute the Laplacian matrix of a graph given its adjacency matrix.
    """
    degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
    return degree_matrix - adjacency_matrix


def connectivity_loss(adjacency_matrix):
    """
    Encourage the graph to have a single connected component.
    """
    adjacency_matrix = torch.argmax(adjacency_matrix, dim=-1)
    assert adjacency_matrix.dim() == 2, "Currently, we start with a non-batch graph"
    L = laplacian_matrix(adjacency_matrix)
    eigenvalues = torch.linalg.eigvals(L.to(torch.float))  # Compute eigenvalues (symmetric matrix)
    tol = 1e-5
    return torch.sum(torch.abs(eigenvalues) < tol)  # Minimize this loss to encourage connectivity


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, z_T):
        batch_size = 1
        z_t = z_T
        for s_int in reversed(range(0, model.T, model.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones(
                (batch_size, 1), dtype=torch.long, device=z_t.X.device
            )
            z_s = model.sample_zs_from_zt(z_t=z_t, s_int=s_array)

            z_t = z_s
        # Save the intermediate results
        return z_t

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: use a surrogate gradient.
        # Here we use the identity (i.e. pass grad_output unchanged)
        # so that d(output)/d(input) is approximated as 1 everywhere.
        grad_input = grad_output.clone()
        return grad_input


# A helper function to apply the STEFunction.
def sampler_layer(model, z_T):
    return STEFunction.apply(model, z_T)


class LiteProjector:
    def __init__(self):
        pass

    def initialize_graph_from_adj(self):
        assert self.z_t_adj.shape[0] == 1, "Currently, we start with batch size of 1"
        num_nodes = self.z_t_adj.shape[1]
        nx_graph = nx.empty_graph(num_nodes)
        all_rows, all_cols = torch.nonzero(self.z_t_adj.squeeze(0), as_tuple=True)
        for row, col in zip(all_rows, all_cols):
            row, col = row.item(), col.item()
            if col > row:
                selected_label = self.z_t_adj[0, row, col].item()
                # Hard-coded for now. Needs to be fixed
                nx_graph.add_edge(row, col, label=selected_label, cands=[selected_label] * 4)
        # Make it undirected
        nx_graph = nx_graph.to_undirected()
        return nx_graph


# A light version of the ATM Projector
class ATMTreeProjectorLite(ATMTreeProjector, LiteProjector):

    def __init__(self, z_t: PlaceHolder, device: torch.device):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        self.atm_validity_check = ATMLineGraphCheckAndFixModule(device)
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for deletion.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_multi_class_adj_matrix(z_t)

        # add data structure where validity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            # We need to initialize our projector with the current state of the graph
            nx_graph = self.initialize_graph_from_adj()
            # Assign 3D coordinates to nodes
            node_positions = {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # Sometimes the graphs are just wrong. We can not do anything here
            if not self.valid_graph_fn_lite(nx_graph):
                raise ValueError("Invalid graph")
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][
                            (node_1_idx, node_2_idx)] = False  # None of the edges are blocked.


class ATMVanillaTreeProjectorLite(ATMVanillaTreeProjector, LiteProjector):

    def __init__(self, z_t: PlaceHolder, device: torch.device):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        self.atm_validity_check = ATMVanillaLineGraphCheck(device)
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for deletion.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_multi_class_adj_matrix(z_t)

        # assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist
        # add data structure where validity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            # We create a complete graph with the given number of nodes.
            nx_graph = self.initialize_graph_from_adj()
            # Assign 3D coordinates to nodes
            node_positions = {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An fully-connected graph is always a valid graph
            assert self.valid_graph_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][
                            (node_1_idx, node_2_idx)] = False  # None of the edges are blocked.


class TreeProjectorLite(TreeProjector, LiteProjector):
    def __init__(self, z_t: PlaceHolder):
        self.batch_size = z_t.X.shape[0]
        self.nx_graphs_list = []
        if self.can_block_edges:
            # Which edges of the graph should be blocked.
            # These edges would not be considered a candidate for addition.
            # The process improves sampling speed since we need not check these invalid edges in subsequent steps.
            self.blocked_edges = {graph_idx: {} for graph_idx in range(self.batch_size)}

        # initialize adjacency matrix and check no edges
        self.z_t_adj = get_adj_matrix(z_t)
        # We do not start with an empty graph. Hence, ignoring the statement.
        # assert (self.z_t_adj == 0).all()  # no edges in the planar limit dist

        # add data structure where planarity is checked
        for graph_idx in range(self.batch_size):
            # We iterate over each of the graphs and create an empty graph with node coordinate information.
            num_nodes = z_t.node_mask[graph_idx].sum()
            pos = z_t.pos[graph_idx, :num_nodes]
            nx_graph = self.initialize_graph_from_adj()
            # Assign 3D coordinates to nodes
            node_positions = {i: pos[i] for i in range(num_nodes)}
            nx.set_node_attributes(nx_graph, node_positions, name="pos")
            # An empty graph is always a valid graph
            assert self.valid_graph_fn(nx_graph)
            self.nx_graphs_list.append(nx_graph)
            # initialize block edge list
            if self.can_block_edges:
                for node_1_idx in range(num_nodes):
                    for node_2_idx in range(node_1_idx + 1, num_nodes):
                        self.blocked_edges[graph_idx][
                            (node_1_idx, node_2_idx)] = False  # None of the edges are blocked.


def num_connected_component_from_vtp(vtp_file):
    mesh = pyvista.read(vtp_file)
    # Extract edges (assuming the graph is stored as a polydata)
    edges = mesh.lines.reshape(-1, 3)[:, 1:]  # Extracting the edge pairs (ignoring first column)
    # Create a networkx graph
    G = nx.Graph()
    G.add_edges_from(edges)
    # Compute number of connected components
    num_components = nx.number_connected_components(G)
    return num_components


def num_connected_component_from_edges(edge_index: torch.Tensor) -> int:
    edges = edge_index.T.cpu().numpy()  # Extracting the edge pairs (ignoring first column)
    # Create a networkx graph
    G = nx.Graph()
    G.add_edges_from(edges)
    # Compute number of connected components
    num_components = nx.number_connected_components(G)
    return num_components


@hydra.main(version_base="1.3", config_path="./configs", config_name="config_temp")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset
    from ConStruct.datasets.atm_dataset import ATMDataModule, ATMDatasetInfos
    datamodule = ATMDataModule(cfg)
    dataset_infos = ATMDatasetInfos(datamodule)

    # Not needed but, put here for easy testing
    val_sampling_metrics = SamplingMetrics(
        dataset_infos,
        test=False,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )
    test_sampling_metrics = SamplingMetrics(
        dataset_infos,
        test=True,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.test_dataloader(),
    )

    model = Discrete3dDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        val_sampling_metrics=val_sampling_metrics,
        test_sampling_metrics=test_sampling_metrics,
    )
    checkpoint = torch.load(cfg.general.test_only)
    model.load_state_dict(checkpoint["state_dict"])

    # Now, we have the checkpoint loaded. Next, we shall load the VTP file
    target_folder = "/home/chinmayp/workspace/ConStruct/outputs/2025-02-20/18-51-34-atm-equivariant_False-transition_absorbing_edges-noising_None/generated_samples"
    for filename in glob.glob(os.path.join(target_folder, "*.vtp")):
        connected_counter_check = 0
        if num_connected_component_from_vtp(filename) > 1:
            # We try to connect the components for 10 times
            start_time = 250
            batch_size = 1
            # We iterate over the generated samples.
            # Hence, we initialize with the same filename to begin with and then, we update it
            intermediate_filename = filename
            while connected_counter_check < 1:
                try:
                    print("Processing", intermediate_filename)
                    data_sample = load_vtp(dataset_infos, intermediate_filename)
                    data_sample.pos = data_sample.pos.to(device)
                    # # We put everything on the GPU
                    data_sample = data_sample.device_as(data_sample.pos)
                    data_sample.node_mask = data_sample.node_mask.to(device)

                    model.to(device)

                    # Hacky-hack hack
                    start_time_tensor = torch.tensor([start_time]).reshape(1, 1).to(device)
                    z_t = model.noise_model.apply_noise(data_sample, t_int=start_time_tensor)

                    for s_int in reversed(range(1, start_time + 1, model.cfg.general.faster_sampling)):
                        s_array = s_int * torch.ones(
                            (batch_size, 1), dtype=torch.long, device=z_t.X.device
                        )
                        z_s = model.sample_zs_from_zt(z_t=z_t, s_int=s_array)
                        if s_int == start_time:
                            rev_projector = ATMTreeProjectorLite(z_t, device)
                        # Now filter the obtained sample
                        rev_projector.project(z_s)  # In place operation
                        z_t = z_s
                    sampled_s = z_s.collapse(model.collapse_charges)
                    # Save the graph
                    save_vtp(sampled_s, cfg, 'intermediate_results', os.path.basename(filename))
                    # Let us update the intermediate_filename variable
                    intermediate_filename = os.path.join("intermediate_results", os.path.basename(filename))
                    # Get the edge indices
                    # edge_indices = convert_adj_to_graph(sampled_s)
                    # save_vtp(sampled_s, cfg, 'check_something', os.path.basename(filename))
                    # Let us check if the components are already connected.
                    # Then, we can stop the process
                    if num_connected_component_from_vtp(
                            os.path.join("intermediate_results", os.path.basename(filename))) == 1:
                        # if num_connected_component_from_edges(edge_indices) == 1:
                        print(f"Managed to connect for {os.path.basename(filename)}")
                        break
                    connected_counter_check += 1
                    print(f"Failed to connect for {os.path.basename(filename)} --- attempt {connected_counter_check}")
                except Exception as e:
                    print(f"Error in processing {filename}: {e}")
            # We save the resulting graph
            save_vtp(sampled_s, cfg, 'check_something', os.path.basename(filename))
        else:
            # Just copy over the file to destination
            shutil.copy2(filename, os.path.join('check_something', os.path.basename(filename)))
    # Finally, we want to check on the destination folder how many connected components are there
    for filename in glob.glob(os.path.join('check_something', "*.vtp")):
        if num_connected_component_from_vtp(filename) > 1:
            print(f"Check {filename} has multiple connected components")


if __name__ == '__main__':
    main()
