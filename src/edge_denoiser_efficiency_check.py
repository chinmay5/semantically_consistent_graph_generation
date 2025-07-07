import glob
import os

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
from ConStruct.utils import PlaceHolder, save_graph_as_vtp_file
from sd_edit_connectivity_corrector import ATMTreeProjectorLite, ATMVanillaTreeProjectorLite, TreeProjectorLite


def load_vtp(dataset_info, filename):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    if 'radius' in vtk_data.cell_data.keys():
        edge_attr = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)  # Should always be float
    elif 'edge_type' in vtk_data.cell_data.keys():
        edge_attr = torch.tensor(np.asarray(vtk_data.cell_data['edge_type']), dtype=torch.float)
    else:
        raise ValueError("No edge attribute found")
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

    # Normalizing the positions
    graph_data.pos = graph_data.pos - torch.mean(graph_data.pos, dim=0, keepdim=True)
    # Scale the point cloud
    max_distance = np.max(np.linalg.norm(graph_data.pos, axis=1))
    graph_data.pos = graph_data.pos / max_distance

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

@hydra.main(version_base="1.3", config_path="./configs", config_name="config_temp")
def create_dataset(cfg: DictConfig):
    # Hacky-hack hack
    dataset_infos, device, model = model_bootstrap(cfg)
    start_time = 150
    model.to(device)
    target_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/test_data"
    for filename in glob.glob(os.path.join(target_folder, "*.vtp")):
        batch_size = 1
        print("Processing", filename)
        data_sample = load_vtp(dataset_infos, filename)
        data_sample.pos = data_sample.pos.to(device)
        # # We put everything on the GPU
        data_sample = data_sample.device_as(data_sample.pos)
        data_sample.node_mask = data_sample.node_mask.to(device)
        # Hacky-hack hack
        start_time_tensor = torch.tensor([start_time]).reshape(1, 1).to(device)
        z_t = model.noise_model.apply_noise(data_sample, t_int=start_time_tensor)
        sampled_s = z_t.collapse(model.collapse_charges)
        # Save the graph
        save_vtp(sampled_s, cfg, '/mnt/elephant/chinmay/ATM22/atm_link_pred', os.path.basename(filename))



@hydra.main(version_base="1.3", config_path="./configs", config_name="config_temp")
def main(cfg: DictConfig):
    dataset_infos, device, model = model_bootstrap(cfg)

    # Now, we have the checkpoint loaded. Next, we shall load the VTP file
    target_folder = "/mnt/elephant/chinmay/ATM22/atm_link_pred"
    for filename in glob.glob(os.path.join(target_folder, "*.vtp")):
    # for filename in ['/mnt/elephant/chinmay/ATM22/atm_link_pred/ATM_129_0000_graph_met.vtp']:
        start_time = 200
        batch_size = 1
        print("Processing", filename)
        data_sample = load_vtp(dataset_infos, filename)
        data_sample.pos = data_sample.pos.to(device)
        # # We put everything on the GPU
        data_sample = data_sample.device_as(data_sample.pos)
        data_sample.node_mask = data_sample.node_mask.to(device)

        model.to(device)

        # Hacky-hack hack
        start_time_tensor = torch.tensor([start_time]).reshape(1, 1).to(device)
        # Remove this commented line later on
        # z_t = model.noise_model.apply_noise(data_sample, t_int=start_time_tensor)
        z_t = data_sample
        z_t.t_int = start_time_tensor
        z_t.t = start_time_tensor.float() / model.noise_model.T

        # Save this noisy graph.
        # We will use it later on for our model generation

        for s_int in reversed(range(1, start_time + 1, model.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones(
                (batch_size, 1), dtype=torch.long, device=z_t.X.device
            )
            z_s = model.sample_zs_from_zt(z_t=z_t, s_int=s_array)
            if s_int == start_time:
                rev_projector = ATMTreeProjectorLite(z_t, device)
                # rev_projector = ATMVanillaTreeProjectorLite(z_t, device)
                # rev_projector = TreeProjectorLite(z_t)
            # Now filter the obtained sample
            rev_projector.project(z_s)  # In place operation
            z_t = z_s
        sampled_s = z_s.collapse(model.collapse_charges)
        # Save the graph
        save_vtp(sampled_s, cfg, 'check_something', os.path.basename(filename))


def model_bootstrap(cfg):
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
    return dataset_infos, device, model


if __name__ == '__main__':
    # create_dataset()
    main()
