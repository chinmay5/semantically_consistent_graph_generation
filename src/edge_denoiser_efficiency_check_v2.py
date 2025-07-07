import glob
import os
from collections import Counter

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
from ConStruct.projector.projector_3d_utils import TreeProjector
from ConStruct.utils import PlaceHolder, save_graph_as_vtp_file
from sd_edit_connectivity_corrector import ATMTreeProjectorLite, ATMVanillaTreeProjectorLite, TreeProjectorLite


class EdgeTypeCounter(object):
    def __init__(self, E: torch.Tensor, mask: torch.Tensor):
        super(EdgeTypeCounter, self).__init__()
        edge_labels = torch.argmax(E, dim=-1)[mask]
        # Use a counter to compute each edge type
        self.edge_attr_list = Counter(edge_labels.tolist())

    def balanced_acc_stats(self, predicted_labels):
        predicted_counter = Counter(predicted_labels.tolist())
        # We compute the accuracy for each category
        acc = {}
        for key, value in predicted_counter.items():
            acc[key] = value / self.edge_attr_list[key]
        balanced_acc = sum(list(acc.values())) / len(acc)
        # print(f"Balanced accuracy = {balanced_acc}")
        return balanced_acc


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

def create_dataset(cfg: DictConfig):
    # Hacky-hack hack
    dataset_infos, device, model = model_bootstrap(cfg)
    start_time = 100
    model.to(device)
    target_folder = "/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/test_data"
    z_t_tensor = []
    for filename in glob.glob(os.path.join(target_folder, "*.vtp")):
        print("Processing", filename)
        data_sample = load_vtp(dataset_infos, filename)
        data_sample.pos = data_sample.pos.to(device)
        # # We put everything on the GPU
        data_sample = data_sample.device_as(data_sample.pos)
        data_sample.node_mask = data_sample.node_mask.to(device)
        # Hacky-hack hack
        start_time_tensor = torch.tensor([start_time]).reshape(1, 1).to(device)
        z_t = model.noise_model.apply_noise(data_sample, t_int=start_time_tensor)
        z_t_tensor.append((z_t, data_sample.E.detach(), os.path.basename(filename)))
        sampled_s = z_t.collapse(model.collapse_charges)
        # Save the graph
        save_vtp(sampled_s, cfg, 'incomplete_sample', os.path.basename(filename))
    return z_t_tensor



@hydra.main(version_base="1.3", config_path="./configs", config_name="config_temp")
def main(cfg: DictConfig):
    dataset_infos, device, model = model_bootstrap(cfg)
    # Now, we have the checkpoint loaded. Next, we shall load the VTP file
    z_t_tensor = create_dataset(cfg)
    expected_labels, predicted_labels, predicted_logits = [], [], []
    for z_t, E, filename in z_t_tensor:
    # for filename in ['/mnt/elephant/chinmay/ATM22/atm_link_pred/ATM_129_0000_graph_met.vtp']:
        start_time = 300
        batch_size = 1
        num_classes = E.shape[-1]
        model.to(device)

        # Hacky-hack hack
        start_time_tensor = torch.tensor([start_time]).reshape(1, 1).to(device)
        # All edges to predict including the background, that should not be predicted
        # remaining_edges_mask = (torch.argmax(E, dim=-1) != torch.argmax(z_t.E, dim=-1))

        # edge_counter = EdgeTypeCounter(E.detach().cpu(), remaining_edges_mask.detach().cpu())
        # Add it to the list
        z_t.t_int = start_time_tensor
        z_t.t = start_time_tensor.float() / model.noise_model.T

        baseline_edges = torch.sum(torch.argmax(z_t.E, dim=-1) !=0)
        print(f"Number of baseline edges = {baseline_edges}")

        # Save this noisy graph.
        # We will use it later on for our model generation

        for s_int in reversed(range(1, start_time + 1, model.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones(
                (batch_size, 1), dtype=torch.long, device=z_t.X.device
            )
            z_s = model.sample_zs_from_zt(z_t=z_t, s_int=s_array)
            pred = model.forward(z_s)
            if s_int == start_time:
                # rev_projector = ATMTreeProjectorLite(z_t, device)
                rev_projector = ATMVanillaTreeProjectorLite(z_t, device)
                # rev_projector = TreeProjectorLite(z_t)
            # Now filter the obtained sample
            rev_projector.project(z_s)  # In place operation
            z_t = z_s
        sampled_s = z_s.collapse(model.collapse_charges)
        # Make diagonal inf
        for i in range(E.shape[1]):
            pred.E[0, i, i, 0] = 1000

        # We save the graph to check validity later.
        # We already make sure to use the base filename.
        save_vtp(sampled_s, cfg, 'intermediate_results', filename)
        # We compare how many edges were correctly predicted
        # correct = torch.sum((torch.argmax(E, dim=-1) == sampled_s.E) & (torch.argmax(E, dim=-1) != 0))
        # total = torch.sum(torch.argmax(E, dim=-1) !=0)
        # print(f"Found {correct - baseline_edges}/{total - baseline_edges} edges")
        # accuracy += (correct - baseline_edges) / (total - baseline_edges)
        # We look at the new edges.
        # At this point, we have the mask for only those locations where the labels are matching.
        # The list contains only valid samples that might be lower than the GT number.
        # new_edge_mask = (torch.argmax(E, dim=-1) == sampled_s.E) & (torch.argmax(E, dim=-1) != 0) & remaining_edges_mask
        # avg_bal_acc_ours += edge_counter.balanced_acc_stats(torch.argmax(z_s.E, dim=-1).cpu()[new_edge_mask.cpu()])
        # edge_counter_list.append(edge_counter)
        # We just use the numpy alternative
        expected_labels.append(torch.argmax(E, dim=-1).flatten().cpu())
        predicted_labels.append(torch.argmax(z_s.E, dim=-1).flatten().cpu())
        predicted_logits.append(torch.softmax(pred.E, dim=-1).view(-1, num_classes).detach().cpu())
    predicted_labels_numpy = torch.cat(predicted_labels).numpy()
    expected_labels_numpy = torch.cat(expected_labels).numpy()
    predicted_logits_numpy = torch.cat(predicted_logits).numpy()
    # We use the normal sklearn library for computing the results
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        roc_auc_score
    )
    accuracy = accuracy_score(expected_labels_numpy, predicted_labels_numpy)
    balanced_acc = balanced_accuracy_score(expected_labels_numpy, predicted_labels_numpy)
    # Compute Macro-Averaged Metrics.
    macro_precision = precision_score(expected_labels_numpy, predicted_labels_numpy, average='macro', zero_division=0)
    macro_recall = recall_score(expected_labels_numpy, predicted_labels_numpy, average='macro', zero_division=0)
    macro_f1 = f1_score(expected_labels_numpy, predicted_labels_numpy, average='macro', zero_division=0)
    # We also compute the roc_auc_score
    auroc_score = roc_auc_score(expected_labels_numpy, predicted_logits_numpy, multi_class='ovr', average='macro')
    print("-------------------------------------------------------------------")
    print(f"Accuracy = {accuracy}")
    print(f"Balanced accuracy = {balanced_acc}")
    print(f"Macro Precision = {macro_precision}")
    print(f"Macro Recall = {macro_recall}")
    print(f"Macro F1 = {macro_f1}")
    print(f"AUROC = {auroc_score}")
    print("-------------------------------------------------------------------")
    # print("v2 balanced_acc = ", avg_bal_acc_ours / len(z_t_tensor))
    # balanced_acc = 0
    # for edge_counter in edge_counter_list:
    #     balanced_acc += edge_counter.balanced_acc_stats(predicted_labels_numpy)
    # print("Balanced accuracy = ", balanced_acc / len(edge_counter_list))
    # return accuracy / len(z_t_tensor)


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
