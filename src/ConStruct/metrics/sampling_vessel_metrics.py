from typing import List

import networkx as nx
import numpy as np
import torch
import torch_geometric
import wandb
from torch_geometric.data import Data

from src.ConStruct.analysis.dist_helper import compute_mmd, emd, gaussian_tv
from src.ConStruct.metrics.spectre_utils import SpectreSamplingMetrics


class VesselSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=[
                "degree",
            ],
        )
        self.train_vessel_graphs = self.loader_to_graphs(train_dataloader)
        self.val_vessel_graphs = self.loader_to_graphs(val_dataloader)

    def loader_to_graphs(self, loader):
        vessel_graphs = []
        for batch in loader:
            for vessel_graph in batch.to_data_list():
                vessel_graphs.append(vessel_graph)
        return vessel_graphs

    @classmethod
    def from_placeholder(cls, placeholder):
        adj = placeholder.E.cpu().numpy()
        if placeholder.node_mask is not None:
            n = torch.sum(placeholder.node_mask)
            adj = adj[:n, :n]
        nx_graph = nx.from_numpy_array(adj)
        # Delete weight on edges
        for _, _, data in nx_graph.edges(data=True):
            del data["weight"]
        return torch_geometric.utils.from_networkx(nx_graph)

    def forward(self, generated_graphs: list, current_epoch, local_rank):
        to_log = super().forward(generated_graphs, current_epoch, local_rank)
        if local_rank == 0:
            print(
                f"Computing Vessel sampling metrics between {sum([placeholder.X.shape[0] for placeholder in generated_graphs])} generated graphs and {len(self.val_graphs)}"
            )

        generated_vessel_graphs = []
        for batch in generated_graphs:
            graph_placeholders = batch.split()
            for placeholder in graph_placeholders:
                vessel_graph = VesselSamplingMetrics.from_placeholder(placeholder)
                generated_vessel_graphs.append(vessel_graph)

        # TLS features
        if local_rank == 0:
            print("Computing TLS features stats...")

        vessel_stats = compute_vessel_stats(
            generated_vessel_graphs,
            self.val_vessel_graphs,
            bins=10,
            compute_emd=self.compute_emd,
        )

        for key, value in vessel_stats.items():
            to_log[f"vessel_metrics/{key}"] = value
            if wandb.run:
                wandb.run.summary[f"vessel_metrics/{key}"] = value
        return to_log

    def reset(self):
        super().reset()


def compute_vessel_stats(generated_vessel_graphs, val_vessel_graphs, bins, compute_emd):
    """Compute vessel features for a set of graphs.

    Args:
        generated_vessel_graphs (list): List of Vessel graphs to compute the degree.
        val_vessel_graphs (list): List of Vessel graphs to compute the degree.

    Returns:

    """

    # Extract TLS features
    generated_degree_hists = vessel_graphs_to_degree_features_hists(generated_vessel_graphs, bins)
    val_degree_features_hists = vessel_graphs_to_degree_features_hists(val_vessel_graphs, bins)

    # Compute Vessel features stats
    vessel_stats = {}
    for key in generated_degree_hists.keys():
        generated_sample = [generated_degree_hists[key]]
        val_sample = [val_degree_features_hists[key]]
        if compute_emd:
            mmd_dist = compute_mmd(
                val_sample,
                generated_sample,
                kernel=emd,
            )
        else:
            mmd_dist = compute_mmd(
                val_sample,
                generated_sample,
                kernel=gaussian_tv,
            )
        vessel_stats[key] = mmd_dist

    return vessel_stats


def vessel_graphs_to_degree_features_hists(vessel_graphs: List[Data], bins):
    degree_list = []
    for vessel_graph in vessel_graphs:
        degree = torch_geometric.utils.degree(vessel_graph.edge_index[0])
        degree_list.extend(degree.tolist())

    vessel_features_grouped = {}
    for key in ('degree', ):
        vessel_features_grouped[key] = degree_list

    # Generate histograms
    vessel_hists = {}
    for key in vessel_features_grouped.keys():
        values_list = vessel_features_grouped[key]
        vessel_hists[key], _ = np.histogram(
            values_list, bins=bins, range=(0, 1), density=False
        )

    return vessel_features_grouped
