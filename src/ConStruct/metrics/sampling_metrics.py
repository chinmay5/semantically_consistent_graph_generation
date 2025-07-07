from collections import Counter
from typing import List

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torchmetrics import MeanMetric, MaxMetric, Metric, MeanAbsoluteError

from src.ConStruct.metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)
from src.ConStruct.utils import PlaceHolder


class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test, train_loader=None, val_loader=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.test = test
        self.stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )

        self.disconnected = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.node_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()
        self.mean_no_cycles = MeanMetric()
        self.deg_histogram = DegreeHistogramMetric(self.stat)

        if dataset_infos.is_vascular:
            from src.ConStruct.metrics.sampling_vessel_metrics import VesselSamplingMetrics
            self.domain_metrics = VesselSamplingMetrics(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
            )
        else:
            from src.ConStruct.metrics.spectre_utils import (
                Comm20SamplingMetrics,
                GridSamplingMetrics,
                TreeSamplingMetrics,
                SBMSamplingMetrics,
                EgoSamplingMetrics,
                ProteinSamplingMetrics,
            )

            if dataset_infos.dataset_name == "comm-20":
                self.domain_metrics = Comm20SamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "grid":
                self.domain_metrics = GridSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "tree":
                self.domain_metrics = TreeSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "sbm":
                self.domain_metrics = SBMSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "protein":
                self.domain_metrics = ProteinSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "ego":
                self.domain_metrics = EgoSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_infos.dataset_name} not implemented"
                )

    def reset(self):
        for metric in [
            self.mean_components,
            self.max_components,
            self.disconnected,
            self.num_nodes_w1,
            self.node_types_tv,
            self.edge_types_tv,
            self.mean_no_cycles,
            self.deg_histogram,
        ]:
            metric.reset()
        self.domain_metrics.reset()

    def compute_all_metrics(self, generated_graphs: list, current_epoch, local_rank):
        """Compare statistics of the generated data with statistics of the val/test set"""

        # Number of nodes
        self.num_nodes_w1(number_nodes_distance(generated_graphs, self.stat.num_nodes))

        # Node types
        node_type_tv, node_tv_per_class = node_types_distance(
            generated_graphs, self.stat.atom_types, save_histogram=self.test
        )
        self.node_types_tv(node_type_tv)

        # Edge types
        edge_types_tv, edge_tv_per_class = bond_types_distance(
            generated_graphs, self.stat.bond_types, save_histogram=self.test
        )
        self.edge_types_tv(edge_types_tv)

        # Components
        device = self.disconnected.device
        connected_comp = connected_components(generated_graphs).to(device)
        self.disconnected(connected_comp > 1)
        self.mean_components(connected_comp)
        self.max_components(connected_comp)

        # No cycles
        no_cycles_ratios = no_cycles_ratio(generated_graphs).to(device)
        self.mean_no_cycles(no_cycles_ratios)


        # Degree distributions
        self.deg_histogram(generated_graphs)

        # Log
        key = "val_sampling" if not self.test else "test_sampling"
        (
            generated_deg_hist,
            diff_deg_hist,
            abs_diff_deg_hist,
        ) = self.deg_histogram.get_hists_to_log(target_hist=self.stat.degree_hist[0])
        to_log = {
            f"{key}/NumNodesW1": self.num_nodes_w1.compute().item(),
            f"{key}/NodeTypesTV": self.node_types_tv.compute().item(),
            f"{key}/EdgeTypesTV": self.edge_types_tv.compute().item(),
            f"{key}/Disconnected": self.disconnected.compute().item() * 100,
            f"{key}/MeanComponents": self.mean_components.compute().item(),
            f"{key}/MaxComponents": self.max_components.compute().item(),
            f"{key}/no_cycles": self.mean_no_cycles.compute().item(),
            f"{key}/generated_deg_hist": wandb.Histogram(
                np_histogram=generated_deg_hist
            ),
            f"{key}/diff_deg_hist": wandb.Histogram(np_histogram=diff_deg_hist),
            f"{key}/abs_diff_deg_hist": wandb.Histogram(np_histogram=abs_diff_deg_hist),
        }

        if self.domain_metrics is not None:
            log_domain_metrics = self.domain_metrics.forward(
                generated_graphs, current_epoch, local_rank
            )
            to_log.update(log_domain_metrics)
            ratios = self.compute_ratios_to_ref(
                reference_metrics=(
                    self.dataset_infos.test_reference_metrics
                    if self.test
                    else self.dataset_infos.val_reference_metrics
                ),
                generated_metrics=log_domain_metrics,
            )
            to_log.update(ratios)

        if wandb.run:
            wandb.log(to_log, commit=False)
        if local_rank == 0:
            print(
                f"Sampling metrics",
                {
                    key: (
                        round(val, 3)
                        if "hist" not in key
                        else [round(el, 3) for el in val.histogram]
                    )
                    for key, val in to_log.items()
                },
            )

        return to_log, edge_tv_per_class

    def compute_ratios_to_ref(self, reference_metrics, generated_metrics):
        def compute_ratio(generated, reference):
            # Protect against division by 0 (when ref is 0)
            return (
                (generated / reference)
                if reference != 0
                else (generated + 1e-8) / (reference + 1e-8)
            )

        ratios = {}
        log_key = "test" if self.test else "val"
        if self.dataset_infos.is_molecular:
            for key in ["fcd score"]:  # TODO: Add more metrics?
                ratios[f"{log_key}_ratio/{key}"] = compute_ratio(
                    generated=generated_metrics[f"{log_key}_sampling/{key}"],
                    reference=reference_metrics[f"val_sampling/{key}"],
                )
            ratios[f"{log_key}_ratio/average"] = sum(ratios.values()) / len(ratios)
        elif self.dataset_infos.is_tls:
            # TLS entries
            tls_ratios = {}
            metrics_to_ratio = [
                metric
                for metric in generated_metrics
                if (
                        "tls_metric" in metric
                        and "frac" not in metric
                        and "mean_tls_validity" not in metric
                )
            ]
            for key in metrics_to_ratio:
                tls_ratios[f"{log_key}_ratio/{key.split('/')[-1]}"] = compute_ratio(
                    generated=generated_metrics[key],
                    reference=reference_metrics[key],
                )
            tls_ratios[f"{log_key}_ratio/average_tls"] = sum(tls_ratios.values()) / len(
                tls_ratios
            )
            # Spectre entries
            spectre_ratios = {}
            for key, val in generated_metrics.items():
                if key in self.domain_metrics.metrics_list:
                    spectre_ratios[f"{log_key}_ratio/{key}"] = compute_ratio(
                        generated=generated_metrics[key],
                        reference=reference_metrics[key],
                    )
            spectre_ratios[f"{log_key}_ratio/average_spectre"] = sum(
                spectre_ratios.values()
            ) / len(spectre_ratios)
            # Merge
            ratios = {**tls_ratios, **spectre_ratios}
        else:  # spectre datasets
            for key, val in generated_metrics.items():
                if key in self.domain_metrics.metrics_list:
                    ratios[f"{log_key}_ratio/{key}"] = compute_ratio(
                        generated=generated_metrics[key],
                        reference=reference_metrics[key],
                    )
            ratios[f"{log_key}_ratio/average"] = sum(ratios.values()) / len(ratios)
        return ratios


def number_nodes_distance(generated_graphs: List[PlaceHolder], dataset_counts):
    """each element of generated graphs is a batch of graphs."""
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1, device=generated_graphs[0].X.device)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for batch in generated_graphs:
        num_nodes = torch.sum(batch.node_mask, dim=1)
        c.update(num_nodes.tolist())

    generated_n = counter_to_tensor(c).to(reference_n.device)
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(
        generated_graphs: List[PlaceHolder], target, save_histogram=False
):
    device = generated_graphs[0].X.device
    target = target.to(device)
    generated_distribution = torch.zeros_like(target)

    for batch in generated_graphs:
        unique, counts = torch.unique(batch.X[batch.X >= 0], return_counts=True)
        for u, c in zip(unique.to(device), counts.to(device)):
            generated_distribution[u] += c

    if save_histogram:
        np.save("generated_atom_types.npy", generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(
        generated_graphs: List[PlaceHolder], target, save_histogram=False
):
    device = generated_graphs[0].X.device
    target = target.to(device)
    generated_distribution = torch.zeros_like(target)

    for batch in generated_graphs:
        unique, counts = torch.unique(batch.E[batch.E >= 0], return_counts=True)
        for u, c in zip(unique.to(device), counts.to(device)):
            generated_distribution[u] += c

    if save_histogram:
        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class


def connected_components(generated_graphs: List[PlaceHolder]):
    all_num_components = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            n = torch.sum(mask)
            edge_mat = edge_mat[:n, :n]
            adj = (edge_mat > 0).int()
            num_components, _ = sp.csgraph.connected_components(adj.cpu().numpy())
            all_num_components.append(num_components)
    all_num_components = torch.tensor(
        all_num_components, device=generated_graphs[0].X.device
    )
    return all_num_components


def no_cycles_ratio(generated_graphs: List[PlaceHolder]):
    no_cycles_list = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            n = torch.sum(mask)
            edge_mat = edge_mat[:n, :n]
            adj = (edge_mat > 0).int()
            # nx_graph = nx.from_numpy_matrix(adj.cpu().numpy())
            nx_graph = nx.from_numpy_array(adj.cpu().numpy())
            no_cycles_list.append(nx.is_forest(nx_graph))
    no_cycles_tg = torch.tensor(no_cycles_list, device=generated_graphs[0].X.device)
    return no_cycles_tg



class DegreeHistogramMetric(Metric):
    def __init__(self, dataset_stats, upper_margin=5):
        super().__init__()
        self.max_possible_degree = max(dataset_stats.num_nodes.keys()) - 1
        max_target_degree = dataset_stats.degree_hist[1][-1]
        self.num_bins = (
                min(
                    max_target_degree + upper_margin + 1,  # +1 for last bin
                    self.max_possible_degree,
                )
                + 1  # +1 for degree 0
        )  # ,
        self.add_state(
            "histogram",
            default=torch.zeros(self.num_bins),
            dist_reduce_fx="sum",
        )

    def update(self, local_generated_graphs) -> None:
        local_hist = self.compute_deg_histogram(local_generated_graphs)
        self.histogram += local_hist

    def compute(self):
        """Return normalized histogram.

        Since it is in compute, self.histogram will be synced (reduced, i.e., summed) across all GPUs, and thus we can normalize it.
        """
        return self.histogram / self.histogram.sum()

    def compute_deg_histogram(self, generated_graphs: List[PlaceHolder]):
        generated_degrees = []
        for batch in generated_graphs:
            for edge_mat, mask in zip(batch.E, batch.node_mask):
                n = torch.sum(mask)
                edge_mat = edge_mat[:n, :n]
                adj = (edge_mat > 0).int()
                degrees = torch.sum(adj, dim=0)
                generated_degrees.extend(degrees.tolist())

        generated_hist = self.get_histogram(generated_degrees)

        return generated_hist

    def get_histogram(self, generated_degrees):
        generated_degrees = torch.tensor(
            generated_degrees, dtype=torch.float32, device=self.histogram.device
        )
        # torch.histogram does not work with CUDA. This function does
        gen_hist = torch.histc(
            generated_degrees,
            self.max_possible_degree + 1,
            min=0,
            max=self.max_possible_degree,
        )
        large_degrees_sum = torch.sum(gen_hist[self.num_bins - 1:])
        gen_hist = torch.cat(
            (gen_hist[: self.num_bins - 1], large_degrees_sum.unsqueeze(0))
        )

        # Not normalized because different local histograms will be summed later
        return gen_hist

    def get_hists_to_log(self, target_hist):
        self_normalized_hist = self.compute()  # get hist synced across GPUs

        # Output numpy objects to easen interface with wandb
        self_normalized_hist = self_normalized_hist.cpu().numpy()
        # pad shorter histogram
        if len(target_hist) < len(self_normalized_hist):
            padding = (0, len(self_normalized_hist) - len(target_hist))
            target_hist = np.pad(target_hist, padding)
        else:
            padding = (0, len(target_hist) - len(self_normalized_hist))
            self_normalized_hist = np.pad(self_normalized_hist, padding)

        # Output numpy objects to easen interface with wandb
        diff_hist = self_normalized_hist - target_hist
        assert diff_hist.sum() < 1e-3  # Assert both are normalized to the same value
        abs_diff_hist = np.abs(diff_hist)

        bin_edges = np.concatenate(
            (np.arange(self.num_bins), [self.max_possible_degree])
        )
        generated_hist_to_log = self_normalized_hist, bin_edges
        diff_hist_to_log = diff_hist, bin_edges
        abs_diff_hist_to_log = abs_diff_hist, bin_edges

        # Assert target hist sums to 1
        return generated_hist_to_log, diff_hist_to_log, abs_diff_hist_to_log


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples
