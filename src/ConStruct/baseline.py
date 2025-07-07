import math
import os
import pickle
import time
from collections import Counter
from itertools import combinations_with_replacement, combinations

import wandb
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import numpy as np
import networkx as nx

from ConStruct import utils
from ConStruct.datasets.tls_dataset import CellGraph, PHENOTYPE_DECODER


class BaselineModel:
    def __init__(self, cfg, dataset_infos, val_sampling_metrics, test_sampling_metrics):
        if not dataset_infos.is_tls:
            raise ValueError("Baseline model only works for TLS datasets.")

        self.cfg = cfg
        self.dataset_infos = dataset_infos

        # Validation metrics
        self.val_sampling_metrics = val_sampling_metrics
        self.test_sampling_metrics = test_sampling_metrics

    def train(self, datamodule):
        train_graphs_list = []
        train_set = datamodule.train_dataloader()
        for batch in train_set:
            for graph in batch.to_data_list():
                cell_graph = CellGraph.from_torch_geometric(graph)
                train_graphs_list.append(cell_graph)
        num_train_graphs = len(train_graphs_list)

        # Get distribution of number of nodes in each graph in the dataset
        print("=== Computing probabilities of number of nodes ===")
        num_nodes_counter = Counter(
            [graph.number_of_nodes() for graph in train_graphs_list]
        )
        num_nodes_distribution = {
            num_nodes: counts / num_train_graphs
            for num_nodes, counts in num_nodes_counter.items()
        }
        print("=== Done ===")

        # Get distribution of nodes types
        print("=== Computing probabilities of node phenotypes ===")
        node_phenotypes_counter = Counter()
        for cell_graph in tqdm(train_graphs_list):
            graph_phenotypes = cell_graph.get_phenotypes_list()

            node_phenotypes_counter.update(graph_phenotypes)
            # node_phenotypes_counter.update(
            #     graph_phenotypes.reshape(
            #         -1,
            #     )
            # )
        total_num_nodes = sum(node_phenotypes_counter.values())
        node_phenotypes_distribution = {
            node_phenotype: counts / total_num_nodes
            for node_phenotype, counts in node_phenotypes_counter.items()
        }
        print("=== Done ===")

        # Get distribution of number of edges in each graph in the dataset
        print("=== Computing probabilities of edges between phenotype pairs ===")
        phenotype_pairwise_combinations = {
            combination: Counter()
            for combination in combinations_with_replacement(
                sorted(PHENOTYPE_DECODER), 2
            )  # sorted so that the combinations are lexicographicaly ordered
        }
        for cell_graph in tqdm(train_graphs_list):
            for node_1_idx, node_2_idx in combinations(
                range(cell_graph.number_of_nodes()), 2
            ):
                edge_vertices_phenotypes = tuple(
                    sorted(
                        (
                            cell_graph.get_cell_phenotype_from_idx(node_1_idx),
                            cell_graph.get_cell_phenotype_from_idx(node_2_idx),
                        )
                    )  # Two phenotypes sorted lexicographically
                )
                phenotype_pairwise_combinations[edge_vertices_phenotypes].update(
                    [cell_graph.has_edge(node_1_idx, node_2_idx)]
                )

        phenotype_pair_edge_probs = {
            phenotype_pair: (
                counter[1] / sum(counter.values()) if sum(counter.values()) > 0 else 0.0
            )
            for phenotype_pair, counter in phenotype_pairwise_combinations.items()
        }
        print("=== Done ===")

        # Store values
        self.num_nodes_distribution = num_nodes_distribution
        self.node_phenotypes_distribution = node_phenotypes_distribution
        self.phenotype_pair_edge_probs = phenotype_pair_edge_probs

    def sample_n_graphs(self, n_graphs_to_sample):
        print(f"=== Sampling {n_graphs_to_sample} graphs using baseline model ===")
        # Sample number of nodes for each graph
        num_nodes_sorted = sorted(self.num_nodes_distribution)
        num_nodes_probs = [
            self.num_nodes_distribution[num_nodes] for num_nodes in num_nodes_sorted
        ]
        num_nodes_sampled = np.random.choice(
            num_nodes_sorted, size=n_graphs_to_sample, p=num_nodes_probs
        )

        # Build each graph
        graph_list = []
        for graph_idx in tqdm(range(n_graphs_to_sample)):
            # Sample phenotypes for nodes of each graph
            num_nodes = num_nodes_sampled[graph_idx]
            node_phenotypes = np.random.choice(
                list(self.node_phenotypes_distribution.keys()),
                size=num_nodes,
                p=list(self.node_phenotypes_distribution.values()),
            )
            node_phenotypes_dict = {
                node_id: node_phenotype
                for node_id, node_phenotype in enumerate(node_phenotypes)
            }
            # Sample edges for each graph
            edge_list = []
            for node_1_idx, node_2_idx in combinations(range(num_nodes), 2):
                edge_vertices_phenotypes = tuple(
                    sorted(
                        (
                            node_phenotypes[node_1_idx],
                            node_phenotypes[node_2_idx],
                        )
                    )
                )  # sorted so that the combinations are lexicographicaly ordered
                edge_probability = self.phenotype_pair_edge_probs[
                    edge_vertices_phenotypes
                ]
                if np.random.binomial(1, edge_probability):
                    edge_list.append([node_1_idx, node_2_idx])

            # Build graph
            gen_cell_graph = CellGraph(nx.Graph())
            gen_cell_graph.add_nodes_from(range(num_nodes))
            nx.set_node_attributes(
                gen_cell_graph, values=node_phenotypes_dict, name="phenotype"
            )
            gen_cell_graph.add_edges_from(edge_list)

            graph_list.append(gen_cell_graph)

        return graph_list

    def test(self):
        utils.setup_wandb(self.cfg)
        generated_graphs = self.sample_n_graphs(
            self.cfg.general.final_model_samples_to_generate
        )
        gen_placeholders = []
        for gen_graph in generated_graphs:
            gen_placeholders.append(gen_graph.to_placeholder(self.dataset_infos))

        to_log, _ = self.test_sampling_metrics.compute_all_metrics(
            generated_graphs=gen_placeholders,
            current_epoch=0,
            local_rank=0,
        )

        if wandb.run:
            print("Logging to wandb")
            wandb.log(to_log)
        print(to_log)
        wandb.finish()
