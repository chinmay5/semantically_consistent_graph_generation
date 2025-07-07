import copy
import math
import os
import pathlib
from collections import Counter, defaultdict
import pyvista

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveDuplicatedEdges, BaseTransform
from torch_geometric.utils import k_hop_subgraph, contains_isolated_nodes, \
    contains_self_loops
from tqdm import tqdm

from src.ConStruct.analysis.topological_analysis import SimplicialComplex
from src.ConStruct.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.ConStruct.datasets.dataset_utils import Statistics, save_pickle, load_pickle, compute_reference_metrics
from src.ConStruct.utils import PlaceHolder
from src.environment_setup import PROJECT_ROOT_DIR


def load_atm_file(graph_filename):
    """
    Loads the cow dataset as a file and returns the pytorch geometric graph object
    :param graph_filename: Absolute path of the file to load
    :return: pyG graph file
    """
    vtk_data = pyvista.read(graph_filename)
    pos = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    # We center the points
    pos = pos - torch.mean(pos, dim=0, keepdim=True)
    # Scale the point cloud
    max_distance = np.max(np.linalg.norm(pos, axis=1))
    pos = pos / max_distance
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    edge_attr = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.long)
    edges = edges.T
    x = torch.ones((pos.size(0), 1))
    # We will make the graph undirected
    edges, edge_attr = torch_geometric.utils.to_undirected(edge_index=edges, edge_attr=edge_attr, reduce="min")
    graph_data = Data(x=x, edge_index=edges, edge_attr=edge_attr, pos=pos)
    return graph_data


class NodeCoordJitter(BaseTransform):
    def __init__(self, mean=0, std=0.025):
        super(NodeCoordJitter, self).__init__()
        self.mean = mean
        self.stddev = std

    def __call__(self, data):
        # Updating the point cloud coordinates
        data.pos = data.pos + torch.randn_like(data.pos) * self.stddev + self.mean  # mu + sigma * N(0, 1)
        # Center the coordinates again
        # Make sure the jitter is low enough to not move it very far from COM
        data.pos = data.pos - torch.mean(data.pos, 0, True)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ATMGraphDataset(Dataset):
    def __init__(self, dataset_name, split, root, use_radius=False, transform=None, node_jitter=True):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.is_directed = False
        # self.subgraph_aug = hasattr(cfg, 'use_aug')
        base_load_dir = os.path.basename(root)
        self.atm_file_path = '/mnt/elephant/chinmay/ATM22/atm_metric_graphs_oversampled_mst_discrete_verified/train_data'
        self.preprocssed_graph_save_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', base_load_dir)
        self.raw_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', f'raw_{base_load_dir}')
        self.processed_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.preprocssed_graph_save_dir, exist_ok=True)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        indices_file = self.raw_paths[file_idx[self.split]]
        if not os.path.exists(indices_file):
            self.download()
        self.indices = pd.read_csv(indices_file)["filenames"].tolist()
        # We also encode the number of "atom" types.
        # In our case, these are the node degree types
        # Load any data object
        self.num_degree_categories = 1
        self.num_edge_categories = 5  # 4 edge types and 1 to represent missing connection
        # We can use custom names here if needed but degree in themselves are self-explanatory
        stats_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', f'statistics_{base_load_dir}', f"{split}.pkl")
        if not os.path.exists(stats_path):
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            self.compute_statistics(stats_path)
        self.statistics = load_pickle(stats_path)
        self.cleanup_transform = RemoveDuplicatedEdges(key='edge_attr', reduce='min')
        self.node_jitter = node_jitter
        self.coord_jitter = NodeCoordJitter()
        self.use_radius = use_radius

    @property
    def raw_file_names(self):
        return ['train_filename.csv', 'val_filename.csv', 'test_filename.csv']

    @property
    def raw_paths(self):
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        return [self.split + '_filename.csv']

    def download(self):
        print("Creating fresh data split")
        all_graph_filenames = []
        for base, _, filename_list in os.walk(self.atm_file_path):
            for filename in filename_list:
                if filename.endswith('met.vtp'):
                    all_graph_filenames.append(os.path.join(self.atm_file_path, base, os.path.basename(filename)))
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        num_graphs = len(all_graph_filenames)
        if num_graphs == 1:
            # This is debug mode
            print(f'Debug mode:- Dataset sizes: train 1, val 1, test 1')

            graph_data = load_atm_file(all_graph_filenames[0])
            graph_filename = all_graph_filenames[0]
            if not hasattr(graph_data, 'y'):
                # Provide a dummy graph level label
                graph_data.y = torch.zeros((1, 0), dtype=torch.float)
                torch.save(graph_data, os.path.join(self.preprocssed_graph_save_dir, os.path.basename(graph_filename)))
            train_filenames, val_filenames, test_filenames = [graph_filename], [graph_filename], [graph_filename]
        else:
            train_filenames, val_filenames, test_filenames = [], [], []
            test_len = int(round(num_graphs * 0.1))
            train_len = int(round((num_graphs - test_len) * 0.9))
            val_len = num_graphs - train_len - test_len
            # For 100 -> 81 train, 9 val and 10 test
            indices = torch.randperm(num_graphs, generator=g_cpu)
            print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]
            test_indices = indices[train_len + val_len:]

            for i, graph_filename in enumerate(all_graph_filenames):
                name = os.path.split(graph_filename)[1].replace("vtp", "pt")
                if i in train_indices:
                    train_filenames.append(name)
                elif i in val_indices:
                    val_filenames.append(name)
                elif i in test_indices:
                    test_filenames.append(name)
                else:
                    raise ValueError(f'Index {i} not in any split')
                # We have decided the splits. Now we do the last bit of processing
                graph_data = load_atm_file(graph_filename)
                graph_data.y = torch.zeros((1, 0), dtype=torch.float)
                torch.save(graph_data, os.path.join(self.preprocssed_graph_save_dir, name))

        # Convert the list to a pandas DataFrame
        train_data = pd.DataFrame({"filenames": train_filenames})
        val_data = pd.DataFrame({"filenames": val_filenames})
        test_data = pd.DataFrame({"filenames": test_filenames})
        # Save the DataFrame to a CSV file
        train_data.to_csv(self.raw_paths[0], index=False)
        val_data.to_csv(self.raw_paths[1], index=False)
        test_data.to_csv(self.raw_paths[2], index=False)

    def __getitem__(self, item):
        file_name = self.indices[item]
        graph_data = torch.load(os.path.join(self.preprocssed_graph_save_dir, file_name))
        if self.split == 'train':
            if self.transform is not None:
                graph_data = self.transform(graph_data)
            if self.node_jitter:
                graph_data = self.coord_jitter(graph_data)
        if not self.use_radius:
            graph_data.edge_radius = None
        graph_data.file_name = file_name
        return graph_data

    def __len__(self):
        return len(self.indices)
        # Sometimes, we return fewer samples for quick debugging
        # return 8

    def compute_statistics(self, stats_path):
        # We already have indices.
        print(f"Computing statistics for {self.split=}")
        num_nodes = Counter()
        edge_types = torch.zeros(self.num_edge_categories)
        # Not sure why the angles discretization is so large.
        all_edge_angles = np.zeros((self.num_degree_categories, 180 * 10 + 1))
        # Compute the bond lengths separately for each bond type
        all_edge_lengths = {idx: Counter() for idx in range(self.num_edge_categories)}
        # Let us also compute values for the betti numbers
        betti_val_dict = {idx: Counter() for idx in range(3)}  # betti 0, 1 and 2
        # Computing the degree information as well
        degree_list = []
        for file_name in tqdm(self.indices):
            graph_data = torch.load(f"{self.preprocssed_graph_save_dir}/{file_name}")
            # count edge types
            edge_types += F.one_hot(graph_data.edge_attr, num_classes=self.num_edge_categories).sum(dim=0)
            # count the number of nodes
            N = graph_data.x.size(0)
            num_nodes[N] += 1
            self.update_edge_length_info(graph_data, all_edge_lengths)
            self.update_edge_angle_info(graph_data, all_edge_angles)
            self.update_betti_vals(graph_data, betti_val_dict)
            degree_list.extend(torch_geometric.utils.degree(graph_data.edge_index[1]).numpy().tolist())

        edge_types = edge_types / edge_types.sum()
        edge_lengths = self.normalize_edge_lengths(all_edge_lengths)
        edge_angles = self.normalize_edge_angles(all_edge_angles)
        betti_vals = self.normalize_betti_vals(betti_val_dict)
        max_degree = max(degree_list)
        hist, bin_edges = np.histogram(
            degree_list, bins=np.arange(max_degree + 2), density=True
        )
        degree_hist = hist, bin_edges.astype(int)
        # We have computed all the statistics now
        stats = Statistics(num_nodes=num_nodes, atom_types=torch.tensor([1]), bond_types=edge_types,
                           bond_lengths=edge_lengths, degree_hist=degree_hist,
                           bond_angles=edge_angles, betti_vals=betti_vals)
        print(stats)
        save_pickle(stats, stats_path)

    def update_edge_length_info(self, graph_data, all_edge_lengths):
        cdists = torch.cdist(graph_data.pos.unsqueeze(0), graph_data.pos.unsqueeze(0)).squeeze(0)
        edge_distances = cdists[graph_data.edge_index[0], graph_data.edge_index[1]]
        for edge_type in range(self.num_edge_categories):
            # bond_type_mask = torch.argmax(graph_data.edge_attr, dim=1) == edge_type
            bond_type_mask = graph_data.edge_attr == edge_type
            distances_to_consider = edge_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_edge_lengths[edge_type][d.item()] += 1

    def normalize_edge_lengths(self, all_edge_lengths):
        for bond_type in range(self.num_edge_categories):
            s = sum(all_edge_lengths[bond_type].values())
            for d, count in all_edge_lengths[bond_type].items():
                all_edge_lengths[bond_type][d] = count / s
        return all_edge_lengths

    def update_edge_angle_info(self, graph_data, all_edge_angles):
        assert not torch.isnan(graph_data.pos).any()
        node_types = torch.argmax(graph_data.x, dim=1)
        for i in range(graph_data.x.size(0)):
            neighbors, _, _, _ = k_hop_subgraph(i, num_hops=1, edge_index=graph_data.edge_index,
                                                relabel_nodes=False, directed=self.is_directed,
                                                num_nodes=graph_data.x.size(0), flow='target_to_source')
            # All the degree one nodes are unfortunately skipped in this evaluation.
            # Hence, it is more about non-degree 1 nodes and their connectivity
            for j in neighbors:
                for k in neighbors:
                    if j == k or i == j or i == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(i, j, k)
                    a = graph_data.pos[j] - graph_data.pos[i]
                    b = graph_data.pos[k] - graph_data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6))
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)

                    all_edge_angles[node_types[i].item(), bin] += 1

    def normalize_edge_angles(self, all_edge_angles):
        s = all_edge_angles.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        all_bond_angles = all_edge_angles / s
        return all_bond_angles

    def update_betti_vals(self, graph_data, betti_val_dict):
        edges = graph_data.edge_index.T
        edge_list = edges.numpy().tolist()
        betti_val_info = SimplicialComplex(edge_list)
        for betti_number in [0, 1, 2]:
            # A counter is internally a dictionary
            val = betti_val_info.betti_number(betti_number)
            betti_val_dict[betti_number][val] += 1

    def normalize_betti_vals(self, betti_val_dict):
        """
        Structure is as follows
        {
        Betti 0:
            1 -> 10,
            2 -> 8
            3 -> 25
        Betti 1:
            1 -> 5
            2 -> 5
            ...
        }
        So, we are computing the probability distribution for this discrete random variable.
        """
        # Again, category 0 is the "pseudo connection"
        for betti_number in [0, 1, 2]:
            s = sum(betti_val_dict[betti_number].values())
            for component, count in betti_val_dict[betti_number].items():
                betti_val_dict[betti_number][component] = count / s
        return betti_val_dict


class ATMDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = ATMGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='train', root=cfg.dataset.datadir,
                                        use_radius=cfg.dataset.use_radius)
        val_dataset = ATMGraphDataset(dataset_name=self.cfg.dataset.name,
                                      split='val', root=cfg.dataset.datadir,
                                      use_radius=cfg.dataset.use_radius)
        test_dataset = ATMGraphDataset(dataset_name=self.cfg.dataset.name,
                                       split='test', root=cfg.dataset.datadir,
                                      use_radius=cfg.dataset.use_radius)
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}

        datasets_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        super().__init__(cfg, datasets=datasets_dict)


class ATMDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.statistics = datamodule.statistics
        self.name = 'vessel_graphs'
        self.num_atom_types = 1  # Absence of an atom is also a type
        super().complete_infos(datamodule.statistics)

        print("Distribution of number of nodes", self.n_nodes)
        np.savetxt('n_counts.txt', self.n_nodes.numpy())
        print("Distribution of edge types", self.edge_types)
        np.savetxt('edge_types.txt', self.edge_types.numpy())
        y_out = 0
        y_in = 1
        self.input_dims = PlaceHolder(X=1, E=self.edge_types.size(0), charges=0, y=y_in, pos=3)
        self.output_dims = PlaceHolder(X=1, E=self.edge_types.size(0), charges=0, y=y_out, pos=3)
        self.is_molecular = False
        self.is_tls = False
        self.is_vascular = True
        self.atom_types = torch.tensor([1])
        compute_reference_metrics(self, datamodule)

    def to_one_hot(self, pl):
        # Need to override specifically for the vessel dataset.
        # The node attributes are not present and we just ignore it in the one-hot encoding computation
        pl.E = F.one_hot(pl.E, num_classes=self.num_edge_types).float()
        if self.num_charge_types > 1:
            pl.charges = F.one_hot(
                pl.charges + 1, num_classes=self.num_charge_types
            ).float()
        else:
            pl.charges = pl.X.new_zeros((*pl.X.shape[:-1], 0))
        return pl.mask(pl.node_mask)

import hydra
@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def check_deg_c_N(cfg):
    datamodule = ATMDataModule(cfg)
    dataset_infos = ATMDatasetInfos(datamodule)
    print(dataset_infos)


def utility_fn(graph_data):
    self_loop = contains_self_loops(graph_data.edge_index)
    list_of_tuple_edges = graph_data.edge_index.T.numpy().tolist()
    edge_dict = defaultdict(int)
    for edges in list_of_tuple_edges:
        edge_dict[tuple(edges)] += 1
    dup_edges = [(edge, count) for edge, count in edge_dict.items()
                 if count > 1]
    return self_loop, dup_edges


def compute_min_max_pos(split):
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', 'pt_files')
    dataset = ATMGraphDataset(dataset_name='sample', split=split, root=save_path)
    min_val, max_val = float("inf"), -float("inf")
    radius_info = []
    num_node = []
    isolated_node = 0
    degree = []
    for idx in tqdm(range(len(dataset))):
        graph_data = dataset[idx]
        self_loop, count_arr = utility_fn(graph_data=graph_data)
        if self_loop:
            print(f"{graph_data=}")
        if len(count_arr) > 0:
            print(f"{count_arr=}")
        if idx <= 5:
            print(graph_data.pos)
            print(f"{torch.mean(graph_data.pos, dim=0)=}")
        min_val = min(torch.min(graph_data.pos), min_val)
        max_val = max(torch.max(graph_data.pos), max_val)
        radius_info.extend(graph_data.edge_attr.cpu().numpy().tolist())
        num_node.append(graph_data.x.size(0))
        isolated_node += contains_isolated_nodes(graph_data.edge_index)
        degree.extend(torch_geometric.utils.degree(graph_data.edge_index[1]).numpy().tolist())
    print(f"{split=} - {min_val=} and {max_val=}")
    print(f"{Counter(radius_info)=}")
    print(f"{sorted(Counter(num_node))=}")
    print(f"{sorted(Counter(degree))=}")
    print(f"{isolated_node=}")
    return dataset


def get_edge_attr_counts():
    class dummy_obj(object):
        def __init__(self):
            super(dummy_obj, self).__init__()

    # Now we use the dummy object for random configuration
    cfg = dummy_obj()
    cfg.use_aug = True
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', 'pt_files')
    dataset = ATMGraphDataset(dataset_name='sample', split='train', root=save_path, cfg=cfg)
    all_attr = []
    for idx in range(len(dataset)):
        edge_type = dataset[idx].edge_attr
        all_attr.append(edge_type)
    all_attr_tensor = torch.cat(all_attr)
    edge_type_sum = all_attr_tensor.sum(dim=0)
    print("Num Edge count distribution")
    print(edge_type_sum / edge_type_sum.sum())


if __name__ == '__main__':
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'atm', 'pt_files')
    dataset = ATMGraphDataset(dataset_name='atm', split='test', root=save_path)
    check_deg_c_N()
    compute_min_max_pos(split='train')
    # datasets = compute_min_max_pos(split='train')
    # from pyinstrument import Profiler
    #
    # with Profiler(interval=0.1) as profiler:
    #     save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessel', 'pt_files')
    #     datasets = compute_min_max_pos(split='val')
    #     print(f"{len(datasets)=}")
    #     # datasets = VesselGraphDataset(dataset_name='sample', split='train', root=save_path)
    #     print(datasets[0])
    #     print(datasets[0].y)
    #     print(datasets[0].pos)
    #     print(torch.mean(datasets[0].pos, dim=0))
    #     exec_configs()
    # profiler.print()
