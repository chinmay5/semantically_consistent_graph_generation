import abc

from src.ConStruct.diffusion.distributions import DistributionNodes
from src.ConStruct.utils import PlaceHolder
import torch
import torch.nn.functional as F
from torch_geometric.data.lightning import LightningDataset


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def prepare_dataloader(self):
        self.dataloaders = {}
        self.dataloaders["train"] = self.train_dataloader()
        self.dataloaders["val"] = self.val_dataloader()
        self.dataloaders["test"] = self.test_dataloader()

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def atom_types(self):
        num_classes = None
        for data in self.dataloaders["train"]:
            num_classes = data.x.shape[1]

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders["train"]):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.dataloaders["train"]:
            num_classes = data.edge_attr.shape[1]

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.dataloaders["train"]):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    @abc.abstractmethod
    def to_one_hot(self, pl):
        pl.X = F.one_hot(pl.X, num_classes=self.num_atom_types).float()
        pl.E = F.one_hot(pl.E, num_classes=self.num_edge_types).float()
        if self.num_charge_types > 1:
            pl.charges = F.one_hot(
                pl.charges + 1, num_classes=self.num_charge_types
            ).float()
        else:
            pl.charges = pl.X.new_zeros((*pl.X.shape[:-1], 0))
        return pl.mask(pl.node_mask)

    def one_hot_charges(self, charges):
        if self.num_charge_types > 1:
            charges = F.one_hot(charges + 1, num_classes=self.num_charge_types).float()

        return charges

    def complete_infos(self, statistics):
        # atom and edge type information
        self.num_edge_types = len(statistics["train"].bond_types)
        self.edge_types = statistics["train"].bond_types
        if (
            statistics["train"].charge_types is None
            or len(statistics["train"].charge_types[0])
            == 1  # with only a single charge type, it is equivalent with no charges
        ):
            self.num_charge_types = 0
            self.charge_types = statistics["train"].atom_types.new_zeros(
                (*statistics["train"].atom_types.shape[:-1], 0)
            )
            self.charges_marginals = self.charge_types.new_zeros(0)
        else:
            self.num_charge_types = len(statistics["train"].charge_types[0])
            self.charge_types = statistics["train"].charge_types
            self.charges_marginals = (self.charge_types * self.atom_types[:, None]).sum(
                dim=0
            )

        # Train + val + test for n_nodes
        train_n_nodes = statistics["train"].num_nodes
        val_n_nodes = statistics["val"].num_nodes
        test_n_nodes = statistics["test"].num_nodes
        max_n_nodes = max(
            max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys())
        )
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value
        self.n_nodes = n_nodes / n_nodes.sum()

        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        # dimensions settings
        self.input_dims = PlaceHolder(
            X=self.num_atom_types,
            charges=self.num_charge_types,
            E=self.num_edge_types,
            y=1,
        )  # y=1 for the time information

        self.output_dims = PlaceHolder(
            X=self.num_atom_types,
            charges=self.num_charge_types,
            E=self.num_edge_types,
            y=0,
        )
