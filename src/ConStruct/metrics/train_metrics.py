import torch
from torchmetrics import Metric, MetricCollection, MeanSquaredError
from torch import Tensor
import wandb
import torch.nn as nn
from src.ConStruct.metrics.abstract_metrics import (
        CrossEntropyMetric,
)


class TrainLoss(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        # Including the radius loss information
        self.radius_loss = MeanSquaredError(sync_on_compute=False, dist_sync_on_step=False)

        self.lambda_train = lambda_train

    def forward(self, masked_pred, masked_true, log: bool):
        """Compute train metrics. Warning: the predictions and the true values are masked, but the relevant entriese
        need to be computed before calculating the loss

        masked_pred, masked_true: placeholders
        log : boolean."""

        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_X = masked_true.X[node_mask]  # q x 4
        masked_pred_X = masked_pred.X[node_mask]  # q x 4

        diag_mask = (
            ~torch.eye(n, device=node_mask.device, dtype=torch.bool)
            .unsqueeze(0)
            .repeat(bs, 1, 1)
        )
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]  # r x 5
        true_E = masked_true.E[edge_mask]  # r x 5

        # Let us check for the radius information
        loss_radius = 0.0
        if masked_true.radius is not None:
            masked_pred_radius = masked_pred.radius[edge_mask]
            true_radius = masked_true.radius[edge_mask]
            # Ensuring that the valid edges have a non zero radius
            assert (true_radius[true_E.argmax(dim=-1) > 0] > 0).all()
            # Computing the radius loss
            loss_radius = self.radius_loss(masked_pred_radius, true_radius) if true_radius.numel() > 0 else 0.0

        # Check that the masking is correct
        assert (true_X != 0.0).any(dim=-1).all()
        assert (true_E != 0.0).any(dim=-1).all()
        # assert (true_charges != 0.0).any(dim=-1).all()

        if masked_true.charges.numel() > 0:
            true_charges = masked_true.charges[node_mask]  # q x 3
            masked_pred_charges = masked_pred.charges[node_mask]  # q x 3

            loss_charges = self.charges_loss(masked_pred_charges, true_charges)
        else:
            loss_charges = 0.0

        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E) if true_E.numel() > 0 else 0.0
        loss_y = (
            self.y_loss(masked_pred.y, masked_true.y)
            if masked_true.y.numel() > 0
            else 0.0
        )


        batch_loss = (
            self.lambda_train[0] * loss_X
            + self.lambda_train[1] * loss_charges
            + self.lambda_train[2] * loss_E
            + self.lambda_train[3] * loss_y
            + self.lambda_train[2] * loss_radius  # Same scaling term as the loss for edges
        )

        to_log = (
            {
                "train_loss/X_CE": (
                    self.lambda_train[0] * self.node_loss.compute()
                    if true_X.numel() > 0
                    else -1
                ),
                "train_loss/charges_CE": (
                    self.lambda_train[1] * self.charges_loss.compute()
                    if masked_true.charges.numel() > 0
                    else -1
                ),
                "train_loss/E_CE": (
                    self.lambda_train[2] * self.edge_loss.compute()
                    if true_E.numel() > 0
                    else -1.0
                ),
                "train_loss/radius_MSE": (
                    self.lambda_train[2] * self.radius_loss.compute()
                    if true_E.numel() > 0
                    else -1.0
                ),
                "train_loss/y_CE": (
                    self.lambda_train[3] * self.y_loss.compute()
                    if masked_true.y.numel() > 0
                    else -1.0
                ),
                "train_loss/batch_loss": batch_loss.item(),
            }
            if log
            else None
        )

        if log and wandb.run:
            wandb.log(to_log, commit=True)
        return batch_loss, to_log

    def reset(self):
        for metric in [self.node_loss, self.charges_loss, self.edge_loss, self.y_loss, self.radius_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1.0
        )
        epoch_charges_loss = (
            self.charges_loss.compute().item() if self.charges_loss.total_samples > 0 else -1.0
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1.0
        )
        # Including the radius loss
        epoch_radius_loss = (
            self.radius_loss.compute() if self.radius_loss.total > 0 else -1.0
        )
        epoch_y_loss = (
            self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1.0
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/charges_CE": epoch_charges_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/radius_MSE": epoch_radius_loss,
            "train_epoch/y_CE": epoch_y_loss,
        }
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log


class TrainMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred, masked_true, log: bool):
        # Unpack Placeholders
        masked_pred_X, masked_pred_E = masked_pred.X, masked_pred.E
        true_X, true_E = masked_true.X, masked_true.E
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log["train/" + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log["train/" + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, local_rank):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log["train_epoch/epoch" + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log["train_epoch/epoch" + key] = val.item()

        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = round(val.item(), 3)
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = round(val.item(), 3)

        if local_rank == 0:
            print(
                f"Epoch {current_epoch}: {epoch_atom_metrics} -- {epoch_bond_metrics}"
            )

        return to_log


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {
            "H": HydrogenCE,
            "C": CarbonCE,
            "N": NitroCE,
            "O": OxyCE,
            "F": FluorCE,
            "B": BoronCE,
            "Br": BrCE,
            "Cl": ClCE,
            "I": IodineCE,
            "P": PhosphorusCE,
            "S": SulfurCE,
            "Se": SeCE,
            "Si": SiCE,
        }

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class CEPerClass(Metric):
    full_state_update = False

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


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)
