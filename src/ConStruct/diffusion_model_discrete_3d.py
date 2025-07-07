import math
import os
import pickle
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb

from src.ConStruct import utils
from src.ConStruct.analysis.visualization import Visualizer
from src.ConStruct.datasets.adaptive_loader import effective_batch_size
from src.ConStruct.diffusion import diffusion_utils
# from ConStruct.metrics.sampling_molecular_metrics import Molecule
from src.ConStruct.diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
from src.ConStruct.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.ConStruct.diffusion.noise_model import (
    DiscreteUniformTransition,
    MarginalTransition,
    AbsorbingTransition,
    AbsorbingEdgesTransition,
)
from src.ConStruct.metrics.abstract_metrics import NLL
from src.ConStruct.metrics.abstract_metrics import TrainAbstractMetrics
from src.ConStruct.metrics.abstract_metrics import XKl, ChargesKl, EKl
from src.ConStruct.metrics.train_metrics import TrainLoss
from src.ConStruct.metrics.train_metrics import TrainMolecularMetrics
from src.ConStruct.models.equivariant_pos_transformer_model import EquivariantGraph3DTransformer
from src.ConStruct.models.pos_transformer_model import Graph3DTransformer
from src.ConStruct.projector.projector_3d_utils import BaseValidProjector, \
    CoWProjector, CoWVanillaProjector, ATMTreeProjector, \
    ATMVanillaTreeProjector

from src.ConStruct.utils import save_graphs


class Discrete3dDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos, val_sampling_metrics, test_sampling_metrics):
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        self.nodes_dist = dataset_infos.nodes_dist
        self.dataset_infos = dataset_infos
        if cfg.model.extra_features:
            self.extra_features = ExtraFeatures(cfg=cfg, dataset_infos=dataset_infos)
            self.input_dims = self.extra_features.update_input_dims(
                dataset_infos.input_dims
            )
        else:
            self.extra_features = DummyExtraFeatures()
            self.input_dims = dataset_infos.input_dims

        if cfg.model.extra_molecular_features and dataset_infos.is_molecular:
            self.domain_features = ExtraMolecularFeatures(dataset_infos)
            self.input_dims = self.domain_features.update_input_dims(self.input_dims)
        else:
            self.domain_features = DummyExtraFeatures()
        self.output_dims = dataset_infos.output_dims

        # Train metrics
        self.train_loss = TrainLoss(self.cfg.model.lambda_train)

        if dataset_infos.is_molecular:
            self.train_metrics = TrainMolecularMetrics(dataset_infos)
        else:
            self.train_metrics = TrainAbstractMetrics()

        # Validation metrics
        self.val_metrics = torchmetrics.MetricCollection(
            [
                XKl(),
                ChargesKl(),
                EKl(),
            ]
        )
        self.val_nll = NLL()
        self.val_sampling_metrics = val_sampling_metrics

        # Test metrics
        self.test_metrics = torchmetrics.MetricCollection(
            [
                XKl(),
                ChargesKl(),
                EKl(),
            ]
        )
        self.test_nll = NLL()
        self.test_sampling_metrics = test_sampling_metrics

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "val_sampling_metrics",
                "test_sampling_metrics",
                "dataset_infos",
            ]
        )

        if cfg.model.equivariant:
            print("Using Equivariant model")
            self.model = EquivariantGraph3DTransformer(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                dropout=cfg.model.dropout,
                dropout_in_and_out=cfg.model.dropout_in_and_out,
            )
        else:
            print("Using Non-Equivariant model")
            self.model = Graph3DTransformer(
                input_dims=self.input_dims,
                n_layers=cfg.model.n_layers,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=self.output_dims,
                dropout=cfg.model.dropout,
                dropout_in_and_out=cfg.model.dropout_in_and_out,
            )

        if cfg.model.transition == "uniform":
            self.noise_model = DiscreteUniformTransition(
                output_dims=self.output_dims, cfg=cfg
            )
        elif cfg.model.transition == "marginal":
            print(
                f"Marginal distribution of the classes: nodes: {self.dataset_infos.atom_types} --"
                f" edges: {self.dataset_infos.edge_types} --"
                f"charges: {self.dataset_infos.charges_marginals}"
            )

            self.noise_model = MarginalTransition(
                x_marginals=self.dataset_infos.atom_types,
                e_marginals=self.dataset_infos.edge_types,
                charges_marginals=self.dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
                cfg=cfg,
            )
        elif cfg.model.transition == "absorbing":
            print("Absorbing transition model")
            self.noise_model = AbsorbingTransition(
                cfg=cfg,
                output_dims=self.output_dims,
            )
        elif cfg.model.transition == "absorbing_edges":
            print("Absorbing transition model with edges")
            self.noise_model = AbsorbingEdgesTransition(
                cfg=cfg,
                x_marginals=self.dataset_infos.atom_types,
                e_marginals=self.dataset_infos.edge_types,
                charges_marginals=self.dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
            )
        else:
            assert ValueError(
                f"Transition type '{cfg.model.transition}' not implemented."
            )

        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

        self.visualizer = Visualizer(dataset_infos=dataset_infos)

        self.collapse_charges = (
            self.dataset_infos.collapse_charges
            if hasattr(self.dataset_infos, "collapse_charges")
            else None
        )
        self.total_proj_updates = 0  # Needed during the rebuttal operation

    def forward(self, z_t):
        assert z_t.node_mask is not None
        extra_features = self.extra_features(z_t)
        extra_domain_features = self.domain_features(z_t)

        # Need to copy to preserve dimensions in transition to z_{t-1} in sampling (prevent changing dimensions of z_t
        model_input = z_t.copy()
        model_input.X = torch.cat(
            (z_t.X, z_t.charges, extra_features.X, extra_domain_features.X), dim=2
        ).float()
        model_input.E = torch.cat(
            (z_t.E, extra_features.E, extra_domain_features.E), dim=3
        ).float()
        model_input.y = torch.hstack(
            (z_t.y, extra_features.y, extra_domain_features.y, z_t.t)
        ).float()

        return self.model(model_input)

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self.forward(z_t)

        loss, tl_log_dict = self.train_loss(
            masked_pred=pred,
            masked_true=dense_data,
            log=i % self.log_every_steps == 0,
        )

        tm_log_dict = self.train_metrics(
            masked_pred=pred,
            masked_true=dense_data,
            log=i % self.log_every_steps == 0,
        )
        if tl_log_dict is not None:
            self.log_dict(tl_log_dict, batch_size=self.BS)
        if tm_log_dict is not None:
            self.log_dict(tm_log_dict, batch_size=self.BS)

        return loss

    def on_fit_start(self) -> None:
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

        # Set a different seed for each GPU (initial seed gets the seed from pl.seed_everything)
        # torch.random.manual_seed(torch.initial_seed() + self.local_rank)

    def on_train_epoch_start(self) -> None:
        self.print("Starting epoch", self.current_epoch)
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.print(f"Train epoch {self.current_epoch} ends")
        tle_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch} finished: "
            f"X: {tle_log['train_epoch/x_CE'] :.2f} --"
            f" charges: {tle_log['train_epoch/charges_CE']:.2f} --"
            f" E: {tle_log['train_epoch/E_CE'] :.2f} --"
            f" radius: {tle_log['train_epoch/radius_MSE'] :.2f} --"
            f" y: {tle_log['train_epoch/y_CE'] :.2f} -- {time.time() - self.start_epoch_time:.1f}s "
        )
        self.log_dict(tle_log, batch_size=self.BS)
        tme_log = self.train_metrics.log_epoch_metrics(
            self.current_epoch, self.local_rank
        )
        if tme_log is not None:
            self.log_dict(tme_log, batch_size=self.BS)
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)
        if self.cfg.general.memory_summary:
            print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_metrics.reset()
        self.val_sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self.forward(z_t)
        nll, log_dict = self.compute_val_loss(
            pred, z_t, clean_data=dense_data, test=False
        )
        return {"loss": nll}, log_dict

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_metrics.compute()]
        log_dict = {
            "val/epoch_NLL": metrics[0],
            "val/X_kl": metrics[1]["XKl"] * self.T,
            "val/E_kl": metrics[1]["EKl"] * self.T,
            "val/charges_kl": metrics[1]["ChargesKl"] * self.T if not torch.any(
                metrics[1]["ChargesKl"].isnan()) else 0.0
        }
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True)
        if wandb.run:
            wandb.log(log_dict)

        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.2f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = "".join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        self.val_counter += 1
        if self.name == "debug" or (
                self.val_counter % self.cfg.general.sample_every_val == 0
                and self.current_epoch > 0
        ):
            self.print(f"Sampling start")
            start = time.time()
            gen = self.cfg.general
            samples = self.sample_n_graphs(
                samples_to_generate=math.ceil(
                    gen.samples_to_generate / max(self._trainer.num_devices, 1)
                ),
                chains_to_save=gen.chains_to_save if self.local_rank == 0 else 0,
                samples_to_save=gen.samples_to_save if self.local_rank == 0 else 0,
                test=False,
            )
            print(
                f"Sampled {len(samples)} batches on local rank {self.local_rank}. Sampling took {time.time() - start:.2f} seconds\n"
            )
            print(f"Computing sampling metrics on {self.local_rank}...")
            self.val_sampling_metrics.compute_all_metrics(
                generated_graphs=samples,
                current_epoch=self.current_epoch,
                local_rank=self.local_rank,
            )
        self.print(f"Val epoch {self.current_epoch} ends")


    def on_test_epoch_start(self) -> None:
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        # Set a different seed for each GPU (initial seed gets the seed from pl.seed_everything)
        # torch.random.manual_seed(torch.initial_seed() + self.local_rank)

        self.test_nll.reset()
        self.test_metrics.reset()
        self.test_sampling_metrics.reset()

    def test_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self.forward(z_t)
        nll, log_dict = self.compute_val_loss(
            pred, z_t, clean_data=dense_data, test=True
        )
        return {"loss": nll}, log_dict

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        metrics = [self.test_nll.compute(), self.test_metrics.compute()]
        test_nll = metrics[0]
        self.print(f"Test loss: {test_nll :.4f}")
        log_dict = {
            "test/epoch_NLL": metrics[0],
            "test/X_kl": metrics[1]["XKl"] * self.T,
            "test/E_kl": metrics[1]["EKl"] * self.T,
            "test/charges_kl": metrics[1]["ChargesKl"] * self.T if not torch.any(
                metrics[1]["ChargesKl"].isnan()) else 0.0
        }
        self.log_dict(log_dict, sync_dist=True)

        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.4f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = "".join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])

        if wandb.run:
            wandb.log(log_dict)

        print(f"Sampling start on GR{self.global_rank}")
        start = time.time()
        to_sample = math.ceil(
            self.cfg.general.final_model_samples_to_generate
            / max(self._trainer.num_devices, 1)
        )
        self.print(
            f"Samples to generate: {to_sample} for each of the {max(self._trainer.num_devices, 1)} devices"
        )
        self.print(f"Samples to save: {self.cfg.general.final_model_samples_to_save}")
        samples = self.sample_n_graphs(
            samples_to_generate=to_sample,
            chains_to_save=self.cfg.general.final_model_chains_to_save,
            samples_to_save=self.cfg.general.final_model_samples_to_save,
            test=True,
        )
        # Save the samples list as pickle to a file that depends on the local rank
        # This is needed to avoid overwriting the same file on different GPUs
        with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
            pickle.dump(samples, f)

        print("Saving the generated graphs")
        # This line is used to sync between gpus
        # self._trainer.strategy.barrier()
        self.trainer.strategy.barrier()
        save_graphs(samples, save_dir=os.path.join(os.getcwd(), "generated_samples"), cfg=self.cfg)
        print("Saved.")
        # print("Computing sampling metrics...")
        #
        # # Load the pickles of the other GPUs
        # samples = []
        # for i in range(self._trainer.num_devices):
        #     with open(f"generated_samples_rank{i}.pkl", "rb") as f:
        #         samples.extend(pickle.load(f))
        #
        # self.test_sampling_metrics.compute_all_metrics(
        #     generated_graphs=samples,
        #     current_epoch=self.current_epoch,
        #     local_rank=self.local_rank,
        # )
        print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")
        print(f"Test ends.")

        # Close wandb
        wandb.finish()

    def kl_prior(self, clean_data, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones(
            (clean_data.X.size(0), 1), dtype=torch.long, device=clean_data.X.device
        )
        Ts = self.T * ones
        Qtb = self.noise_model.get_Qt_bar(t_int=Ts)

        # Compute transition probabilities
        probX = clean_data.X @ Qtb.X + 1e-7  # (bs, n, dx_out)
        probE = clean_data.E @ Qtb.E.unsqueeze(1) + 1e-7  # (bs, n, n, de_out)
        probc = clean_data.charges @ Qtb.charges + 1e-7
        probX = probX / probX.sum(dim=-1, keepdims=True)
        probE = probE / probE.sum(dim=-1, keepdims=True)
        probc = probc / probc.sum(dim=-1, keepdims=True)
        assert probX.shape == clean_data.X.shape

        # bs, n, _ = probX.shape
        limit_dist = self.noise_model.get_limit_dist().device_as(probX)

        # Set masked rows to limit dist, so its KL to limit dist is 0 and doesn't contribute to loss
        probX[~node_mask] = limit_dist.X.float()
        diag_mask = ~torch.eye(
            node_mask.size(1), device=node_mask.device, dtype=torch.bool
        ).unsqueeze(0)
        probE[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = (
            limit_dist.E.float()
        )

        # Compute Kl divergences
        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist.X[None, None, :], reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(),
            target=limit_dist.E[None, None, None, :],
            reduction="none",
        )

        if limit_dist.charges.numel() > 0:
            probc[~node_mask] = limit_dist.charges.float()
            kl_distance_c = F.kl_div(
                input=probc.log(),
                target=limit_dist.charges[None, None, :],
                reduction="none",
            )
        else:
            kl_distance_c = torch.zeros_like(kl_distance_X)

        return (
                diffusion_utils.sum_except_batch(kl_distance_X)
                + diffusion_utils.sum_except_batch(kl_distance_E)
                + diffusion_utils.sum_except_batch(kl_distance_c)
        ).mean()

    def compute_Lt(self, clean_data, pred, z_t, s_int, node_mask, test):
        t_int = z_t.t_int
        pred = utils.PlaceHolder(
            X=F.softmax(pred.X, dim=-1),
            charges=F.softmax(pred.charges, dim=-1),
            E=F.softmax(pred.E, dim=-1),
            node_mask=clean_data.node_mask,
            y=None,
        )

        Qtb = self.noise_model.get_Qt_bar(z_t.t_int)
        Qsb = self.noise_model.get_Qt_bar(s_int)
        Qt = self.noise_model.get_Qt(t_int)

        # Compute distributions to compare with KL
        bs, n, d = clean_data.X.shape
        prob_true = diffusion_utils.posterior_distributions(
            clean_data=clean_data, noisy_data=z_t, Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(
            clean_data=pred, noisy_data=z_t, Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true = diffusion_utils.mask_distributions(prob_true, node_mask)
        prob_pred = diffusion_utils.mask_distributions(prob_pred, node_mask)
        prob_true.X = torch.log(prob_true.X + 1e-7)
        prob_true.E = torch.log(prob_true.E + 1e-7)
        if prob_true.charges is not None:
            prob_true.charges = torch.log(prob_true.charges + 1e-7)

        # Compute metrics
        metrics = (self.test_metrics if test else self.val_metrics)(
            prob_pred, prob_true
        )

        return self.T * (
                (metrics["XKl"] if not torch.any(torch.isnan(metrics["XKl"])) else 0.0)
                + (
                    metrics["ChargesKl"]
                    if not torch.any(torch.isnan(metrics["ChargesKl"]))
                    else 0.0
                )
                + (metrics["EKl"] if not torch.any(torch.isnan(metrics["EKl"])) else 0.0)
        )

    def compute_val_loss(self, pred, z_t, clean_data, test=False):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
        pred: (batch_size, n, total_features)
        noisy_data: dict
        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
        node_mask : (bs, n)
        Output: nll (size 1)
        """
        node_mask = z_t.node_mask
        t_int = z_t.t_int
        s_int = t_int - 1

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.nodes_dist.log_prob(N).mean()

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(clean_data, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(clean_data, pred, z_t, s_int, node_mask, test)

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        log_dict = {
            "kl prior": kl_prior,
            "Estimator loss terms": loss_all_t,
            "log_pn": log_pN,
            "test_nll" if test else "val_nll": nll,
        }

        return nll, log_dict

    @torch.no_grad()
    def sample_batch(
            self,
            n_nodes: list,
            number_chain_steps: int = 50,
            batch_id: int = 0,
            keep_chain: int = 0,
            save_final: int = 0,
            test=True,
    ):
        """
        :param batch_id: int
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        print(
            f"Sampling a batch with {len(n_nodes)} graphs on local rank {self.local_rank}."
            f" Saving {save_final} visualization and {keep_chain} full chains."
        )
        assert keep_chain >= 0
        assert save_final >= 0
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask)

        assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()
        assert number_chain_steps < self.T

        chains = utils.PlaceHolder(
            X=torch.zeros((number_chain_steps, keep_chain, n_max), dtype=torch.long),
            charges=torch.zeros(
                (number_chain_steps, keep_chain, n_max), dtype=torch.long
            ),
            E=torch.zeros((number_chain_steps, keep_chain, n_max, n_max)),
            y=None,
            pos=torch.zeros((number_chain_steps, keep_chain, n_max, 3), dtype=torch.float),
            radius=torch.zeros((number_chain_steps, keep_chain, n_max, n_max))
        )

        z_t = z_T

        # Create planarity preserving objects
        if self.cfg.model.rev_proj == "cow_proj":
            print("Using our multi-label projector")
            rev_projector = CoWProjector(z_t, z_t.node_mask.device)
        elif self.cfg.model.rev_proj == 'cow_proj_vanilla':
            print("CoW vanilla projector")
            rev_projector = CoWVanillaProjector(z_t, z_t.node_mask.device)
        # ATM specific
        elif self.cfg.model.rev_proj == "atm_proj":
            print("Using our ATM multi-label projector")
            rev_projector = ATMTreeProjector(z_t, z_t.node_mask.device)
        elif self.cfg.model.rev_proj == "atm_proj_vanilla":
            print("Using ATM Vanilla projector")
            rev_projector = ATMVanillaTreeProjector(z_t, z_t.node_mask.device)
        else:
            rev_projector = BaseValidProjector(z_t)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T, self.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones(
                (batch_size, 1), dtype=torch.long, device=z_t.X.device
            )
            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)

            # Planarity preserving
            if self.cfg.model.rev_proj:
                rev_projector.project(z_s)

            z_t = z_s

            # Save the first keep_chain graphs
            if (s_int * number_chain_steps) % self.T == 0:
                write_index = (
                        number_chain_steps - 1 - ((s_int * number_chain_steps) // self.T)
                )
                discrete_z_t = z_t.collapse(self.collapse_charges)
                chains.X[write_index] = discrete_z_t.X[:keep_chain]
                chains.E[write_index] = discrete_z_t.E[:keep_chain]
                chains.pos[write_index] = discrete_z_t.pos[:keep_chain]
                if discrete_z_t.charges.numel() > 0:
                    chains.charges[write_index] = discrete_z_t.charges[:keep_chain]
                if discrete_z_t.radius.numel() > 0:
                    chains.radius[write_index] = discrete_z_t.radius[:keep_chain]

        # Extra information requested during rebuttal
        # print(f"Total number of corrections for the batch = {rev_projector.top_cow_validity_check.counter}")
        # self.total_proj_updates += rev_projector.top_cow_validity_check.counter

        # Sample final data
        sampled_s = z_t.collapse(self.collapse_charges)
        X, E, charges, Pos, radius = sampled_s.X, sampled_s.E, sampled_s.charges, sampled_s.pos, sampled_s.radius
        # X, E, charges = discrete_z_s.X, discrete_z_s.E, discrete_z_s.charges

        # Prepare the chain for saving
        if keep_chain > 0:
            chains.X[-1] = X[
                           :keep_chain
                           ]  # Overwrite last frame with the resulting X, E
            chains.E[-1] = E[:keep_chain]
            if sampled_s.charges.numel() > 0:
                chains.charges[-1] = charges[:keep_chain]
            if sampled_s.pos != None:
                chains.pos[-1] = sampled_s.pos[:keep_chain]
            if sampled_s.radius != None:
                chains.radius[-1] = sampled_s.radius[:keep_chain]

        final_batch = sampled_s

        # Visualize chains
        if keep_chain > 0:
            self.print("Batch sampled. Visualizing chains starts!")
            chains_path = os.path.join(
                os.getcwd(),
                f"chains/epoch{self.current_epoch}/",
                f"batch{batch_id}_GR{self.global_rank}",
            )
            os.makedirs(chains_path, exist_ok=True)

            self.visualizer.visualize_chains(
                chains=chains,
                num_nodes=n_nodes,
                chain_path=chains_path,
                batch_id=batch_id,
                local_rank=self.local_rank,
            )

        # Visualize the final graphs
        current_path = os.getcwd()
        result_path = os.path.join(
            current_path,
            f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
        )
        self.visualizer.visualize(
            path=result_path,
            graphs=final_batch,
            atom_decoder=(
                self.dataset_infos.atom_decoder
                if hasattr(self.dataset_infos, "atom_decoder")
                else None
            ),
            num_graphs_to_visualize=save_final,
        )
        self.print("Done.")

        # During testing, move the graphs to cpu so that they can be aggregated for metrics computation
        if test:
            final_batch = final_batch.device_as(torch.zeros(1, device="cpu"))

        return final_batch

    def sample_zs_from_zt(self, z_t, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        pred = self.forward(z_t)
        return self.noise_model.sample_zs_from_zt_and_pred(
            z_t=z_t, pred=pred, s_int=s_int
        )

    def sample_n_graphs(
            self,
            samples_to_generate: int,
            chains_to_save: int,
            samples_to_save: int,
            test: bool,
    ):
        if samples_to_generate <= 0:
            return []

        chains_left_to_save = chains_to_save

        samples = []
        # The first graphs are sampled without sorting the sizes, so that the visualizations are not biased
        first_sampling = min(samples_to_generate, max(samples_to_save, chains_to_save))
        if first_sampling > 0:
            n_nodes = self.nodes_dist.sample_n(first_sampling, self.device)
            current_max_size = 0
            current_n_list = []
            for i, n in enumerate(n_nodes):
                potential_max_size = max(current_max_size, n)
                if self.cfg.dataset.adaptive_loader:
                    potential_ebs = effective_batch_size(
                        potential_max_size,
                        self.cfg.train.reference_batch_size,
                        sampling=True,
                    )
                else:
                    potential_ebs = int(
                        1.8
                        * self.cfg.train.batch_size  # 1.8 because No need to make a backward pass
                    )
                if potential_ebs > len(current_n_list) or len(current_n_list) == 0:
                    current_n_list.append(n)
                    current_max_size = potential_max_size
                else:
                    chains_save = max(min(chains_left_to_save, len(current_n_list)), 0)
                    samples.append(
                        self.sample_batch(
                            n_nodes=current_n_list,
                            batch_id=i,
                            save_final=len(current_n_list),
                            keep_chain=chains_save,
                            number_chain_steps=self.number_chain_steps,
                            test=test,
                        )
                    )
                    chains_left_to_save -= chains_save
                    current_n_list = [n]
                    current_max_size = n
            chains_save = max(min(chains_left_to_save, len(current_n_list)), 0)
            samples.append(
                self.sample_batch(
                    n_nodes=current_n_list,
                    batch_id=i + 1,
                    save_final=len(current_n_list),
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                    test=test,
                )
            )
            if samples_to_generate - first_sampling <= 0:
                return samples

        n_nodes = self.nodes_dist.sample_n(
            samples_to_generate - first_sampling, self.device
        )

        if self.cfg.dataset.adaptive_loader:
            # The remaining graphs are sampled in decreasing graph size
            n_nodes = torch.sort(n_nodes, descending=True)[0]

        max_size = 0
        current_n_list = []
        for i, n in enumerate(n_nodes):
            max_size = max(max_size, n)
            potential_ebs = (
                effective_batch_size(
                    max_size, self.cfg.train.reference_batch_size, sampling=True
                )
                if self.cfg.dataset.adaptive_loader
                else 1.8 * self.cfg.train.batch_size
            )
            if potential_ebs > len(current_n_list) or len(current_n_list) == 0:
                current_n_list.append(n)
            else:
                samples.append(self.sample_batch(n_nodes=current_n_list, test=test))
                current_n_list = [n]
                max_size = n
        samples.append(self.sample_batch(n_nodes=current_n_list, test=test))
        return samples
