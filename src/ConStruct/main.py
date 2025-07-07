try:
    import graph_tool as gt
except ModuleNotFoundError:
    print("Graph tool not found, non molecular datasets cannot be used")

import hydra
import pytorch_lightning as pl
import torch

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.warnings import PossibleUserWarning

# import utils
from src.ConStruct.diffusion_model_discrete_3d import Discrete3dDenoisingDiffusion
# from baseline import BaselineModel
from src.ConStruct.metrics.sampling_metrics import SamplingMetrics

torch.cuda.empty_cache()
# warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset

    # Choose the dataset
    if dataset_config.name == "atm":
        from src.ConStruct.datasets.atm_dataset import ATMDataModule, ATMDatasetInfos
        datamodule = ATMDataModule(cfg)
        dataset_infos = ATMDatasetInfos(datamodule)
    elif dataset_config.name == "cow":
        from src.ConStruct.datasets.cow_dataset import CoWDataModule, CoWDatasetInfos
        datamodule = CoWDataModule(cfg)
        dataset_infos = CoWDatasetInfos(datamodule)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg.dataset))

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

    if not hasattr(cfg.model, "is_baseline"):
        model = Discrete3dDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            val_sampling_metrics=val_sampling_metrics,
            test_sampling_metrics=test_sampling_metrics,
        )

        # need to ignore metrics because otherwise ddp tries to sync them
        params_to_ignore = [
            "module.model.dataset_infos",
            "module.model.val_sampling_metrics",
            "module.model.test_sampling_metrics",
        ]
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model, params_to_ignore
        )

        callbacks = []
        if cfg.train.save_model:
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"checkpoints/{cfg.general.name}",
                filename="{epoch}",
                monitor="val/epoch_NLL",
                save_top_k=5,
                mode="min",
                every_n_epochs=1,
            )
            last_ckpt_save = ModelCheckpoint(
                dirpath=f"checkpoints/{cfg.general.name}",
                filename="last",
                every_n_epochs=1,
            )
            callbacks.append(last_ckpt_save)
            callbacks.append(checkpoint_callback)

        is_debug_run = cfg.general.name == "debug"
        if is_debug_run:
            print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run")
            print(
                "[WARNING]: Run is called 'debug' -- it will run with only 1 CPU core"
            )
        use_gpu = torch.cuda.is_available() and not is_debug_run
        trainer = Trainer(
            gradient_clip_val=cfg.train.clip_grad,
            # strategy="ddp",
            accelerator="gpu" if use_gpu else "cpu",
            # devices=-1 if use_gpu else 1,
            devices=[0] if use_gpu else 1,
            max_epochs=cfg.train.n_epochs,
            check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
            fast_dev_run=is_debug_run,
            enable_progress_bar=False,
            callbacks=callbacks,
            log_every_n_steps=1 if is_debug_run else 50,
            logger=[],
        )

        if not cfg.general.test_only:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
            # if cfg.general.name not in ["debug", "test"]:
            trainer.test(model, datamodule=datamodule)
        else:
            # Start by evaluating test_only_path
            # for i in range(5):
            # For now, we only perform a single test run. Ideally, we can run multiple test runs with different seeds
            # The latter would make our results more robust.
            for i in range(1):
                new_seed = i * 1000
                pl.seed_everything(new_seed)
                cfg.train.seed = new_seed
                trainer.test(
                    model, datamodule=datamodule, ckpt_path=cfg.general.test_only
                )


if __name__ == "__main__":
    main()
