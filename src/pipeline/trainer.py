import os
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Dict, List

import lightning as L

from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from eo_lib.pipeline.logger import CustomWandbLogger
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    GradientAccumulationScheduler,
    OnExceptionCheckpoint,
    LearningRateMonitor,
    ModelCheckpoint,
)
from ray.tune.integration.pytorch_lightning import (
    TuneReportCheckpointCallback,
)
from ray.train.lightning import RayDDPStrategy
from ray.train.lightning import RayLightningEnvironment


import torch
from yacs.config import CfgNode as CN
from utils.cuda import get_device
from utils.reproducibility import set_random_seed
from src import REPO_LOCATION


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

torch.set_float32_matmul_precision("medium")
#torch.cuda.set_device(1)


class ClearCacheCallback(Callback):
    """Callback to clear CUDA cache at the end of each training epoch.
    This can help manage memory usage.
    """

    def on_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


class StopOnNanGradientCallback(Callback):
    def on_after_backward(self, trainer, pl_module):
        if any(
            torch.isnan(param.grad).any()
            for param in pl_module.parameters()
            if param.grad is not None
        ):
            trainer.should_stop = True


class Trainer(L.Trainer):
    def __init__(
        self,
        accelerator: str,
        callback_list: List[Callable],
        default_root_dir: Path,
        device_count: str | int,
        epochs: int,
        experiment_name: str,
        project_name: str,
        logging_interval: str,
        params: Dict,
        seed: int,
        strategy: Callable | str,
        plugins: None | List[Callable] = None,
        use_wandblogger: bool = True,
        use_wandb_offline: bool = False,
    ):

        set_random_seed(seed=seed)
        default_root_dir.mkdir(parents=True, exist_ok=True)
        self.train_logger = self.init_logger(
            logs_dir=default_root_dir / "logs",
            project_name=project_name,
            experiment_name=experiment_name,
            use_wandblogger=use_wandblogger,
            use_wandb_offline=use_wandb_offline,
            log_checkpoint=logging_interval,
        )
        self.wandb_logger = None
        self.tensorboard_logger = None

        super().__init__(
            accelerator=accelerator,
            devices=device_count,
            callbacks=callback_list,
            logger=self.train_logger,
            max_epochs=epochs,
            default_root_dir=default_root_dir,
            strategy=strategy,
            plugins=plugins,
            **params,
        )

    @classmethod
    def check_log_checkpoint_and_precision(cls, log_checkpoint, precision):

        # values for assertion
        # taken from pytorch lightning trainer documentation
        assert log_checkpoint in {"all", True}
        assert precision in {
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "64",
            "32",
            "16",
            "bf16",
        }

    @classmethod
    def init_callbacks(cls, lr_scheduler_name, logging_interval):

        return [
            ClearCacheCallback(),
            DeviceStatsMonitor(cpu_stats=True),
            LearningRateMonitor(
                log_weight_decay=True if lr_scheduler_name in {"cyclic"} else False,
                log_momentum=True if lr_scheduler_name in {"sgd", "cyclic"} else False,
                logging_interval=logging_interval,
            ),
        ]

    @abstractmethod
    def init_logger(
        self,
        experiment_name: str,
        project_name: str,
        logs_dir: Path,
        use_wandblogger: bool,
        use_wandb_offline: bool,
        log_checkpoint: str | bool,
    ) -> Logger:
        raise NotImplementedError


class HpoTrainer(Trainer):
    def __init__(
        self,
        config: CN,
        metric: str = "loss",
        hw_device: str | None = None,
    ) -> None:

        Trainer.check_log_checkpoint_and_precision(
            log_checkpoint=config.logging.log_checkpoint,
            precision=config.params.precision,
        )

        hw_device = get_device() if not hw_device else hw_device
        if not hasattr(config.logging, "default_root_dir"):
            default_root_dir: Path = (
                REPO_LOCATION / "pl_logs" / config.logging.experiment_name
            )
        else:
            default_root_dir = Path(config.logging.default_root_dir)

        callback_list = Trainer.init_callbacks(
            lr_scheduler_name=config.lr_scheduler.name,
            logging_interval=config.logging.logging_interval,
        )
        callback_list += [
            TuneReportCheckpointCallback(
                {metric: f"val_{metric}"},
                filename="ray-tune-exp-{epoch:02d}-{val_loss:.4f}.ckpt",
                on="validation_end",
            ),
            StopOnNanGradientCallback(),
        ]

        super().__init__(
            accelerator="auto",
            callback_list=callback_list,
            default_root_dir=default_root_dir,
            device_count=config.device_count,
            devices=config.devices if hasattr(config, "devices") else "auto",
            epochs=config.epochs,
            experiment_name=config.logging.experiment_name,
            project_name=config.loggging.project_name,
            logging_interval=config.logging.logging_interval,
            params=dict(config.params),
            plugins=[RayLightningEnvironment()],
            seed=config.seed,
            strategy=RayDDPStrategy(find_unused_parameters=False),
        )

        self.config = config
        self.config.params.device_type = hw_device
        if self.is_global_zero:
            self.train_logger[-1].log_hyperparams(self.config.params)

    def init_logger(
        self,
        experiment_name: str,
        project_name: str, 
        logs_dir: Path,
        use_wandblogger: bool,
        use_wandb_offline: bool,
        log_checkpoint: str | bool,
    ) -> Logger:

        logger = [
            CSVLogger(save_dir=logs_dir / "csv_logs", name=experiment_name),
        ]

        try:
            if use_wandblogger:
                wandb_dir = logs_dir / "wandb_logs"
                wandb_dir.mkdir(parents=True, exist_ok=True)
                log_model = log_checkpoint if not use_wandb_offline else False
                os.environ["WANDB_MODE"] = (
                    "online" if not use_wandb_offline else "offline"
                )

                wandb_logger = CustomWandbLogger(
                    save_dir=wandb_dir,
                    project=project_name,
                    name=experiment_name,
                    offline=use_wandb_offline,
                    log_model=log_model,
                )
                logger.append(wandb_logger)
                self.wandb_logger = wandb_logger

        except ModuleNotFoundError:
            pass

        return logger


class PipelineTrainer(Trainer):
    def __init__(
        self,
        config: CN,
        hw_device: str | None = None,
    ) -> None:

        Trainer.check_log_checkpoint_and_precision(
            log_checkpoint=config.logging.log_checkpoint,
            precision=config.params.precision,
        )

        hw_device = get_device() if not hw_device else hw_device
        if not hasattr(config.logging, "default_root_dir"):
            default_root_dir: Path = (
                REPO_LOCATION / "pl_logs" / config.logging.experiment_name
            )
        else:
            default_root_dir = Path(config.logging.default_root_dir)

        callback_list = Trainer.init_callbacks(
            lr_scheduler_name=config.lr_scheduler.name,
            logging_interval=config.logging.logging_interval,
        )

        callback_list += [
            OnExceptionCheckpoint(default_root_dir / "checkpoints/exceptions"),
            ModelCheckpoint(
                dirpath=default_root_dir / "checkpoints" / "results",
                **dict(config.model_checkpoint),
            ),
        ]

        if config.use_early_stopping:
            callback_list.append(EarlyStopping(**dict(config.early_stopping)))
        if config.use_grad_accumulation_scheduler:
            callback_list.append(
                GradientAccumulationScheduler(
                    scheduling={
                        config.grad_accumulator_start_epoch: config.accumulate_grad_batches
                    }
                )
            )

        super().__init__(
            accelerator="auto",
            callback_list=callback_list,
            default_root_dir=default_root_dir,
            device_count=config.device_count,
            devices=config.devices if hasattr(config, "devices") else "auto",
            epochs=config.epochs,
            experiment_name=config.logging.experiment_name,
            project_name=config.loggging.project_name,
            logging_interval=config.logging.logging_interval,
            params=dict(config.params),
            seed=config.seed,
            strategy=(
                DDPStrategy(process_group_backend=config.ddp_strategy)
                if config.strategy == "ddp"
                else config.strategy
            ),
        )

        self.config = config
        self.config.params.device_type = hw_device
        if self.is_global_zero:
            self.train_logger[-1].log_hyperparams(self.config.params)

    def init_logger(
        self,
        experiment_name: str,
        project_name: str,
        logs_dir: Path,
        use_wandblogger: bool,
        use_wandb_offline: bool,
        log_checkpoint: str | bool,
    ) -> Logger:

        logger = [
            CSVLogger(save_dir=logs_dir / "csv_logs", name=experiment_name),
            TensorBoardLogger(
                save_dir=logs_dir / "tensorboard",
                name=experiment_name,
                log_graph=False,
            ),
        ]
        self.tensorboard_logger = logger[-1]
        try:
            if use_wandblogger:
                wandb_dir = logs_dir / "wandb_logs"
                wandb_dir.mkdir(parents=True, exist_ok=True)
                log_model = log_checkpoint if not use_wandb_offline else False
                os.environ["WANDB_MODE"] = (
                    "online" if not use_wandb_offline else "offline"
                )

                wandb_logger = CustomWandbLogger(
                    save_dir=wandb_dir,
                    project=project_name,
                    name=experiment_name,
                    offline=use_wandb_offline,
                    log_model=log_model,
                )
                logger.append(wandb_logger)
                self.wandb_logger = wandb_logger

        except ModuleNotFoundError:
            pass

        return logger
    
