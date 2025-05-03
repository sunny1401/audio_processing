import logging
from abc import abstractmethod


from src.utils.cuda import get_device
from src.utils.reproducibility import set_random_seed
from src.utils.read_config import load_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Pipeline:
    def __init__(self) -> None:

        self._logger = logging.getLogger(Pipeline.__qualname__)

        device = get_device()
        self._logger.info(f"Device that will be used is : {device}")
        set_random_seed()

    def _build_datamodule(self):
        raise NotImplementedError

    @abstractmethod
    def _build_model(self):

        raise NotImplementedError

    @abstractmethod
    def train(self):
        self._logger.info("Starting training")

        if not hasattr(self, "trainer"):
            raise AttributeError("Trainer should be defined in _build_model")
        
        if not hasattr(self, "data_module_ins"):
            raise AttributeError("Please call the setup function")

        self.trainer.fit(
            train_dataloaders=self.data_module_ins.train_dataloader(),
            val_dataloaders=self.data_module_ins.val_dataloader(),
            ckpt_path=(
                self.model_cfg.trainer.checkpoint_path
                if self.model_cfg.trainer.load_from_checkpoint
                else "last"
            ),
        )

        self._logger.info("Logged Metrics:", self.trainer.logged_metrics)
        self._logger.info("Callback Metrics:", self.trainer.callback_metrics)

        self.trainer.logger.finalize("done")

        self._logger.info("Train process complete")

    def setup(self):
        self._build_datamodule()
        self._logger.info("Data Module Initialized")

        self.model = self._build_model()
        self._logger.info(f"Objective Loaded: {self.model}")

