from argparse import ArgumentParser
from pathlib import Path
from yacs.config import CfgNode as CN

from src.experiments.dialect_recognition.dataset import get_dataloaders
from src.pipeline.base import Pipeline
from src.pipeline.training import PretrainingPipeline
from src.models.yoho import YOHO, yoho_detection_loss


class YohoLitModel(PretrainingPipeline):

    def __init__(self, batch_size: int, optimizer_details: CN, lr_scheduler_details: CN, model_cfg: CN, metric: str = "loss", automatic_optimization: bool = True):
        super().__init__(batch_size, optimizer_details, lr_scheduler_details, metric, automatic_optimization)

        self.model = YOHO(
            model_cfg.data.in_shape,
            model_cfg.data.num_classes
        )

    def forward(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = yoho_detection_loss(preds=outputs, targets=labels, num_classes=self.model.num_classes)
        return loss


class YohoPipeline(Pipeline):

    def __init__(self, model_cfg: Path, trainer_type: str = "pipeline") -> None:
        super().__init__(model_cfg, trainer_type)
        self.model = YohoLitModel(
            batch_size = self.trainer_params.data.batch_size,
            optimizer_details=self.trainer_params.optimizer_details,
            lr_scheduler_details=self.trainer_params.lr_scheduler,
            model_cfg=self.trainer_params
        )

    def build_model(self):
        return self.model

    def build_ds(self):
        return get_dataloaders(
            input_dir=self.trainer_params.data.input_dir, 
            batch_size=self.trainer_params.data.batch_size, 
            seed=self.trainer_params.data.seed
        )




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-mcfg", "--model-cfg", required=True, type=Path)
    args = parser.parse_args()

    pipeline = YohoPipeline(args.model_cfg)
    train_dl, val_dl = pipeline.build_ds()
    pipeline.train(train_dl, val_dl)
