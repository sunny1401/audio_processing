from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader


class FeatureExtractorClassification:

    def __init__(
        self, 
        ckpt_path: Path | str,
        save_dir: Path | str,
        module_path: Path,
        train_dl: DataLoader, 
        test_dl: DataLoader, 
        val_dl: Optional[DataLoader],
        **kwargs
    ):

        self.module_path = module_path
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.val_dl = val_dl
        self.model = self.load_model(ckpt_path, **kwargs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def load_model(self, ckpt_path: Path, **kwargs):
        raise NotImplementedError
    

    def extract_save_features(self, dataloader, split: str, return_for_segmentation: bool):
        raise NotImplementedError
    
    def forward(self, return_for_segmentation: bool = False):

        iter_list = [
            (self.train_dl, "train"),  (self.test_dl, "test"), (self.val_dl, "valid")
        ] if self.val_dl else [
            (self.train_dl, "train"),  (self.test_dl, "test")
        ]

        for items in iter_list:

            dl, split = items
            self.extract_save_features(dl, split, return_for_segmentation)
            print(f"Data for {split} saved at {self.save_dir / split}.npz")
