import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split, DataLoader

class SpectrogramDS(Dataset):

    def __init__(self, input_dir: Path):

        self.input_files = [i.with_suffix(".pt") for i in input_dir.iterdir()]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        file_path = self.input_files[index]

        data = torch.load(file_path)
        return data["spectrogram"].transpose(2, 1), data["events"]


def get_dataloaders(input_dir: Path, batch_size: int, seed: int):
    ds = SpectrogramDS(input_dir=input_dir)

    val_len = int(0.15 * len(ds))
    train_len = len(ds) - val_len

    train_ds, val_ds = random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
