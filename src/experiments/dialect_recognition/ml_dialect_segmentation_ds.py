import argparse

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.datasets.audio_spectrograms import AudioToSpectrogramConvertor


CONSTANT_ENV = dict(
    Music=1, Noisy=2, Clean=3, Car=4
)

CONSTANT_LABEL = dict(
    Khaliji=0, 
    Najdi=1, 
    Hijazi=2, 
    ModernStandardArabic=3
)


class MLDialectSegmentation(AudioToSpectrogramConvertor):

    COLUMN_MAP = dict(
       FileName="source", 
       SegmentStart="start", 
       SegmentEnd="end", 
       SegmentID="seg_file", 
       SegmentLength="duration",
       Environment="env"
    )

    def __init__(
        self, input_dir: Path, 
        mode: str,
        output_dir: Path, 
        segment_duration: int, 
        sample_rate: int,
    ):
        super().__init__(
            input_dir, mode, output_dir, segment_duration, sample_rate
        )

    def load_labels(self, input_dir: Path):
        df = pd.read_csv(input_dir/ f"{self.mode}_segmentation.csv")
        if self.mode == "train":
            self.COLUMN_MAP["SpeakerDialect"]= "label"

    def __getitem__(self, index):

        wv_path = self.labels.source.iloc[index]
        label_df = self.labels.loc[self.labels.source == wv_path].reset_index(drop=True)

        wv = self.load_mono_audio(wv_path)
        duration = self.get_audio_duration(wv_path)

        self._process_audio(wv, duration, label_df, wv_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sr",
        "--sample-rate",
        default=16000,
        type=int,
        help="Sample rate to used to parse the audio"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Path to save output to"
    )
    parser.add_argument(
        "-i","--input-dir",
        type=Path,
        help="Path to read input from"   
    )
    parser.add_argument(
        "-sd",
        "--segment-duration",
        type=int,
        help="Length of an audio segment ot be processed",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "test"],
        type=str,
        default="Train or val mode for which data is being processed"
    )
    args = parser.parse_args()

    dialect_processing_ds = MLDialectSegmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate,
        mode=args.mode
    )   

    for i in tqdm(range(len(dialect_processing_ds)), desc="Processing segments"):
        dialect_processing_ds[i]