from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torch.utils.data import Dataset


class AudioToSpectrogramConvertor(Dataset):

    N_MELS: int = 64
    NNFT: int = 1024
    HOP_LENGTH: int = 512
    def __init__(self, input_dir: Path, mode: str, output_dir: Path, segment_duration: int, sample_rate: int):

        self.mode = mode
        self.labels = self.load_labels(input_dir)
        self.segment_duration = segment_duration
        self.output_dir = output_dir
        self.sr = sample_rate
        self.mel = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.NNFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            power=2.0
        )
        self.input_dir = input_dir

    @abstractmethod
    def load_labels(self, input_dir: Path):
        raise NotImplementedError

    @abstractmethod
    def process_labels(self, label_path: Path):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def get_audio_duration(self, path: str | Path) -> float:
        metadata = torchaudio.info(str(path))
        return metadata.num_frames / metadata.sample_rate

    def load_mono_audio(self, file_path: str | Path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sr:
            resampler = T.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def get_logmel_spectrogram(self, audio: torch.Tensor):
        min_required = self.NNFT
        if audio.shape[-1] < min_required:
            pad_len = min_required - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        return self.mel(audio)

    def _process_audio(self, wv, duration, labels: pd.DataFrame, base_path: Path):
        num_segments = int(np.ceil(duration / self.segment_duration))

        for idx in range(num_segments):
            seg_start = idx * self.segment_duration
            seg_end = min(seg_start + self.segment_duration, duration)
            segment = wv[:, int(seg_start * self.sr): int(seg_end * self.sr)]

            mel_spec_segment = self.get_logmel_spectrogram(segment)  # [1, mel, time]
            output_data = dict(spectrogram=mel_spec_segment)
            donot_save = False

            if self.mode == "train":
                df_seg = labels[(labels["start"] >= seg_start) & (labels["end"] <= seg_end)]

                events = []
                for _, row in df_seg.iterrows():
                    rel_start = float(row["start"] - seg_start)
                    rel_end = float(row["end"] - seg_start)
                    events.append(dict(start=rel_start, end=rel_end, label=int(row["label"])))

                if events:
                    output_data["events"] = events

                else:
                    donot_save = True

            if not donot_save:
                out_path = self.output_dir / f"{base_path.stem}_seg{idx:03d}.pt"
                torch.save(output_data, out_path)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError