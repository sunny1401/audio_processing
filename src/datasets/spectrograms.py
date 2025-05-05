import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


def get_logmel_spectrogram(audio: torch.Tensor, sr: int, n_fft=2048, hop_length=512, n_mels=64) -> torch.Tensor:
    mel = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )(audio)
    log_mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    return log_mel


