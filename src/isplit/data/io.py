"""Audio I/O via soundfile only (no torchaudio -- see plan notes on its
I/O-layer deprecation and FFmpeg system dependency). Resampling via
scipy.signal.resample_poly keeps the DSP layer to numpy+scipy+soundfile.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a mono waveform as float32 in [-1, 1]. Multi-channel files are
    averaged down to mono. Resamples to `target_sr` if given and different
    from the file's native rate.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    sf.write(str(path), wav, sr)


def resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wav
    g = np.gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(wav, up, down).astype(np.float32)
