"""On-disk feature cache: one `.npz` per (encoder, split, utterance[, condition]),
with one array per layer, stored as float16. `.npz` + a plain path convention
was chosen over h5py to avoid libhdf5/locking friction on Windows at this data
scale (see plan notes).
"""

from pathlib import Path

import numpy as np


def cache_path(
    cache_dir: str | Path,
    encoder_name: str,
    split: str,
    utt_id: str,
    condition: str | None = None,
) -> Path:
    """condition (e.g. 'noise=n1_snr=10.0' or 'channel=telephone') distinguishes
    synthetically-augmented versions of the same base utterance so environment/
    channel pairs cache both conditions separately.
    """
    stem = utt_id if condition is None else f"{utt_id}__{condition}"
    return Path(cache_dir) / encoder_name / split / f"{stem}.npz"


def has_cache(path: str | Path) -> bool:
    return Path(path).exists()


def save_features(path: str | Path, features: dict[int, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer_{i:02d}": np.asarray(arr, dtype=np.float16) for i, arr in features.items()}
    tmp_path = path.with_suffix(".npz.tmp")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(path)  # atomic on the same filesystem -- avoids half-written cache files


def load_features(path: str | Path) -> dict[int, np.ndarray]:
    with np.load(path) as data:
        return {int(k.split("_")[1]): data[k] for k in data.files}
