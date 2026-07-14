"""Glue between cached encoder features + pair tables and the subspace/causal
math: loading paired feature matrices for a given (encoder, layer, factor),
and estimating that factor's subspace via intervention covariance.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from isplit.encoders.cache import cache_path, load_features
from isplit.encoders.extract import pool_mean
from isplit.subspace.intervention_covariance import estimate_subspace_from_deltas


@lru_cache(maxsize=100_000)
def _pooled_all_layers(
    features_dir: str, encoder_name: str, split: str, utt_id: str, condition: str | None
) -> tuple[np.ndarray, ...]:
    """Every layer's mean-pooled vector for one cached utterance, memoized.

    Callers ask for one (utterance, layer) at a time, but the cache file holds
    all 13 layers together -- so a naive read decompresses the whole file and
    pools all 13 layers to return one of them, then does it again for the next
    layer. Memoizing the pooled result turns each file into one read instead of
    13, which is what makes the layer sweeps tractable. Tuple-of-arrays (not a
    dict) so the return value is cheap to index and not accidentally mutable.
    """
    features = load_features(cache_path(features_dir, encoder_name, split, utt_id, condition))
    pooled = pool_mean(features)
    return tuple(pooled[i] for i in sorted(pooled))


@lru_cache(maxsize=20_000)
def _frames_layer(
    features_dir: str, encoder_name: str, split: str, utt_id: str, condition: str | None, layer: int
) -> np.ndarray:
    features = load_features(cache_path(features_dir, encoder_name, split, utt_id, condition))
    return features[layer].astype(np.float32)


def load_pooled_layer(
    features_dir: str | Path, encoder_name: str, split: str, utt_id: str, condition: str | None, layer: int
) -> np.ndarray:
    return _pooled_all_layers(str(features_dir), encoder_name, split, utt_id, condition)[layer]


def load_frames_layer(
    features_dir: str | Path, encoder_name: str, split: str, utt_id: str, condition: str | None, layer: int
) -> np.ndarray:
    """Frame-level (T, D) features for one layer -- the content (CTC) probe needs
    the time axis that pooling throws away.
    """
    return _frames_layer(str(features_dir), encoder_name, split, utt_id, condition, layer)


def load_paired_features(
    features_dir: str | Path,
    encoder_name: str,
    layer: int,
    pairs: pd.DataFrame,
    factor: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (features_a, features_b) matrices, one row per pair, for the given
    factor's pair table. Handles both the (a_utt_id, b_utt_id)-keyed schema
    (content/speaker) and the (base_utt_id, a_condition, b_condition)-keyed
    schema (environment/channel).
    """
    a_list, b_list = [], []
    for _, row in pairs.iterrows():
        split = row["split"]
        if factor in ("content", "speaker"):
            a = load_pooled_layer(features_dir, encoder_name, split, row["a_utt_id"], None, layer)
            b = load_pooled_layer(features_dir, encoder_name, split, row["b_utt_id"], None, layer)
        elif factor == "environment":
            from isplit.data.augment import encode_environment_condition

            cond_a = encode_environment_condition(row["a_noise_id"], row["a_snr_db"])
            cond_b = encode_environment_condition(row["b_noise_id"], row["b_snr_db"])
            a = load_pooled_layer(features_dir, encoder_name, split, row["base_utt_id"], cond_a, layer)
            b = load_pooled_layer(features_dir, encoder_name, split, row["base_utt_id"], cond_b, layer)
        elif factor == "channel":
            from isplit.data.augment import encode_channel_condition

            cond_a = encode_channel_condition(row["a_channel"])
            cond_b = encode_channel_condition(row["b_channel"])
            a = load_pooled_layer(features_dir, encoder_name, split, row["base_utt_id"], cond_a, layer)
            b = load_pooled_layer(features_dir, encoder_name, split, row["base_utt_id"], cond_b, layer)
        else:
            raise ValueError(f"unknown factor: {factor!r}")
        a_list.append(a)
        b_list.append(b)
    return np.stack(a_list).astype(np.float32), np.stack(b_list).astype(np.float32)


def fit_factor_subspace(
    features_dir: str | Path,
    encoder_name: str,
    layer: int,
    pairs: pd.DataFrame,
    factor: str,
    rank: int | None = None,
    energy_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: load a factor's train-split pairs' cached features for one
    (encoder, layer), estimate its intervention-covariance subspace.
    """
    train_pairs = pairs[pairs["split"] == "train"]
    features_a, features_b = load_paired_features(features_dir, encoder_name, layer, train_pairs, factor)
    deltas = features_b - features_a
    return estimate_subspace_from_deltas(deltas, rank=rank, energy_threshold=energy_threshold)
