"""Paired-intervention construction (I-SPLIT core input, section 7): builds
(a, b) example pairs that differ in exactly one generative factor.

- speaker pairs: same VCTK prompt_id (identical text), different speaker_id
- content pairs: same speaker_id, different prompt_id (different text)
- environment pairs: same base utterance, differing only in overlaid
  noise clip / SNR (a synthetic DSP intervention applied at feature-extraction time)
- channel pairs: same base utterance, differing only in channel transform
  (clean / telephone-bandpass / mu-law)

A single speaker-disjoint train/held-out split is computed once and applied
consistently across all four factor types, so held-out speakers never leak
into training pairs for any factor (see tests/data/test_pairs.py).
"""

import numpy as np
import pandas as pd

CHANNEL_CONDITIONS = ("clean", "telephone", "mulaw")


def split_speakers(
    speaker_ids: list[str] | np.ndarray, held_out_fraction: float, seed: int
) -> tuple[set, set]:
    speakers = np.array(sorted(set(speaker_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n_held = max(1, int(round(len(speakers) * held_out_fraction)))
    held_out = set(speakers[:n_held])
    train = set(speakers[n_held:])
    return train, held_out


def assign_split_by_speaker(
    df: pd.DataFrame, speaker_cols: list[str], train_speakers: set, held_out_speakers: set
) -> pd.DataFrame:
    """Tag each pair 'train' / 'held_out' based on the speaker(s) it involves,
    dropping any pair whose speakers straddle both splits (it belongs cleanly
    to neither, so keeping it would leak held-out identity into training).
    """
    if df.empty:
        return df.assign(split=pd.Series(dtype=object))

    def _split(row: pd.Series) -> str | None:
        speakers_in_row = {row[c] for c in speaker_cols if pd.notna(row[c])}
        if speakers_in_row <= train_speakers:
            return "train"
        if speakers_in_row <= held_out_speakers:
            return "held_out"
        return None

    df = df.copy()
    df["split"] = df.apply(_split, axis=1)
    return df[df["split"].notna()].reset_index(drop=True)


def make_speaker_pairs(vctk_manifest: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """(a, b) pairs sharing a prompt_id (identical text) but different speaker_id."""
    rng = np.random.default_rng(seed)
    grouped = vctk_manifest.groupby("prompt_id")
    groups = {pid: g for pid, g in grouped if g["speaker_id"].nunique() >= 2}
    prompt_ids = list(groups)
    if not prompt_ids:
        return pd.DataFrame(
            columns=["pair_id", "factor", "a_utt_id", "b_utt_id", "a_speaker_id", "b_speaker_id", "prompt_id"]
        )

    rows, seen = [], set()
    attempts, max_attempts = 0, n_pairs * 30 + 100
    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1
        pid = prompt_ids[rng.integers(0, len(prompt_ids))]
        group = groups[pid]
        speakers = group["speaker_id"].unique()
        sa, sb = rng.choice(speakers, size=2, replace=False)
        a_row = group[group.speaker_id == sa].iloc[0]
        b_row = group[group.speaker_id == sb].iloc[0]
        key = (a_row.utt_id, b_row.utt_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pair_id": len(rows),
                "factor": "speaker",
                "a_utt_id": a_row.utt_id,
                "b_utt_id": b_row.utt_id,
                "a_speaker_id": sa,
                "b_speaker_id": sb,
                "prompt_id": pid,
            }
        )
    return pd.DataFrame(rows)


def make_content_pairs(vctk_manifest: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """(a, b) pairs sharing a speaker_id but different prompt_id (different text)."""
    rng = np.random.default_rng(seed)
    grouped = vctk_manifest.groupby("speaker_id")
    groups = {sid: g for sid, g in grouped if len(g) >= 2}
    speakers = list(groups)
    if not speakers:
        return pd.DataFrame(
            columns=["pair_id", "factor", "a_utt_id", "b_utt_id", "speaker_id", "a_prompt_id", "b_prompt_id"]
        )

    rows, seen = [], set()
    attempts, max_attempts = 0, n_pairs * 30 + 100
    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1
        sid = speakers[rng.integers(0, len(speakers))]
        group = groups[sid]
        idx_a, idx_b = rng.choice(len(group), size=2, replace=False)
        a_row, b_row = group.iloc[idx_a], group.iloc[idx_b]
        if a_row.prompt_id == b_row.prompt_id:
            continue
        key = (a_row.utt_id, b_row.utt_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pair_id": len(rows),
                "factor": "content",
                "a_utt_id": a_row.utt_id,
                "b_utt_id": b_row.utt_id,
                "speaker_id": sid,
                "a_prompt_id": a_row.prompt_id,
                "b_prompt_id": b_row.prompt_id,
            }
        )
    return pd.DataFrame(rows)


def make_environment_pairs(
    vctk_manifest: pd.DataFrame,
    noise_manifest: pd.DataFrame,
    snr_levels_db: list[float],
    n_pairs: int,
    seed: int,
    include_clean: bool = True,
) -> pd.DataFrame:
    """(a, b) pairs sharing the same base utterance, differing only in which
    noise clip / SNR gets overlaid at feature-extraction time.
    """
    rng = np.random.default_rng(seed)
    conditions: list[tuple[str | None, float | None]] = [("clean", None)] if include_clean else []
    for nid in noise_manifest["noise_id"].to_numpy():
        for snr in snr_levels_db:
            conditions.append((nid, float(snr)))
    if len(conditions) < 2:
        return pd.DataFrame(
            columns=["pair_id", "factor", "base_utt_id", "speaker_id", "a_noise_id", "a_snr_db", "b_noise_id", "b_snr_db"]
        )

    utt_ids = vctk_manifest["utt_id"].to_numpy()
    speaker_of = dict(zip(vctk_manifest["utt_id"], vctk_manifest["speaker_id"], strict=True))

    rows, seen = [], set()
    attempts, max_attempts = 0, n_pairs * 30 + 100
    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1
        utt_id = utt_ids[rng.integers(0, len(utt_ids))]
        ia, ib = rng.choice(len(conditions), size=2, replace=False)
        cond_a, cond_b = conditions[ia], conditions[ib]
        key = (utt_id, cond_a, cond_b)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pair_id": len(rows),
                "factor": "environment",
                "base_utt_id": utt_id,
                "speaker_id": speaker_of[utt_id],
                "a_noise_id": cond_a[0],
                "a_snr_db": cond_a[1],
                "b_noise_id": cond_b[0],
                "b_snr_db": cond_b[1],
            }
        )
    return pd.DataFrame(rows)


def make_channel_pairs(vctk_manifest: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """(a, b) pairs sharing the same base utterance, differing only in channel
    transform (clean / telephone-bandpass / mu-law).
    """
    rng = np.random.default_rng(seed)
    utt_ids = vctk_manifest["utt_id"].to_numpy()
    speaker_of = dict(zip(vctk_manifest["utt_id"], vctk_manifest["speaker_id"], strict=True))

    rows, seen = [], set()
    attempts, max_attempts = 0, n_pairs * 30 + 100
    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1
        utt_id = utt_ids[rng.integers(0, len(utt_ids))]
        ca, cb = rng.choice(len(CHANNEL_CONDITIONS), size=2, replace=False)
        key = (utt_id, ca, cb)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pair_id": len(rows),
                "factor": "channel",
                "base_utt_id": utt_id,
                "speaker_id": speaker_of[utt_id],
                "a_channel": CHANNEL_CONDITIONS[ca],
                "b_channel": CHANNEL_CONDITIONS[cb],
            }
        )
    return pd.DataFrame(rows)


def build_all_pairs(
    vctk_manifest: pd.DataFrame,
    noise_manifest: pd.DataFrame,
    snr_levels_db: list[float],
    n_pairs_per_factor: dict[str, int],
    held_out_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Build all four factor pair sets against one shared speaker-disjoint
    split, so "held-out speaker" means the same thing for every factor.
    """
    train_speakers, held_out_speakers = split_speakers(
        vctk_manifest["speaker_id"], held_out_fraction, seed
    )

    speaker_pairs = make_speaker_pairs(vctk_manifest, n_pairs_per_factor["speaker"], seed)
    content_pairs = make_content_pairs(vctk_manifest, n_pairs_per_factor["content"], seed)
    environment_pairs = make_environment_pairs(
        vctk_manifest, noise_manifest, snr_levels_db, n_pairs_per_factor["environment"], seed
    )
    channel_pairs = make_channel_pairs(vctk_manifest, n_pairs_per_factor["channel"], seed)

    return {
        "speaker": assign_split_by_speaker(
            speaker_pairs, ["a_speaker_id", "b_speaker_id"], train_speakers, held_out_speakers
        ),
        "content": assign_split_by_speaker(
            content_pairs, ["speaker_id"], train_speakers, held_out_speakers
        ),
        "environment": assign_split_by_speaker(
            environment_pairs, ["speaker_id"], train_speakers, held_out_speakers
        ),
        "channel": assign_split_by_speaker(
            channel_pairs, ["speaker_id"], train_speakers, held_out_speakers
        ),
    }
