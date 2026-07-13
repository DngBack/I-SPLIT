"""Held-out / out-of-distribution robustness (paper Claim 3): does causal
selectivity estimated on controlled VCTK interventions predict behavior on
(a) held-out VCTK speakers, (b) unseen factor *combinations*, and (c) natural,
non-synthetic LibriSpeech audio never touched by subspace estimation?
"""

import numpy as np
import pandas as pd

from isplit.probes.linear import LinearProbe
from isplit.subspace.pipeline import load_pooled_layer


def cross_corpus_speaker_accuracy(
    features_dir: str,
    encoder_name: str,
    layer: int,
    vctk_manifest: pd.DataFrame,
    librispeech_manifest: pd.DataFrame,
    utt_to_split: dict[str, str],
) -> float:
    """Train a speaker-id probe on VCTK train speakers, evaluate speaker
    *verification-style* separability on LibriSpeech (a corpus never used for
    subspace estimation) by checking same-speaker vs different-speaker
    utterance-pair similarity in the pooled feature space -- a coarse but
    dependency-free proxy for cross-corpus generalization (no probe can
    directly predict LibriSpeech speaker *identities* it never trained on).
    """
    train_rows = vctk_manifest[vctk_manifest["utt_id"].map(utt_to_split) == "train"]
    x_train = np.stack(
        [load_pooled_layer(features_dir, encoder_name, "train", uid, None, layer) for uid in train_rows["utt_id"]]
    )
    y_train = train_rows["speaker_id"].to_numpy()
    probe = LinearProbe().fit(x_train, y_train)

    ls_rows = librispeech_manifest.groupby("speaker_id").filter(lambda g: len(g) >= 2)
    if ls_rows.empty:
        return float("nan")
    x_ls = np.stack(
        [load_pooled_layer(features_dir, encoder_name, "held_out", uid, None, layer) for uid in ls_rows["utt_id"]]
    )
    embeddings = probe.predict_proba(x_ls)  # use probe's probability simplex as an embedding proxy
    speakers = ls_rows["speaker_id"].to_numpy()

    same_sims, diff_sims = [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            (same_sims if speakers[i] == speakers[j] else diff_sims).append(sim)
    if not same_sims or not diff_sims:
        return float("nan")
    return float(np.mean(same_sims) - np.mean(diff_sims))  # positive -> same-speaker pairs more similar


def held_out_vs_train_css_gap(train_css: float, held_out_css: float) -> float:
    """Simple generalization-gap summary: how much causal selectivity drops
    from train-split pairs to held-out-speaker pairs. Near 0 = good generalization.
    """
    if np.isnan(train_css) or np.isnan(held_out_css):
        return float("nan")
    return float(train_css - held_out_css)


def joint_unseen_combination_pairs(pairs: pd.DataFrame, seen_condition_cols: list[str]) -> pd.DataFrame:
    """Filter a factor's held-out pairs down to combinations of conditions
    that never co-occurred in the train split -- the "unseen joint
    intervention" generalization test.
    """
    if pairs.empty:
        return pairs
    train_pairs = pairs[pairs["split"] == "train"]
    seen_combos = set(tuple(row) for row in train_pairs[seen_condition_cols].itertuples(index=False))
    held = pairs[pairs["split"] == "held_out"]
    mask = ~held[seen_condition_cols].apply(tuple, axis=1).isin(seen_combos)
    return held[mask]
