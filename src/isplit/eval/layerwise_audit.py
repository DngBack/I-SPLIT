"""Layer-wise audit (paper Claim 1 evidence): for each (encoder, layer,
factor), compute decodability (linear probe accuracy), geometric orthogonality
(principal angles / IEI between factor subspaces), and causal selectivity
(interchange CSS) side by side, so they can be correlated against each other.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from isplit.causal.entanglement import irreducible_entanglement_index
from isplit.causal.interchange import swap_factor
from isplit.causal.metrics import causal_selectivity_score, classification_preserve, classification_transfer
from isplit.probes.linear import LinearProbe
from isplit.subspace.pipeline import load_pooled_layer


def train_label_probe(
    features_dir: str | Path,
    encoder_name: str,
    layer: int,
    manifest: pd.DataFrame,
    label_col: str,
    split_col_values: dict[str, str],
) -> tuple[LinearProbe, dict[str, np.ndarray]]:
    """Train a linear probe to predict `label_col` from pooled features.
    `split_col_values` maps utt_id -> 'train'/'held_out' (from the speaker split).
    """
    train_rows = manifest[manifest["utt_id"].map(split_col_values) == "train"]
    x_train = np.stack(
        [load_pooled_layer(features_dir, encoder_name, "train", uid, None, layer) for uid in train_rows["utt_id"]]
    )
    y_train = train_rows[label_col].to_numpy()
    probe = LinearProbe().fit(x_train, y_train)
    return probe, {"x_train": x_train, "y_train": y_train}


def layerwise_decodability(
    features_dir: str | Path,
    encoder_name: str,
    layer: int,
    manifest: pd.DataFrame,
    label_col: str,
    utt_to_split: dict[str, str],
) -> float:
    """Held-out accuracy of a linear probe predicting `label_col`."""
    train_rows = manifest[manifest["utt_id"].map(utt_to_split) == "train"]
    held_rows = manifest[manifest["utt_id"].map(utt_to_split) == "held_out"]
    if train_rows.empty or held_rows.empty:
        return float("nan")

    x_train = np.stack(
        [load_pooled_layer(features_dir, encoder_name, "train", uid, None, layer) for uid in train_rows["utt_id"]]
    )
    y_train = train_rows[label_col].to_numpy()
    x_held = np.stack(
        [load_pooled_layer(features_dir, encoder_name, "held_out", uid, None, layer) for uid in held_rows["utt_id"]]
    )
    y_held = held_rows[label_col].to_numpy()

    probe = LinearProbe().fit(x_train, y_train)
    return probe.score(x_held, y_held)


def layerwise_causal_selectivity(
    features_dir: str | Path,
    encoder_name: str,
    layer: int,
    factor_subspaces: dict[str, np.ndarray],
    held_out_pairs: pd.DataFrame,
    target_probe: LinearProbe,
    content_probe: LinearProbe | None,
    factor: str,
) -> float:
    """Average CSS over held-out interchange swaps for `factor`'s subspace:
    Transfer measured by `target_probe` on the swapped representation,
    Preserve measured by `content_probe` (a proxy off-target probe) staying
    close to the original.
    """
    from isplit.subspace.pipeline import load_paired_features

    if held_out_pairs.empty:
        return float("nan")

    features_a, features_b = load_paired_features(features_dir, encoder_name, layer, held_out_pairs, factor)
    basis = factor_subspaces[factor]

    css_scores = []
    for i in range(len(features_a)):
        swapped = swap_factor(features_a[i], features_b[i], basis)
        transfer_pred = target_probe.predict(swapped[None, :])[0]
        transfer_true = target_probe.predict(features_b[i][None, :])[0]
        transfer = classification_transfer(transfer_pred, transfer_true)

        if content_probe is not None:
            preserve_pred = content_probe.predict(swapped[None, :])[0]
            preserve_true = content_probe.predict(features_a[i][None, :])[0]
            preserve = classification_preserve(preserve_pred, preserve_true)
        else:
            preserve = 1.0

        css_scores.append(causal_selectivity_score(preserve, transfer))
    return float(np.mean(css_scores))


def pairwise_subspace_entanglement(factor_subspaces: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(factor_subspaces)
    rows = []
    for i, name_i in enumerate(names):
        for name_j in names[i + 1 :]:
            iei = irreducible_entanglement_index(factor_subspaces[name_i], factor_subspaces[name_j])
            rows.append({"factor_a": name_i, "factor_b": name_j, "iei": iei})
    return pd.DataFrame(rows)
