"""Causal evaluation metrics for interchange interventions (section 6):
Preserve, Transfer, Causal Selectivity Score (CSS), plus small distance
utilities used to turn probe outputs into the [0, 1] scores those combine.
"""

import numpy as np
from scipy.linalg import subspace_angles


def clipped_score(distance: float) -> float:
    """Generic divergence -> [0, 1] score mapping used by both Preserve and
    Transfer: 1 - distance, clipped to [0, 1] so a distance > 1 (e.g. an
    unnormalized metric) doesn't produce a negative score.
    """
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def classification_preserve(pred_label_swapped, true_label_a) -> float:
    """Preserve score for a classification head: 1.0 iff the swapped
    representation still predicts a's original label for this (non-target) factor.
    """
    return 1.0 if pred_label_swapped == true_label_a else 0.0


def classification_transfer(pred_label_swapped, target_label_b) -> float:
    """Transfer score for a classification head: 1.0 iff the swapped
    representation now predicts b's label for the target factor.
    """
    return 1.0 if pred_label_swapped == target_label_b else 0.0


def probability_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors, normalized to
    [0, 1] (JS divergence in nats is bounded by ln(2); dividing by ln(2) gives
    a bounded, comparable-across-distributions score of 0=identical, 1=disjoint).
    """
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(js / np.log(2))


def causal_selectivity_score(preserve: float, transfer: float, eps: float = 1e-8) -> float:
    """Harmonic mean of Preserve and Transfer: a high CSS requires both
    preserving off-target content and transferring the target factor.
    """
    if preserve <= 0 or transfer <= 0:
        return 0.0
    return float(2 * preserve * transfer / (preserve + transfer + eps))


def principal_angles(basis_1: np.ndarray, basis_2: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Principal angles between two subspaces (ascending), via scipy's
    numerically-stable SVD-based implementation.
    """
    angles = subspace_angles(basis_1, basis_2)
    return np.degrees(angles) if degrees else angles
