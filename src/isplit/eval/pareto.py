"""Orthogonal-vs-oblique preservation/erasure Pareto frontier (paper Claim 2):
sweep rank and ridge strength tau, compare I-SPLIT's oblique projector against
the orthogonal-removal baseline on content-preservation vs. nuisance-erasure.
"""

import numpy as np
import pandas as pd

from isplit.subspace.projection import fit_oblique, orthogonal_projection_remove, reconstruct_block


def preservation_erasure_curve(
    content_features: np.ndarray,
    nuisance_features_true: np.ndarray,
    content_basis: np.ndarray,
    nuisance_basis: np.ndarray,
    tau_values: list[float],
    content_probe,
    nuisance_probe,
    content_labels_true,
    nuisance_labels_true,
) -> pd.DataFrame:
    """content_features: raw (mixed) pooled representations to be cleaned.
    Compares, for each tau, the oblique projector's content-preservation
    (content probe accuracy on the reconstructed/cleaned representation vs.
    the label from the *original* representation) against nuisance-erasure
    (1 - nuisance probe accuracy on the cleaned representation).
    """
    basis = np.concatenate([content_basis, nuisance_basis], axis=1)
    content_block = slice(0, content_basis.shape[1])
    rows = []

    orth_cleaned = orthogonal_projection_remove(content_features, nuisance_basis)
    orth_preserve = content_probe.score(orth_cleaned, content_labels_true)
    orth_erasure = 1.0 - nuisance_probe.score(orth_cleaned, nuisance_labels_true)
    rows.append({"method": "orthogonal", "tau": None, "content_preservation": orth_preserve, "nuisance_erasure": orth_erasure})

    for tau in tau_values:
        a_hat = fit_oblique(basis, content_features, tau)
        cleaned = reconstruct_block(basis, a_hat, content_block)
        preserve = content_probe.score(cleaned, content_labels_true)
        erasure = 1.0 - nuisance_probe.score(cleaned, nuisance_labels_true)
        rows.append({"method": "oblique", "tau": tau, "content_preservation": preserve, "nuisance_erasure": erasure})

    return pd.DataFrame(rows)


def pareto_frontier_area(df: pd.DataFrame, x_col: str = "nuisance_erasure", y_col: str = "content_preservation") -> float:
    """Trapezoidal area under the (erasure, preservation) frontier -- a single
    scalar summary of the tradeoff curve, higher is better.
    """
    sorted_df = df.sort_values(x_col)
    # np.trapz (not np.trapezoid, which needs numpy>=2.0 -- this repo pins numpy<2.0
    # for wider compatibility with the installed torch/transformers CPU wheels)
    return float(np.trapz(sorted_df[y_col], sorted_df[x_col]))
