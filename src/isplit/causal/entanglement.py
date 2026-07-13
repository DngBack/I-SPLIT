"""Irreducible Entanglement Index (IEI, section 6.4) and pairwise entanglement
reporting across a set of estimated factor subspaces.
"""

import numpy as np
import pandas as pd

from isplit.causal.metrics import principal_angles


def irreducible_entanglement_index(basis_1: np.ndarray, basis_2: np.ndarray) -> float:
    """IEI = mean_i cos^2(theta_i) over principal angles theta_i between the two
    subspaces. 1.0 = identical subspaces, 0.0 = fully orthogonal.

    Note this is a *geometric* overlap measure -- unlike Preserve/Transfer/CSS
    it is not itself a causal claim, and the paper's central point is that IEI
    and causal selectivity can disagree (two subspaces can look separated
    geometrically yet fail an interchange test, or vice versa).
    """
    angles = principal_angles(basis_1, basis_2)
    if angles.size == 0:
        return 0.0
    return float(np.mean(np.cos(angles) ** 2))


def pairwise_entanglement_matrix(subspaces: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(subspaces)
    mat = np.zeros((len(names), len(names)))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            mat[i, j] = (
                1.0 if i == j else irreducible_entanglement_index(subspaces[name_i], subspaces[name_j])
            )
    return pd.DataFrame(mat, index=names, columns=names)
