"""Small MLP probe (nonlinear leakage check, ablation section 12): a low-linear-
probe score doesn't prove erasure -- this catches nonlinear residual information
a linear probe would miss.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MLPProbe:
    hidden_dim: int = 64
    max_iter: int = 500
    seed: int = 0
    scaler: Any = None
    clf: Any = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MLPProbe":
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(features)
        self.clf = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim,),
            max_iter=self.max_iter,
            random_state=self.seed,
            # No early stopping: this probe exists to measure how much of a factor
            # is *nonlinearly* recoverable, and nonlinear targets (XOR-like leakage)
            # sit on a long loss plateau early on. Stopping on it aborts after a few
            # dozen iterations and understates the leakage it is meant to detect.
            early_stopping=False,
        )
        self.clf.fit(self.scaler.transform(features), labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(np.atleast_2d(features)))

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        return float(self.clf.score(self.scaler.transform(features), labels))
