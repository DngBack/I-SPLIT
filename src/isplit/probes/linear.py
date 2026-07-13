"""Linear probe wrapper (logistic regression) used for speaker / environment /
channel classification on pooled utterance-level features.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearProbe:
    scaler: Any = None
    clf: Any = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        self.scaler = StandardScaler().fit(features)
        self.clf = LogisticRegression(max_iter=2000)
        self.clf.fit(self.scaler.transform(features), labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(np.atleast_2d(features)))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(np.atleast_2d(features)))

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        return float(self.clf.score(self.scaler.transform(features), labels))
