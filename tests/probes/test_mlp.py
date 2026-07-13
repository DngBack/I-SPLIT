import numpy as np

from isplit.probes.mlp import MLPProbe


def test_mlp_probe_fits_nonlinear_xor_pattern():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal((n, 4))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)

    probe = MLPProbe(hidden_dim=16, max_iter=300, seed=0).fit(x, y)
    acc = probe.score(x, y)
    assert acc > 0.8


def test_mlp_probe_predict_shape():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 3))
    y = rng.integers(0, 2, size=100)
    probe = MLPProbe(hidden_dim=8, max_iter=100, seed=0).fit(x, y)
    preds = probe.predict(x[:10])
    assert preds.shape == (10,)
