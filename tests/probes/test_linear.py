import numpy as np

from isplit.probes.linear import LinearProbe


def test_linear_probe_fits_linearly_separable_data():
    rng = np.random.default_rng(0)
    n, d = 200, 10
    x = rng.standard_normal((n, d))
    y = (x[:, 0] > 0).astype(int)

    probe = LinearProbe().fit(x, y)
    acc = probe.score(x, y)
    assert acc > 0.9


def test_linear_probe_predict_proba_sums_to_one():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 5))
    y = rng.integers(0, 3, size=100)
    probe = LinearProbe().fit(x, y)
    proba = probe.predict_proba(x[:5])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_linear_probe_predict_single_example():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 4))
    y = (x[:, 0] > 0).astype(int)
    probe = LinearProbe().fit(x, y)
    pred = probe.predict(x[0])
    assert pred.shape == (1,)
