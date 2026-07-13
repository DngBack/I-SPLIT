import numpy as np

from isplit.encoders.cache import cache_path, has_cache, load_features, save_features


def test_cache_path_distinguishes_conditions():
    p1 = cache_path("results/features", "wav2vec2-base", "train", "p225_001")
    p2 = cache_path("results/features", "wav2vec2-base", "train", "p225_001", condition="noise=n1_snr=10.0")
    assert p1 != p2
    assert p1.parent == p2.parent


def test_save_and_load_features_roundtrip_exact_for_fp16_representable_values(tmp_path):
    features = {
        0: np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        1: np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    }
    path = tmp_path / "utt.npz"
    save_features(path, features)
    loaded = load_features(path)

    assert set(loaded) == set(features)
    for layer, arr in features.items():
        assert np.allclose(loaded[layer], arr, atol=1e-3)


def test_save_features_fp16_precision_loss_is_small():
    rng = np.random.default_rng(0)
    original = rng.standard_normal((10, 768)).astype(np.float32)
    features = {0: original}

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "utt.npz"
        save_features(path, features)
        loaded = load_features(path)

    relative_error = np.linalg.norm(loaded[0].astype(np.float32) - original) / np.linalg.norm(original)
    assert relative_error < 1e-2  # fp16 has ~3 decimal digits of precision


def test_has_cache_reflects_file_existence(tmp_path):
    path = tmp_path / "utt.npz"
    assert not has_cache(path)
    save_features(path, {0: np.zeros((2, 4), dtype=np.float32)})
    assert has_cache(path)


def test_save_features_is_idempotent(tmp_path):
    path = tmp_path / "utt.npz"
    features = {0: np.array([[1.0, 2.0]], dtype=np.float32)}
    save_features(path, features)
    first = load_features(path)
    save_features(path, features)
    second = load_features(path)
    assert np.array_equal(first[0], second[0])
