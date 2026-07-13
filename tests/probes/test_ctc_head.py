import numpy as np

from isplit.probes.ctc_head import CTCHead, cer, greedy_decode, text_to_indices


def test_text_to_indices_roundtrip_via_greedy_decode():
    text = "HELLO WORLD"
    indices = text_to_indices(text)
    assert all(i > 0 for i in indices)  # 0 is reserved for blank

    # build a fake log_probs sequence that argmaxes to exactly `indices` with no repeats/blanks
    vocab_size = max(indices) + 5
    log_probs = np.full((len(indices), vocab_size), -10.0)
    for t, idx in enumerate(indices):
        log_probs[t, idx] = 0.0
    decoded = greedy_decode(log_probs)
    assert decoded == text


def test_greedy_decode_collapses_repeats_and_drops_blanks():
    # sequence: blank, A, A, blank, B -> should decode to "AB"
    from isplit.probes.ctc_head import CHAR_TO_IDX

    a_idx, b_idx = CHAR_TO_IDX["A"], CHAR_TO_IDX["B"]
    vocab_size = max(a_idx, b_idx) + 2
    seq = [0, a_idx, a_idx, 0, b_idx]
    log_probs = np.full((len(seq), vocab_size), -10.0)
    for t, idx in enumerate(seq):
        log_probs[t, idx] = 0.0
    assert greedy_decode(log_probs) == "AB"


def test_cer_identical_strings_is_zero():
    assert cer("HELLO", "HELLO") == 0.0


def test_cer_empty_reference_and_hypothesis_is_zero():
    assert cer("", "") == 0.0


def test_ctc_head_overfits_tiny_synthetic_dataset():
    rng = np.random.default_rng(0)
    feature_dim = 16
    texts = ["AB", "BA", "AA"]
    # give each text a distinctive, easily-separable feature pattern per character
    features = []
    for text in texts:
        t = len(text) * 3  # a few frames per character
        feat = rng.standard_normal((t, feature_dim)).astype(np.float32) * 0.01
        for i, ch in enumerate(text):
            idx = text_to_indices(ch)[0]
            feat[i * 3 : (i + 1) * 3, idx % feature_dim] += 5.0
        features.append(feat)

    head = CTCHead(feature_dim=feature_dim, lr=0.05, epochs=200, seed=0).fit(features, texts)

    total_cer = 0.0
    for feat, text in zip(features, texts, strict=True):
        pred = head.predict_text(feat)
        total_cer += cer(pred, text)
    assert total_cer / len(texts) < 0.5
