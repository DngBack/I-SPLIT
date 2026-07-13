"""Frozen forward-pass feature extraction: one waveform in, one dict of
per-layer hidden states out. `output_hidden_states=True` gives layer 0 (the
post-CNN input embeddings) through layer `num_layers` (the final transformer
layer), matching the paper's "layer-wise audit" design.
"""

import numpy as np


def extract_hidden_states(spec, wav: np.ndarray, sr: int, device: str = "cpu") -> dict[int, np.ndarray]:
    """Run the frozen encoder on one waveform under torch.no_grad().

    Returns {layer_idx: (T, D) float16 array} for every hidden-state layer.
    """
    import torch

    inputs = spec.feature_extractor(wav, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = spec.model(**inputs, output_hidden_states=True)
    return {i: hs[0].cpu().numpy().astype(np.float16) for i, hs in enumerate(outputs.hidden_states)}


def pool_mean(features: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Utterance-level mean pooling over time, per layer -- used for
    speaker/environment/channel probes (content probing stays frame-level).
    """
    return {i: arr.astype(np.float32).mean(axis=0) for i, arr in features.items()}
