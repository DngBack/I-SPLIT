"""Registry of the 4 frozen HuggingFace speech encoders I-SPLIT audits:
wav2vec2-base, HuBERT-base, WavLM-base, data2vec-audio-base. All loadable via
`transformers.AutoModel`/`AutoFeatureExtractor` with no extra conversion work
(ContentVec was ruled out as a 4th encoder for exactly this reason -- it's
not natively supported).
"""

from dataclasses import dataclass
from typing import Any

DEFAULT_ENCODERS: dict[str, tuple[str, int]] = {
    "wav2vec2-base": ("facebook/wav2vec2-base", 12),
    "hubert-base": ("facebook/hubert-base-ls960", 12),
    "wavlm-base": ("microsoft/wavlm-base", 12),
    "data2vec-audio-base": ("facebook/data2vec-audio-base", 12),
}


@dataclass
class EncoderSpec:
    name: str
    hf_id: str
    num_layers: int
    model: Any
    feature_extractor: Any


_LOADED: dict[str, EncoderSpec] = {}


def load_encoder(name: str, device: str = "cpu") -> EncoderSpec:
    """Load (and cache in-process) a frozen encoder in eval mode on `device`."""
    cache_key = f"{name}:{device}"
    if cache_key in _LOADED:
        return _LOADED[cache_key]
    if name not in DEFAULT_ENCODERS:
        raise ValueError(f"unknown encoder {name!r}, expected one of {list(DEFAULT_ENCODERS)}")

    from transformers import AutoFeatureExtractor, AutoModel

    hf_id, num_layers = DEFAULT_ENCODERS[name]
    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    spec = EncoderSpec(
        name=name, hf_id=hf_id, num_layers=num_layers, model=model, feature_extractor=feature_extractor
    )
    _LOADED[cache_key] = spec
    return spec
