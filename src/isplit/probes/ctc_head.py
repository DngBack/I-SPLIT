"""Character-level CTC linear probe (content probe, section 4 of the plan):
a lightweight, dependency-light proxy for "does this representation still
carry linguistic content." Avoids phoneme-level alignment (which would need
phonemizer/espeak-ng, a Windows-unfriendly system dependency) by working
directly on raw prompt/transcript text with greedy-decoded CER via `jiwer`.
"""

from dataclasses import dataclass

import numpy as np

BLANK_IDX = 0
_ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(_ALPHABET)}  # 0 reserved for CTC blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(_ALPHABET)}
VOCAB_SIZE = len(_ALPHABET) + 1


def text_to_indices(text: str) -> list[int]:
    text = text.upper()
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def _normalize_for_scoring(text: str) -> str:
    """Restrict text to exactly the characters the CTC head's alphabet can
    produce (see _ALPHABET). The reference text is raw VCTK transcript text
    (mixed case, with punctuation outside the alphabet); scoring it as-is
    against a hypothesis that can only ever be uppercase A-Z/space/apostrophe
    means every letter is counted as an error from case mismatch alone --
    CER stays pinned near 1.0 regardless of transcription quality.
    """
    text = text.upper()
    return "".join(c for c in text if c in CHAR_TO_IDX)


def greedy_decode(log_probs: "np.ndarray") -> str:
    """log_probs: (T, vocab_size). Collapse repeats then drop blanks."""
    best_path = log_probs.argmax(axis=-1)
    chars = []
    prev = None
    for idx in best_path:
        idx = int(idx)
        if idx != prev and idx != BLANK_IDX:
            chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)


def cer(hypothesis: str, reference: str) -> float:
    import jiwer

    reference = _normalize_for_scoring(reference)
    hypothesis = _normalize_for_scoring(hypothesis)
    if len(reference.strip()) == 0:
        return 0.0 if len(hypothesis.strip()) == 0 else 1.0
    return float(jiwer.cer(reference, hypothesis))


@dataclass
class CTCHead:
    """A single linear layer D -> vocab_size trained with CTC loss on frozen
    features. Deliberately minimal (no LM, no beam search) -- this is a
    probing proxy, not a competitive ASR system; absolute CER at pilot scale
    will be high, what matters is the *relative* CER before vs. after an
    intervention/projection.
    """

    feature_dim: int
    lr: float = 1e-3
    epochs: int = 30
    seed: int = 0
    device: str = "cpu"
    model: "object" = None

    def fit(self, features: list[np.ndarray], texts: list[str]) -> "CTCHead":
        import torch
        from torch.nn.utils.rnn import pad_sequence

        torch.manual_seed(self.seed)
        self.model = torch.nn.Linear(self.feature_dim, VOCAB_SIZE).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ctc_loss = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

        feat_tensors = [torch.from_numpy(np.asarray(f, dtype=np.float32)) for f in features]
        target_lists = [text_to_indices(t) for t in texts]
        input_lengths = torch.tensor([f.shape[0] for f in feat_tensors], dtype=torch.long)
        target_lengths = torch.tensor([max(len(t), 1) for t in target_lists], dtype=torch.long)
        targets = torch.cat(
            [torch.tensor(t if t else [BLANK_IDX], dtype=torch.long) for t in target_lists]
        ).to(self.device)
        padded_feats = pad_sequence(feat_tensors, batch_first=False).to(self.device)  # (T, N, D)
        # input_lengths/target_lengths must stay on CPU -- torch.nn.CTCLoss requires it
        # regardless of where log_probs/targets live.

        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(padded_feats)  # (T, N, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
        return self

    def predict_text(self, feature: np.ndarray) -> str:
        import torch

        with torch.no_grad():
            logits = self.model(torch.from_numpy(np.asarray(feature, dtype=np.float32)).to(self.device))
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return greedy_decode(log_probs)
