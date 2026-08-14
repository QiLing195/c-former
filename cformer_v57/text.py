from __future__ import annotations

import hashlib
import re
import unicodedata

import torch
from torch.nn import functional as F


_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";"}))
    return _SPACE.sub(" ", text).strip()


class HashedTextEncoder:
    """Stateless signed feature hashing for noisy demo text.

    It provides an auditable lexical adapter, not a claim of semantic language
    understanding. Character n-grams make punctuation/spacing variants less brittle.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _features(self, text: str):
        normalized = normalize_text(text)
        compact = re.sub(r"[^a-z0-9\u3400-\u9fff]", "", normalized)
        for token in _TOKEN.findall(normalized):
            # Stable object numbers are governed identity anchors, not ordinary words.
            weight = 12.0 if re.search(r"\d{4,}", token) else 2.0
            yield f"w:{token}", weight
        for size, weight in ((2, 1.0), (3, 1.5)):
            for index in range(max(0, len(compact) - size + 1)):
                yield f"c{size}:{compact[index:index + size]}", weight

    def encode(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.dimensions, dtype=torch.float32)
        seen: set[str] = set()
        for feature, weight in self._features(text):
            # Repeated boilerplate must not drown a single target anchor in long text.
            if feature in seen:
                continue
            seen.add(feature)
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            sign = 1.0 if (value >> 63) == 0 else -1.0
            vector[index] += sign * weight
        return F.normalize(vector, dim=0) if vector.abs().sum() else vector

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.encode(text) for text in texts])
