from __future__ import annotations

import re
from typing import Iterable

import torch

# 中英混合：拉丁字母/数字连串（含连字符/点分隔的版本号）作为一个 token，汉字逐字。
# 例：GPT-5.2 -> ["gpt-5-2"]；通义千问3.7 -> ["通","义","千","问","3.7"]
_PIECE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*|[\u3400-\u9fff]")


class MixedTokenizer:
    """Character/word hybrid tokenizer over Latin + Chinese text.

    Vocab is built from the corpus actually fed to it, so real model names
    (GPT, Qwen3.7, DeepSeek-V4-Pro) and Chinese descriptions are all known
    tokens rather than <unk>.
    """

    PAD = 0
    UNK = 1

    def __init__(self, texts: Iterable[str]) -> None:
        pieces = sorted({piece for text in texts for piece in _PIECE.findall(text.lower())})
        self.tokens = ("<pad>", "<unk>", *pieces)
        self.index = {token: offset for offset, token in enumerate(self.tokens)}

    def tokenize(self, text: str) -> list[str]:
        return _PIECE.findall(text.lower())

    def encode(self, text: str, length: int) -> tuple[torch.Tensor, float]:
        pieces = self.tokenize(text)[:length]
        ids = [self.index.get(piece, self.UNK) for piece in pieces]
        coverage = sum(token_id != self.UNK for token_id in ids) / max(1, len(ids))
        ids.extend([self.PAD] * (length - len(ids)))
        return torch.tensor(ids, dtype=torch.long), coverage

    @property
    def size(self) -> int:
        return len(self.tokens)
