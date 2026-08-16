from __future__ import annotations

import random
from typing import Sequence

import torch

from cformer_v60 import ChineseAliasWorld, ChineseSemanticObject, LEXICONS

# Region near-synonym confusion clusters. Region axis indices follow the
# ALIAS order: 0北部 1南部 2东部 3西部 4中央 5海岸 6腹地 7高原 8冰原
# 9赤道城区 10城镇 11田野 12河流上段 13河流下段 14港湾 15边界地区.
REGION_NEAR_GROUPS = (
    (0, 1),  # 北部 / 南部
    (2, 3),  # 东部 / 西部
    (4, 6, 7, 8),  # 中央 / 腹地 / 高原 / 冰原  (V6.0b failure cluster)
    (5, 14),  # 海岸 / 港湾
    (10, 11),  # 城镇 / 田野
    (12, 13),  # 河流上段 / 河流下段
)

# OOV typo characters (guaranteed absent from the frozen tokenizer vocab).
TYPO_POOL = ("録", "提", "衬", "滘", "淸", "塬", "壙", "濱", "県", "峠")

_PUNCTUATION = frozenset("，、；。？！")

# Content characters: every character appearing in any alias word. Typo
# augmentation must corrupt only these, never function words, so the model
# learns to lean on intact context instead of learning to ignore OOV chars.
_CONTENT_CHARS = frozenset(
    char
    for lexicon in LEXICONS
    for word in lexicon.alias
    for char in word
)


def region_partners(region: int) -> tuple[int, ...]:
    for group in REGION_NEAR_GROUPS:
        if region in group:
            return tuple(value for value in group if value != region)
    return ()


class RegionAugmentedWorld:
    """V6.0 world plus region-near, typo and conflict training augmentations."""

    def __init__(self, scale: int = 65536, seed: int = 60) -> None:
        self.world = ChineseAliasWorld(scale, seed=seed)
        self._rng = random.Random(seed + scale)

    # -- augmentation primitives ----------------------------------------------

    def region_negative_values(self, values: Sequence[int]) -> tuple[int, int, int, int] | None:
        partners = region_partners(int(values[2]))
        if not partners:
            return None
        region = partners[self._rng.randrange(len(partners))]
        return (int(values[0]), int(values[1]), region, int(values[3]))

    @staticmethod
    def conflict_negative_values(values: Sequence[int]) -> tuple[int, int, int, int]:
        # Same name + region, mode shifted by one (a real existing object).
        return (int(values[0]), int(values[1]), int(values[2]), (int(values[3]) + 1) % 16)

    def typo_query(self, values: Sequence[int], variant: int) -> str:
        text = self.world.query_text(values, variant)
        positions = [
            index for index, char in enumerate(text)
            if char in _CONTENT_CHARS
        ]
        if not positions:
            return text
        position = positions[self._rng.randrange(len(positions))]
        char = TYPO_POOL[self._rng.randrange(len(TYPO_POOL))]
        return text[:position] + char + text[position + 1 :]

    # -- training batch ---------------------------------------------------------

    def training_batch(
        self,
        objects: Sequence[ChineseSemanticObject],
        variants: Sequence[int],
        *,
        typo_rate: float = 0.15,
        region_rate: float = 0.4,
        seed_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Return (query_tokens, positive_tokens, [neg1_tokens, neg2_tokens]).

        Every sample gets exactly two hard negatives: the first is a region
        near-synonym swap (falling back to the V6.0 standard negative when the
        region has no partner or the rate misses), the second is always the
        V6.0 standard hard negative. The conflict negative is exposed as a
        data primitive but excluded from training after iteration 1 regressed
        the formal set.
        """
        self._rng = random.Random(seed_offset)
        query_texts = []
        first_negatives = []
        second_negatives = []
        for obj, variant in zip(objects, variants):
            if self._rng.random() < typo_rate:
                query_texts.append(self.typo_query(obj.values, int(variant)))
            else:
                query_texts.append(self.world.query_text(obj.values, int(variant)))
            region_neg = self.region_negative_values(obj.values)
            if region_neg is not None and self._rng.random() < region_rate:
                first_negatives.append(ChineseSemanticObject(-1, region_neg))
            else:
                first_negatives.append(self.world.hard_negative(obj, int(variant)))
            second_negatives.append(self.world.hard_negative(obj, int(variant)))

        query_tokens = torch.stack(
            [
                self.world.tokenizer.encode(text, self.world.query_length)[0]
                for text in query_texts
            ]
        )
        positive_tokens = self.world.encode_candidates(objects)
        negatives = [
            self.world.encode_candidates(first_negatives),
            self.world.encode_candidates(second_negatives),
        ]
        return query_tokens, positive_tokens, negatives
