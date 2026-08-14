from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Iterable, Sequence

import torch


_WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexicon:
    canonical: tuple[str, ...]
    alias: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.canonical) != len(self.alias):
            raise ValueError("canonical and alias vocabularies must have equal size")
        if len(set(self.canonical + self.alias)) != len(self.canonical + self.alias):
            raise ValueError("semantic words must be unique")


COLOR = Lexicon(
    tuple("red blue green gold silver black white purple orange yellow brown gray pink cyan lime indigo maroon teal navy beige coral mint lavender magenta bronze ochre pearl charcoal scarlet jade cream plum".split()),
    tuple("crimson azure emerald amber argent onyx ivory violet tangerine lemon umber slate rose turquoise chartreuse cobalt burgundy aqua midnight sand salmon spearmint lilac fuchsia copper sienna opal graphite vermilion malachite vanilla mauve".split()),
)
DOMAIN = Lexicon(
    tuple("navigation medicine energy education materials weather traffic communication agriculture finance geology robotics astronomy ecology logistics security acoustics optics chemistry biology architecture aviation maritime computing diplomacy forestry mining textiles ceramics mechanics hydrology archaeology".split()),
    tuple("routing clinical power learning substance climate transit signaling farming banking earthworks automation stargazing environment freight protection soundwork photonics reactions lifescience building flight seafaring informatics negotiation woodland excavation fabrics pottery machinery waterstudy antiquities".split()),
)
REGION = Lexicon(
    tuple("north south east west central coastal inland alpine polar tropical urban rural upper lower inner outer river desert forest island valley plateau harbor frontier orbital lunar solar deep shallow near far open closed remote local".split()),
    tuple("arctic austral sunrise sunset middle littoral interior mountain icebelt equatorial city countryside high low internal external fluvial arid woodland insular basin tableland port border celestial moonward sunward profound surface adjacent distant exposed sealed secluded nearby".split()),
)
MODE = Lexicon(
    tuple("stable changing active dormant direct inverse expanding contracting heating cooling rising falling emitting absorbing rotating static".split()),
    tuple("steady shifting awake sleeping forward mirrored growing shrinking warming chilling ascending descending radiating takingin spinning fixed".split()),
)

LEXICONS = (COLOR, DOMAIN, REGION, MODE)
RADICES = tuple(len(lexicon.canonical) for lexicon in LEXICONS)
MAX_WORLD_SIZE = math.prod(RADICES)


@dataclass(frozen=True)
class SemanticObject:
    """An evaluation label plus semantic attributes; the label is never rendered."""

    label: int
    values: tuple[int, int, int, int]


class WordTokenizer:
    PAD = 0
    UNK = 1

    def __init__(self) -> None:
        words = {
            "find", "object", "whose", "purpose", "is", "with", "and", "area", "behavior", "appears",
            "registered", "device", "used", "within", "relation", "links", "to",
            "transformation", "remains", "evidence", "describes",
        }
        for lexicon in LEXICONS:
            words.update(lexicon.canonical)
            words.update(lexicon.alias)
        self.words = ("<pad>", "<unk>", *sorted(words))
        self.index = {word: offset for offset, word in enumerate(self.words)}

    def tokenize(self, text: str) -> list[str]:
        return _WORD.findall(text.lower())

    def encode(self, text: str, length: int) -> tuple[torch.Tensor, float]:
        tokens = self.tokenize(text)
        ids = [self.index.get(token, self.UNK) for token in tokens[:length]]
        coverage = sum(token_id != self.UNK for token_id in ids) / max(1, len(ids))
        ids.extend([self.PAD] * (length - len(ids)))
        return torch.tensor(ids, dtype=torch.long), coverage

    @property
    def size(self) -> int:
        return len(self.words)


class OpenAliasWorld:
    """Compositional world with no parseable object IDs or identity links in text."""

    field_length = 7
    query_length = 18

    def __init__(self, scale: int, *, seed: int = 59) -> None:
        if not 1 < scale <= MAX_WORLD_SIZE:
            raise ValueError(f"scale must be in [2, {MAX_WORLD_SIZE}]")
        self.scale = scale
        self.seed = seed
        self.tokenizer = WordTokenizer()

    @staticmethod
    def _decode(value: int) -> tuple[int, int, int, int]:
        result = []
        for radix in RADICES:
            result.append(value % radix)
            value //= radix
        return tuple(result)  # type: ignore[return-value]

    def _permute(self, value: int) -> int:
        """Cycle-walk a keyed 20-bit Feistel permutation into the valid world."""
        while True:
            left, right = value >> 10, value & 1023
            for round_index in range(4):
                mixed = (
                    right * (0x1F5 + round_index * 2)
                    + self.seed * 0x27D4EB2D
                    + round_index * 0x165667B1
                )
                mixed ^= mixed >> 11
                left, right = right, (left ^ mixed) & 1023
            value = (left << 10) | right
            if value < MAX_WORLD_SIZE:
                return value

    def object_at(self, label: int) -> SemanticObject:
        if not 0 <= label < self.scale:
            raise IndexError(label)
        mixed = self._permute(label)
        return SemanticObject(label, self._decode(mixed))

    @staticmethod
    def family_fold(values: Sequence[int], folds: int = 5) -> int:
        # The split depends on the semantic combination, never on a rendered ID.
        return (values[0] * 3 + values[1] * 5 + values[2] * 7 + values[3] * 11) % folds

    @staticmethod
    def candidate_evidence(values: Sequence[int]) -> tuple[str, str, str, str]:
        color, domain, region, mode = (
            LEXICONS[field].canonical[int(value)] for field, value in enumerate(values)
        )
        return (
            f"registered {color} {domain} device",
            f"used within {region} area behavior {mode}",
            f"relation links {domain} to {region}",
            f"transformation remains {mode} and appears {color}",
        )

    @staticmethod
    def query_text(values: Sequence[int], *, omit_mode: bool = False) -> str:
        color, domain, region, mode = (
            LEXICONS[field].alias[int(value)] for field, value in enumerate(values)
        )
        ending = "" if omit_mode else f" and appears {mode}"
        return (
            f"find object whose purpose is {domain} within {region} area, "
            f"with {color} evidence{ending}"
        )

    def encode_queries(
        self, objects: Sequence[SemanticObject], *, omit_mode: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [
            self.tokenizer.encode(self.query_text(obj.values, omit_mode=omit_mode), self.query_length)
            for obj in objects
        ]
        return torch.stack([item[0] for item in encoded]), torch.tensor(
            [item[1] for item in encoded], dtype=torch.float32
        )

    def encode_candidates(self, objects: Sequence[SemanticObject]) -> torch.Tensor:
        rows = []
        for obj in objects:
            fields = [
                self.tokenizer.encode(text, self.field_length)[0]
                for text in self.candidate_evidence(obj.values)
            ]
            rows.append(torch.stack(fields))
        return torch.stack(rows)

    def objects(self, labels: Iterable[int]) -> list[SemanticObject]:
        return [self.object_at(int(label)) for label in labels]

    def heldout_objects(self, count: int, *, fold: int = 4) -> list[SemanticObject]:
        rng = random.Random(self.seed + 1009 * self.scale + fold)
        labels = list(range(self.scale))
        rng.shuffle(labels)
        result = []
        for label in labels:
            obj = self.object_at(label)
            if self.family_fold(obj.values) == fold:
                result.append(obj)
            if len(result) == count:
                break
        return result

    def ambiguous_objects(self, count: int) -> list[SemanticObject]:
        """Return objects whose first three fields collide with another live object."""
        groups: dict[tuple[int, int, int], list[SemanticObject]] = {}
        for label in range(self.scale):
            obj = self.object_at(label)
            groups.setdefault(obj.values[:3], []).append(obj)
        candidates = [items[0] for items in groups.values() if len(items) > 1]
        rng = random.Random(self.seed + 65537 + self.scale)
        rng.shuffle(candidates)
        return candidates[:count]

    def training_objects(
        self, count: int, *, holdout_fold: int = 4, seed_offset: int = 0
    ) -> list[SemanticObject]:
        rng = random.Random(self.seed + count + 7919 * seed_offset)
        result: list[SemanticObject] = []
        seen: set[tuple[int, int, int, int]] = set()
        while len(result) < count:
            values = tuple(rng.randrange(radix) for radix in RADICES)
            if values in seen or self.family_fold(values) == holdout_fold:
                continue
            seen.add(values)
            result.append(SemanticObject(-1, values))
        return result

    @staticmethod
    def hard_negative(obj: SemanticObject) -> SemanticObject:
        values = list(obj.values)
        values[3] = (values[3] + 1) % RADICES[3]
        return SemanticObject(-1, tuple(values))  # type: ignore[arg-type]

    @staticmethod
    def assert_identity_free(texts: Iterable[str]) -> None:
        for text in texts:
            if re.search(r"\d", text):
                raise AssertionError(f"parseable number leaked into semantic text: {text}")
