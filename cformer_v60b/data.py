from __future__ import annotations

from dataclasses import dataclass

from cformer_v60 import ChineseAliasWorld, LEXICONS

# Distinctive fixed phrases of the six V6.0 templates. A blind query must
# never contain any of them, otherwise it could be produced by the training
# generator and the set would not be blind.
FORBIDDEN_FRAGMENTS = (
    "请寻找位于",
    "我需要一个外观",
    "目标用途是",
    "忽略背景中的其他装置",
    "排除",
    "不要",
)

# Hand-authored natural Chinese query patterns (12), distinct from the six
# V6.0 templates. All content words come from the frozen alias vocabularies
# so that the frozen checkpoint and tokenizer remain directly applicable.
BLIND_PATTERNS = (
    "请帮我查找{region}的{domain}设备，外观{appearance}，目前{mode}",
    "我需要一台{appearance}的{domain}装置，位置在{region}，状态{mode}",
    "{region}那边用于{domain}的{appearance}设备，还在{mode}吗",
    "目标对象是{region}的{appearance}{domain}，它当前{mode}",
    "要{appearance}外观的{domain}，地点{region}，运行{mode}",
    "不是{wrong_appearance}，要{appearance}的{domain}，位于{region}，状态{mode}",
    "请定位{region}的{appearance}{domain}装置，其状态为{mode}",
    "{domain}用途的装置，{region}区域内，外观{appearance}，目前{mode}",
    "查找外观{appearance}、位于{region}、用于{domain}且{mode}的对象",
    "那台{region}的{appearance}{domain}，现在是不是{mode}",
    "忽略别的设备，只要{region}的{appearance}{domain}，状态{mode}",
    "查询{region}的{appearance}{domain}装置，它的实际状态是{mode}",
)

_EXPECTED = ("known", "hard", "ambiguous", "disambiguated", "unknown", "conflict")


@dataclass(frozen=True)
class BlindQuery:
    text: str
    target: tuple[int, int, int, int] | None
    expected: str
    note: str

    def __post_init__(self) -> None:
        if self.expected not in _EXPECTED:
            raise ValueError(f"unknown expected category: {self.expected}")


class BlindSet:
    """Hand-authored blind set over the V6.0 four-axis object space.

    The world scale is fixed to 65,536 (=16**4) so that every value tuple in
    [0,16)**4 denotes a real object and any target can be placed in the bank.
    """

    def __init__(self) -> None:
        self.world = ChineseAliasWorld(65536, seed=60)
        self.queries: list[BlindQuery] = []
        self.ambiguous_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
        self._build()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _alias(axis: int, value: int) -> str:
        return LEXICONS[axis].alias[int(value)]

    @staticmethod
    def _fold(values: tuple[int, int, int, int]) -> int:
        return (values[0] * 3 + values[1] * 5 + values[2] * 7 + values[3] * 11) % 5

    def coverage(self, query: BlindQuery) -> float:
        _, cov = self.world.tokenizer.encode(query.text, self.world.query_length)
        return float(cov)

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        queries: list[BlindQuery] = []

        # 12 template-external pattern queries over holdout (fold 4) objects.
        for index, obj in enumerate(self.world.heldout_objects(12)):
            appearance, domain, region, mode = obj.values
            text = BLIND_PATTERNS[index % len(BLIND_PATTERNS)].format(
                appearance=self._alias(0, appearance),
                domain=self._alias(1, domain),
                region=self._alias(2, region),
                mode=self._alias(3, mode),
                wrong_appearance=self._alias(0, (appearance + 8) % 16),
            )
            queries.append(
                BlindQuery(
                    text=text,
                    target=tuple(obj.values),
                    expected="known",
                    note=f"pattern P{index + 1}",
                )
            )

        # 4 bespoke hand-written queries, targets chosen on holdout fold 4.
        # Indices follow the ALIAS vocabularies (alias index == canonical index).
        bespoke = (
            ((2, 11, 4, 0), "帮我看下中央那台翠绿的自动机械，它目前是保持不变吗", "定制口语"),
            ((4, 12, 7, 8), "那台银白的星体观测装置位于高原，目前温度升高", "定制书面"),
            ((7, 8, 11, 6), "要淡紫的农事生产装置放在乡村，现在持续扩大", "定制口语"),
            ((13, 13, 5, 0), "位于海岸的青碧环境保护装置状态保持不变", "定制书面"),
        )
        for values, text, note in bespoke:
            if self._fold(values) != 4:
                raise AssertionError(f"bespoke target not on holdout fold: {values}")
            queries.append(BlindQuery(text=text, target=values, expected="known", note=note))

        # 3 typo / colloquial hard cases (one character corrupted to an OOV char).
        hard = (
            ((2, 11, 4, 0), "帮我看下中央那台翠録的自动机械，它目前是保持不变吗", "错字 録→绿"),
            ((4, 12, 7, 8), "那台银白的星提观测装置位于高原，目前温度升高", "错字 提→体"),
            ((7, 8, 11, 6), "要淡紫的农事生产装置放在乡衬，现在持续扩大", "错字 衬→村"),
        )
        for values, text, note in hard:
            queries.append(BlindQuery(text=text, target=values, expected="hard", note=note))

        # 5 same-name pairs: identical appearance+domain, differing region/mode.
        # Values use the ALIAS vocabularies (alias index == canonical index).
        pairs = (
            ((0, 0, 0, 1), (0, 0, 1, 0), "那台赤红导航装置现在在哪里"),
            ((1, 1, 2, 8), (1, 1, 3, 9), "蔚蓝临床救治装置的状态是什么"),
            ((3, 2, 5, 6), (3, 2, 6, 14), "金黄动力供给装置现在如何"),
            ((5, 4, 7, 2), (5, 4, 8, 15), "墨黑材质研究装置的状态如何"),
            ((8, 11, 14, 10), (8, 11, 15, 11), "橘红自动机械装置现在怎么样"),
        )
        for first, second, text in pairs:
            if first[:2] != second[:2]:
                raise AssertionError("same-name pair does not share name axes")
            self.ambiguous_pairs.append((first, second))
            queries.append(
                BlindQuery(text=text, target=None, expected="ambiguous", note=f"同名对 {first}/{second}")
            )

        # 3 name + region + mode queries that must resolve to a unique object.
        # Targets are on holdout fold 4 and share the pair name families.
        disambiguated = (
            ((0, 0, 0, 9), "北部那台赤红导航装置现在是温度降低"),
            ((1, 1, 2, 12), "蔚蓝临床救治装置在东部并对外释放"),
            ((3, 2, 5, 0), "金黄动力供给装置在沿海并保持不变"),
        )
        for values, text in disambiguated:
            queries.append(BlindQuery(text=text, target=values, expected="disambiguated", note="同名+区域+状态"))

        # 6 natural unknowns: OOV-dense text or deliberately partial info.
        unknown = (
            ("今天天气不错，我们去公园散步吧", "OOV 密集自然口语"),
            ("量子香薰治疗仪", "OOV 密集新实体"),
            ("红色导航装置", "仅两个规范轴，信息缺失"),
            ("给我讲讲这款设备的历史沿革和设计理念", "OOV 密集提问"),
            ("青碧环境保护设备的运行区域在月球", "OOV 区域词"),
            ("那台装置会飞的还会潜水", "OOV 密集描述"),
        )
        for text, note in unknown:
            queries.append(BlindQuery(text=text, target=None, expected="unknown", note=note))

        # 3 conflict probes (reported, not gated).
        conflict = (
            ("北部那台赤红导航装置现在是正在改变吗", "对 (0,0,0,1) 断言冲突状态"),
            ("有人说北部那台赤红导航装置保持不变，又有人说它在正在改变", "双状态断言"),
            ("沿海金黄动力供给装置究竟是持续扩大还是持续转动", "双候选引用"),
        )
        for text, note in conflict:
            queries.append(BlindQuery(text=text, target=None, expected="conflict", note=note))

        for query in queries:
            self.world.assert_identity_free([query.text])
            for fragment in FORBIDDEN_FRAGMENTS:
                if fragment in query.text:
                    raise AssertionError(f"blind query contains V6.0 template fragment {fragment!r}: {query.text}")
        self.queries = queries

    # -- queries by category ------------------------------------------------

    def by_expected(self, category: str) -> list[BlindQuery]:
        return [query for query in self.queries if query.expected == category]
