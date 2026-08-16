from __future__ import annotations

from dataclasses import dataclass
import random

from cformer_v60 import ChineseAliasWorld, LEXICONS
from cformer_v60c.data import TYPO_POOL, _CONTENT_CHARS

from .data import BLIND_PATTERNS, BlindSet

# OOV-dense natural unknown texts (more variants than the blind set's six).
EXTRA_UNKNOWN_TEXTS = (
    "今天天气不错，我们去公园散步吧",
    "量子香薰治疗仪",
    "给我讲讲这款设备的历史沿革和设计理念",
    "那台装置会飞的还会潜水",
    "红色导航装置",
    "青碧环境保护设备的运行区域在月球",
    "这台机器能做出咖啡和三明治吗",
    "帮我写一首关于秋天的诗",
    "昨天开会的纪要放在哪里了",
    "这种材料抗腐蚀还是耐高温",
    "附近有没有二十四小时营业的药店",
    "那本书的作者是谁来着",
    "把这段代码的性能优化一下",
    "这个按钮点了没反应是怎么回事",
    "下周的航班几点起飞",
    "这件商品的退货政策是什么",
    "邻居家的狗总在半夜叫怎么办",
    "这个数学公式怎么推导",
    "如何在服务器上配置防火墙",
    "那道菜要炖多久才入味",
)


@dataclass(frozen=True)
class CalibrationQuery:
    text: str
    target: tuple[int, int, int, int] | None
    expected: str  # known | typo | ambiguous | unknown
    note: str


class CalibrationSet:
    """Held-out calibration set, disjoint from the V6.0b blind set targets.

    Built from the same 12 template-external patterns but on NEW fold-4
    objects, plus typo, ambiguous and unknown queries. Its only purpose is
    threshold calibration; the blind set remains the final evaluation set.
    """

    def __init__(self) -> None:
        self.world = ChineseAliasWorld(65536, seed=60)
        blind = BlindSet()
        self.blind_targets = {query.target for query in blind.queries if query.target is not None}
        self.blind_texts = {query.text for query in blind.queries}
        self.queries: list[CalibrationQuery] = self._build()

    @staticmethod
    def _alias(axis: int, value: int) -> str:
        return LEXICONS[axis].alias[int(value)]

    def _build(self) -> list[CalibrationQuery]:
        queries: list[CalibrationQuery] = []
        rng = random.Random(60)

        # known: 12 patterns x 20 NEW fold-4 objects (disjoint from blind targets)
        heldout = self.world.heldout_objects(400)
        known_objects = [obj for obj in heldout if obj.values not in self.blind_targets][:240]
        for index, obj in enumerate(known_objects):
            pattern = BLIND_PATTERNS[index % len(BLIND_PATTERNS)]
            appearance, domain, region, mode = obj.values
            text = pattern.format(
                appearance=self._alias(0, appearance),
                domain=self._alias(1, domain),
                region=self._alias(2, region),
                mode=self._alias(3, mode),
                wrong_appearance=self._alias(0, (appearance + 8) % 16),
            )
            queries.append(CalibrationQuery(text, tuple(obj.values), "known", f"cal P{index % 12}"))

        # typo: corrupt one content character on 24 known-object queries
        for obj in known_objects[:24]:
            appearance, domain, region, mode = obj.values
            text = BLIND_PATTERNS[0].format(
                appearance=self._alias(0, appearance),
                domain=self._alias(1, domain),
                region=self._alias(2, region),
                mode=self._alias(3, mode),
                wrong_appearance=self._alias(0, (appearance + 8) % 16),
            )
            positions = [i for i, char in enumerate(text) if char in _CONTENT_CHARS]
            if positions:
                position = positions[rng.randrange(len(positions))]
                char = TYPO_POOL[rng.randrange(len(TYPO_POOL))]
                text = text[:position] + char + text[position + 1 :]
            queries.append(CalibrationQuery(text, tuple(obj.values), "typo", "cal typo"))

        # ambiguous: the 5 pair names (name-only queries must be rejected)
        for name_text in (
            "那台赤红导航装置现在在哪里",
            "蔚蓝临床救治装置的状态是什么",
            "金黄动力供给装置现在如何",
            "墨黑材质研究装置的状态如何",
            "橘红自动机械装置现在怎么样",
        ):
            queries.append(CalibrationQuery(name_text, None, "ambiguous", "cal name-only"))

        # unknown: OOV-dense / partial texts
        for text in EXTRA_UNKNOWN_TEXTS:
            if text not in self.blind_texts:
                queries.append(CalibrationQuery(text, None, "unknown", "cal unknown"))

        for query in queries:
            self.world.assert_identity_free([query.text])
        return queries
