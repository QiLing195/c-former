from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable, Sequence

import torch


_TOKEN = re.compile(r"[a-z]+|[\u3400-\u9fff]")


@dataclass(frozen=True)
class ChineseLexicon:
    canonical: tuple[str, ...]
    alias: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.canonical) != 16 or len(self.alias) != 16:
            raise ValueError("V6.0 semantic axes must contain sixteen values")


APPEARANCE = ChineseLexicon(
    tuple("红色 蓝色 绿色 金色 银色 黑色 白色 紫色 橙色 黄色 棕色 灰色 粉色 青色 靛色 铜色".split()),
    tuple("赤红 蔚蓝 翠绿 金黄 银白 墨黑 纯白 淡紫 橘红 明黄 栗色 石灰 桃粉 青碧 靛蓝 古铜".split()),
)
DOMAIN = ChineseLexicon(
    tuple("导航 医疗 能源 教育 材料 气象 交通 通信 农业 金融 地质 机器人 天文 生态 物流 安防".split()),
    tuple("路径指引 临床救治 动力供给 教学培训 材质研究 天气观测 运输调度 信息联络 农事生产 资金管理 地层勘探 自动机械 星体观测 环境保护 货物配送 安全防护".split()),
)
REGION = ChineseLexicon(
    tuple("北方 南方 东方 西方 中部 沿海 内陆 高原 极地 热带 城市 乡村 上游 下游 港口 边境".split()),
    tuple("北部 南部 东部 西部 中央 海岸 腹地 高地 冰原 赤道城区 城镇 田野 河流上段 河流下段 港湾 边界地区".split()),
)
MODE = ChineseLexicon(
    tuple("稳定 变化 活跃 休眠 正向 逆向 扩张 收缩 加热 冷却 上升 下降 发射 吸收 旋转 静止".split()),
    tuple("保持不变 正在改变 正在运作 暂停工作 向前运行 反向运行 持续扩大 逐渐缩小 温度升高 温度降低 向上移动 向下移动 对外释放 向内接收 持续转动 停止移动".split()),
)

LEXICONS = (APPEARANCE, DOMAIN, REGION, MODE)
MAX_WORLD_SIZE = 16**4


_QUERY_TEMPLATES = (
    "请寻找位于{region}、用于{domain}、呈现{appearance}并且{mode}的对象",
    "我需要一个外观{appearance}的{domain}装置，它处在{region}而且{mode}",
    "目标用途是{domain}，位置在{region}，外观为{appearance}，目前{mode}",
    "忽略背景中的其他装置，只查询{region}内承担{domain}用途、表现{appearance}且{mode}的目标",
    "不要{wrong_appearance}的对象，需要{appearance}外观；用途是{domain}，位于{region}并且{mode}",
    "排除{wrong_mode}状态；查找{region}的{appearance}{domain}装置，它实际{mode}",
)


@dataclass(frozen=True)
class ChineseSemanticObject:
    label: int
    values: tuple[int, int, int, int]


class ChineseCharacterTokenizer:
    PAD = 0
    UNK = 1

    def __init__(self) -> None:
        text_parts: list[str] = []
        for lexicon in LEXICONS:
            text_parts.extend(lexicon.canonical)
            text_parts.extend(lexicon.alias)
        text_parts.extend(re.sub(r"\{[^}]+\}", "", template) for template in _QUERY_TEMPLATES)
        text_parts.extend((
            "登记名称装置",
            "对象属性位于运行状态",
            "关系服务领域并关联区域",
            "变化保持模式外观表现",
            "未知概念没有登记证据",
        ))
        tokens = sorted({token for part in text_parts for token in _TOKEN.findall(part.lower())})
        self.tokens = ("<pad>", "<unk>", *tokens)
        self.index = {token: offset for offset, token in enumerate(self.tokens)}

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN.findall(text.lower())

    def encode(self, text: str, length: int) -> tuple[torch.Tensor, float]:
        tokens = self.tokenize(text)[:length]
        ids = [self.index.get(token, self.UNK) for token in tokens]
        coverage = sum(token_id != self.UNK for token_id in ids) / max(1, len(ids))
        ids.extend([self.PAD] * (length - len(ids)))
        return torch.tensor(ids, dtype=torch.long), coverage

    @property
    def size(self) -> int:
        return len(self.tokens)


class ChineseAliasWorld:
    """Identity-free Chinese object world with order and negation hard cases."""

    query_length = 64
    field_length = 24

    def __init__(self, scale: int, *, seed: int = 60) -> None:
        if not 1 < scale <= MAX_WORLD_SIZE:
            raise ValueError(f"scale must be in [2, {MAX_WORLD_SIZE}]")
        self.scale = scale
        self.seed = seed
        self.tokenizer = ChineseCharacterTokenizer()

    def _permute(self, value: int) -> int:
        left, right = value >> 8, value & 255
        for round_index in range(4):
            mixed = right * (93 + round_index * 2) + self.seed * 131 + round_index * 197
            mixed ^= mixed >> 7
            left, right = right, (left ^ mixed) & 255
        return (left << 8) | right

    @staticmethod
    def _decode(value: int) -> tuple[int, int, int, int]:
        return (
            value & 15,
            (value >> 4) & 15,
            (value >> 8) & 15,
            (value >> 12) & 15,
        )

    def object_at(self, label: int) -> ChineseSemanticObject:
        if not 0 <= label < self.scale:
            raise IndexError(label)
        return ChineseSemanticObject(label, self._decode(self._permute(label)))

    @staticmethod
    def family_fold(values: Sequence[int], folds: int = 5) -> int:
        return (values[0] * 3 + values[1] * 5 + values[2] * 7 + values[3] * 11) % folds

    @staticmethod
    def candidate_evidence(values: Sequence[int]) -> tuple[str, str, str, str]:
        appearance, domain, region, mode = (
            LEXICONS[field].canonical[int(value)] for field, value in enumerate(values)
        )
        return (
            f"登记名称为{appearance}{domain}装置",
            f"对象属性是位于{region}，运行状态为{mode}",
            f"关系证据表明它服务{domain}领域并关联{region}区域",
            f"变化证据表明它保持{mode}模式，外观表现为{appearance}",
        )

    @staticmethod
    def query_text(values: Sequence[int], variant: int = 0) -> str:
        appearance, domain, region, mode = (
            LEXICONS[field].alias[int(value)] for field, value in enumerate(values)
        )
        # Offset eight is self-inverse in a sixteen-value axis. Therefore both
        # "not A, choose B" and "not B, choose A" occur with identical token
        # multisets; an order-insensitive model cannot exploit an unordered pair.
        wrong_appearance = APPEARANCE.alias[(int(values[0]) + 8) % 16]
        wrong_mode = MODE.alias[(int(values[3]) + 8) % 16]
        return _QUERY_TEMPLATES[variant % len(_QUERY_TEMPLATES)].format(
            appearance=appearance,
            domain=domain,
            region=region,
            mode=mode,
            wrong_appearance=wrong_appearance,
            wrong_mode=wrong_mode,
        )

    def encode_queries(
        self,
        objects: Sequence[ChineseSemanticObject],
        variants: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if variants is None:
            variants = [index % len(_QUERY_TEMPLATES) for index in range(len(objects))]
        encoded = [
            self.tokenizer.encode(self.query_text(obj.values, int(variant)), self.query_length)
            for obj, variant in zip(objects, variants)
        ]
        return torch.stack([item[0] for item in encoded]), torch.tensor(
            [item[1] for item in encoded], dtype=torch.float32
        )

    def encode_candidates(self, objects: Sequence[ChineseSemanticObject]) -> torch.Tensor:
        rows = []
        for obj in objects:
            rows.append(
                torch.stack(
                    [
                        self.tokenizer.encode(text, self.field_length)[0]
                        for text in self.candidate_evidence(obj.values)
                    ]
                )
            )
        return torch.stack(rows)

    def objects(self, labels: Iterable[int]) -> list[ChineseSemanticObject]:
        return [self.object_at(int(label)) for label in labels]

    def training_objects(
        self, count: int, *, holdout_fold: int = 4, seed_offset: int = 0
    ) -> list[ChineseSemanticObject]:
        rng = random.Random(self.seed + count + seed_offset * 7919)
        result: list[ChineseSemanticObject] = []
        seen: set[tuple[int, int, int, int]] = set()
        while len(result) < count:
            values = tuple(rng.randrange(16) for _ in range(4))
            if values in seen or self.family_fold(values) == holdout_fold:
                continue
            seen.add(values)
            result.append(ChineseSemanticObject(-1, values))
        return result

    def heldout_objects(self, count: int, *, fold: int = 4) -> list[ChineseSemanticObject]:
        rng = random.Random(self.seed + self.scale * 1009 + fold)
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

    def ambiguous_objects(self, count: int) -> list[ChineseSemanticObject]:
        groups: dict[tuple[int, int, int], list[ChineseSemanticObject]] = {}
        for label in range(self.scale):
            obj = self.object_at(label)
            groups.setdefault(obj.values[:3], []).append(obj)
        candidates = [items[0] for items in groups.values() if len(items) > 1]
        rng = random.Random(self.seed + self.scale + 65537)
        rng.shuffle(candidates)
        return candidates[:count]

    @staticmethod
    def partial_query_text(values: Sequence[int]) -> str:
        appearance, domain, region = (
            LEXICONS[field].alias[int(values[field])] for field in range(3)
        )
        return f"请查找{region}区域中用于{domain}且外观为{appearance}的对象，状态信息没有说明"

    def encode_partial_queries(
        self, objects: Sequence[ChineseSemanticObject]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [
            self.tokenizer.encode(self.partial_query_text(obj.values), self.query_length)
            for obj in objects
        ]
        return torch.stack([item[0] for item in encoded]), torch.tensor(
            [item[1] for item in encoded], dtype=torch.float32
        )

    @staticmethod
    def hard_negative(obj: ChineseSemanticObject, variant: int) -> ChineseSemanticObject:
        values = list(obj.values)
        if variant % len(_QUERY_TEMPLATES) == 4:
            values[0] = (values[0] + 8) % 16
        elif variant % len(_QUERY_TEMPLATES) == 5:
            values[3] = (values[3] + 8) % 16
        else:
            values[3] = (values[3] + 5) % 16
        return ChineseSemanticObject(-1, tuple(values))  # type: ignore[arg-type]

    @staticmethod
    def assert_identity_free(texts: Iterable[str]) -> None:
        for text in texts:
            if re.search(r"\d", text) or "object_id" in text.lower() or "label" in text.lower():
                raise AssertionError(f"identity leakage in text: {text}")
