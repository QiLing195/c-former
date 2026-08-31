# -*- coding: utf-8 -*-
"""V6.3 PreciseMatch：确定性精确匹配前置层（身份解析的架构加法）。

实证（diag_exact_match_coverage）：heldout 273 条 name 查询 100% 原样包含对象全名，
而神经层只做到 ~85%——让神经层重复做词法匹配能 100% 完成的事是浪费。
本层在神经层**之前**：对象全名/别名词法命中即返回（100%），未命中才交神经层。
与远程 V61c「精确别名 B-tree 命中即返回」同路线。

职责边界（对齐 C-Former「确定性优先，神经只做检索」）：
- 命中：100% 确定（全名/别名在问题文本中原样出现）；
- 未命中：返回 None，交神经层（描述性指代如"OpenAI 家最新那个"）；
- 近名歧义（"GPT" 裸系列名）：不属于精确命中（无唯一全名），交结构歧义规则。

本模块不依赖 torch，纯标准库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def norm(text: str) -> str:
    """归一化：去空白、转小写、全角冒号转半角——与 MixedTokenizer 的分词习惯对齐。"""
    return re.sub(r"\s+", "", text).lower().replace("：", ":")


@dataclass
class ExactHit:
    object_id: str
    matched: str  # 命中的全名或别名
    via: str      # "full_name" | "alias"


class PreciseMatch:
    """精确匹配前置层：对象全名 + 别名表，命中即返回。

    命中纪律（保持精确层 100% 正确，宁可未命中也不要错挂）：
    - 边界完整匹配：命中词在查询中的前后必须是「非字母数字」边界，
      避免短名截胡长名（`gpt-5` 截胡 `gpt-5-2`、`o1` 截胡 `doubao 1.5 pro`）；
    - 长词优先：全名/别名按长度降序尝试；
    - 未命中返回 None，交神经层（描述性指代）。
    """

    _BOUNDARY = re.compile(r"[a-z0-9]")

    def __init__(self, objects: list[dict]) -> None:
        # 全名表：norm(name) -> object_id（长名优先，避免"GPT-4"吞掉"GPT-4.1"）
        self.full_names: dict[str, str] = {}
        # 别名表：norm(alias) -> object_id
        self.aliases: dict[str, str] = {}
        for obj in objects:
            object_id = obj["id"]
            name = obj.get("name", "")
            if name:
                self.full_names[norm(name)] = object_id
            evidence_name = obj.get("evidence", {}).get("名称", "")
            # "也常写作X" / "也常被称作X、Y"
            for marker in ("也常写作", "也常被称作"):
                for m in re.finditer(re.escape(marker) + r"(.+?)(?:，|$)", evidence_name):
                    for piece in re.split(r"[、,，]", m.group(1)):
                        piece = piece.strip()
                        if piece and len(piece) >= 2:
                            self.aliases[norm(piece)] = object_id
        # 统一词表（全名 + 别名合并），长词优先——"GPT-5 最新版"(别名,10字)
        # 必须排在 "GPT-5"(全名,5字) 之前，否则别名被全名截胡
        table: dict[str, str] = {}
        for k, v in self.full_names.items():
            table.setdefault(k, v)
        for k, v in self.aliases.items():
            table.setdefault(k, v)  # 冲突时全名优先（对象名是权威）
        self._sorted = sorted(table.items(), key=lambda kv: -len(kv[0]))

    @classmethod
    def _bounded_hit(cls, text: str, needle: str) -> bool:
        """needle 在 text 中出现且前后是非字母数字边界（边界完整匹配）。"""
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx < 0:
                return False
            before_ok = idx == 0 or not cls._BOUNDARY.search(text[idx - 1])
            after = idx + len(needle)
            after_ok = after >= len(text) or not cls._BOUNDARY.search(text[after])
            if before_ok and after_ok:
                return True
            start = idx + 1

    def hit(self, text: str) -> ExactHit | None:
        """命中返回 ExactHit；未命中返回 None（交神经层）。长词优先 + 边界完整匹配。"""
        normalized = norm(text)
        for name, object_id in self._sorted:
            if name and len(name) >= 2 and self._bounded_hit(normalized, name):
                via = "full_name" if name in self.full_names else "alias"
                return ExactHit(object_id=object_id, matched=name, via=via)
        return None
