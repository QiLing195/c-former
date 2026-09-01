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
    """归一化：全角→半角、去空白、转小写、统一分隔符、繁→简、口语版本号。

    顺序（幂等）：
    1. 全角 ASCII → 半角；常见全角标点 → 半角；
    2. 繁简体映射（核心字对表 + 转写）；
    3. 口语版本号：「5点2/5 点 2」→「5-2」，「五二」→「52」；
    4. 统一分隔符：`·`/`．`/`－`/`／`/`_` → `-`；
    5. 去空白、转小写。
    """
    text = text.translate(str.maketrans({
        "　": " ", "，": ",", "。": ".", "（": "(", "）": ")",
        "：": ":", "；": ";", "？": "?", "！": "!", "·": "-", "．": "-", "－": "-",
    }))
    text = text.translate(str.maketrans(
        {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}  # 全角 ASCII → 半角
    ))
    # 繁→简（核心字对：覆盖对象名/别名/证据中的高频繁体字）
    text = text.translate(str.maketrans(SIMPLIFIED_MAP))
    # 口语版本号：「5点2」「5 点 2」「5.2」→「5-2」；「五二」→「52」
    text = re.sub(r"(\d)\s*[点點]\s*(\d)", r"\1-\2", text)
    text = re.sub(r"[一二三四五六七八九零〇]+", _cn_numeral, text)
    text = re.sub(r"[\s/＿_]+", "", text)
    return text.lower()


_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_numeral(match: "re.Match[str]") -> str:
    """中文数字串 → 阿拉伯数字（仅支持 0-99，超出原样保留）。"""
    s = match.group(0)
    if len(s) > 2:
        return s
    value = 0
    for ch in s:
        if ch not in _CN_DIGITS:
            return s
        value = value * 10 + _CN_DIGITS[ch]
    return str(value)


# 繁→简核心映射（覆盖高频繁体字；OpenCC 全集为生产选项，本表为轻量子集）
SIMPLIFIED_MAP = {
    "義": "义", "問": "问", "稱": "称", "為": "为", "麼": "么", "這": "这",
    "裡": "里", "關": "关", "係": "系", "變": "变", "學": "学", "體": "体",
    "國": "国", "開": "开", "發": "发", "佈": "布", "與": "与", "對": "对",
    "們": "们", "說": "说", "話": "话", "還": "还", "來": "来", "東": "东",
    "際": "际", "後": "后", "麼": "么", "萬": "万", "點": "点", "產": "产",
    "優": "优", "曆": "历", "曆": "历", "構": "构", "標": "标", "廣": "广",
    "視": "视", "評": "评", "證": "证", "認": "认", "讓": "让", "讀": "读",
    "當": "当", "會": "会", "寫": "写", "復": "复", "見": "见", "長": "长",
}


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
        # 无分隔变体：`gpt-5-2` 同时登记 `gpt52`，匹配 `gpt5.2` 这类口语写法
        self._loose: dict[str, str] = {}
        for k, v in table.items():
            stripped = k.replace("-", "")
            if stripped and stripped != k and len(stripped) >= 2:
                self._loose.setdefault(stripped, v)  # 冲突时保留先登记的（长词优先序）
        self._sorted = sorted(table.items(), key=lambda kv: -len(kv[0]))
        self._sorted_loose = sorted(self._loose.items(), key=lambda kv: -len(kv[0]))

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
        # 1) 严格词表（含分隔符），边界完整匹配
        for name, object_id in self._sorted:
            if name and len(name) >= 2 and self._bounded_hit(normalized, name):
                via = "full_name" if name in self.full_names else "alias"
                return ExactHit(object_id=object_id, matched=name, via=via)
        # 2) 无分隔变体（`gpt5.2` ↔ `gpt-5-2`）：两边去 `-` 后比较
        stripped_query = normalized.replace("-", "")
        for name, object_id in self._sorted_loose:
            if name and len(name) >= 2 and self._bounded_hit(stripped_query, name):
                via = "full_name" if name.replace("-", "") in {
                    k.replace("-", "") for k in self.full_names} else "alias"
                return ExactHit(object_id=object_id, matched=name, via=via)
        return None
