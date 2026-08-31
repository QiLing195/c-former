# -*- coding: utf-8 -*-
"""V6.3 QueryUnderstanding：把自然语言问题解析为结构化查询（理解层 v1，确定性）。

为什么需要它：身份层盲测 5.6–19.4%（V62 报告）、神经递归 v2 只有随机特征——
「GPT 家族现在最牛的是哪款？」这类**措辞变体**问题，小对比模型在训练查询上
是记忆而非泛化。C-Former 哲学：确定性治理优先，神经只做检索。理解问题同样
走确定性词法解析，而不是指望神经网络泛化。

职责（在身份层/递归层**之前**）：
    问题文本
      → 意图分类（意图词表）：latest / earliest / predecessor / identity /
        attribute / ambiguous / unknown
      → 锚定：系列名 / 公司名 / 别名（从数据集自动构建词表）
      → 时间切点 as_of（"截至 2024 年"）
      → 结构化 Query{intent, series, company, as_of, matched_terms}
      → 路由：latest/predecessor → 递归层；identity → 身份层；ambiguous → 结构歧义

理解层只做「问题 → 结构化意图」，不做答案；答不出来返回 intent=unknown
（与 V6.0b 盲测「未知零误支持」同纪律）。

本模块不依赖 torch，纯标准库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 意图词表

LATEST_WORDS = (
    "最新", "现在最新", "最近发布", "最新一代", "最新款", "最新版", "现在出到",
    "出到第", "家族现在", "现在最牛", "目前最强", "当前旗舰", "现在旗舰",
    "最新旗舰", "最近有什么新", "刚发布", "新一代", "目前最新",
)
EARLIEST_WORDS = ("最早", "初代", "第一代", "首个", "第一个", "开山")
PREDECESSOR_WORDS = (
    "前一代", "前代", "上一代", "上一部", "前身", "直接前代", "前一部",
    "的前任", "继承", "后继", "续集", "上一任",
)
IDENTITY_WORDS = ("介绍", "是什么", "是哪个", "介绍一下", "我想了解", "帮我查", "资料", "是什么模型", "是什么电影", "是什么国家")
ATTRIBUTE_WORDS = ("首都", "导演", "类型", "语言", "年份", "哪一年", "上映", "发布", "开发者", "公司", "谁开发的", "谁拍的")
AMBIGUOUS_WORDS = ("有哪些", "一共有", "是不是", "哪一个", "系列有", "家族有")
UNKNOWN_WORDS = ("写一段", "散文", "下雨", "会怎么样", "给我讲个故事")

# ---------------------------------------------------------------- 时间切点

AS_OF_PATTERNS = (
    re.compile(r"截至\s*((?:19|20)\d{2})\s*年"),
    re.compile(r"截止\s*((?:19|20)\d{2})\s*年"),
    re.compile(r"到\s*((?:19|20)\d{2})\s*年(?:为止)?"),
    re.compile(r"((?:19|20)\d{2})\s*年(?:的)?时候"),
)

# ---------------------------------------------------------------- 数据结构


@dataclass
class Query:
    intent: str  # latest | earliest | predecessor | identity | attribute | ambiguous | unknown
    series: str | None = None
    company: str | None = None
    as_of: int | None = None
    matched_terms: list[str] = field(default_factory=list)
    raw: str = ""


class QueryUnderstanding:
    """确定性理解层：从数据集对象构建词表，把问题解析为结构化 Query。"""

    def __init__(self, objects: list[dict]) -> None:
        self.series_names: set[str] = set()
        self.company_names: set[str] = set()
        self.alias_map: dict[str, str] = {}  # 别名 -> 对象 name
        for obj in objects:
            series = obj.get("series")
            company = obj.get("company")
            name = obj.get("name", "")
            if series and series not in ("", "科幻", "爱情", "动作", "剧情", "动画",
                                         "战争", "悬疑", "武侠", "喜剧", "奇幻", "亚洲",
                                         "欧洲", "北美洲", "南美洲", "非洲", "大洋洲"):
                self.series_names.add(series)
            if company and company not in ("国家", "电影", "AI"):
                self.company_names.add(company)
            if name:
                self.alias_map[name] = name
                for alias in obj.get("evidence", {}).get("名称", "").split("、"):
                    alias = alias.strip()
                    if alias and len(alias) <= 8:
                        self.alias_map[alias] = name
            # 名称证据里的别名（"也常被称作X、Y"）
            evidence_name = obj.get("evidence", {}).get("名称", "")
            m = re.search(r"也常被称作(.+?)(?:，|$)", evidence_name)
            if m:
                for alias in m.group(1).split("、"):  # 含中英文混合分隔
                    alias = alias.strip()
                    if alias and len(alias) <= 12:
                        self.alias_map[alias] = name
        # 词表排序（长词优先，避免"最新"吞掉"最新版"）
        self._sorted_series = sorted(self.series_names, key=len, reverse=True)
        self._sorted_companies = sorted(self.company_names, key=len, reverse=True)

    # ------------------------------------------------------------ 解析入口

    def parse(self, text: str) -> Query:
        query = Query(intent="unknown", raw=text)

        # 0) 时间切点
        for pattern in AS_OF_PATTERNS:
            match = pattern.search(text)
            if match:
                query.as_of = int(match.group(1))
                break

        # 1) 明显未知（先于一切：开放域问题不硬猜）
        if any(word in text for word in UNKNOWN_WORDS):
            return query

        # 2) 锚定（理解层只做解析，不保证答案存在）
        for series in self._sorted_series:
            if series in text or series.replace("系列", "") in text:
                query.series = series
                break
        for company in self._sorted_companies:
            if company in text:
                query.company = company
                break
        # 3) 别名锚定 → 命中即身份查询（对象名唯一）
        for alias, name in sorted(self.alias_map.items(), key=lambda kv: -len(kv[0])):
            if alias and len(alias) >= 2 and alias in text:
                query.series = name  # 用对象名作为系列键（身份查询用）
                query.intent = "identity"
                query.matched_terms.append(f"alias:{alias}")
                return query

        # 4) 意图分类（优先级：前代 > 最早 > 最新 > 歧义 > 身份 > 属性）
        def first_hit(words) -> str | None:
            for word in words:
                if word in text:
                    return word
            return None

        if (hit := first_hit(PREDECESSOR_WORDS)) is not None:
            query.intent = "predecessor"
            query.matched_terms.append(hit)
        elif (hit := first_hit(EARLIEST_WORDS)) is not None:
            query.intent = "earliest"
            query.matched_terms.append(hit)
        elif (hit := first_hit(LATEST_WORDS)) is not None:
            query.intent = "latest"
            query.matched_terms.append(hit)
        elif (hit := first_hit(AMBIGUOUS_WORDS)) is not None:
            query.intent = "ambiguous"
            query.matched_terms.append(hit)
        elif (hit := first_hit(IDENTITY_WORDS)) is not None:
            query.intent = "identity"
            query.matched_terms.append(hit)
        elif (hit := first_hit(ATTRIBUTE_WORDS)) is not None:
            query.intent = "attribute"
            query.matched_terms.append(hit)

        # 5) 带系列/公司但无意图词 → 歧义（"GPT 是哪个模型"）
        if query.intent == "unknown" and (query.series or query.company):
            query.intent = "ambiguous"
            query.matched_terms.append("bare-series")
        return query
