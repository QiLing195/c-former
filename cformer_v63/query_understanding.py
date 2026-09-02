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
    # 注意：裸"最新"不在此列——"最新报道/最新消息"是新闻用语，非"最新模型"语义
    "最新模型", "最新一代", "最新款", "最新版", "最新旗舰", "现在最新",
    "现在出到", "出到第", "家族现在", "现在最牛", "目前最强", "当前旗舰",
    "现在旗舰", "最近发布", "最近有什么新", "最近上映", "最近出",
    "刚发布", "新一代", "目前最新", "最牛", "出到", "现在最强",
    "最新的是", "最新的是啥", "现在出到第", "最近的是", "最近上映的那部",
    "最近出的", "最新开源", "最新推出的", "最新放出来的",
)
EARLIEST_WORDS = ("最早", "初代", "第一代", "首个", "第一个", "开山")
PREDECESSOR_WORDS = (
    "前一代", "前代", "上一代", "上一部", "前身", "直接前代", "前一部",
    "的前任", "上一任",
)
# 续集/后继 = 后一部（X 的续集 → X 的后继，方向与 predecessor 相反）
SUCCESSOR_WORDS = ("续集", "后继", "后一部", "下一代", "下一部")
# "继承"歧义词：带"现在/现行/如今" → latest（谁现在继承=现行继承国）；
# 否则由语境判（"前身政体是什么" 无继承词 → predecessor 由"前身"覆盖）
INHERIT_LATEST_WORDS = ("现在由谁继承", "现在继承", "现行继承", "如今的继承", "由谁继承", "解体后")
IDENTITY_WORDS = ("介绍", "是什么", "是哪个", "介绍一下", "我想了解", "帮我查", "资料", "是什么模型", "是什么电影", "是什么国家")
ATTRIBUTE_WORDS = ("首都", "导演", "类型", "语言", "年份", "哪一年", "开发者", "谁开发的", "谁拍的")
AMBIGUOUS_WORDS = ("有哪些", "一共有", "是不是", "哪一个", "系列有", "家族有")
UNKNOWN_WORDS = ("写一段", "散文", "下雨", "会怎么样", "给我讲个故事")

# 公司口语别名（锚定用，会进 Query.company；"X家" 由 __init__ 补全）
COMPANY_ALIASES = {
    "OpenAI": "OpenAI", "谷歌": "Google", "Google": "Google",
    "深度求索": "深度求索", "DeepSeek": "深度求索", "月之暗面": "月之暗面",
    "智谱": "智谱AI", "阿里": "阿里巴巴", "阿里巴巴": "阿里巴巴",
    "字节": "字节跳动", "字节跳动": "字节跳动", "Meta": "Meta", "Meta 家": "Meta",
    "微软": "Microsoft", "百度": "百度", "腾讯": "腾讯", "讯飞": "科大讯飞",
    "Anthropic": "Anthropic", "xAI": "xAI", "Mistral": "Mistral AI",
}

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
        self.company_aliases: dict[str, str] = dict(COMPANY_ALIASES)
        self.alias_map: dict[str, str] = {}  # 别名 -> 对象 name
        company_series: dict[str, dict[str, int]] = {}  # company -> {series: count}
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
                if series:
                    company_series.setdefault(company, {}).setdefault(series, 0)
                    company_series[company][series] += 1
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
        # 公司 → 主系列（对象数最多的系列；"OpenAI家最新" → GPT）
        self.company_main_series: dict[str, str] = {
            company: max(counts, key=counts.get) for company, counts in company_series.items()
        }
        # 对象名 → 所属系列（反向锚定："苏联解体后…" → 苏联对象 → 苏联继承系列）
        self.object_to_series: dict[str, str] = {}
        for obj in objects:
            oid = obj.get("id")
            series = obj.get("series")
            if oid and series and series in self.series_names:
                self.object_to_series[oid] = series
                name = obj.get("name")
                if name:
                    self.object_to_series.setdefault(name, series)
        # 公司口语别名：补 "X家" 形式（"OpenAI家""谷歌家"）
        extra_company = {}
        for alias, canonical in self.company_aliases.items():
            if alias.endswith("家"):
                continue
            extra_company[f"{alias}家"] = canonical
        self.company_aliases.update(extra_company)
        # 系列 → 成员版本号集合（库外版本越界检测：GPT-6 vs 库内 GPT-1..5.2）
        import re as _re
        self.series_version_numbers: dict[str, set[float]] = {}
        # 子系列前缀 → 最大版本（DeepSeek 的 "R2" → ("R", 2)；"GPT-5.2" → 整名模式）
        self.series_prefix_versions: dict[str, dict[str, float]] = {}
        for obj in objects:
            series = obj.get("series")
            name = obj.get("name", "")
            if series not in self.series_names or not name:
                continue
            # 抽 "GPT-5.2" 里的 5.2 / "Claude 5" 里的 5
            for match in _re.finditer(r"(?:^|[\s\-—])(\d+(?:\.\d+)?)", name):
                try:
                    self.series_version_numbers.setdefault(series, set()).add(
                        float(match.group(1)))
                except ValueError:
                    pass
            # 抽前缀-数字型子系列：DeepSeek-R2 → 前缀 "R" 版本 2；Kimi-K2.6 → "K" 2.6
            for match in _re.finditer(r"[\s\-—]([A-Za-z\u4e00-\u9fff]{1,3})[-—]?(\d+(?:\.\d+)?)$", name):
                prefix = match.group(1)
                try:
                    version = float(match.group(2))
                except ValueError:
                    continue
                bucket = self.series_prefix_versions.setdefault(series, {})
                bucket[prefix] = max(bucket.get(prefix, 0.0), version)
        # 词表排序（长词优先，避免"最新"吞掉"最新版"）
        self._sorted_series = sorted(self.series_names, key=len, reverse=True)
        self._sorted_companies = sorted(self.company_names, key=len, reverse=True)
        self._sorted_company_aliases = sorted(self.company_aliases.items(), key=lambda kv: -len(kv[0]))

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

        # 2) 意图分类（优先于对象名锚定！latest/predecessor 有意图词时
        #    绝不能被身份锚定截胡——「GPT-5.2 的前一代」必须是 predecessor）
        def first_hit(words) -> str | None:
            for word in words:
                if word in text:
                    return word
            return None

        intent = None
        matched = None
        for words, name in (
            (SUCCESSOR_WORDS, "successor"),
            (INHERIT_LATEST_WORDS, "latest"),   # "现在由谁继承" → latest（先于纯 predecessor）
            (PREDECESSOR_WORDS, "predecessor"),
            (EARLIEST_WORDS, "earliest"),
            (LATEST_WORDS, "latest"),
            (AMBIGUOUS_WORDS, "ambiguous"),
            (IDENTITY_WORDS, "identity"),
            (ATTRIBUTE_WORDS, "attribute"),
        ):
            hit = first_hit(words)
            if hit is not None:
                intent = name
                matched = hit
                break

        # 3) 锚定公司（口语别名优先，"OpenAI家" → OpenAI）
        for alias, canonical in self._sorted_company_aliases:
            if alias in text:
                query.company = canonical
                break
        if query.company is None:
            for company in self._sorted_companies:
                if company in text:
                    query.company = company
                    break

        # 4) 锚定系列（含"X系列"形式）
        for series in self._sorted_series:
            if series in text or f"{series.replace('系列', '')}系列" in text \
               or series.replace("系列", "") in text:
                query.series = series
                break

        # 5) 对象名/别名锚定：仅当意图是 identity/unknown 时用对象名解析身份；
        #    latest/predecessor/earliest 已带系列/公司，不需要对象名
        if intent in ("identity", None):
            for alias, name in sorted(self.alias_map.items(), key=lambda kv: -len(kv[0])):
                if alias and len(alias) >= 2 and alias in text:
                    query.series = name
                    query.matched_terms.append(f"alias:{alias}")
                    if intent is None:
                        intent = "identity"
                    break

        # 6) 意图落地 + 歧义兜底
        if matched:
            query.matched_terms.append(matched)
        if intent is not None:
            query.intent = intent
            # latest/successor/predecessor/earliest 需要系列：依次试 对象名→系列、公司主系列
            if query.series is None and intent in ("latest", "successor", "earliest", "predecessor"):
                # 反向锚定：文本含对象名（"苏联解体后…"→"苏联"对象→苏联继承系列）
                for oid_or_name, series in sorted(self.object_to_series.items(),
                                                  key=lambda kv: -len(kv[0])):
                    if oid_or_name and len(oid_or_name) >= 2 and oid_or_name in text:
                        query.series = series
                        query.matched_terms.append(f"object:{oid_or_name}")
                        break
            if query.series is None and query.company in self.company_main_series \
               and intent in ("latest", "earliest", "predecessor"):
                query.series = self.company_main_series[query.company]
        elif query.series or query.company:
            query.intent = "ambiguous"  # 带系列/公司但无意图词 → 歧义
            query.matched_terms.append("bare-series")

        # 7) 版本越界 → unknown（库外版本：GPT-6 / Claude 5 / Gemini 4 不在库中）
        #    歧义 ≠ 未知：歧义是库内有多个候选；越界版本号是库外，必须显式 unknown。
        #    判据：版本号不在库内集合，且大于库内该系列最大版本（4.8 在库、问 5 → 越界）。
        if query.series and query.intent in ("ambiguous", "identity", None):
            known_versions = self.series_version_numbers.get(query.series)
            if known_versions:
                max_known = max(known_versions)
                for match in re.finditer(r"(?:^|[^.\d])(\d+(?:\.\d+)?)(?:[^.\d]|$)", text):
                    try:
                        version = float(match.group(1))
                    except ValueError:
                        continue
                    if version > max_known and version not in known_versions:
                        query.intent = "unknown"
                        query.matched_terms.append(f"version-oob:{match.group(1)}")
                        break
            # 子系列越界：DeepSeek-R3（前缀 R 库内最大 R2）→ unknown
            if query.intent in ("ambiguous", "identity", None):
                prefixes = self.series_prefix_versions.get(query.series)
                if prefixes:
                    for prefix in sorted(prefixes, key=len, reverse=True):
                        for match in re.finditer(
                            re.escape(prefix) + r"[-—]?(\d+(?:\.\d+)?)", text
                        ):
                            try:
                                version = float(match.group(1))
                            except ValueError:
                                continue
                            if version > prefixes[prefix]:
                                query.intent = "unknown"
                                query.matched_terms.append(f"subseries-oob:{prefix}{match.group(1)}")
                                break
                        if query.intent == "unknown":
                            break
        return query
