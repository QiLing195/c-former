# -*- coding: utf-8 -*-
"""LLM 语义后端：用大模型做「语义匹配 + 覆盖判定」，替代关键词锚定与词表探针。

技术问题背景（#1 + #2）：
  - #1 关键词检索无语义：用户问"什么时候会被开除"，条款写"予以劝退" → 关键词 miss；
  - #2 空白识别靠人工 probe_rules：每加一个域就要手写规则，脆且不通用。

本模块把两件事合并为**一次 LLM 调用**：
  给定「问题 + 候选条款（id/标题/内容）」，让模型判断：
    a. 哪些条款与问题语义相关（检索）；
    b. 相关条款是否**明确回答**了问题（覆盖判定 → covered/gap）。
输出严格 JSON，便于程序解析。

设计约束：
  - 不引入本地模型依赖（只依赖 requests）→ 部署镜像仍可保持轻量；
  - 无 API key 或调用失败时**自动降级**（返回 None，由 GovLayer 回退关键词）；
  - 结果带缓存（同一问题不重复计费）；
  - 只把**候选条款**交给模型（大规模库先用关键词粗筛，避免全量塞进 prompt）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import requests

API_URL = "https://api.deepseek.com/chat/completions"
API_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是企业知识库的检索与覆盖判定助手。给定一个员工问题和若干制度条款，你需要：
1. 判断哪些条款与问题**语义相关**（即使措辞不同也算相关，例如"什么时候会被开除"与"予以劝退"相关）；
2. 判断这些相关条款是否**明确回答**了问题（有具体规则/标准/流程即为明确回答；只有泛泛表述则不算）。

只输出 JSON，不要任何其他文字：
{"relevant": ["条款id", ...], "covered": true 或 false, "missing": "若未覆盖，一句话说明缺少什么规则"}"""


@dataclass
class SemanticResult:
    relevant_ids: list[str]
    covered: bool
    missing: str = ""
    cached: bool = False


class LLMSemanticBackend:
    """LLM 语义匹配 + 覆盖判定后端（可注入 GovLayer）。"""

    def __init__(self, api_key: str | None = None, model: str = API_MODEL,
                 timeout: int = 30, max_candidates: int = 20) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.max_candidates = max_candidates
        self._cache: dict[str, SemanticResult] = {}
        self.calls = 0            # 统计实际 API 调用次数（演示/审计用）
        self.cache_hits = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ---- 核心：一次调用完成检索 + 覆盖判定 ----
    def match(self, question: str, candidates: list[dict],
              filter_fn: Callable[[str], str] | None = None) -> SemanticResult | None:
        """candidates: [{id, title, content}]（应为已按权限过滤后的内容）。
        filter_fn: 可选的内容过滤（如按角色裁剪字段）——确保只把可见内容交给模型。"""
        if not self.available or not candidates:
            return None
        candidates = candidates[: self.max_candidates]
        # 缓存键：问题 + 候选集合（含内容，保证权限裁剪后的结果也被区分）
        key_src = question + "|" + "|".join(
            f"{c['id']}:{hashlib.md5(c['content'].encode()).hexdigest()[:8]}" for c in candidates
        )
        key = hashlib.md5(key_src.encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            hit = self._cache[key]
            return SemanticResult(hit.relevant_ids, hit.covered, hit.missing, cached=True)

        blocks = []
        for c in candidates:
            content = filter_fn(c["content"]) if filter_fn else c["content"]
            blocks.append(f"[{c['id']}] 标题：{c['title']}\n内容：{content}")
        user_prompt = f"【员工问题】{question}\n\n【候选条款】\n" + "\n\n".join(blocks)

        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": user_prompt}],
                      "temperature": 0.0, "max_tokens": 300},
                timeout=self.timeout,
            )
            self.calls += 1
            if resp.status_code != 200:
                return None
            text = resp.json()["choices"][0]["message"]["content"].strip()
            result = self._parse(text)
            if result is None:
                return None
            self._cache[key] = result
            return result
        except Exception:  # noqa: BLE001 —— LLM 不可用时降级，不阻断主流程
            return None

    @staticmethod
    def _parse(text: str) -> SemanticResult | None:
        """稳健解析模型输出（容忍 markdown 代码块/多余文字）。"""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        ids = data.get("relevant") or []
        if not isinstance(ids, list):
            ids = [str(ids)]
        return SemanticResult(
            relevant_ids=[str(i) for i in ids],
            covered=bool(data.get("covered", False)),
            missing=str(data.get("missing", "") or ""),
        )
