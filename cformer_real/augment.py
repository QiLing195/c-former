# -*- coding: utf-8 -*-
"""Query paraphrase augmentation for the real-corpus line.

Variants are semantics-preserving rewrites of the generated known queries.
Templates deliberately reuse only vocabulary that already occurs in the main
dataset, so `MixedTokenizer` coverage stays at 1.0 (enforced by a unit test).
"""

from __future__ import annotations


def query_variants(text: str, meta: dict | None) -> list[str]:
    """Return the original text plus paraphrases built from company/series meta.

    Without usable meta (hand-written extras), only the original is returned:
    hand-written queries are never machine-rewritten.
    """
    if not meta:
        return [text]
    company = meta.get("company")
    series = meta.get("series")
    if not company or not series:
        return [text]
    return [
        f"{company} 的 {series} 系列最新模型是什么？",
        f"{series} 系列最新的模型是哪一个？",
        f"{company} 的 {series} 系列最新模型叫什么？",
        f"{company} 发布的 {series} 系列最新模型是什么？",
        f"{series} 系列中最新的一个模型是什么？",
    ]
