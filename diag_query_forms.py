# -*- coding: utf-8 -*-
"""句式泛化诊断：统计 train/heldout 的 name 查询句式分布，验证「heldout 句式未见于训练」假设。

问题：小对比模型对训练查询是记忆而非泛化（V62 报告），若 heldout 的 name 句式
（如"我想了解X这个模型"）在训练集里完全没出现过，则 name 13.5% 的失败主因
是句式泛化缺口，而不是对象/证据问题——提升路径 = 训练增加句式多样性。

用法：
    D:/conda/envs/cformer-gpu/python.exe diag_query_forms.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai_models_dataset.json"


def form_of(text: str) -> str:
    """把对象名替换为 X，得到句式模板。"""
    return re.sub(r"[A-Za-z0-9\u4e00-\u9fff.：·]+", "X", text)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    queries = data["queries"]
    by_split = {"train": Counter(), "heldout": Counter()}
    for q in queries:
        if q.get("kind") != "known":
            continue
        if q.get("subtype") not in ("name", "alias"):
            continue
        split = q.get("split", "train")
        by_split[split][form_of(q["text"])] += 1

    train_forms = set(by_split["train"])
    heldout_forms = set(by_split["heldout"])
    print(f"train 句式数: {len(train_forms)}")
    print(f"heldout 句式数: {len(heldout_forms)}")
    print(f"heldout 句式中有多少未出现在 train: {len(heldout_forms - train_forms)}")
    print()
    print("=== heldout 独有句式（train 未见过）===")
    for form, count in sorted(by_split["heldout"].items()):
        seen = "TRAIN-OK" if form in train_forms else "**UNSEEN**"
        print(f"{seen:12s} {form}  (x{count})")
    print()
    print("=== train 句式（供对照）===")
    for form, count in sorted(by_split["train"].items()):
        print(f"{form}  (x{count})")


if __name__ == "__main__":
    main()
