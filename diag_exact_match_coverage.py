# -*- coding: utf-8 -*-
"""精确匹配覆盖诊断：heldout name 查询里，有多少可以靠「对象名/别名词法命中」直接答对。

问题：身份层神经部分 name heldout ~85%，但若查询文本原样包含对象名或别名，
一个确定性精确匹配层（词法表命中即返回）可以 100% 接住——这是远程 V61c
「精确别名 B-tree 命中即返回」路线的本地验证。

诊断输出（artifacts/diag_exact_match_coverage.json）：
- heldout name 查询总数；
- 其中「文本含对象全名」的数量与占比；
- 其中「文本含任一别名」的数量与占比；
- 未被精确覆盖的（需神经层处理）数量。

用法：
    D:/conda/envs/cformer-gpu/python.exe diag_exact_match_coverage.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai_models_dataset.json"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().replace("：", ":")


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    objects = data["objects"]
    queries = data["queries"]

    # 对象全名 + 别名表（从证据「名称」字段抽别名）
    full_names: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for obj in objects:
        name = obj["name"]
        object_id = obj["id"]
        full_names[norm(name)] = object_id
        evidence_name = obj.get("evidence", {}).get("名称", "")
        for m in re.finditer(r"也常被称作(.+?)(?:，|$)", evidence_name):
            for alias in m.group(1).split("、"):  # 中文顿号 + 英文逗号
                for piece in re.split(r"[、,，]", alias):
                    piece = piece.strip()
                    if piece and len(piece) >= 2:
                        aliases[norm(piece)] = object_id
    for obj in objects:
        for alias_key in ("也常写作", "也常被称作"):
            pass

    heldout_names = [q for q in queries if q.get("split") == "heldout"
                     and q.get("subtype") == "name"]
    covered_by_full = []
    covered_by_alias = []
    uncovered = []
    for q in heldout_names:
        text = norm(q["text"])
        target = q["target_id"]
        target_name = next((o["name"] for o in objects if o["id"] == target), None)
        if target_name and norm(target_name) in text:
            covered_by_full.append(q)
            continue
        if target in aliases.values():
            # 是否文本命中该对象某个别名
            obj_aliases = [a for a, oid in aliases.items() if oid == target]
            if any(a in text for a in obj_aliases):
                covered_by_alias.append(q)
                continue
        uncovered.append(q)

    report = {
        "heldout_name_total": len(heldout_names),
        "covered_by_full_name": len(covered_by_full),
        "covered_by_alias": len(covered_by_alias),
        "uncovered": len(uncovered),
        "coverage_rate": round((len(covered_by_full) + len(covered_by_alias)) / len(heldout_names), 4),
        "uncovered_samples": [q["text"] for q in uncovered[:20]],
    }
    output = ROOT / "artifacts" / "diag_exact_match_coverage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", **{k: v for k, v in report.items() if k != "uncovered_samples"}},
                     ensure_ascii=False))
    print("uncovered 样例:")
    for text in report["uncovered_samples"]:
        print(f"  - {text}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
