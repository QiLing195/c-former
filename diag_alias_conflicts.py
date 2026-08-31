# -*- coding: utf-8 -*-
"""别名精确命中冲突诊断：找出 alias 查询中「精确命中对象 ≠ 目标对象」的案例。

现象：eval_identity_mixed 中 alias precise_hit_ratio=1.0 但 accuracy=88.4%——
所有 alias 查询都命中了 PreciseMatch，但有些命中返回的对象不是查询目标，
说明别名表存在错挂（同名别名指向错误对象 / 子串误命中）。

输出：artifacts/diag_alias_conflicts.json（冲突案例 + 别名表冲突分析）。

用法：
    D:/conda/envs/cformer-gpu/python.exe diag_alias_conflicts.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from cformer_v63.precise_match import PreciseMatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai_models_dataset.json"


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    objects = data["objects"]
    queries = data["queries"]
    precise = PreciseMatch(objects)
    id_to_name = {o["id"]: o["name"] for o in objects}

    conflicts = []
    alias_holders: dict[str, list[str]] = defaultdict(list)  # alias -> [object_id,...]
    for obj in objects:
        evidence_name = obj.get("evidence", {}).get("名称", "")
        for marker in ("也常写作", "也常被称作"):
            start = 0
            while True:
                idx = evidence_name.find(marker, start)
                if idx < 0:
                    break
                rest = evidence_name[idx + len(marker):]
                end = rest.find("，")
                chunk = rest if end < 0 else rest[:end]
                for piece in chunk.split("、"):  # 中文顿号
                    piece = piece.strip()
                    if piece and len(piece) >= 2:
                        alias_holders[piece].append(obj["id"])
                start = idx + len(marker)

    alias_conflicts = {a: ids for a, ids in alias_holders.items() if len(set(ids)) > 1}

    for q in queries:
        if q.get("subtype") != "alias" or q.get("split") != "heldout":
            continue
        hit = precise.hit(q["text"])
        target = q["target_id"]
        if hit is not None and hit.object_id != target:
            conflicts.append({
                "text": q["text"],
                "target_id": target,
                "target_name": id_to_name.get(target),
                "hit_id": hit.object_id,
                "hit_name": id_to_name.get(hit.object_id),
                "matched": hit.matched,
                "via": hit.via,
            })

    report = {
        "alias_conflicts_in_table": {k: v for k, v in list(alias_conflicts.items())[:20]},
        "n_alias_conflicts": len(conflicts),
        "conflicts": conflicts[:40],
    }
    output = ROOT / "artifacts" / "diag_alias_conflicts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done",
                      "n_alias_conflicts": len(conflicts),
                      "table_conflicts": {k: v for k, v in list(alias_conflicts.items())[:20]}},
                     ensure_ascii=False))
    for c in conflicts[:20]:
        print(f"  {c['text']} -> target={c['target_name']}({c['target_id']}) "
              f"hit={c['hit_name']}({c['hit_id']}) matched={c['matched']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
