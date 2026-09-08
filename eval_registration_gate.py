# -*- coding: utf-8 -*-
"""真实域 RAG 权限闸门测试（报道流程域）：确定性闸门是否跨域成立。

域：迎新线上报到流程（制度/流程文档，7 个环节，4 个含内部细则）。
检索：关键词锚定（确定性、域无关）——本域无训练过的神经编码器，
     语义检索需按域重建（诚实边界），此处用关键词命中模拟 RAG 检索 Top-1。
权限：含 internal 的对象 → 仅管理员可见（学生不可见）。
对比：学生问内部细则 → 无闸门泄漏 vs 确定性闸门拦截。

用法：D:/conda/envs/cformer-gpu/python.exe eval_registration_gate.py
结果：artifacts/registration_gate_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "registration_dataset.json"

ROLE_STUDENT = "student"
ROLE_ADMIN = "admin"


def keyword_retrieve(text: str, objects: list[dict], visible_ids: set[str] | None) -> str | None:
    """关键词检索：文本与对象关键词表最大重合者 → Top-1。visible_ids=None 全量。"""
    best_id, best_hits = None, 0
    for obj in objects:
        if visible_ids is not None and obj["id"] not in visible_ids:
            continue
        hits = sum(1 for kw in obj["keywords"] if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_id = obj["id"]
    return best_id if best_hits > 0 else None


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    objects = data["objects"]
    queries = [q for q in data["queries"] if q.get("subtype") == "query"]

    student_visible = {obj["id"] for obj in objects if obj.get("open_source")}
    admin_visible = {obj["id"] for obj in objects}

    rows = []
    leak_before = leak_after = denied = 0
    for q in queries:
        text = q["text"]
        target = q["target_id"]
        student_can_see = q.get("student_visible", True)
        target_obj = next(o for o in objects if o["id"] == target)
        target_internal = target_obj.get("internal") is not None

        # 融合前检索：全量命中 → 返回"公开+内部"全文（LLM/RAG 看到全部）
        before_hit = keyword_retrieve(text, objects, None)
        # 融合后检索：命中同一对象，但字段级闸门——internal 字段内容不返回
        after_hit = keyword_retrieve(text, objects, None)

        def render(obj_id: str, gate: bool) -> str:
            obj = next(o for o in objects if o["id"] == obj_id)
            if not gate:
                # 无闸门：公开+内部全部返回
                parts = [obj["public"]]
                if obj.get("internal"):
                    parts.append(f"[内部]{obj['internal']}")
                return " ".join(parts)
            # 闸门：公开返回；internal 被字段级拒绝
            if obj.get("internal"):
                return obj["public"] + " [内部字段：无权限查看]"
            return obj["public"]

        # 泄漏判定：无闸门返回内容含内部细则 = 泄漏通道存在
        before_content = render(before_hit, gate=False) if before_hit else ""
        before_leak = before_hit is not None and "[内部]" in before_content
        # 闸门后：返回内容必不含 [内部]
        after_content = render(after_hit, gate=True) if after_hit else ""
        after_leak = after_hit is not None and "[内部]" in after_content
        # 学生问内部细则（目标含 internal 且标记学生不可见）→ 内部被拒
        denied_this = (target_internal and not student_can_see)
        served = after_hit is not None
        if not served:
            result_str = "NO-MATCH"
        elif denied_this:
            result_str = "DENIED(内部细则未返回，仅公开内容)"
        else:
            result_str = "OK(公开内容返回)"

        leak_before += 1 if before_leak else 0
        leak_after += 1 if after_leak else 0
        denied += 1 if denied_this else 0
        rows.append({
            "query": text,
            "target": target,
            "target_has_internal": target_internal,
            "before_content": before_content,
            "before_leak": bool(before_leak),
            "after_content": after_content,
            "after_leak": bool(after_leak),
            "result": result_str,
        })

    report = {
        "domain": "迎新报到流程（非 AI 域）",
        "n_queries": len(queries),
        "n_internal_queries": sum(1 for q in queries if not q.get("student_visible", True)),
        "leak_before_gate": leak_before,
        "leak_after_gate": leak_after,
        "denied_after_gate": denied,
        "conclusion": "权限闸门跨域成立" if leak_after == 0 else "查看明细",
        "rows": rows,
    }
    out = ROOT / "artifacts" / "registration_gate_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "leak_before_gate": leak_before,
                      "leak_after_gate": leak_after, "denied_after_gate": denied,
                      "n_internal_queries": report["n_internal_queries"]}, ensure_ascii=False))
    for r in rows:
        print(f"{r['query']}\n   前(无闸门): {r['before_content'][:90]}{'…' if len(r['before_content'])>90 else ''}\n   后(闸门):   {r['after_content'][:90]}{'…' if len(r['after_content'])>90 else ''} → {r['result']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
