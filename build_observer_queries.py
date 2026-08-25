# -*- coding: utf-8 -*-
"""生成观测点（observer）查询集：同一模型库、不同观测点 -> 不同但正确的答案。

用法：先运行 build_ai_models_dataset.py 生成对象库，再运行本脚本：
    D:/conda/envs/cformer-gpu/python.exe build_observer_queries.py
输出：data/observer_queries.json

设计要点（对齐路线图 §8）：
- 观测点是「确定性可见性过滤器」，不是知识副本：visible_labels 在生成期算好；
- 身份不变性：同一对象在不同合法观测点下解析结果一致；
- 权限/防泄漏：观测点不可见的对象，即使存在于对象库，也不得被解析出来；
- 关键协议：identity_text（不含观测点）只给神经身份模型，observer 只做确定性过滤——
  身份解析先于观测点注入，观测点从不进入身份模型的输入。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai_models_dataset.json"
OUT = ROOT / "data" / "observer_queries.json"

CHINESE_COMPANIES = {
    "阿里巴巴", "深度求索", "月之暗面", "智谱AI", "MiniMax", "字节跳动",
    "腾讯", "百度", "阶跃星辰", "零一万物", "百川智能", "昆仑万维",
}

# 观测点 -> 对象过滤谓词（确定性，只依赖结构化字段）
OBSERVERS = {
    "开源": lambda o: bool(o["open_source"]),
    "闭源": lambda o: not o["open_source"],
    "中国": lambda o: o["company"] in CHINESE_COMPANIES,
    "美国": lambda o: o["region"] == "美国",
    "2026年": lambda o: o["year"] == 2026,
    "2025年": lambda o: o["year"] == 2025,
    "编程": lambda o: "编程" in o["note"] or "编码" in o["note"] or "SWE" in o["note"] or "代理" in o["note"],
    "旗舰": lambda o: "旗舰" in o["note"],
    "推理": lambda o: "推理" in o["note"],
    "多模态": lambda o: "多模态" in o["note"],
}


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    objects = data["objects"]

    series_order: dict[str, list[dict]] = {}
    for obj in objects:
        series_order.setdefault(obj["series"], []).append(obj)

    queries = []

    # A) 系列最新（按观测点过滤后按年份取最新）：不同观测点可能给出不同答案
    for series, objs in series_order.items():
        for observer, predicate in OBSERVERS.items():
            visible = [obj for obj in objs if predicate(obj)]
            if not visible:
                continue
            latest_visible = max(visible, key=lambda obj: (obj["year"], objs.index(obj)))
            queries.append({
                "text": f"从{observer}视角看，{series}系列最新的模型是哪个？",
                "identity_text": f"{series}系列最新的模型是哪个？",
                "observer": observer,
                "kind": "selection",
                "target_label": latest_visible["label"],
                "visible_labels": [obj["label"] for obj in visible],
            })

    # B) 身份不变性：同一对象在不同观测点下身份一致
    for obj in objects[:12]:
        for observer in ("开源", "闭源", "中国"):
            visible = [o for o in objects if OBSERVERS[observer](o)]
            if obj["label"] not in [o["label"] for o in visible]:
                continue
            queries.append({
                "text": f"从{observer}视角看，介绍一下{obj['name']}",
                "identity_text": f"介绍一下{obj['name']}",
                "observer": observer,
                "kind": "invariance",
                "target_label": obj["label"],
                "visible_labels": [o["label"] for o in visible],
            })

    # C) 权限/防泄漏：观测点不可见的对象不得被解析出来（对每个不可见对象都生成）
    for series, objs in series_order.items():
        for observer, predicate in OBSERVERS.items():
            invisible = [obj for obj in objs if not predicate(obj)]
            visible = [obj for obj in objs if predicate(obj)]
            if not invisible or not visible:
                continue
            for forbidden in invisible:
                queries.append({
                    "text": f"从{observer}视角看，{series}系列里有{forbidden['name']}吗？",
                    "identity_text": f"{series}系列里有{forbidden['name']}吗？",
                    "observer": observer,
                    "kind": "permission",
                    "target_label": -1,
                    "forbidden_label": forbidden["label"],
                    "visible_labels": [obj["label"] for obj in visible],
                })

    payload = {
        "meta": {
            "description": "观测点查询集：身份解析(神经) + 观测点可见性过滤(确定性) 的联合验证",
            "observers": list(OBSERVERS),
            "queries": {kind: sum(1 for q in queries if q["kind"] == kind) for kind in ("selection", "invariance", "permission")},
        },
        "queries": queries,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
