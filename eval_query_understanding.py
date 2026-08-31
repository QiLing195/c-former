# -*- coding: utf-8 -*-
"""V6.3 理解层盲测：措辞变体问题端到端验证（理解层 → 递归层）。

核心问题：「GPT 家族现在最牛的是哪款？」这类**同义改写**问题，
理解层能否解析出 {intent=latest, series=GPT} 并路由到递归层答对。

盲测集设计（诚实原则）：
- 每题都改写措辞，**不出现系列原名**（GPT→"OpenAI 家"、Qwen→"阿里的通义"等），
  或只给公司名/模糊指代——模拟真实口语；
- 每题标注 ground truth 系列名（评测用，不喂给解析器）；
- 期望行为：理解层从公司名/别名锚定到系列 → 递归层答对；
- 若理解层无法锚定（intent=unknown 或 series=None）→ 计为「未理解」，
  即使答案猜对也不算数——测的是理解，不是检索。

用法：
    D:/conda/envs/cformer-gpu/python.exe eval_query_understanding.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cformer_v63 import RecursiveResolver, RelationGraph
from cformer_v63.query_understanding import QueryUnderstanding

ROOT = Path(__file__).resolve().parent

# (改写问题, 期望系列名, 期望意图, 期望答案对象 id 或 None)
BLIND = [
    # ---- latest 类（不出现系列原名）----
    ("OpenAI 家现在最牛的是哪款？", "GPT", "latest", "gpt-5-2"),
    ("谷歌现在出到第几代了？", "Gemini", "latest", "gemini-3-5-pro"),
    ("Meta 最新放出来的开源大模型叫什么？", "Llama", "latest", "llama-4"),
    ("阿里的通义现在家族最新的是哪款？", "Qwen", "latest", "qwen3-7-max"),
    ("字节家的豆包现在出到哪个版本了？", "豆包", "latest", "豆包 1.5 Pro".lower().replace(" ", "-")),
    ("深度求索家现在最新的是啥？", "DeepSeek", "latest", "deepseek-v4-pro"),
    ("月之暗面最新一代是哪个？", "Kimi", "latest", "kimi-k2-6"),
    ("智谱现在最强的开源编程模型是哪个？", "GLM", "latest", "glm-5-3"),
    ("漫威宇宙最近上映的那部是？", "漫威宇宙", "latest", "复仇者联盟4-终局之战"),
    ("指环王系列最近出的是哪部？", "指环王系列", "latest", "霍比特人"),
    ("苏联解体后现在继承的是哪个国家？", "苏联继承", "latest", "俄罗斯"),
    ("奥斯曼帝国现在由谁继承？", "奥斯曼帝国继承", "latest", "土耳其"),
    # ---- predecessor 类 ----
    ("OpenAI 家 GPT-5.2 的前一代是谁？", "GPT", "predecessor", "gpt-5-1"),
    ("哪个模型的直接前代是 GPT-4o？", "GPT", "predecessor", "gpt-4-1"),
    ("漫威宇宙里复仇者联盟4 的上一部是？", "漫威宇宙", "predecessor", "复仇者联盟"),
    ("哈利波特系列魔法石那部的续集是哪部？", "哈利波特系列", "predecessor", "哈利·波特与死亡圣器"),
    ("苏联的前身政体是什么？", "苏联继承", "predecessor", None),  # 前身是链头，无前代
    # ---- identity 类（改写锚定）----
    ("介绍一下 OpenAI 家那个叫 GPT 的闭源旗舰", "GPT", "identity", "gpt-5-2"),
    ("阿里的开源模型 Qwen3.6 是什么？", "Qwen3.6", "identity", "qwen3-6"),
    # ---- 未知（开放域，不得硬猜）----
    ("帮我写一首关于秋天的诗", None, "unknown", None),
    ("明天会下雨吗", None, "unknown", None),
    # ---- 歧义（系列名 + 无意图词）----
    ("GPT 是哪一个模型？", "GPT", "ambiguous", None),
    ("漫威宇宙系列有哪些电影？", "漫威宇宙", "ambiguous", None),
]


def main() -> None:
    # 加载三域对象（理解层词表 + 递归层图）
    all_objects: list[dict] = []
    for path in (ROOT / "data" / "ai_models_dataset.json",
                 ROOT / "data" / "movies_dataset.json",
                 ROOT / "data" / "countries_recursion.json"):
        all_objects.extend(json.loads(path.read_text(encoding="utf-8"))["objects"])
    understanding = QueryUnderstanding(all_objects)
    graph = RelationGraph(all_objects)
    resolver = RecursiveResolver(graph, max_depth=4)

    def resolve(text: str) -> tuple[str, str | None, str | None]:
        """端到端：理解层 → 递归层；identity/unknown/ambiguous 返回解析结果。"""
        query = understanding.parse(text)
        if query.intent in ("unknown", "ambiguous"):
            return query.intent, query.series, None
        if query.intent == "identity":
            return query.intent, query.series, None
        if query.series is None:
            return "unknown", None, None  # 无法锚定 = 未理解
        if query.intent == "latest":
            result = resolver.latest_of_series(query.series, world_version=query.as_of)
        elif query.intent == "earliest":
            heads = graph.series_heads(query.series)
            result = None
            if heads:
                head = min(heads, key=lambda h: graph.year_of(h))
                result = resolver.chain(head, 0)
        elif query.intent == "predecessor":
            # predecessor 需要对象名；盲测中该字段通过 query.series 传对象 id 语义——
            # 此处按「最近一次 latest 的链尾」无法定位，改为：若 series 指向具体对象则查表
            obj = next((o for o in all_objects if o["id"] == query.series), None)
            if obj and obj.get("predecessor"):
                result = resolver.predecessor_of(query.series)
            else:
                return "predecessor", query.series, None
        else:
            return query.intent, query.series, None
        if result is None:
            return query.intent, query.series, None
        if result.ok:
            return query.intent, query.series, result.answer_id
        return query.intent, query.series, f"!{result.reason}"

    rows = []
    ok = total = 0
    for text, expected_series, expected_intent, expected_answer in BLIND:
        intent, series, answer = resolve(text)
        # 评测：理解正确（intent+series 匹配）且答案正确
        understood = (intent == expected_intent) and (
            expected_series is None or series == expected_series)
        correct = understood and (
            expected_answer is None or answer == expected_answer)
        if expected_answer is None:
            correct = understood  # 未知/歧义只要求理解层不硬答
        ok += 1 if correct else 0
        total += 1
        rows.append({"text": text, "expected": (expected_intent, expected_series, expected_answer),
                     "got": (intent, series, answer), "correct": correct,
                     "understood": understood})

    report = {
        "accuracy": ok / total if total else float("nan"),
        "counts": {"total": total, "correct": ok},
        "rows": rows,
    }
    output = ROOT / "artifacts" / "v63_understanding_blind.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "accuracy": report["accuracy"],
                      "counts": report["counts"]}, ensure_ascii=False))
    for row in rows:
        mark = "OK " if row["correct"] else "XX "
        print(f"{mark}{row['text']}\n    expected={row['expected']}\n    got      ={row['got']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
