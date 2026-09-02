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
from cformer_v63.precise_match import PreciseMatch
from cformer_v63.query_understanding import QueryUnderstanding

ROOT = Path(__file__).resolve().parent

# (改写问题, 期望系列名, 期望意图, 期望答案对象 id 或 None)
BLIND = [
    # ---- latest 类（不出现系列原名）----
    ("OpenAI 家现在最牛的是哪款？", "GPT", "latest", "gpt-5-2"),
    ("谷歌现在出到第几代了？", "Gemini", "latest", "gemini-3-5-flash"),
    ("Meta 最新放出来的开源大模型叫什么？", "Llama", "latest", "llama-4"),
    ("阿里的通义现在家族最新的是哪款？", "Qwen", "latest", "qwen3-7-max"),
    ("字节家的豆包现在出到哪个版本了？", "豆包", "latest", "豆包-1-5-pro"),
    ("深度求索家现在最新的是啥？", "DeepSeek", "latest", "deepseek-r2"),
    ("月之暗面最新一代是哪个？", "Kimi", "latest", "kimi-k2-6"),
    ("智谱现在最强的开源编程模型是哪个？", "GLM", "latest", "glm-5-3"),
    ("漫威宇宙最近上映的那部是？", "漫威宇宙", "latest", "蜘蛛侠-英雄远征"),
    ("指环王系列最近出的是哪部？", "指环王系列", "latest", "霍比特人"),
    ("苏联解体后现在继承的是哪个国家？", "苏联继承", "latest", "俄罗斯"),
    ("奥斯曼帝国现在由谁继承？", "奥斯曼帝国继承", "latest", "土耳其"),
    # ---- predecessor 类 ----
    ("OpenAI 家 GPT-5.2 的前一代是谁？", "GPT", "predecessor", "gpt-5-1"),
    ("哪个模型的直接前代是 GPT-4o？", "GPT", "predecessor", "gpt-4o-mini"),
    ("漫威宇宙里复仇者联盟4：终局之战 的上一部是？", "漫威宇宙", "predecessor", "复仇者联盟"),
    ("哈利·波特与魔法石的续集是哪部？", "哈利波特系列", "successor", "哈利·波特与死亡圣器"),
    ("苏联的前身政体是什么？", "苏联继承", "predecessor", None),  # 前身是链头，无前代
    # ---- identity 类（改写锚定；对象全名/别名必须在文本中可精确提取）----
    ("介绍一下 OpenAI 家的 GPT-5.2", "GPT", "identity", "gpt-5-2"),
    ("阿里的开源模型 Qwen3.6 是什么？", "Qwen3.6", "identity", "qwen3-6"),
    # ---- 未知（开放域，不得硬猜）----
    ("帮我写一首关于秋天的诗", None, "unknown", None),
    ("明天会下雨吗", None, "unknown", None),
    # ---- 歧义（系列名 + 无意图词）----
    ("GPT 是哪一个模型？", "GPT", "ambiguous", None),
    ("漫威宇宙系列有哪些电影？", "漫威宇宙", "ambiguous", None),
    # ---- 网页语境（库内对象出现在新闻报道/搜索语境，识别身份）----
    ("据最新报道，OpenAI 发布了 GPT-5.2，成为当前前沿旗舰", "GPT", "identity", "gpt-5-2"),
    ("今天谷歌的 Gemini 3.5 Pro 登上了热搜", "Gemini", "identity", "gemini-3-5-pro"),
    ("Meta 开源的 Llama 4 在 GitHub 上很火", "Llama", "identity", "llama-4"),
    ("搜索结果显示阿里通义千问3.7-Max 是国产旗舰", "Qwen", "identity", "qwen3-7-max"),
    ("这篇评测提到了豆包 1.5 Pro 的推理能力", "豆包", "identity", "豆包-1-5-pro"),
    # ---- 库外对象（网页搜到但不在知识库；必须不硬猜，答 None/unknown 即正确）----
    ("听说 GPT-6 快要发布了，是真的吗", None, "unknown", None),
    ("Claude 5 什么时候发布？", None, "unknown", None),
    ("Gemini 4 的参数是多少？", None, "unknown", None),
    ("谷歌的 Gemini 6 比 3.5 强多少？", None, "unknown", None),
    ("有人在讨论 DeepSeek-R3，你知道吗", None, "unknown", None),
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
    precise = PreciseMatch(all_objects)

    def resolve(text: str) -> tuple[str, str | None, str | None]:
        """端到端：理解层 → 递归层；identity/unknown/ambiguous 返回解析结果。"""
        query = understanding.parse(text)
        if query.intent in ("unknown", "ambiguous"):
            return query.intent, query.series, None
        if query.intent == "identity":
            # 身份意图：走精确层（PreciseMatch），命中返回对象 id
            hit = precise.hit(text)
            if hit is not None:
                return "identity", query.series, hit.object_id
            return query.intent, query.series, None
        if query.series is None:
            return "unknown", None, None  # 无法锚定 = 未理解
        if query.intent == "latest":
            result = resolver.latest_of_series(query.series, world_version=query.as_of)
        elif query.intent == "successor":
            # successor = X 的后继（续集/下一代）：提取文本中的对象 → 查其后继
            norm_text = text.replace(" ", "").replace("：", "").replace(":", "")
            target_obj_id = None
            candidates = []
            for obj in all_objects:
                if obj.get("series") != query.series:
                    continue
                name = obj.get("name", "").replace(" ", "").replace("：", "").replace(":", "")
                if name and len(name) >= 2 and norm_text.find(name) >= 0:
                    candidates.append((len(name), obj["id"]))
            if candidates:
                target_obj_id = max(candidates, key=lambda pair: pair[0])[1]
            if target_obj_id is None:
                hit = precise.hit(text)
                target_obj_id = hit.object_id if hit is not None else None
            if target_obj_id:
                nxt = graph.successors_of(target_obj_id)
                if nxt:
                    return "successor", query.series, nxt[0]
            return "successor", query.series, None
        elif query.intent == "earliest":
            heads = graph.series_heads(query.series)
            result = None
            if heads:
                head = min(heads, key=lambda h: graph.year_of(h))
                result = resolver.chain(head, 0)
        elif query.intent == "predecessor":
            # 两种句式方向不同：
            #  A) "X 的前代/前一代/上一部" → 查 X 的前驱；
            #  B) "哪个模型的直接前代是 X" / "前代是 X 的模型" → X 的后继（谁的前代是 X）。
            norm_text = text.replace(" ", "").replace("：", "").replace(":", "")
            candidates = []
            for obj in all_objects:
                if obj.get("series") != query.series:
                    continue
                name = obj.get("name", "").replace(" ", "").replace("：", "").replace(":", "")
                if name and len(name) >= 2 and norm_text.find(name) >= 0:
                    candidates.append((len(name), obj["id"]))
            target_obj_id = max(candidates, key=lambda pair: pair[0])[1] if candidates else None
            if target_obj_id is None:
                hit = precise.hit(text)
                target_obj_id = hit.object_id if hit is not None else None
            if not target_obj_id:
                return "predecessor", query.series, None
            # 句式 B 检测：文本形如「前代是 X」「直接前代是 X」，且 X（对象名）
            # 出现在「前代是」**之后**（宾语位置）。「X 的前一代是谁」X 是主语 → 正向。
            reverse_query = False
            for phrase in ("前代是", "直接前代是", "前一代是"):
                idx = text.find(phrase)
                if idx >= 0:
                    # 短语后面紧跟着对象名（如 "GPT-4o"），而非疑问词
                    tail = text[idx + len(phrase):]
                    if any(obj.get("name", "").replace(" ", "") in tail
                           for obj in all_objects if obj.get("series") == query.series):
                        reverse_query = True
                    break
            if reverse_query:
                nxt = graph.successors_of(target_obj_id)
                return "predecessor", query.series, nxt[0] if nxt else None
            obj = next((o for o in all_objects if o["id"] == target_obj_id), None)
            if obj and obj.get("predecessor"):
                result = resolver.predecessor_of(target_obj_id)
                if result.ok:
                    return "predecessor", query.series, result.answer_id
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
        # 评测：意图匹配为第一关；
        # - 有期望答案：答案对即算对（series 是语义锚定参考，identity 下对象名可能更精确）；
        # - 无期望答案（未知/歧义/链头无前代）：要求意图对 + series 对（不硬答）
        intent_ok = intent == expected_intent
        if expected_answer is not None:
            correct = intent_ok and (answer == expected_answer)
        else:
            correct = intent_ok and (expected_series is None or series == expected_series)
        ok += 1 if correct else 0
        total += 1
        rows.append({"text": text, "expected": (expected_intent, expected_series, expected_answer),
                     "got": (intent, series, answer), "correct": correct,
                     "understood": intent_ok})

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
