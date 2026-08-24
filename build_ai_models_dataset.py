# -*- coding: utf-8 -*-
"""把主流 AI 模型系列展开成 122 个对象的四证据数据集（查询多样化版）。

用法：
    D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py

输出：data/ai_models_dataset.json
要点：
- 每个对象有 2 训练 + 1 留出（heldout）名称查询；
- 有前代的对象加 1 训练 + 1 留出前代推理查询；
- 每个系列加 5 训练 + 2 留出「最新模型」推理查询；
- 歧义/未知查询也分 train/heldout，heldout 一律不参与训练。
注意：版本号/年份来自 2026 检索与常识，需人工核对后进入正式对象库。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (company, region, open_source, series, [(name, year, note)])
SERIES = [
    ("OpenAI", "美国", False, "GPT", [
        ("GPT-1", "2018", ""), ("GPT-2", "2019", ""), ("GPT-3", "2020", ""),
        ("GPT-3.5", "2022", ""), ("GPT-4", "2023", ""), ("GPT-4 Turbo", "2023", ""),
        ("GPT-4o", "2024", "多模态"), ("GPT-4o mini", "2024", "轻量"),
        ("GPT-4.1", "2025", ""), ("GPT-5", "2025", "前沿旗舰"),
        ("GPT-5.1", "2026", ""), ("GPT-5.2", "2026", "当前前沿闭源旗舰"),
    ]),
    ("OpenAI", "美国", False, "o 推理", [
        ("o1", "2024", "推理"), ("o1-mini", "2024", "推理轻量"),
        ("o3", "2025", "推理"), ("o3-mini", "2025", "推理轻量"),
    ]),
    ("Anthropic", "美国", False, "Claude", [
        ("Claude 1", "2023", ""), ("Claude 2", "2023", ""), ("Claude 2.1", "2023", ""),
        ("Claude 3 Opus", "2024", "旗舰"), ("Claude 3 Sonnet", "2024", "均衡"),
        ("Claude 3 Haiku", "2024", "轻量"), ("Claude 3.5 Sonnet", "2024", "均衡"),
        ("Claude 3.5 Haiku", "2024", "轻量"), ("Claude Opus 4", "2025", "旗舰"),
        ("Claude Sonnet 4", "2025", "均衡"), ("Claude Haiku 4", "2025", "轻量"),
        ("Claude Opus 4.5", "2025", "旗舰"), ("Claude Opus 4.7", "2026", "旗舰"),
        ("Claude Opus 4.8", "2026", "SWE-bench 领先"),
    ]),
    ("Google", "美国", False, "Gemini", [
        ("Gemini 1.0", "2023", ""), ("Gemini 1.5 Pro", "2024", "旗舰"),
        ("Gemini 1.5 Flash", "2024", "轻量"), ("Gemini 2.0 Flash", "2024", "轻量"),
        ("Gemini 2.0 Pro", "2025", "旗舰"), ("Gemini 2.5 Pro", "2025", "旗舰"),
        ("Gemini 2.5 Flash", "2025", "轻量"), ("Gemini 3 Pro", "2025", "旗舰"),
        ("Gemini 3 Flash", "2025", "轻量"), ("Gemini 3.5 Pro", "2026", "代理编码旗舰"),
        ("Gemini 3.5 Flash", "2026", "最快代理编码"),
    ]),
    ("Meta", "美国", True, "Llama", [
        ("Llama 1", "2023", ""), ("Llama 2", "2023", ""),
        ("Llama 3", "2024", ""), ("Llama 3.1", "2024", ""),
        ("Llama 3.2", "2024", "多模态"), ("Llama 3.3", "2024", ""),
        ("Llama 4", "2026", "开源多模态旗舰"),
    ]),
    ("xAI", "美国", False, "Grok", [
        ("Grok 1", "2023", ""), ("Grok 1.5", "2024", ""), ("Grok 2", "2024", ""),
        ("Grok 3", "2025", "旗舰"), ("Grok 4", "2025", "旗舰"), ("Grok 5", "2026", "旗舰"),
    ]),
    ("Mistral AI", "法国", True, "Mistral", [
        ("Mistral 7B", "2023", ""), ("Mixtral 8x7B", "2023", "MoE"),
        ("Mistral Small", "2024", "轻量"), ("Mistral Medium", "2024", "均衡"),
        ("Mistral Large 2", "2024", "旗舰"), ("Mistral Nemo", "2024", "轻量"),
        ("Mistral Large 3", "2025", "旗舰"), ("Mistral Small 3", "2025", "轻量"),
    ]),
    ("阿里巴巴", "中国", True, "Qwen", [
        ("Qwen 1", "2023", ""), ("Qwen 1.5", "2024", ""),
        ("Qwen2", "2024", ""), ("Qwen2.5", "2024", ""),
        ("Qwen3", "2025", "开源旗舰"), ("Qwen3.5", "2025", "开源旗舰"),
        ("Qwen3.6", "2026", "开源旗舰"), ("Qwen3.7-Max", "2026", "闭源国产旗舰第一"),
        ("Qwen-VL", "2024", "多模态"), ("Qwen2.5-Coder", "2024", "编程"),
    ]),
    ("深度求索", "中国", True, "DeepSeek", [
        ("DeepSeek-V1", "2024", ""), ("DeepSeek-V2", "2024", "MoE"),
        ("DeepSeek-V2.5", "2024", ""), ("DeepSeek-V3", "2024", "开源旗舰"),
        ("DeepSeek-R1", "2025", "推理"), ("DeepSeek-R1-Zero", "2025", "推理"),
        ("DeepSeek-V3.1", "2025", "开源旗舰"), ("DeepSeek-V4", "2026", "开源旗舰"),
        ("DeepSeek-V4-Pro", "2026", "开源旗舰"),
    ]),
    ("月之暗面", "中国", True, "Kimi", [
        ("Kimi K1", "2024", ""), ("Kimi K1.5", "2025", "推理"),
        ("Kimi K2", "2025", "开源"), ("Kimi K2.5", "2025", "开源旗舰"),
        ("Kimi-K2.6", "2026", "开源旗舰"),
    ]),
    ("智谱AI", "中国", True, "GLM", [
        ("GLM-4", "2024", ""), ("GLM-4.5", "2025", ""), ("GLM-4.6", "2025", ""),
        ("GLM-5", "2025", ""), ("GLM-5.1", "2026", "开源"),
        ("GLM-5.3", "2026", "开源编程最强"), ("GLM-Z1", "2025", "推理"),
    ]),
    ("MiniMax", "中国", True, "MiniMax", [
        ("MiniMax M1", "2025", ""), ("MiniMax M2", "2025", "开源"),
        ("MiniMax M2.5", "2025", "开源"), ("MiniMax M2.7", "2026", "开源"),
    ]),
    ("字节跳动", "中国", False, "豆包", [
        ("豆包 1.0", "2023", ""), ("豆包 1.5", "2025", "商用"),
        ("豆包 1.5 Pro", "2026", "推理突出"), ("Seed 1.5", "2025", "多模态"),
        ("Seed 2.0", "2025", "多模态"),
    ]),
    ("腾讯", "中国", False, "混元", [
        ("混元", "2023", ""), ("混元 Turbo", "2024", "商用"),
        ("混元 Large", "2024", "商用"), ("混元 T1", "2025", "推理"),
    ]),
    ("百度", "中国", False, "文心", [
        ("文心 ERNIE 3.0", "2021", ""), ("文心 ERNIE 3.5", "2023", ""),
        ("文心一言 4.0", "2023", "商用"), ("文心一言 4.5", "2025", "商用"),
        ("文心 4.5 Turbo", "2025", "商用"),
    ]),
    ("阶跃星辰", "中国", True, "Step", [
        ("Step-1", "2024", ""), ("Step-2", "2024", "开源"), ("Step-3", "2025", "开源"),
    ]),
    ("零一万物", "中国", True, "Yi", [
        ("Yi-1.5", "2024", "开源"), ("Yi-Lightning", "2024", "开源"),
        ("Yi-2", "2025", "开源"),
    ]),
    ("百川智能", "中国", True, "Baichuan", [
        ("Baichuan-13B", "2023", "开源"), ("Baichuan2", "2023", "开源"),
        ("Baichuan4", "2024", "开源"),
    ]),
    ("昆仑万维", "中国", True, "Skywork", [
        ("Skywork-13B", "2023", "开源"), ("Skywork 4.0", "2024", "开源"),
    ]),
]


def _prev(model_list, index):
    return model_list[index - 1][0] if index > 0 else None


def build():
    objects = []
    queries = []
    label = 0
    for company, region, open_source, series, models in SERIES:
        series_model_ids = []
        series_count = len(models)
        for index, (name, year, note) in enumerate(models):
            prev = _prev(models, index)
            is_latest = index == series_count - 1
            alias_note = "" if " " not in name else name.replace(" ", "")
            evidence = {
                "名称": f"这个模型的全称是 {name}，属于 {series} 系列"
                       + (f"，也常写作 {alias_note}" if alias_note and alias_note != name else ""),
                "属性": f"它由 {company} 开发，是{'开源' if open_source else '闭源'}模型"
                        + (f"，{note}" if note else ""),
                "关系": f"它在 {series} 系列中"
                        + (f"，前一代是 {prev}" if prev else "，是该系列早期版本")
                        + ("，是该系列最新版本" if is_latest else ""),
                "变化": f"它于 {year} 年发布",
            }
            object_id = name.lower().replace(" ", "-").replace(".", "-")
            objects.append({
                "id": object_id,
                "label": label,
                "name": name,
                "company": company,
                "region": region,
                "series": series,
                "open_source": open_source,
                "year": year,
                "note": note,
                "evidence": evidence,
            })
            series_model_ids.append(object_id)
            label += 1

            # 名称查询：2 训练 + 1 留出（身份在查询里，测基本匹配与句式稳健）
            queries.append({"text": f"介绍一下{name}这个模型", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
            queries.append({"text": f"{name}是什么模型？", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
            queries.append({"text": f"我想了解{name}这个模型", "target_id": object_id, "kind": "known", "subtype": "name", "split": "heldout"})

            # 前代推理查询：1 训练 + 1 留出（测「关系」证据）
            if prev:
                queries.append({"text": f"在{series}系列中，前一代是{prev}的模型是哪个？", "target_id": object_id, "kind": "known", "subtype": "predecessor", "split": "train"})
                queries.append({"text": f"哪个模型的直接前代是{prev}？", "target_id": object_id, "kind": "known", "subtype": "predecessor", "split": "heldout"})

        # 系列「最新模型」推理查询：5 训练 + 2 留出
        latest_id = series_model_ids[-1]
        for text in (
            f"{company}的{series}系列最新模型是什么？",
            f"{series}系列现在最新的是哪一款？",
            f"哪个是{company}最新的{series}模型？",
            f"{series}家族当前最新的版本是哪个？",
            f"{series}系列现在出到哪个版本了？",
        ):
            queries.append({"text": text, "target_id": latest_id, "kind": "known", "subtype": "latest", "split": "train"})
        for text in (
            f"{series}系列最近有什么新旗舰？",
            f"{company}刚发布的{series}模型是哪一个？",
        ):
            queries.append({"text": text, "target_id": latest_id, "kind": "known", "subtype": "latest", "split": "heldout"})

        # 歧义查询：2 训练 + 1 留出
        queries.append({"text": f"{series}是哪一个模型？", "target_id": None, "kind": "ambiguous", "split": "train"})
        queries.append({"text": f"{company}的{series}系列有哪些模型？", "target_id": None, "kind": "ambiguous", "split": "train"})
        queries.append({"text": f"{series}系列一共有哪些版本？", "target_id": None, "kind": "ambiguous", "split": "heldout"})

    # 跨系列已知（训练用）
    queries.extend([
        {"text": "OpenAI 最新发布的旗舰模型是哪一个？", "target_id": "gpt-5-2", "kind": "known", "subtype": "latest", "split": "train"},
        {"text": "谷歌推出的代理编码旗舰模型叫什么？", "target_id": "gemini-3-5-pro", "kind": "known", "subtype": "latest", "split": "train"},
        {"text": "Meta 最新的开源模型叫什么？", "target_id": "llama-4", "kind": "known", "subtype": "latest", "split": "train"},
    ])

    # 未知查询：训练 6 + 留出 5
    queries.extend([
        {"text": "GPT-6 是什么时候发布的？", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "介绍一下百度的文心 5.0", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "OpenAI 的 o5 推理模型参数量是多少？", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "Claude Opus 5 的最新版本是什么？", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "Mistral 的 Medium 最新型号是什么？", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "帮我写一段关于秋天的散文", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "DeepSeek 的 R2 推理模型什么时候发布？", "target_id": None, "kind": "unknown", "split": "heldout"},
        {"text": "谷歌的 Gemini 4 有什么新特性？", "target_id": None, "kind": "unknown", "split": "heldout"},
        {"text": "智谱的 GLM-6 参数规模是多少？", "target_id": None, "kind": "unknown", "split": "heldout"},
        {"text": "介绍一下字节跳动的豆包 2.0 模型", "target_id": None, "kind": "unknown", "split": "heldout"},
        {"text": "明天会下雨吗", "target_id": None, "kind": "unknown", "split": "heldout"},
    ])

    return {
        "meta": {
            "dataset": "ai_models_dataset",
            "description": "主流 AI 大模型四证据数据集（查询多样化版），版本号需人工核对",
            "status": "自动生成；查询分 train/heldout，heldout 不参与训练；版本号与年份需人工核对",
            "objects": len(objects),
            "queries": {
                "known": sum(1 for q in queries if q["kind"] == "known"),
                "ambiguous": sum(1 for q in queries if q["kind"] == "ambiguous"),
                "unknown": sum(1 for q in queries if q["kind"] == "unknown"),
                "train": sum(1 for q in queries if q.get("split", "train") == "train"),
                "heldout": sum(1 for q in queries if q.get("split", "heldout") == "heldout"),
            },
        },
        "objects": objects,
        "queries": queries,
    }


def main():
    dataset = build()
    output = ROOT / "data" / "ai_models_dataset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"objects={dataset['meta']['objects']} queries={dataset['meta']['queries']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
