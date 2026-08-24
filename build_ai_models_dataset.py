# -*- coding: utf-8 -*-
r"""把主流 AI 模型系列展开成 200+ 个对象的四证据数据集。

用法：
    D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py

输出：data/ai_models_dataset.json（覆盖旧版 18 对象种子集）。

事实来源与核对约定：
- 条目默认为编写时已知事实；凡年份/版本号/命名口径不确定或超出可靠记忆的，
  一律在条目第 4 位标 True（needs_review），并在 meta 中汇总计数；
- needs_review=True 的对象在人工或检索核对之前不得进入正式对象库。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REVIEW = True  # 条目第 4 位写 True 表示 needs_review


def _entry(name, year, note="", needs_review=False):
    return (name, year, note, needs_review)


# (company, region, open_source, series, [(name, year, note[, needs_review])])
SERIES = [
    ("OpenAI", "美国", False, "GPT", [
        _entry("GPT-1", "2018"), _entry("GPT-2", "2019"),
        _entry("GPT-3", "2020"), _entry("GPT-3.5", "2022"),
        _entry("GPT-4", "2023"), _entry("GPT-4 Turbo", "2023"),
        _entry("GPT-4o", "2024", "多模态"), _entry("GPT-4o mini", "2024", "轻量"),
        _entry("GPT-4.1", "2025"), _entry("GPT-5", "2025", "前沿旗舰"),
        _entry("GPT-5.1", "2025"), _entry("GPT-5.2", "2025", "前沿闭源旗舰"),
        _entry("GPT-5.4", "2026", "旗舰，1M上下文"),
    ]),
    ("OpenAI", "美国", False, "o 推理", [
        _entry("o1", "2024", "推理"), _entry("o1-mini", "2024", "推理轻量"),
        _entry("o1-pro", "2025", "推理旗舰"),
        _entry("o3", "2025", "推理"), _entry("o3-mini", "2025", "推理轻量"),
        _entry("o4-mini", "2025", "推理轻量"),
    ]),
    ("Anthropic", "美国", False, "Claude", [
        _entry("Claude 1", "2023"), _entry("Claude 2", "2023"),
        _entry("Claude 2.1", "2023"),
        _entry("Claude 3 Opus", "2024", "旗舰"), _entry("Claude 3 Sonnet", "2024", "均衡"),
        _entry("Claude 3 Haiku", "2024", "轻量"),
        _entry("Claude 3.5 Sonnet", "2024", "均衡"), _entry("Claude 3.5 Haiku", "2024", "轻量"),
        _entry("Claude Opus 4", "2025", "旗舰"), _entry("Claude Sonnet 4", "2025", "均衡"),
        _entry("Claude Haiku 4", "2025", "轻量"),
        _entry("Claude Sonnet 4.5", "2025", "均衡"),
        _entry("Claude Haiku 4.5", "2025", "轻量"),
        _entry("Claude Opus 4.5", "2025", "旗舰"),
        _entry("Claude Opus 4.6", "2026", ""),
        _entry("Claude Sonnet 4.6", "2026", "均衡"),
        _entry("Claude Opus 4.7", "2026", "旗舰"),
        _entry("Claude Opus 4.8", "2026", "旗舰，1M上下文"),
        _entry("Claude Sonnet 5", "2026", "均衡"),
        _entry("Claude Opus 5", "2026", "旗舰"),
    ]),
    ("Google", "美国", False, "Gemini", [
        _entry("Gemini 1.0", "2023"),
        _entry("Gemini 1.5 Pro", "2024", "旗舰"), _entry("Gemini 1.5 Flash", "2024", "轻量"),
        _entry("Gemini 2.0 Flash", "2024", "轻量"), _entry("Gemini 2.0 Pro", "2025", "旗舰"),
        _entry("Gemini 2.5 Pro", "2025", "旗舰"), _entry("Gemini 2.5 Flash", "2025", "轻量"),
        _entry("Gemini 2.5 Flash-Lite", "2025", "轻量"),
        _entry("Gemini 3 Pro", "2025", "旗舰"), _entry("Gemini 3 Flash", "2025", "轻量"),
        _entry("Gemini 3.1 Pro", "2026", "预览版核心升级"),
        _entry("Gemini 3.5 Flash", "2026", "代理编码"),
        _entry("Gemini 3.5 Flash-Lite", "2026", "轻量"),
        _entry("Gemini 3.6 Flash", "2026", "最新Flash"),
    ]),
    ("Google", "美国", True, "Gemma", [
        _entry("Gemma", "2024", "开源轻量"),
        _entry("Gemma 2", "2024", "开源"),
        _entry("Gemma 3", "2025", "开源多模态"),
    ]),
    ("Google", "美国", False, "PaLM", [
        _entry("PaLM 2", "2023", "多语言"),
    ]),
    ("Meta", "美国", True, "Llama", [
        _entry("Llama 1", "2023"), _entry("Llama 2", "2023"),
        _entry("Llama 3", "2024"), _entry("Llama 3.1", "2024"),
        _entry("Llama 3.2", "2024", "多模态"), _entry("Llama 3.3", "2024"),
        _entry("Llama 4 Scout", "2025", "开源多模态"),
        _entry("Llama 4 Maverick", "2025", "开源多模态旗舰"),
    ]),
    ("Meta", "美国", True, "Code Llama", [
        _entry("Code Llama", "2023", "代码开源"),
    ]),
    ("xAI", "美国", False, "Grok", [
        _entry("Grok 1", "2023"), _entry("Grok 1.5", "2024"), _entry("Grok 2", "2024"),
        _entry("Grok 3", "2025", "旗舰"), _entry("Grok 4", "2025", "旗舰"),
        _entry("Grok 4.5", "2026", "当前消费级旗舰"),
        # Grok 5 截至 2026-08 仍未发布，不收录
    ]),
    ("Mistral AI", "法国", True, "Mistral", [
        _entry("Mistral 7B", "2023"), _entry("Mixtral 8x7B", "2023", "MoE"),
        _entry("Mistral Small", "2024", "轻量"), _entry("Mistral Medium", "2024", "均衡"),
        _entry("Mistral Large 2", "2024", "旗舰"),
        _entry("Codestral", "2024", "代码"), _entry("Mistral Nemo", "2024", "轻量"),
        _entry("Ministral 8B", "2024", "端侧边缘"), _entry("Pixtral", "2024", "多模态"),
        _entry("Mistral Small 3", "2025", "轻量"), _entry("Magistral Small", "2025", "推理"),
        _entry("Mistral Large 3", "2025", "旗舰MoE"),
    ]),
    ("阿里巴巴", "中国", True, "Qwen", [
        _entry("Qwen 1", "2023"), _entry("Qwen 1.5", "2024"),
        _entry("Qwen2", "2024"),
        _entry("Qwen-VL", "2023", "多模态"),
        _entry("Qwen2.5", "2024"), _entry("Qwen2.5-Coder", "2024", "编程"),
        _entry("Qwen2.5-Max", "2025", "闭源旗舰"),
        _entry("QwQ-32B", "2025", "推理"),
        _entry("Qwen3", "2025", "开源旗舰"), _entry("Qwen3-Coder", "2025", "代理编程"),
        _entry("Qwen3.5", "2026", "开源旗舰"),
        _entry("Qwen3.5-Omni", "2026", "全模态"),
        _entry("Qwen3.6", "2026", "开源旗舰", REVIEW),
        _entry("Qwen3.7-Max", "2026", "闭源智能体旗舰"),
    ]),
    ("深度求索", "中国", True, "DeepSeek", [
        _entry("DeepSeek LLM", "2023", "初代开源"),
        _entry("DeepSeek-V2", "2024", "MoE"), _entry("DeepSeek-V2.5", "2024"),
        _entry("DeepSeek-Coder-V2", "2024", "代码MoE"),
        _entry("DeepSeek-V3", "2024", "开源旗舰"),
        _entry("DeepSeek-R1-Zero", "2025", "推理"), _entry("DeepSeek-R1", "2025", "推理"),
        _entry("DeepSeek-V3.1", "2025", "开源旗舰"),
        _entry("DeepSeek-V4", "2026", "开源预览，1M上下文"),
        _entry("DeepSeek-V4-Flash", "2026", "高效轻量"),
        _entry("DeepSeek-V4-Pro", "2026", "开源旗舰"),
    ]),
    ("月之暗面", "中国", True, "Kimi", [
        _entry("Moonshot v1", "2023", "初代长上下文"),
        _entry("Kimi K1.5", "2025", "推理"),
        _entry("Kimi K2", "2025", "开源"),
        _entry("Kimi K2 Thinking", "2025", "开源思考"),
        _entry("Kimi K2.5", "2026", "开源多模态Agent"),
        _entry("Kimi-K2.6", "2026", "开源多模态Agent旗舰"),
    ]),
    ("智谱AI", "中国", True, "GLM", [
        _entry("GLM-4", "2024"),
        _entry("GLM-4.5", "2025"), _entry("GLM-4.5-Air", "2025", "轻量开源"),
        _entry("GLM-Z1", "2025", "推理"), _entry("GLM-4.6", "2025"),
        _entry("GLM-4.7", "2025", "年底迭代"),
        _entry("GLM-5", "2026", "开源编程Agent旗舰"),
        _entry("GLM-5.1", "2026", "长程增强"), _entry("GLM-5.2", "2026", "1M无损上下文"),
        _entry("GLM-5.3", "2026", "开源编程最强"),
    ]),
    ("MiniMax", "中国", True, "MiniMax", [
        _entry("MiniMax Text-01", "2025", "长上下文"),
        _entry("MiniMax M1", "2025", "推理MoE"), _entry("MiniMax M2", "2025", "开源"),
        _entry("MiniMax M2.5", "2026", "开源"),
        _entry("MiniMax M2.7", "2026", "Agent旗舰，自我进化"),
    ]),
    ("字节跳动", "中国", False, "豆包", [
        _entry("豆包 1.0", "2023"), _entry("豆包 1.5", "2025", "商用"),
        _entry("豆包 1.5 Pro", "2025", "商用"),
        _entry("豆包 1.6", "2025", "256K思考模型"),
        _entry("豆包 2.0", "2026", "Pro/Lite/Mini系列旗舰"),
    ]),
    ("腾讯", "中国", False, "混元", [
        _entry("混元", "2023"), _entry("混元 Turbo", "2024", "商用"),
        _entry("混元 Large", "2024", "商用"), _entry("混元 T1", "2025", "推理"),
    ]),
    ("百度", "中国", False, "文心", [
        _entry("文心 ERNIE 3.0", "2021"), _entry("文心 ERNIE 3.5", "2023"),
        _entry("文心一言 4.0", "2023", "商用"), _entry("ERNIE X1", "2025", "推理"),
        _entry("文心一言 4.5", "2025", "商用"), _entry("文心 4.5 Turbo", "2025", "商用"),
    ]),
    ("阶跃星辰", "中国", True, "Step", [
        _entry("Step-1V", "2023", "多模态"),
        _entry("Step-2", "2025", "万亿MoE"), _entry("Step-3", "2025", "开源多模态推理"),
    ]),
    ("零一万物", "中国", True, "Yi", [
        _entry("Yi-1.5", "2024", "开源"), _entry("Yi-Lightning", "2024", "开源"),
        _entry("Yi-2", "2025", "开源", REVIEW),
    ]),
    ("百川智能", "中国", True, "Baichuan", [
        _entry("Baichuan-13B", "2023", "开源"), _entry("Baichuan2", "2023", "开源"),
        _entry("Baichuan4", "2024", "开源"),
    ]),
    ("昆仑万维", "中国", True, "Skywork", [
        _entry("Skywork-13B", "2023", "开源"),
        _entry("Skywork 4o", "2024", "天工4.0多模态，o1版开源"),
    ]),
    ("科大讯飞", "中国", False, "星火", [
        _entry("星火 V1.5", "2023"), _entry("星火 V2.0", "2023"), _entry("星火 V3.0", "2023"),
        _entry("星火 V3.5", "2024"), _entry("星火 4.0", "2024", "旗舰"),
        _entry("星火 4.0 Turbo", "2024", "商用"), _entry("星火 X1", "2025", "推理"),
    ]),
    ("上海人工智能实验室", "中国", True, "书生", [
        _entry("InternLM", "2023"), _entry("InternLM2", "2024"),
        _entry("InternLM2.5", "2024"), _entry("InternVL2.5", "2024", "多模态"),
        _entry("InternLM3", "2025"),
    ]),
    ("面壁智能", "中国", True, "MiniCPM", [
        _entry("MiniCPM", "2024", "端侧"), _entry("MiniCPM-V 2.6", "2024", "多模态端侧"),
        _entry("MiniCPM3", "2024", "端侧"), _entry("MiniCPM4", "2025", "端侧"),
    ]),
    ("华为", "中国", False, "盘古", [
        _entry("盘古 5.0", "2024", "全系列多模态强思维"),
    ]),
    ("微软", "美国", True, "Phi", [
        _entry("Phi-1", "2023", "代码"), _entry("Phi-1.5", "2023"),
        _entry("Phi-2", "2023", "小型"),
        _entry("Phi-3-mini", "2024"), _entry("Phi-3-small", "2024"),
        _entry("Phi-3-medium", "2024"), _entry("Phi-3.5", "2024"),
        _entry("Phi-4", "2024"), _entry("Phi-4-mini", "2025", "轻量"),
        _entry("Phi-4-multimodal", "2025", "多模态"),
    ]),
    ("英伟达", "美国", True, "Nemotron", [
        _entry("Nemotron-4 340B", "2024", "合成数据生成"),
        _entry("Llama-3.1-Nemotron-70B", "2024", "RLHF开源权重"),
        _entry("Llama-3.1-Nemotron-Ultra-253B", "2025", "旗舰开源权重"),
    ]),
    ("亚马逊", "美国", False, "Nova", [
        _entry("Nova Micro", "2024", "文本轻量"), _entry("Nova Lite", "2024", "多模态轻量"),
        _entry("Nova Pro", "2024", "多模态均衡"), _entry("Nova Premier", "2025", "旗舰"),
    ]),
    ("Cohere", "加拿大", True, "Command", [
        _entry("Command R", "2024", "检索增强"), _entry("Command R+", "2024", "旗舰"),
        _entry("Aya Expanse", "2024", "多语言"), _entry("Command R7B", "2024", "轻量"),
        _entry("Command A", "2025", "代理旗舰"),
    ]),
    ("AI21 Labs", "以色列", True, "Jamba", [
        _entry("Jamba", "2024", "混合SSM-MoE"),
        _entry("Jamba 1.5 Mini", "2024", "轻量长上下文"),
        _entry("Jamba 1.5 Large", "2024", "长上下文"),
    ]),
    ("TII", "阿联酋", True, "Falcon", [
        _entry("Falcon 40B", "2023", "开源"), _entry("Falcon 180B", "2023", "旗舰"),
        _entry("Falcon 2", "2024", "多模态"), _entry("Falcon 3", "2024", "轻量系列"),
    ]),
    ("Ai2", "美国", True, "OLMo", [
        _entry("OLMo", "2024", "完全开源"), _entry("OLMo 2", "2024"),
        _entry("OLMo 2 32B", "2025", "旗舰"),
    ]),
    ("Hugging Face", "法国", True, "SmolLM", [
        _entry("SmolLM", "2024", "超小型"), _entry("SmolLM2", "2024", "超小型"),
        _entry("SmolLM3", "2025"),
    ]),
    ("Databricks", "美国", True, "DBRX", [
        _entry("DBRX", "2024", "MoE"),
    ]),
    ("LG AI研究院", "韩国", True, "EXAONE", [
        _entry("EXAONE 3.0", "2024"), _entry("EXAONE 3.5", "2024"),
        _entry("EXAONE 4.0", "2025"),
    ]),
]


# 公司与系列的常见别称（人工整理的指代消解词表；键=查询中可能出现的形式）
COMPANY_ALIASES = {
    "阿里": "阿里巴巴", "通义": "阿里巴巴", "淘宝系": "阿里巴巴",
    "谷歌": "Google", "脸书": "Meta", "Meta公司": "Meta",
    "微软亚洲": "微软", "小艺": "华为",
    "Zhipu": "智谱AI", "智谱": "智谱AI", "清华系": "智谱AI",
    "DeepSeek": "深度求索", "暗面": "月之暗面", "Moonshot": "月之暗面",
    "Kimi公司": "月之暗面", "稀宇": "MiniMax",
    "讯飞": "科大讯飞", "上交系实验室": "上海人工智能实验室",
    "书生": "上海人工智能实验室", "面壁": "面壁智能",
    "StepFun": "阶跃星辰", "阶跃": "阶跃星辰",
    "零一万务": "零一万物", "百川": "百川智能",
    "昆仑": "昆仑万维", "万维": "昆仑万维",
    "字节数科": "字节跳动", "火山引擎": "字节跳动",
}
SERIES_ALIASES = {
    "千问": "Qwen", "通义千问": "Qwen", "Qwen系列": "Qwen",
    "文心大模型": "文心", "ERNIE": "文心",
    "豆包大模型": "豆包", "云雀": "豆包",
    "星火认知": "星火", "Spark": "星火",
    "书生浦语": "书生", "浦语": "书生", "InternLM系列": "书生",
    "小钢炮": "Phi", "混元大模型": "混元",
    "盘古大模型": "盘古", "天工": "Skywork",
    "ChatGPT系列": "GPT", "GPT家族": "GPT",
}


def _object_id(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "-")


def _prev(model_list, index):
    return model_list[index - 1][0] if index > 0 else None


def build():
    objects = []
    queries = []
    label = 0
    for company, region, open_source, series, models in SERIES:
        for index, (name, year, note, needs_review) in enumerate(models):
            prev = _prev(models, index)
            alias_note = "" if " " not in name else name.replace(" ", "")
            evidence = {
                "名称": f"这个模型的全称是 {name}，属于 {series} 系列"
                       + (f"，也常写作 {alias_note}" if alias_note and alias_note != name else ""),
                "属性": f"它由 {company} 开发，是{'开源' if open_source else '闭源'}模型"
                        + (f"，{note}" if note else ""),
                "关系": f"它在 {series} 系列中"
                        + (f"，前一代是 {prev}" if prev else "，是该系列早期版本"),
                "变化": f"它于 {year} 年发布",
            }
            objects.append({
                "id": _object_id(name),
                "label": label,
                "name": name,
                "evidence": evidence,
                "needs_review": bool(needs_review),
                "meta": {"company": company, "region": region,
                         "open_source": open_source, "series": series,
                         "series_index": index},
            })
            label += 1

        # 每个系列生成：最新模型已知查询 + 系列级歧义查询（meta 供改写增强使用）
        latest = models[-1][0]
        objects_by_name = {obj["name"]: obj["label"] for obj in objects}
        queries.append({
            "text": f"{company} 的 {series} 系列最新模型是什么？",
            "target_id": objects[objects_by_name[latest]]["id"],
            "kind": "known",
            "meta": {"company": company, "series": series},
        })
        queries.append({
            "text": f"{series} 是哪一个模型？",
            "target_id": None,
            "kind": "ambiguous",
        })

    # 属性/来源已知查询（跨系列）
    queries.extend([
        {"text": "阿里巴巴登顶国产第一的模型是什么？", "target_id": None, "kind": "ambiguous"},
        {"text": "哪个公司发布了 Claude 系列？", "target_id": None, "kind": "ambiguous"},
        {"text": "OpenAI 最新发布的旗舰模型是哪一个？", "target_id": "gpt-5-4", "kind": "known"},
        {"text": "Meta 最新的开源模型叫什么？", "target_id": "llama-4-maverick", "kind": "known"},
    ])

    # 未知查询（对象库外）
    queries.extend([
        {"text": "GPT-6 是什么时候发布的？", "target_id": None, "kind": "unknown"},
        {"text": "介绍一下百度的文心 5.0", "target_id": None, "kind": "unknown"},
        {"text": "OpenAI 的 o5 推理模型参数量是多少？", "target_id": None, "kind": "unknown"},
        {"text": "Claude Opus 6 的发布日期是哪天？", "target_id": None, "kind": "unknown"},
        {"text": "Mistral 的 Medium 最新型号是什么？", "target_id": None, "kind": "unknown"},
        {"text": "帮我写一段关于秋天的散文", "target_id": None, "kind": "unknown"},
    ])

    reviewed = sum(1 for obj in objects if not obj["needs_review"])
    return {
        "meta": {
            "dataset": "ai_models_dataset",
            "description": "主流 AI 大模型四证据数据集（200+ 对象自动展开版）",
            "status": "自动展开；needs_review=true 的条目在人工或检索核对前不得进入正式对象库",
            "objects": len(objects),
            "objects_verified": reviewed,
            "objects_needs_review": len(objects) - reviewed,
            "queries": {"known": sum(1 for q in queries if q["kind"] == "known"),
                        "ambiguous": sum(1 for q in queries if q["kind"] == "ambiguous"),
                        "unknown": sum(1 for q in queries if q["kind"] == "unknown")},
        },
        "company_aliases": COMPANY_ALIASES,
        "series_aliases": SERIES_ALIASES,
        "objects": objects,
        "queries": queries,
    }


def main():
    dataset = build()
    output = ROOT / "data" / "ai_models_dataset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = dataset["meta"]
    print(f"objects={meta['objects']} verified={meta['objects_verified']} "
          f"needs_review={meta['objects_needs_review']} queries={meta['queries']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
