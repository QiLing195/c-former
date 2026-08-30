# -*- coding: utf-8 -*-
"""把主流 AI 模型系列展开成约 200 个对象的四证据数据集（查询多样化版）。

用法：
    D:/conda/envs/cformer-gpu/python.exe build_ai_models_dataset.py

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
        ("GPT-4.1", "2025", ""), ("GPT-4.1 mini", "2025", "轻量"),
        ("GPT-4.1 nano", "2025", "轻量"), ("GPT-5", "2025", "前沿旗舰"),
        ("GPT-5.1", "2026", ""), ("GPT-5.2", "2026", "当前前沿闭源旗舰"),
    ]),
    ("OpenAI", "美国", False, "o 推理", [
        ("o1", "2024", "推理"), ("o1-mini", "2024", "推理轻量"),
        ("o3", "2025", "推理"), ("o3-mini", "2025", "推理轻量"),
        ("o4", "2026", "推理"), ("o4-mini", "2026", "推理轻量"),
    ]),
    ("Anthropic", "美国", False, "Claude", [
        ("Claude 1", "2023", ""), ("Claude 2", "2023", ""), ("Claude 2.1", "2023", ""),
        ("Claude 3 Opus", "2024", "旗舰"), ("Claude 3 Sonnet", "2024", "均衡"),
        ("Claude 3 Haiku", "2024", "轻量"), ("Claude 3.5 Sonnet", "2024", "均衡"),
        ("Claude 3.5 Haiku", "2024", "轻量"), ("Claude Opus 4", "2025", "旗舰"),
        ("Claude Sonnet 4", "2025", "均衡"), ("Claude Haiku 4", "2025", "轻量"),
        ("Claude Opus 4.5", "2025", "旗舰"), ("Claude Sonnet 4.5", "2025", "均衡"),
        ("Claude Haiku 4.5", "2025", "轻量"), ("Claude Opus 4.7", "2026", "旗舰"),
        ("Claude Opus 4.8", "2026", "SWE-bench 领先"),
    ]),
    ("Google", "美国", False, "Gemini", [
        ("Gemini 1.0", "2023", ""), ("Gemini 1.5 Pro", "2024", "旗舰"),
        ("Gemini 1.5 Flash", "2024", "轻量"), ("Gemini 2.0 Flash", "2024", "轻量"),
        ("Gemini 2.0 Flash-Lite", "2025", "轻量"), ("Gemini 2.0 Pro", "2025", "旗舰"),
        ("Gemini 2.5 Pro", "2025", "旗舰"), ("Gemini 2.5 Flash", "2025", "轻量"),
        ("Gemini 3 Pro", "2025", "旗舰"), ("Gemini 3 Flash", "2025", "轻量"),
        ("Gemini 3.5 Pro", "2026", "代理编码旗舰"), ("Gemini 3.5 Flash", "2026", "最快代理编码"),
    ]),
    ("Meta", "美国", True, "Llama", [
        # 变体（variant=True：不进链，predecessor=None）：尺寸规格
        ("Llama 2 7B", "2023", "轻量", True), ("Llama 2 13B", "2023", "均衡", True),
        ("Llama 2 70B", "2023", "旗舰", True),
        ("Llama 3 8B", "2024", "轻量", True), ("Llama 3 70B", "2024", "旗舰", True),
        ("Llama 3.1 8B", "2024", "轻量", True), ("Llama 3.1 70B", "2024", "均衡", True),
        ("Llama 3.1 405B", "2024", "旗舰", True), ("Llama 3.2 1B", "2024", "轻量", True),
        ("Llama 3.2 3B", "2024", "轻量", True), ("Llama 3.2 11B", "2024", "均衡", True),
        ("Llama 4 Scout", "2026", "轻量开源", True), ("Llama 4 Maverick", "2026", "均衡开源", True),
        # 主链（代数序，严格时间单调）
        ("Llama 1", "2023", ""), ("Llama 2", "2023", ""),
        ("Llama 3", "2024", ""), ("Llama 3.1", "2024", ""),
        ("Llama 3.2", "2024", "多模态"), ("Llama 3.3", "2024", "70B 旗舰"),
        ("Llama 4", "2026", "开源多模态旗舰"),
    ]),
    ("xAI", "美国", False, "Grok", [
        ("Grok 1", "2023", ""), ("Grok 1.5", "2024", ""), ("Grok 2", "2024", ""),
        ("Grok 2.5", "2025", ""), ("Grok 3", "2025", "旗舰"),
        ("Grok 4", "2025", "旗舰"), ("Grok 5", "2026", "旗舰"),
    ]),
    ("Mistral AI", "法国", True, "Mistral", [
        ("Mistral 7B", "2023", ""), ("Mixtral 8x7B", "2023", "MoE"),
        ("Mistral Small", "2024", "轻量"), ("Mistral Medium", "2024", "均衡"),
        ("Mistral Large 2", "2024", "旗舰"), ("Mistral Nemo", "2024", "轻量"),
        ("Codestral", "2024", "编程"), ("Mathstral", "2024", "数学"),
        ("Mixtral 8x22B", "2024", "MoE"), ("Mistral 7B v0.3", "2024", "开源"),
        ("Mistral Large 3", "2025", "旗舰"), ("Mistral Small 3", "2025", "轻量"),
        ("Mistral Small 3.1", "2026", "轻量"),
    ]),
    ("Microsoft", "美国", True, "Phi", [
        ("Phi-1", "2023", "轻量"), ("Phi-1.5", "2023", "轻量"), ("Phi-2", "2023", "轻量"),
        ("Phi-3-mini", "2024", "轻量"), ("Phi-3-small", "2024", "轻量"),
        ("Phi-3.5", "2024", "轻量"), ("Phi-4", "2024", "轻量"),
    ]),
    ("Amazon", "美国", False, "Nova", [
        ("Nova Lite", "2024", "轻量"), ("Nova Pro", "2024", "均衡"),
        ("Nova Premier", "2024", "旗舰"),
    ]),
    ("NVIDIA", "美国", True, "Nemotron", [
        ("Nemotron-4", "2024", "开源"), ("Nemotron-4-340B", "2024", "开源"),
        ("Nemotron-H", "2025", "开源"),
    ]),
    ("IBM", "美国", True, "Granite", [
        ("Granite-7B", "2023", "开源"), ("Granite-13B", "2023", "开源"),
        ("Granite-20B", "2024", "开源"),
    ]),
    ("Cohere", "加拿大", True, "Command", [
        ("Command R", "2024", "开源"), ("Command R+", "2024", "开源"),
        ("Command A", "2025", "旗舰"),
    ]),
    ("Databricks", "美国", True, "DBRX", [
        ("DBRX", "2024", "开源"),
    ]),
    ("AI21", "以色列", True, "Jamba", [
        ("Jamba", "2024", "开源"), ("Jamba 1.5", "2024", "开源"),
    ]),
    ("Apple", "美国", True, "Apple", [
        ("OpenELM", "2024", "开源"), ("Apple Foundation Model", "2025", "闭源"),
    ]),
    ("TII", "阿联酋", True, "Falcon", [
        ("Falcon-40B", "2023", "开源"), ("Falcon-180B", "2023", "开源"),
        ("Falcon-3", "2024", "开源"),
    ]),
    ("Naver", "韩国", False, "HyperCLOVA", [
        ("HyperCLOVA X", "2023", "闭源"), ("HyperCLOVA X2", "2025", "闭源"),
    ]),
    ("LG", "韩国", True, "EXAONE", [
        ("EXAONE 1.0", "2024", "开源"), ("EXAONE 2.0", "2025", "开源"),
        ("EXAONE 3.0", "2025", "开源"),
    ]),
    ("阿里巴巴", "中国", True, "Qwen", [
        # 变体（variant=True：不进链）：尺寸规格 / 多模态 / 专项能力
        ("Qwen2.5 0.5B", "2024", "轻量", True), ("Qwen2.5 1.5B", "2024", "轻量", True),
        ("Qwen2.5 3B", "2024", "轻量", True), ("Qwen2.5 7B", "2024", "均衡", True),
        ("Qwen2.5 14B", "2024", "均衡", True), ("Qwen2.5 32B", "2025", "旗舰", True),
        ("Qwen2.5 72B", "2025", "旗舰", True),
        ("Qwen3 0.6B", "2025", "轻量", True), ("Qwen3 1.7B", "2025", "轻量", True),
        ("Qwen3 4B", "2025", "轻量", True), ("Qwen3 8B", "2025", "均衡", True),
        ("Qwen3 14B", "2025", "均衡", True), ("Qwen3 32B", "2026", "旗舰", True),
        ("Qwen3 235B", "2026", "超大型", True),
        ("Qwen-VL", "2024", "多模态", True), ("Qwen2.5-Coder", "2024", "编程", True),
        ("Qwen2.5-Math", "2024", "数学", True), ("Qwen3-VL", "2025", "多模态", True),
        # 主链（代数序，严格时间单调）
        ("Qwen 1", "2023", ""), ("Qwen 1.5", "2024", ""),
        ("Qwen2", "2024", ""), ("Qwen2.5", "2024", ""),
        ("Qwen3", "2025", "开源旗舰"), ("Qwen3.5", "2025", "开源旗舰"),
        ("Qwen3.6", "2026", "开源旗舰"), ("Qwen3.7-Max", "2026", "闭源国产旗舰第一"),
    ]),
    ("深度求索", "中国", True, "DeepSeek", [
        ("DeepSeek-V1", "2024", ""), ("DeepSeek-V2", "2024", "MoE"),
        ("DeepSeek-V2.5", "2024", ""), ("DeepSeek-V3", "2024", "开源旗舰"),
        ("DeepSeek-Coder-V2", "2024", "编程"),
        ("DeepSeek-R1", "2025", "推理"), ("DeepSeek-R1-Zero", "2025", "推理"),
        ("DeepSeek-V3.1", "2025", "开源旗舰"), ("DeepSeek-Coder", "2025", "编程"),
        ("DeepSeek-V3 671B", "2025", "MoE 超大型"), ("DeepSeek-R1-0528", "2025", "推理"),
        ("DeepSeek-V4", "2026", "开源旗舰"), ("DeepSeek-V4-Pro", "2026", "开源旗舰"),
        ("DeepSeek-R2", "2026", "推理"),
    ]),
    ("月之暗面", "中国", True, "Kimi", [
        ("Kimi K1", "2024", ""), ("Kimi K1.5", "2025", "推理"),
        ("Kimi K2", "2025", "开源"), ("Kimi K2.5", "2025", "开源旗舰"),
        ("Kimi-K2.6", "2026", "开源旗舰"),
    ]),
    ("智谱AI", "中国", True, "GLM", [
        # 变体：推理专用
        ("GLM-Z1", "2025", "推理", True),
        # 主链（严格时间单调）
        ("GLM-4", "2024", ""), ("GLM-4.5", "2025", ""), ("GLM-4.6", "2025", ""),
        ("GLM-5", "2025", ""), ("GLM-4.7", "2026", "开源"),
        ("GLM-5.1", "2026", "开源"), ("GLM-5.3", "2026", "开源编程最强"),
    ]),
    ("MiniMax", "中国", True, "MiniMax", [
        ("MiniMax M1", "2025", ""), ("MiniMax M1.5", "2025", "轻量"),
        ("MiniMax M2", "2025", "开源"), ("MiniMax M2.5", "2025", "开源"),
        ("MiniMax M2.7", "2026", "开源"),
    ]),
    ("字节跳动", "中国", False, "豆包", [
        ("豆包 1.0", "2023", ""), ("豆包 1.5", "2025", "商用"),
        ("豆包 1.5 Pro", "2026", "推理突出"),
    ]),
    ("字节跳动", "中国", False, "Seed", [
        ("Seed 1.5", "2025", "多模态"), ("Seed 2.0", "2025", "多模态"),
        ("Seed 3.0", "2026", "多模态"),
    ]),
    ("腾讯", "中国", False, "混元", [
        ("混元", "2023", ""), ("混元 Turbo", "2024", "商用"),
        ("混元 Large", "2024", "商用"), ("混元 T1", "2025", "推理"),
        ("混元 A1", "2026", "推理"),
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
    ("商汤科技", "中国", False, "SenseChat", [
        ("SenseChat-5", "2024", ""), ("SenseChat-5.5", "2025", ""),
        ("日日新 5.5", "2025", "商用"),
    ]),
    ("科大讯飞", "中国", False, "星火", [
        ("星火 v3.0", "2023", ""), ("星火 v3.5", "2024", ""),
        ("星火 v4.0", "2024", ""), ("星火 v5.0", "2025", "商用"),
    ]),
    ("上海人工智能实验室", "中国", True, "InternLM", [
        ("InternLM-1.0", "2023", "开源"), ("InternLM-1.5", "2024", "开源"),
        ("InternLM2", "2024", "开源"), ("InternLM2.5", "2024", "开源"),
        ("InternLM3", "2025", "开源"), ("InternLM3.5", "2026", "开源"),
    ]),
    ("阿里巴巴", "中国", True, "通义万相", [
        ("Wan 1.0", "2024", "多模态"), ("Wan 1.5", "2024", "多模态"),
        ("Wan 2.0", "2025", "多模态"),
    ]),
    ("华为", "中国", False, "盘古", [
        ("盘古 1.0", "2021", ""), ("盘古 2.0", "2022", ""), ("盘古 3.0", "2023", "商用"),
    ]),
    ("网易有道", "中国", False, "子曰", [
        ("子曰 1.0", "2023", ""), ("子曰 2.0", "2024", ""),
    ]),
    ("快手", "中国", False, "可灵", [
        ("可灵 1.0", "2024", "视频生成"), ("可灵 2.0", "2025", "视频生成"),
    ]),
    ("Google", "美国", True, "Gemma", [
        ("Gemma 2 2B", "2024", "开源轻量"), ("Gemma 2 9B", "2024", "开源"),
        ("Gemma 2 27B", "2024", "开源旗舰"), ("Gemma 3 4B", "2025", "开源轻量"),
        ("Gemma 3 12B", "2025", "开源"), ("Gemma 3 27B", "2025", "开源旗舰"),
    ]),
    ("AI2", "美国", True, "OLMo", [
        ("OLMo 7B", "2024", "开源"), ("OLMo2 7B", "2025", "开源"),
    ]),
    ("AI2", "美国", True, "Molmo", [
        ("Molmo 7B", "2024", "开源多模态"), ("Molmo 72B", "2024", "开源多模态"),
    ]),
    ("BigScience", "国际", True, "BLOOM", [
        ("BLOOM 176B", "2022", "开源"), ("BLOOMZ 176B", "2022", "开源"),
    ]),
    ("Hugging Face", "美国", True, "Zephyr", [
        ("Zephyr 7B", "2024", "开源"), ("Zephyr 7B Beta", "2024", "开源"),
    ]),
    ("Hugging Face", "美国", True, "SmolLM", [
        ("SmolLM 135M", "2024", "开源轻量"), ("SmolLM 1.7B", "2024", "开源轻量"),
        ("SmolLM2 1.7B", "2024", "开源轻量"),
    ]),
    ("Stability AI", "英国", True, "StableLM", [
        ("StableLM 3B", "2023", "开源"), ("StableLM 7B", "2023", "开源"),
        ("StableCode 3B", "2023", "开源编程"),
    ]),
    ("Reka AI", "美国", True, "Reka", [
        ("Reka Core", "2024", "开源"), ("Reka Flash", "2024", "开源"),
        ("Reka Edge", "2024", "开源轻量"),
    ]),
    ("Cerebras", "美国", True, "Cerebras-GPT", [
        ("Cerebras-GPT 7B", "2023", "开源"), ("Cerebras-GPT 13B", "2023", "开源"),
    ]),
    ("H2O.ai", "美国", True, "h2oGPT", [
        ("h2oGPT 7B", "2023", "开源"), ("h2oGPT 12B", "2023", "开源"),
    ]),
    ("Aleph Alpha", "德国", False, "Luminous", [
        ("Luminous Supreme", "2022", "闭源"), ("Luminous Extended", "2022", "闭源"),
    ]),
    ("智谱AI", "中国", True, "ChatGLM", [
        ("ChatGLM 6B", "2023", "开源"), ("ChatGLM2 6B", "2023", "开源"),
        ("ChatGLM3 6B", "2023", "开源"),
    ]),
    ("复旦大学", "中国", True, "MOSS", [
        ("MOSS", "2023", "开源"),
    ]),
    ("中国移动", "中国", False, "九天", [
        ("九天 1.0", "2023", "闭源"), ("九天 2.0", "2024", "闭源"),
    ]),
    ("中国电信", "中国", False, "星辰", [
        ("星辰 1.0", "2023", "闭源"), ("星辰 2.0", "2024", "闭源"),
    ]),
    ("中国联通", "中国", False, "元景", [
        ("元景 1.0", "2024", "闭源"),
    ]),
    ("北京智源", "中国", True, "Aquila", [
        ("Aquila-7B", "2023", "开源"), ("Aquila-34B", "2023", "开源"),
    ]),
    ("中科闻歌", "中国", True, "雅意", [
        ("雅意", "2023", "开源"),
    ]),
    ("MiniMax", "中国", False, "海螺", [
        ("海螺", "2024", "视频生成"),
    ]),
    ("昆仑万维", "中国", False, "天工", [
        ("天工 3.0", "2024", "闭源"),
    ]),
    ("京东", "中国", False, "言犀", [
        ("言犀 1.0", "2023", "闭源"),
    ]),
    ("阅文集团", "中国", False, "妙笔", [
        ("妙笔", "2023", "闭源"),
    ]),
]


# 真实别名表：只收录「唯一指向该对象」的别名（近名/系列级模糊别名会制造歧义，
# 那些应该走 ambiguous 查询而不是 known）。别名会进证据文本 + 生成别名查询。
ALIASES = {
    "GPT-4o": ["GPT-4 Omni", "Omni"],
    "GPT-5.2": ["GPT-5 最新版"],
    "GPT-3.5": ["GPT-3.5 Turbo"],
    "Qwen3.7-Max": ["千问3.7", "通义千问3.7"],
    "Qwen3.6": ["千问3.6", "通义千问3.6"],
    "Qwen2.5-Coder": ["通义代码模型"],
    "Claude Opus 4.8": ["Claude 4.8", "Opus 4.8"],
    "Claude Opus 4.7": ["Opus 4.7"],
    "Gemini 3.5 Pro": ["Gemini 3.5 旗舰"],
    "Gemini 3.5 Flash": ["Gemini Flash 3.5"],
    "DeepSeek-V4-Pro": ["DeepSeek V4 Pro", "深度求索V4 Pro"],
    "DeepSeek-R1": ["深度求索R1"],
    "Kimi-K2.6": ["Kimi K2.6", "月之暗面K2.6"],
    "GLM-5.3": ["智谱GLM 5.3"],
    "GLM-5.1": ["智谱GLM 5.1"],
    "GLM-Z1": ["智谱Z1"],
    "豆包 1.5 Pro": ["豆包1.5Pro", "Doubao 1.5 Pro"],
    "文心一言 4.5": ["文心4.5"],
    "混元 T1": ["混元T1", "Hunyuan T1"],
    "混元 A1": ["Hunyuan A1"],
    "Llama 4": ["Meta Llama 4"],
    "Llama 3.1": ["Llama 3.1 405B"],
    "Mistral Large 3": ["Mistral Large 最新版"],
    "Grok 5": ["xAI Grok 5"],
    "o3": ["OpenAI o3"],
    "o4": ["OpenAI o4"],
    "Phi-3-mini": ["Phi-3"],
    "MiniMax M2.7": ["MiniMax M2 最新版"],
    "InternLM3.5": ["书生 InternLM"],
    "星火 v5.0": ["讯飞星火5", "Spark 5"],
    "盘古 3.0": ["华为盘古"],
    "可灵 2.0": ["快手可灵"],
    "Nemotron-H": ["英伟达Nemotron"],
    "Command A": ["Cohere Command A"],
}


def build():
    objects = []
    queries = []
    label = 0
    for company, region, open_source, series, models in SERIES:
        series_model_ids = []
        main_names = [m[0] for m in models if len(m) < 4 or not m[3]]
        last_main_name = None
        for name, year, note, *extra in models:
            is_variant = bool(extra and extra[0] is True)
            prev = last_main_name if not is_variant else None
            if not is_variant:
                last_main_name = name
            is_latest = (not is_variant) and (name == main_names[-1])
            alias_note = "" if " " not in name else name.replace(" ", "")
            aliases = ALIASES.get(name, ())
            evidence = {
                "名称": f"这个模型的全称是 {name}，属于 {series} 系列"
                       + (f"，也常写作 {alias_note}" if alias_note and alias_note != name else "")
                       + (f"，也常被称作{'、'.join(aliases)}" if aliases else ""),
                "属性": f"它由 {company} 开发，是{'开源' if open_source else '闭源'}模型"
                        + (f"，{note}" if note else ""),
                "关系": f"它在 {series} 系列中"
                        + (f"，是{series}系列的一个规格/能力变体" if is_variant else
                           (f"，前一代是 {prev}" if prev else "，是该系列早期版本"))
                        + ("，是该系列最新版本" if is_latest else ""),
                "变化": f"它于 {year} 年发布",
            }
            object_id = name.lower().replace(" ", "-").replace(".", "-")
            prev_id = None
            if prev is not None:
                prev_id = prev.lower().replace(" ", "-").replace(".", "-")
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
                "predecessor": prev_id,
                "evidence": evidence,
            })
            if not is_variant:
                series_model_ids.append(object_id)
            label += 1

            # 名称查询：2 训练 + 1 留出（身份在查询里，测基本匹配与句式稳健）
            queries.append({"text": f"介绍一下{name}这个模型", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
            queries.append({"text": f"{name}是什么模型？", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
            queries.append({"text": f"我想了解{name}这个模型", "target_id": object_id, "kind": "known", "subtype": "name", "split": "heldout"})

            # 别名查询：每个别名 3 训练 + 1 留出（加强训练信号，压 alias 短板）
            for alias in aliases:
                queries.append({"text": f"介绍一下{alias}", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
                queries.append({"text": f"{alias}是什么模型？", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
                queries.append({"text": f"帮我查一下{alias}的资料", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
                queries.append({"text": f"我想了解一下{alias}", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "heldout"})

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
