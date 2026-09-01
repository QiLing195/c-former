# -*- coding: utf-8 -*-
"""身份解析变体增强：让模型「学会表达方式」而非枚举每一种说法。

原则（回应模拟暴露的短板）：
- 不给模型背"克劳德4.8=Claude Opus 4.8"这种单点映射，而是给一批**映射种子**，
  让对比学习在「变体查询 ↔ 正确对象」对上学会**关系泛化**；
- 变体维度（种子知识，可扩展）：
  a. 音译：Claude→克劳德、Gemini→谷歌（公司词已覆盖）等；
  b. 同义/公司：谷歌=Google、通义=Qwen、深度=深度求索、Meta=元宇宙（不用）；
  c. 口语版本号：5.2 → 「5 点 2」「5.2 版」「5-2」；
  d. 繁简体：核心字对转写；
  e. 夹带口语：那个什么/帮我看看/知道不/呗；
  f. 语序：X 是啥 / 是啥 X；
  g. 无分隔：GPT5.2。
- 生成规则纯确定性（固定 seed），train 查询追加变体，heldout 不动（保证对比公平）；
- 变体**不写入 ALIASES**（那是精确层词表，仍保留——精确层保底，神经层学泛化）。

用法（在 build_ai_models_dataset.py 之后运行）：
    D:/conda/envs/cformer-gpu/python.exe augment_identity.py
输出：data/ai_models_dataset.json 原地追加变体 train 查询（幂等：重复运行先剔除旧标记）。
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai_models_dataset.json"
AUGMENT_TAG = "__identity_augment__"

# ---- 变体种子（映射关系，模型从中学泛化；新增维度在此扩展）----

# a. 音译：英文名 → 中文音译（常用音译，非穷举）
TRANSLIT = {
    "Claude": "克劳德", "GPT": "吉皮提", "Gemini": "吉米尼", "Llama": "拉玛",
    "Qwen": "千问", "DeepSeek": "深度求索", "Kimi": "基米", "GLM": "智谱",
    "Mistral": "米斯特拉尔", "Grok": "格罗克", "Phi": "法伊", "Nova": "诺瓦",
    "Nemotron": "尼莫", "Granite": "格拉尼特", "Command": "康曼德",
}
# b. 公司/系列同义（短别名 → 对象名中的片段）
SYNONYMS = {
    "谷歌": "Gemini", "Google": "Gemini", "通义": "Qwen", "千问": "Qwen",
    "深度": "DeepSeek", "月之暗面": "Kimi", "智谱": "GLM", "讯飞": "星火",
    "华为": "盘古", "快手": "可灵", "英伟达": "Nemotron", "Meta": "Llama",
}
# d. 繁→简核心字对（与 PreciseMatch.norm 一致）
SIMPLIFIED = {"義": "义", "問": "问", "稱": "称", "為": "为", "麼": "么",
              "這": "这", "裡": "里", "關": "关", "係": "系", "變": "变",
              "體": "体", "國": "国", "開": "开", "發": "发", "與": "与",
              "們": "们", "說": "说", "話": "话", "還": "还", "來": "来",
              "東": "东", "後": "后", "點": "点", "產": "产", "廣": "广",
              "當": "当", "會": "会", "寫": "写", "讀": "读", "認": "认"}
_TRAD_MAP = {v: k for k, v in SIMPLIFIED.items()}

# e. 口语夹带前缀/后缀
FILLERS = ["那个什么", "帮我看看", "你知道", "打听下", "好像有个"]
SUFFIXES = ["来着", "知道不", "是啥", "呗", "哈", "怎么样", "靠谱吗", "好用吗"]

# f. 语序模板
TEMPLATES = [
    "介绍一下{name}",
    "{name}是什么模型？",
    "我想了解{name}这个模型",
    "请介绍一下{name}",
    "{name}这款模型怎么样？",
]


def simplified(text: str) -> str:
    return text.translate(str.maketrans(SIMPLIFIED))


def traditional(text: str) -> str:
    return text.translate(str.maketrans(_TRAD_MAP))


def spoken_version(name: str) -> str:
    """版本号口语化：GPT-5.2 → GPT 5 点 2。"""
    return re.sub(r"(\d)[.\-](\d)", r"\1 点 \2", name)


def gen_variants(obj: dict, rng: random.Random) -> list[str]:
    """为一个对象生成变体查询文本（返回多种措辞，随机取一部分）。"""
    name = obj["name"]
    variants: list[str] = []

    # a. 音译替换（把对象名里的英文片段换成音译）
    translit_name = name
    for en, zh in TRANSLIT.items():
        if en in name and zh not in name:
            translit_name = translit_name.replace(en, zh)
    if translit_name != name:
        variants.append(translit_name)

    # b. 同义替换（公司/系列词）
    for zh, en in SYNONYMS.items():
        if en in name and zh not in name:
            variants.append(name.replace(en, zh))

    # c. 口语版本号
    spoken = spoken_version(name)
    if spoken != name:
        variants.append(spoken)

    # d. 繁体
    trad = traditional(name)
    if trad != name:
        variants.append(trad)

    # g. 无分隔（去连字符/点）
    compact = re.sub(r"[\s.\-]+", "", name)
    if compact != name:
        variants.append(compact)

    # 组合：模板 + 变体名
    texts = []
    for v in variants:
        for template in TEMPLATES:
            texts.append(template.format(name=v))
        # 夹带 + 后缀口语
        texts.append(f"{rng.choice(FILLERS)}{v}{rng.choice(SUFFIXES)}")
    # 语序倒装
    for v in variants[:3]:
        texts.append(f"{rng.choice(SUFFIXES)} {v}")
    # 原名的夹带/后缀（不带变体也生成，增加句式）
    texts.append(f"{rng.choice(FILLERS)}{name}{rng.choice(SUFFIXES)}")
    texts.append(f"{rng.choice(SUFFIXES)} {name}")
    return texts


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    objects = data["objects"]
    queries = data["queries"]

    # 剔除旧增强（幂等）
    queries = [q for q in queries if q.get("tag") != AUGMENT_TAG]

    rng = random.Random(42)
    added = 0
    for obj in objects:
        object_id = obj["id"]
        variants = gen_variants(obj, rng)
        # 每对象最多取 6 个变体查询，避免数据集膨胀失控
        rng.shuffle(variants)
        for text in variants[:6]:
            queries.append({"text": text, "target_id": object_id, "kind": "known",
                            "subtype": "name", "split": "train", "tag": AUGMENT_TAG})
            added += 1

    data["queries"] = queries
    data["meta"]["queries"] = {
        "known": sum(1 for q in queries if q["kind"] == "known"),
        "ambiguous": sum(1 for q in queries if q["kind"] == "ambiguous"),
        "unknown": sum(1 for q in queries if q["kind"] == "unknown"),
        "train": sum(1 for q in queries if q.get("split", "train") == "train"),
        "heldout": sum(1 for q in queries if q.get("split", "heldout") == "heldout"),
        "identity_augment_train": added,
    }
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "added_variant_queries": added,
                      "meta": data["meta"]["queries"]}, ensure_ascii=False))
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
