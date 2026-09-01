# -*- coding: utf-8 -*-
"""身份解析真实场景模拟：模拟真实用户问法，暴露精确层/神经层的边界。

设计原则（先模拟、后调整）：
- 用贴近真实口语/噪音/变体的查询，而非数据集模板；
- 每类场景单独统计「精确层命中率 / 最终答对率」，定位短板在哪层：
  - 精确层应 100% 覆盖的场景却漏了 → norm/词表要调；
  - 描述性指代场景（精确层不命中）→ 看神经层表现，是理解层/神经层的职责；
- 输出失败明细，作为调整输入。

用法：
    D:/conda/envs/cformer-gpu/python.exe sim_identity_scenarios.py --checkpoint artifacts/real_checkpoints/real_seed1.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld
from cformer_v63.precise_match import PreciseMatch

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)

# 场景：(类别, 查询文本, 期望对象 id 或 None)
SCENARIOS = [
    # ---- 标准问询（含全名）----
    ("standard", "GPT-5.2 是什么模型？", "gpt-5-2"),
    ("standard", "介绍一下 Qwen3.6", "qwen3-6"),
    ("standard", "Claude Opus 4.8 是哪个公司的？", "claude-opus-4-8"),
    ("standard", "DeepSeek-V4-Pro 是什么？", "deepseek-v4-pro"),
    # ---- 口语化 / 省略 / 大小写混合 ----
    ("casual", "那个 gpt5.2 靠谱吗？", "gpt-5-2"),
    ("casual", "gpt-5.2 咋样？", "gpt-5-2"),
    ("casual", "qwen3.6 现在还能用吗", "qwen3-6"),
    ("casual", "Claude 4.8 好用吗", "claude-opus-4-8"),
    # ---- 全半角 / 特殊字符 ----
    ("fullwidth", "GPT－5.2 是什么？", "gpt-5-2"),          # 全角连字符
    ("fullwidth", "ＧＰＴ－５.２ 是啥", "gpt-5-2"),          # 全角字母数字
    ("fullwidth", "gpt·5.2", "gpt-5-2"),                     # 间隔号
    # ---- 别名口语 ----
    ("alias", "千问3.7 怎么样？", "qwen3-7-max"),
    ("alias", "通义千问3.7 是哪个？", "qwen3-7-max"),
    ("alias", "深度求索R1 是什么", "deepseek-r1"),
    ("alias", "豆包1.5Pro 好用吗", "豆包-1-5-pro"),
    # ---- 描述性指代（不含全名，精确层应不命中 → 神经层/理解层）----
    ("descriptive", "OpenAI 家最新那个旗舰模型是什么？", None),   # 意图 latest，身份层给不出 → 交递归
    ("descriptive", "谷歌出的那个叫 Gemini 的 3.5 版本", "gemini-3-5-pro"),
    ("descriptive", "阿里开源的千问，现在最强的是哪个？", None),  # 意图 latest
    # ---- 带噪音 / 夹带 / 错字 ----
    ("noisy", "帮我看看 gpt-5.2 这玩意是啥", "gpt-5-2"),
    ("noisy", "那个什么 GPT5.2 来着？", "gpt-5-2"),
    ("noisy", "介绍下 qwen3.6 这个模型呗", "qwen3-6"),
    ("noisy", "克劳德 opus 4.8 知道不", "claude-opus-4-8"),
    # ---- 简称 / 常见变形 ----
    ("abbrev", "claude4.8 是哪个？", "claude-opus-4-8"),
    ("abbrev", "qwen3.6", "qwen3-6"),
    ("abbrev", "kimi k2.6 是啥", "kimi-k2-6"),
    ("abbrev", "glm5.3", "glm-5-3"),
    # ---- 同义词 / 音译 / 意译 / 繁简体（精确层词表外，测试真实泛化）----
    ("synonym", "克劳德 4.8 是哪个？", "claude-opus-4-8"),          # 音译 Claude（已登记克劳德4.8）
    ("synonym", "谷歌的 Gemini 3.5 Pro 怎么样？", "gemini-3-5-pro"),  # 谷歌=Google 同义（Gemini 在词表）
    ("synonym", "Anthropic 家那个 Claude 4.8", "claude-opus-4-8"),   # 公司+名，Claude 4.8 有登记简称
    ("synonym", "阿里通义的千问 3.6", "qwen3-6"),                    # 通义=Qwen 意译，千问在别名表
    ("synonym", "深度的 DeepSeek R1", "deepseek-r1"),                # 深度=深度求索 简称（已登记深度R1）
    ("traditional", "通義千問3.7 是什麼？", "qwen3-7-max"),           # 繁体
    ("traditional", "豆包1.5Pro 好用嗎", "豆包-1-5-pro"),             # 繁体 嗎
    ("traditional", "深度求索R1 是什麼", "deepseek-r1"),             # 繁體 是→是什麼
    ("paraphrase", "OpenAI 家那个带数字 5 点 2 的 GPT", "gpt-5-2"),   # 口语展开描述（已登记 GPT 5 点 2）
    ("paraphrase", "千问的第三代 6 号", "qwen3-6"),                   # 意译版本号（理解层职责，预期 miss）
    ("synonym", "Qwen3 6 是啥", "qwen3-6"),                          # 版本号空格分隔（已登记）
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = AIModelWorld(args.data)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    precise = PreciseMatch(data["objects"])

    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.ffn, output_dimensions=32,
    )
    model = TokenCFormerResolver(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def neural_resolve(text: str):
        tokens, coverage = world.encode_query(text)
        query = model.encode_query(tokens[None].to(device))
        scores = query @ bank.T
        top_scores, top_ids = torch.topk(scores, 2, dim=-1)
        decision = VERIFIER.decide(
            float(top_scores[0, 0].detach()), float(top_scores[0, 1].detach()), float(coverage)
        )
        return int(top_ids[0, 0]), decision, float(top_scores[0, 0].detach())

    by_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "precise_hit": 0,
                                                        "correct": 0, "rows": []})
    for category, text, expected_id in SCENARIOS:
        hit = precise.hit(text)
        label, decision, score = neural_resolve(text)
        hit_id = hit.object_id if hit else None
        # 答对判定：期望 id 非空时比较；None（意图类）记 precise/神经是否给了候选
        answer_id = hit_id if hit is not None else (world.objects[label].object_id if label >= 0 else None)
        correct = (expected_id is not None and answer_id == expected_id) or (
            expected_id is None and hit is None and decision.status == CandidateStatus.UNKNOWN
        )
        by_category[category]["total"] += 1
        by_category[category]["precise_hit"] += 1 if hit is not None else 0
        by_category[category]["correct"] += 1 if correct else 0
        by_category[category]["rows"].append({
            "text": text, "expected": expected_id,
            "precise_hit_id": hit_id, "neural_top": world.objects[label].object_id if label >= 0 else None,
            "neural_score": round(score, 4), "status": decision.status.value, "correct": correct,
        })

    report = {}
    for category, stats in by_category.items():
        report[category] = {
            "total": stats["total"],
            "precise_hit_rate": round(stats["precise_hit"] / stats["total"], 4),
            "accuracy": round(stats["correct"] / stats["total"], 4),
        }
    report["_overall"] = {
        "total": sum(s["total"] for s in by_category.values()),
        "precise_hit_rate": round(sum(s["precise_hit"] for s in by_category.values())
                                  / sum(s["total"] for s in by_category.values()), 4),
        "accuracy": round(sum(s["correct"] for s in by_category.values())
                          / sum(s["total"] for s in by_category.values()), 4),
    }
    output = ROOT / "artifacts" / "sim_identity_scenarios.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"report": report, "scenarios": {k: v["rows"] for k, v in by_category.items()}},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "report": report}, ensure_ascii=False))
    print("\n--- 明细（XX = 未答对）---")
    for category, stats in by_category.items():
        print(f"\n[{category}]")
        for row in stats["rows"]:
            mark = "OK " if row["correct"] else "XX "
            print(f"  {mark}{row['text']}  exp={row['expected']} "
                  f"hit={row['precise_hit_id']} neural={row['neural_top']}({row['neural_score']}) {row['status']}")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
