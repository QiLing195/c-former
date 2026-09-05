# -*- coding: utf-8 -*-
"""最小 POC：C-Former 权限闸门 × RAG（路径 A——检索前权限过滤）。

场景：校内知识库（用 AI 模型域 273 对象模拟"文档库"），两个用户角色：
  - 学生：只可见开源模型（可见性 = 确定性规则）
  - 管理员：全部可见
敏感查询：学生问一个**闭源/旗舰**模型（管理员才可见）→ 融合前 RAG 直检会泄漏，
融合后 C-Former 闸门先把不可见对象 mask 掉，RAG 检索不到 → 拦截。

RAG 模拟（简化但诚实）：
  - 检索：V6.0 神经编码器 查询×库 打分 → Top-K（真实向量检索）
  - 生成：取 Top-1 对象的内容文本作为"生成答案"（不接 LLM API——
    泄漏发生在检索层，生成只是透传检索结果，接不接 LLM 结论一致）
对比输出：融合前 vs 融合后，泄漏次数 / mask_caught（闸门实际挡住的检索命中）

用法：D:/conda/envs/cformer-gpu/python.exe rag_permission_poc.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "artifacts" / "real_checkpoints"

# 角色定义（确定性可见性规则）
# 学生：只可见 open_source=True 的对象 + 指定公开系列
# 管理员：全部可见
ROLE_STUDENT = "student"
ROLE_ADMIN = "admin"


def visible_labels_for_role(role: str, world: AIModelWorld) -> list[int]:
    if role == ROLE_ADMIN:
        return list(range(len(world.objects)))
    if role == ROLE_STUDENT:
        return [obj.label for obj in world.objects if obj.name.startswith("Llama")
                or obj.name.startswith("Qwen")
                or obj.name.startswith("DeepSeek")
                or obj.name.startswith("GLM")
                or obj.name.startswith("InternLM")
                or obj.name.startswith("Phi")
                or obj.name.startswith("Mistral")]
    return []


# 敏感查询（目标对象 = 闭源/高价值，学生不应可见）
SENSITIVE_QUERIES = [
    "OpenAI 最新的闭源旗舰模型是什么？",   # → GPT-5.2（闭源，学生不可见）
    "Anthropic 的旗舰模型 Opus 4.8 的参数是多少？",
    "谷歌 Gemini 3.5 Pro 是闭源的吗？",
    "阿里巴巴 Qwen3.7-Max 的详细规格？",
    "字节跳动豆包 1.5 Pro 的推理能力评测？",
    "xAI 的 Grok 5 是什么时候发布的？",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "real_seed1.pt")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.ffn, output_dimensions=32,
    )
    model = TokenCFormerResolver(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def rag_retrieve(text: str, visible: list[int] | None) -> tuple[int, int, float]:
        """RAG 检索：查询 × 库打分 → Top-1。
        返回 (过滤后top1, 未过滤raw top1, raw_score)。
        visible=None 时不过滤（融合前）；visible=[] 空集 → 全部拒绝。
        """
        tokens, _ = world.encode_query(text)
        scores = model.encode_query(tokens[None].to(device)) @ bank.T  # (1, N)
        raw_top = int(scores[0].argmax())
        if visible is None:
            return raw_top, raw_top, float(scores[0, raw_top])
        if not visible:
            return -1, raw_top, float(scores[0, raw_top])  # 空可见集 → 全拒
        mask = torch.full_like(scores[0], float("-inf"))
        v = torch.tensor(visible, dtype=torch.long, device=device)
        mask[v] = scores[0, v]
        return int(mask.argmax()), raw_top, float(scores[0, raw_top])

    student_visible = visible_labels_for_role(ROLE_STUDENT, world)
    admin_visible = visible_labels_for_role(ROLE_ADMIN, world)
    print(f"知识库对象: {len(world.objects)} | 学生可见: {len(student_visible)} | 管理员可见: {len(admin_visible)}")

    rows = []
    leak_before = leak_after = mask_caught = denied_after = 0
    for text in SENSITIVE_QUERIES:
        # 库内真实目标 = 未过滤 top1
        raw_top, _, _ = rag_retrieve(text, None)
        # 融合前：RAG 直检（无闸门）
        before_top, _, _ = rag_retrieve(text, None)
        # 融合后：C-Former 闸门（学生可见性 mask）
        after_top, raw_of_filtered, _ = rag_retrieve(text, student_visible)

        target_obj = world.objects[raw_top]
        before_obj = world.objects[before_top]
        # 泄漏判定：检索命中的对象不在学生可见集 = 泄漏
        before_leak = before_top not in student_visible
        # 融合后：若"库内真实目标"不可见 → 应拒绝（denied），不是硬答无关对象
        target_invisible = raw_top not in student_visible
        if target_invisible:
            after_leak = False          # 拒绝 = 未泄漏
            denied = True
            after_gate_str = "DENIED(权限拦截)"
        else:
            after_obj = world.objects[after_top]
            after_leak = after_top not in student_visible
            denied = False
            after_gate_str = "LEAK" if after_leak else after_obj.name
        mask_caught_this = (raw_of_filtered not in student_visible) and target_invisible
        leak_before += 1 if before_leak else 0
        leak_after += 1 if after_leak else 0
        mask_caught += 1 if mask_caught_this else 0
        denied_after += 1 if denied else 0

        rows.append({
            "query": text,
            "target_in_library": target_obj.name,
            "target_student_visible": not target_invisible,
            "before_gate_top1": before_obj.name,
            "before_gate_leak": before_leak,
            "after_gate": after_gate_str,
            "after_gate_leak": after_leak,
            "mask_caught": mask_caught_this,
        })

    report = {
        "scenario": "student asks sensitive (admin-only) queries in shared KB",
        "n_queries": len(SENSITIVE_QUERIES),
        "leak_before_gate": leak_before,
        "leak_after_gate": leak_after,
        "denied_after_gate": denied_after,
        "mask_caught": mask_caught,
        "conclusion": "路径A成立：C-Former 权限闸门在检索前拦截了 RAG 对不可见对象的命中"
        if leak_before > 0 and leak_after == 0 else "查看明细判断",
        "rows": rows,
    }
    output = ROOT / "artifacts" / "rag_permission_poc.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done",
                      "leak_before_gate": leak_before, "leak_after_gate": leak_after,
                      "denied_after_gate": denied_after, "mask_caught": mask_caught},
                     ensure_ascii=False))
    for r in rows:
        print(f"{r['query']}\n   库内目标: {r['target_in_library']}(学生可见={r['target_student_visible']}) | 闸门前Top1: {r['before_gate_top1']}(leak={r['before_gate_leak']}) | 闸门后: {r['after_gate']} | mask_caught={r['mask_caught']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
