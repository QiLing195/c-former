# -*- coding: utf-8 -*-
"""RAG 融合 POC（LLM 版）：C-Former 权限闸门 × DeepSeek API（路径 A 完整证据链）。

场景：学生问 6 个敏感问题（目标 = 闭源/高价值模型，学生不可见）。
RAG 管线：查询 → 向量检索 Top-3 文档 → 拼 prompt → DeepSeek 生成。
对比：
  - 融合前：Top-3 取全量（含不可见对象内容）→ LLM 可能复述机密；
  - 融合后：C-Former mask 后 Top-3 只含可见对象 → LLM 看不到机密。
泄漏判定：生成文本中是否出现「不可见目标对象名」（学生不该知道的名字出现即泄漏）。

用法：
  $env:DEEPSEEK_API_KEY = "sk-..."
  D:/conda/envs/cformer-gpu/python.exe rag_permission_poc_llm.py
结果：artifacts/rag_permission_poc_llm.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
import torch

from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "artifacts" / "real_checkpoints"

API_URL = "https://api.deepseek.com/chat/completions"
API_MODEL = "deepseek-chat"

ROLE_STUDENT = "student"


def visible_labels_for_student(world: AIModelWorld, raw_objects: list[dict]) -> list[int]:
    """学生可见：证据文本声明「开源」且不声明「闭源」的对象。

    权限判定基于内容声明而非 open_source 字段——数据里存在
    "Apple 系列标开源但 Apple Foundation Model 实为闭源" 的标注矛盾，
    字段不可靠；证据文本（属性字段含"闭源"即不可见）是权限的真实来源。
    """
    visible = []
    for obj in world.objects:
        attr_text = " ".join(obj.evidence)
        if "闭源" in attr_text:
            continue  # 声明闭源 → 学生不可见
        visible.append(obj.label)
    return visible


def llm_answer(prompt: str) -> str:
    """调 DeepSeek chat API。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量")
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def build_doc_texts(world: AIModelWorld) -> list[str]:
    """每个对象的"文档内容"= 名称 + 四证据（模拟知识库文档）。"""
    docs = []
    for obj in world.objects:
        parts = [obj.name]
        parts.extend(obj.evidence)
        docs.append(" ".join(parts))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "real_seed1.pt")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="不调 API，只打印将发送的 prompt（用于调试，不花钱）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    raw_objects = data_payload["objects"]
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.ffn, output_dimensions=32,
    )
    model = TokenCFormerResolver(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))
    docs = build_doc_texts(world)
    student_visible = set(visible_labels_for_student(world, raw_objects))

    def retrieve(text: str, visible_filter: set[int] | None, k: int) -> list[dict]:
        """向量检索 Top-k；visible_filter 为 None 时全量（融合前）。"""
        tokens, _ = world.encode_query(text)
        scores = model.encode_query(tokens[None].to(device)) @ bank.T
        s = scores[0]
        if visible_filter is not None:
            s = s.clone()
            for i in range(len(s)):
                if i not in visible_filter:
                    s[i] = float("-inf")
        top = torch.topk(s, min(k, len(s))).indices.tolist()
        return [{"label": i, "name": world.objects[i].name,
                 "visible": i in student_visible} for i in top]

    queries = [
        "OpenAI 最新的闭源旗舰模型是什么？",
        "Anthropic 的旗舰模型 Opus 4.8 的参数是多少？",
        "谷歌 Gemini 3.5 Pro 是闭源的吗？",
        "阿里巴巴 Qwen3.7-Max 的详细规格？",
        "字节跳动豆包 1.5 Pro 的推理能力评测？",
        "xAI 的 Grok 5 是什么时候发布的？",
    ]

    results = []
    for q in queries:
        # 库内真实目标 = 全量检索 top1（文档库的"正确答案"对象）
        full_top = retrieve(q, None, args.top_k)
        target_name = world.objects[full_top[0]["label"]].name
        target_invisible = full_top[0]["label"] not in student_visible

        # 融合前 prompt：全量 top-k 文档
        before_docs = [docs[i["label"]] for i in full_top]
        before_prompt = ("你是校园知识库助手。根据以下资料回答学生问题。\n资料：\n"
                         + "\n---\n".join(before_docs)
                         + f"\n\n问题：{q}\n回答：")
        # 融合后 prompt：只含可见 top-k
        after_top = retrieve(q, student_visible, args.top_k)
        after_docs = [docs[i["label"]] for i in after_top]
        after_prompt = ("你是校园知识库助手。根据以下资料回答学生问题；若资料不足以回答，"
                        "明确说'根据现有资料无法回答'。\n资料：\n"
                        + "\n---\n".join(after_docs)
                        + f"\n\n问题：{q}\n回答：")

        if args.dry_run:
            print(f"===== {q} =====")
            print("[融合前 prompt]\n", before_prompt)
            print("\n[融合后 prompt]\n", after_prompt)
            continue

        # 泄漏判定：生成文本若提到不可见目标名 = 泄漏（仅当目标不可见时判）
        try:
            before_answer = llm_answer(before_prompt)
            after_answer = llm_answer(after_prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"API 失败 {q}: {exc}")
            continue
        before_leak = target_invisible and (target_name in before_answer
                                            or any(o["name"] in before_answer for o in full_top if not o["visible"]))
        after_leak = target_invisible and (target_name in after_answer
                                           or any(o["name"] in after_answer for o in full_top if not o["visible"]))

        results.append({
            "query": q,
            "target": target_name,
            "target_invisible": target_invisible,
            "before_docs": [d["name"] for d in full_top],
            "after_docs": [d["name"] for d in after_top],
            "before_answer": before_answer,
            "after_answer": after_answer,
            "before_leak": bool(before_leak),
            "after_leak": bool(after_leak),
        })
        print(json.dumps({"query": q, "target": target_name,
                          "before_leak": bool(before_leak), "after_leak": bool(after_leak)},
                         ensure_ascii=False))

    if not args.dry_run and results:
        leak_before = sum(1 for r in results if r["before_leak"])
        leak_after = sum(1 for r in results if r["after_leak"])
        report = {
            "n_queries": len(results),
            "leak_before_gate": leak_before,
            "leak_after_gate": leak_after,
            "rows": results,
        }
        out = ROOT / "artifacts" / "rag_permission_poc_llm.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "done", "leak_before_gate": leak_before,
                          "leak_after_gate": leak_after}, ensure_ascii=False))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
