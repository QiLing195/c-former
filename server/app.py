# -*- coding: utf-8 -*-
"""C-Former GovLayer Web 服务：把确定性治理层封装成可部署 API + 前端。

设计要点（部署友好）：
  - GovLayer 纯标准库，服务本身不依赖 torch → 镜像小、启动秒级；
  - 治理逻辑（权限/空白/先例）在服务端确定性执行，LLM 只负责"把可见内容写成自然语言"；
  - 可选接入 DeepSeek：**只有权限过滤后的可见内容才进 Prompt**（泄漏在源头掐断）。

接口：
  GET  /api/datasets            → 可用知识库 + 角色列表
  POST /api/ask                 → {dataset, role, question} → 治理后答案 + 依据
  GET  /                        → 前端演示页
  GET  /healthz                 → 健康检查

运行：
  pip install -r requirements-serve.txt
  uvicorn app:app --host 0.0.0.0 --port 8000     # 在 server/ 目录下
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parent
sys.path.insert(0, str(ROOT))

from cformer_v63.governance import GovLayer, load_dataset  # noqa: E402
from cformer_v63.semantic import LLMSemanticBackend  # noqa: E402

DATA_DIR = ROOT / "data"
STATIC_DIR = SERVER_DIR / "static"

app = FastAPI(title="C-Former GovLayer", version="0.2.0")

# 语义后端（#1 检索语义化 + #2 覆盖判定）：有 DEEPSEEK_API_KEY 时启用，
# 否则自动降级为关键词检索 + 词表探针（零依赖可跑）。
SEMANTIC = LLMSemanticBackend()

# ---- 知识库加载（dataset 驱动：新客户只需加一个 gov_*.json）----
_LAYERS: dict[str, GovLayer] = {}
_DATASETS: dict[str, dict] = {}


def _load_all_datasets() -> None:
    for path in sorted(DATA_DIR.glob("gov_*.json")):
        spec = load_dataset(path)
        dataset_id = path.stem.replace("gov_", "")
        _LAYERS[dataset_id] = GovLayer(
            objects=spec["objects"], roles=spec["roles"],
            probe_pairs=spec["probe_rules"], cases=spec["cases"],
            semantic_backend=SEMANTIC if SEMANTIC.available else None,
        )
        _DATASETS[dataset_id] = {
            "id": dataset_id,
            "file": path.name,
            "roles": spec["roles"],
            "n_objects": len(spec["objects"]),
            "n_cases": len(spec["cases"]),
        }


_load_all_datasets()


# ---- 请求/响应模型 ----
class AskRequest(BaseModel):
    dataset: str
    role: str
    question: str


class AskResponse(BaseModel):
    question: str
    role: str
    verdict: str                 # covered | gap | out_of_scope
    answer_text: str             # 治理后返回给用户的内容（LLM 或条款原文）
    visible_sources: list[dict]  # [{id, title, content}] 权限过滤后的依据
    denied_fields: list[str]     # 命中但因权限被截断的条目
    precedents: list[dict]       # 过往先例（仅供参考）
    boundary_note: str           # 边界声明（空白/权限/先例提示）
    typo_corrections: list[str]  # 错别字纠正记录（透明可审计）
    semantic_used: bool          # 是否走了 LLM 语义检索（否则关键词回退）
    llm_used: bool


def _llm_answer(question: str, visible_context: str, verdict: str) -> str | None:
    """可选：把**权限过滤后的可见内容**交给 DeepSeek 生成自然语言答案。
    关键：Prompt 只包含可见内容——不可见内容在检索前已被治理层排除。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None
    if verdict == "gap":
        return None  # 空白问题不交给 LLM（避免编造），走治理层的升级话术
    import requests

    system = ("你是企业知识库助手。只能依据下方【已授权资料】回答；"
              "资料不足时必须回答'根据现有资料无法回答'，不得编造任何制度条款。")
    prompt = f"【已授权资料】\n{visible_context}\n\n【员工提问】{question}\n\n【回答】"
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 300},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001 —— LLM 不可用时降级为条款原文
        return None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "datasets": list(_DATASETS)}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    return list(_DATASETS.values())


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    layer = _LAYERS.get(req.dataset)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {req.dataset}")
    if req.role not in layer.roles:
        raise HTTPException(status_code=400, detail=f"unknown role: {req.role}")

    ans = layer.answer(req.question, req.role)

    # 权限过滤后的可见依据（只有这些内容会被交给 LLM）
    sources = []
    for oid, content in ans.visible_texts.items():
        obj = next(o for o in layer.objects if o["id"] == oid)
        sources.append({"id": oid, "title": obj.get("title", oid),
                        "content": content or "（无权限查看此内容）"})

    visible_context = "\n---\n".join(
        f"《{s['title']}》：{s['content']}" for s in sources if s["content"] and "无权限" not in s["content"]
    )
    # 回答生成：
    #  - restricted（权限截断）：**不交给 LLM**——否则模型会把"你无权限"说成"资料不足"，
    #    必须由治理层直接给出权限声明（这是本系统的核心语义，不能被生成层稀释）；
    #  - gap（知识空白）：也不交给 LLM（避免编造），走治理层升级话术；
    #  - 其余情况才允许 LLM 基于可见内容生成自然语言。
    llm_text = None
    if ans.verdict not in ("restricted", "gap"):
        llm_text = _llm_answer(req.question, visible_context, ans.verdict)

    if ans.verdict == "restricted":
        answer_text = ans.boundary_note or "该问题涉及的信息超出你的角色权限范围。"
    elif llm_text:
        answer_text = llm_text
    elif ans.verdict == "gap":
        answer_text = ("根据现有制度无法回答此问题。" + (ans.boundary_note or ""))
    elif ans.verdict == "out_of_scope":
        answer_text = "此问题不在已登记制度范围内，建议咨询 HR。"
    else:
        answer_text = visible_context or "根据你的权限，没有可展示的内容。"

    return AskResponse(
        question=req.question, role=req.role, verdict=ans.verdict,
        answer_text=answer_text,
        visible_sources=sources,
        denied_fields=ans.denied_fields,
        precedents=[{k: c.get(k) for k in ("case_id", "topic", "ruling", "reasoning",
                                           "approver", "date", "reference_only")}
                    for c in ans.precedents],
        boundary_note=ans.boundary_note or "",
        typo_corrections=ans.typo_corrections,
        semantic_used=ans.semantic_used,
        llm_used=bool(llm_text),
    )


# ---- 前端 ----
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
