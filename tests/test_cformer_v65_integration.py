"""V6.5 集成冒烟：一个入口覆盖全部已验证能力，端到端可组合。"""
from pathlib import Path

import pytest

from cformer_v59 import CandidateStatus
from cformer_v62 import ObserverFrame
from cformer_v65 import UnifiedCFormer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai_models_dataset.json"


@pytest.fixture(scope="module")
def unified():
    # 冒烟只验证接线正确性（reasoner/exact/recursion 均为确定性路径），小步数+CPU 即可
    return UnifiedCFormer(DATA, seed=601, steps=60, device="cpu")


def test_known_query_supported_with_evidence(unified):
    answer = unified.resolve("OpenAI 的 GPT 系列最新模型是什么？", query_type="known")
    assert answer.status == "supported"
    assert answer.object_id == "gpt-5-4"
    assert answer.path == "reasoned"
    assert set(answer.evidence) == {"名称", "属性", "关系", "变化"}
    assert any(t.startswith("anchor=") for t in answer.trace)


def test_exact_alias_and_recursion_paths(unified):
    exact = unified.resolve("GPT-5.4")
    assert exact.status == "supported" and exact.path == "exact"

    hop = unified.resolve("Qwen2.5-Coder 的前一代是什么？")
    assert hop.status == "supported" and hop.path == "recursion"
    assert hop.object_id == "qwen2-5"
    assert "recursion_hops=1" in hop.reason


def test_temporal_snapshot_and_vacuum(unified):
    past = unified.resolve("截至2021年，GPT 系列最新模型是什么？", query_type="known")
    assert past.status == "supported" and past.object_id != "gpt-5-4"

    vacuum = unified.resolve("截至1999年，GPT 系列最新模型是什么？")
    assert vacuum.status == "unknown"
    assert "temporal_no_member_as_of=1999" in vacuum.reason
    assert vacuum.object_id is None


def test_structural_ambiguity_and_unknown_proposal(unified):
    ambiguous = unified.resolve("Gemini 2.5 是哪一个？")
    assert ambiguous.status in ("ambiguous", "unknown")  # 结构规则或 margin 判定

    unknown = unified.resolve("gpt-nine")
    assert unknown.status == "unknown"
    assert unknown.path == "ann"


def test_observer_masking_never_leaks(unified):
    frame = ObserverFrame("us-only", allowed_companies=frozenset({"OpenAI"}))
    denied = unified.resolve("千问系列最新的模型是哪一个？", observer_frame=frame)
    assert denied.status == "access_denied"
    assert denied.object_id is None


def test_single_copy_across_all_resolutions(unified):
    before = unified.index.count
    for text in ("OpenAI 的 GPT 系列最新模型是什么？",
                 "Claude Opus 5 的前一代的前一代是什么？",
                 "截至2020年，DeepSeek 系列最新模型是什么？"):
        unified.resolve(text)
    assert unified.index.count == before == len(unified.world.objects)


def test_ledger_records_proposals_not_auto_verified(unified):
    verified = unified.ledger.connection.execute(
        "SELECT COUNT(*) FROM candidates WHERE status='verified'").fetchone()[0]
    assert verified == 0  # 模型永远不能自我验证
