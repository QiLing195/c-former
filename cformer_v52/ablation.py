from __future__ import annotations

import torch

from cformer_v4 import ReliableCFormer


ABLATION_MODES = ("full", "no_observer", "retrieval_only", "generation_only")


def _observer_query(
    model: ReliableCFormer, question: torch.Tensor, observer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    question_only = model.question_encoder(question[:, None]).squeeze(1)
    observer_vector = model.embedding(observer)
    gate = torch.sigmoid(model.fusion_gate(torch.cat((question_only, observer_vector), dim=-1)))
    conditioned = question_only + gate * model.observer_transform(observer_vector)
    return question_only, conditioned


@torch.no_grad()
def answer_from_cache_ablation(
    model: ReliableCFormer,
    cache: tuple[torch.Tensor, torch.Tensor],
    question: torch.Tensor,
    observer: torch.Tensor,
    allowed: torch.Tensor,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Separate observer influence on retrieval from influence on answer generation."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"unknown ablation mode {mode}")
    exact, index = cache
    question_only, conditioned = _observer_query(model, question, observer)
    retrieval_query = conditioned if mode in ("full", "retrieval_only") else question_only
    generation_query = conditioned if mode in ("full", "generation_only") else question_only

    scores: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    evidence: list[torch.Tensor] = []
    rows = torch.arange(question.shape[0], device=question.device)
    for hop in range(2):
        hop_scores, hop_selected = model._retrieve(retrieval_query, index, allowed, hop)
        scores.append(hop_scores)
        selected.append(hop_selected)
        evidence.append(exact[rows, hop_selected])
        stacked_evidence = torch.stack(evidence, dim=1)
        if mode in ("full", "no_observer"):
            updated = model.query_block(retrieval_query, stacked_evidence)
            retrieval_query = generation_query = updated
        else:
            retrieval_query = model.query_block(retrieval_query, stacked_evidence)
            generation_query = model.query_block(generation_query, stacked_evidence)

    hidden = model.final_norm(generation_query)
    reliability = model._reliability_features(scores, exact)
    return {
        "answer_logits": model.answer_head(hidden),
        "status_logits": model.status_head(hidden) + model.retrieval_status_head(reliability),
        "scores_first": scores[0],
        "scores_second": scores[1],
        "selected_first": selected[0],
        "selected_second": selected[1],
    }
