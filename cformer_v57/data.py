from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from cformer_v56 import CognitiveCandidateAdapter

from .text import HashedTextEncoder, normalize_text


V57_SCALES = (2048, 8192, 32768)
_COLORS = ("红色", "蓝色", "绿色", "金色", "银色", "黑色", "白色", "紫色")
_DOMAINS = ("导航", "医疗", "能源", "教育", "材料", "气象", "交通", "通信")


def _fullwidth(value: str) -> str:
    return "".join(chr(ord(char) + 0xFEE0) if "0" <= char <= "9" else char for char in value)


@dataclass(frozen=True)
class TextRetrievalWorld:
    scale: int
    world_index: int
    encoder: HashedTextEncoder
    entity_features: torch.Tensor
    candidate_features: torch.Tensor
    candidate_kinds: torch.Tensor
    candidate_scopes: torch.Tensor
    entity_kinds: torch.Tensor
    entity_modes: torch.Tensor
    entity_tasks: torch.Tensor
    documents: tuple[str, ...]

    @classmethod
    def build(
        cls, scale: int, world_index: int, dimensions: int = 256
    ) -> "TextRetrievalWorld":
        if scale not in V57_SCALES or scale % 4:
            raise ValueError(f"unsupported V5.7 scale {scale}")
        encoder = HashedTextEncoder(dimensions)
        identity_adapter = CognitiveCandidateAdapter(key_dimensions=8)
        entities = scale // 4
        documents: list[str] = []
        kinds = []
        modes = []
        offset = world_index * 100_000
        for entity in range(entities):
            identifier = offset + entity
            kind = identifier % 2
            mode = (identifier // 3 + world_index) % 2
            color = _COLORS[identifier % len(_COLORS)]
            domain = _DOMAINS[(identifier // len(_COLORS)) % len(_DOMAINS)]
            kind_text = "状态记录" if kind == 0 else "变化规则"
            mode_text = "正向" if mode == 0 else "镜像"
            documents.append(
                f"对象 E{identifier:06d}; 别名 目标{identifier:06d}, 星体-{identifier:06d}; "
                f"类别 {kind_text}; 属性 {color} {domain}; 观察映射 {mode_text}。"
            )
            kinds.append(kind)
            modes.append(mode)
        text_features = encoder.encode_batch(documents)
        identity_features = torch.tensor(
            [
                identity_adapter.key_for_object(world_index * 100_000 + entity)
                for entity in range(entities)
            ],
            dtype=torch.float32,
        )
        entity_features = torch.cat((text_features, identity_features), dim=-1)
        entity_kinds = torch.tensor(kinds)
        entity_modes = torch.tensor(modes)
        entity_tasks = entity_kinds * 2 + entity_modes
        return cls(
            scale,
            world_index,
            encoder,
            entity_features,
            entity_features.repeat_interleave(4, dim=0),
            entity_tasks.repeat_interleave(4),
            torch.arange(1, 5).repeat(entities),
            entity_kinds,
            entity_modes,
            entity_tasks,
            tuple(documents),
        )

    def query_text(self, entity: int, variant: int) -> str:
        identifier = self.world_index * 100_000 + entity
        kind = int(self.entity_kinds[entity])
        mode = int(self.entity_modes[entity])
        kind_text = "状态" if kind == 0 else "变化"
        mode_text = "正向" if mode == 0 else "镜像"
        if variant == 0:
            return f"请查询对象 E{identifier:06d} 的{kind_text}，映射模式{mode_text}。"
        if variant == 1:
            return f"目标{identifier:06d}现在对应哪条{kind_text}信息? 模式={mode_text}"
        if variant == 2:
            noisy = _fullwidth(f"{identifier:06d}")
            return f"请 查 询：星体 - {noisy}；{kind_text}；{mode_text}！"
        distractors = "背景资料 与当前问题无关 " * 18
        return f"{distractors} 最终目标 E{identifier:06d} {kind_text} {mode_text}"

    def encode_queries(self, entities: torch.Tensor, texts: list[str]) -> torch.Tensor:
        text_features = self.encoder.encode_batch(texts)
        identity_adapter = CognitiveCandidateAdapter(key_dimensions=8)
        identity_features = torch.tensor(
            [
                identity_adapter.key_for_object(self.world_index * 100_000 + int(entity))
                for entity in entities
            ],
            dtype=torch.float32,
        )
        return torch.cat((text_features, identity_features), dim=-1)

    def fixed_queries(self, count: int = 100):
        generator = torch.Generator().manual_seed(71_000 + self.scale + self.world_index)
        entities = torch.randperm(self.entity_features.shape[0], generator=generator)[:count]
        variants = torch.arange(count) % 4
        observers = torch.randint(1, 5, (count,), generator=generator)
        texts = [self.query_text(entity, int(variant)) for entity, variant in zip(entities, variants)]
        features = self.encode_queries(entities, texts)
        base_kinds = self.entity_kinds[entities]
        modes = self.entity_modes[entities]
        kinds = self.entity_tasks[entities]
        inverse = base_kinds.bool() ^ modes.bool()
        target_scopes = torch.where(inverse, 5 - observers, observers)
        correct_ids = entities * 4 + target_scopes - 1
        return entities, variants, texts, features, kinds, observers, correct_ids

    def text_shortlists(
        self,
        query_features: torch.Tensor,
        correct_ids: torch.Tensor,
        entity_topk: int = 16,
        query_texts: list[str] | None = None,
    ):
        scores = torch.matmul(query_features, self.entity_features.T)
        top_entities = scores.topk(min(entity_topk, self.entity_features.shape[0]), dim=-1).indices
        # Canonical IDs and registered aliases are deterministic identity evidence.
        # Insert their object shard before lexical candidates instead of asking the
        # neural model to rediscover an already governed identity link.
        if query_texts is not None:
            rows = []
            offset = self.world_index * 100_000
            for row, text in enumerate(query_texts):
                match = re.search(r"(?<!\d)(\d{6})(?!\d)", normalize_text(text))
                entity = int(match.group(1)) - offset if match else -1
                lexical = top_entities[row]
                if 0 <= entity < self.entity_features.shape[0]:
                    lexical = lexical[lexical.ne(entity)]
                    lexical = torch.cat((torch.tensor([entity]), lexical))[:entity_topk]
                rows.append(lexical)
            top_entities = torch.stack(rows)
        scopes = torch.arange(4)[None, None, :]
        candidate_ids = (top_entities[:, :, None] * 4 + scopes).reshape(
            query_features.shape[0], -1
        )
        matches = candidate_ids.eq(correct_ids[:, None])
        recall = matches.any(dim=-1)
        labels = matches.long().argmax(dim=-1)
        lexical_scores = torch.matmul(query_features[:, :256], self.entity_features[:, :256].T)
        pure_lexical_top1 = lexical_scores.argmax(dim=-1).eq(correct_ids // 4)
        return candidate_ids, labels, recall, pure_lexical_top1

    @property
    def text_cache_bytes(self) -> int:
        return self.entity_features.numel() * self.entity_features.element_size()
