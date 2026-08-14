from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class IVFConfig:
    n_centroids: int = 1024
    n_iter: int = 12
    seed: int = 61
    train_sample: int = 32768


class IVFIndex:
    """Cosine inverted-file index over normalized vectors (pure torch).

    One object vector is stored once (single copy); aliases, questions and
    observers never duplicate vectors. Search probes ``nprobe`` centroid
    clusters and scores only the candidates inside them.
    """

    def __init__(self, dimension: int, config: IVFConfig | None = None) -> None:
        self.dimension = dimension
        self.config = config or IVFConfig()
        self.centroids: torch.Tensor | None = None  # (k, d) FP32, normalized
        self.vectors: torch.Tensor | None = None    # (n, d) FP16
        self.ids: list[int] = []
        self.id_array: torch.Tensor | None = None   # (n,) long, same device as vectors
        self.cluster_of: torch.Tensor | None = None  # (n,) long
        self.tombstones: set[int] = set()
        self.tombstone_ids: torch.Tensor | None = None  # (t,) long, same device
        self.version = 0

    # -- build --------------------------------------------------------------

    def train(self, sample: torch.Tensor) -> None:
        if sample.shape[0] < self.config.n_centroids:
            raise ValueError("sample smaller than centroid count")
        vectors = F.normalize(sample.float(), dim=-1)
        k = self.config.n_centroids
        rng = torch.Generator().manual_seed(self.config.seed)
        centroids = vectors[torch.randperm(vectors.shape[0], generator=rng)[:k]].clone()
        for _ in range(self.config.n_iter):
            assignment = (vectors @ centroids.T).argmax(dim=-1)
            for centroid in range(k):
                members = vectors[assignment == centroid]
                if members.shape[0] > 0:
                    centroids[centroid] = F.normalize(members.mean(dim=0), dim=-1)
        self.centroids = centroids

    def add(self, vectors: torch.Tensor, ids: Sequence[int]) -> None:
        if self.centroids is None:
            raise RuntimeError("train() must run before add()")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        normalized = F.normalize(vectors.float(), dim=-1).half()
        if self.vectors is None:
            self.vectors = normalized
            self.ids = list(ids)
            self.id_array = torch.tensor(ids, dtype=torch.long, device=normalized.device)
        else:
            self.vectors = torch.cat((self.vectors, normalized), dim=0)
            self.ids.extend(int(item) for item in ids)
            self.id_array = torch.cat(
                (self.id_array, torch.tensor(ids, dtype=torch.long, device=normalized.device)),
                dim=0,
            )
        cluster = (normalized.float() @ self.centroids.T).argmax(dim=-1)
        self.cluster_of = (
            cluster
            if self.cluster_of is None
            else torch.cat((self.cluster_of, cluster), dim=0)
        )
        self.version += 1

    def remove(self, ids: Sequence[int]) -> None:
        for item in ids:
            self.tombstones.add(int(item))
        if self.tombstone_ids is None:
            self.tombstone_ids = torch.tensor(ids, dtype=torch.long, device=self.vectors.device)
        else:
            extra = torch.tensor(ids, dtype=torch.long, device=self.vectors.device)
            self.tombstone_ids = torch.cat((self.tombstone_ids, extra), dim=0)
        self.version += 1

    # -- search -------------------------------------------------------------

    def search(
        self, query: torch.Tensor, nprobe: int, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scores, object_ids) of the top-k among probed clusters."""
        if self.vectors is None or self.centroids is None:
            raise RuntimeError("empty index")
        q = F.normalize(query.float().reshape(1, -1), dim=-1)
        probe = (q @ self.centroids.T).topk(min(nprobe, self.centroids.shape[0])).indices[0]
        keep = torch.zeros(self.centroids.shape[0], dtype=torch.bool, device=q.device)
        keep[probe] = True
        positions = keep[self.cluster_of].nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            return torch.empty(0), torch.empty(0)
        candidate_ids = self.id_array[positions]
        if self.tombstone_ids is not None and self.tombstone_ids.numel():
            live = ~torch.isin(candidate_ids, self.tombstone_ids)
            positions = positions[live]
            candidate_ids = candidate_ids[live]
        if positions.numel() == 0:
            return torch.empty(0), torch.empty(0)
        scores = q @ self.vectors[positions].float().T
        top = scores.topk(min(topk, positions.numel()))
        return top.values[0], candidate_ids[top.indices[0]]

    # -- persistence ----------------------------------------------------------

    def snapshot(self, path) -> None:
        payload = {
            "dimension": self.dimension,
            "version": self.version,
            "centroids": self.centroids,
            "vectors": self.vectors,
            "ids": self.ids,
            "cluster_of": self.cluster_of,
            "tombstones": sorted(self.tombstones),
            "config": {
                "n_centroids": self.config.n_centroids,
                "n_iter": self.config.n_iter,
                "seed": self.config.seed,
            },
        }
        torch.save(payload, path)

    @classmethod
    def restore(cls, path) -> "IVFIndex":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        index = cls(payload["dimension"], IVFConfig(**payload["config"]))
        index.version = int(payload["version"])
        index.centroids = payload["centroids"]
        index.vectors = payload["vectors"]
        index.ids = list(payload["ids"])
        index.id_array = torch.tensor(index.ids, dtype=torch.long, device=index.vectors.device)
        index.cluster_of = payload["cluster_of"]
        index.tombstones = set(payload["tombstones"])
        if index.tombstones:
            index.tombstone_ids = torch.tensor(
                sorted(index.tombstones), dtype=torch.long, device=index.vectors.device
            )
        return index

    # -- stats ----------------------------------------------------------------

    @property
    def count(self) -> int:
        return 0 if self.vectors is None else int(self.vectors.shape[0])

    @property
    def bytes_per_vector(self) -> int:
        return 2 * self.dimension  # FP16


class QuantizedVectorStore:
    """FP16 or INT8 single-copy vector storage with per-vector scale."""

    def __init__(self, dimension: int, dtype: str = "fp16") -> None:
        if dtype not in ("fp16", "int8"):
            raise ValueError(dtype)
        self.dimension = dimension
        self.dtype = dtype
        self._fp16: torch.Tensor | None = None
        self._int8: torch.Tensor | None = None
        self._scales: torch.Tensor | None = None  # (n, 1) FP16

    def add(self, vectors: torch.Tensor) -> None:
        normalized = F.normalize(vectors.float(), dim=-1)
        if self.dtype == "fp16":
            block = normalized.half()
            if self._fp16 is None:
                self._fp16 = block
            else:
                self._fp16 = torch.cat((self._fp16, block), dim=0)
        else:
            scale = normalized.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
            block = (normalized / scale * 127).round().to(torch.int8)
            if self._int8 is None:
                self._int8 = block
                self._scales = scale.half()
            else:
                self._int8 = torch.cat((self._int8, block), dim=0)
                self._scales = torch.cat((self._scales, scale.half()), dim=0)

    def vector(self, position: int) -> torch.Tensor:
        if self.dtype == "fp16":
            return self._fp16[position].float()
        return (self._int8[position].float() / 127.0) * self._scales[position].float()

    def vectors(self, positions: torch.Tensor) -> torch.Tensor:
        if self.dtype == "fp16":
            return self._fp16[positions].float()
        return (self._int8[positions].float() / 127.0) * self._scales[positions].float()

    @property
    def count(self) -> int:
        tensor = self._fp16 if self.dtype == "fp16" else self._int8
        return 0 if tensor is None else int(tensor.shape[0])

    @property
    def bytes_per_vector(self) -> int:
        if self.dtype == "fp16":
            return 2 * self.dimension
        return self.dimension + 2  # int8 body + FP16 scale


def rerank(
    query: torch.Tensor,
    candidate_ids: torch.Tensor,
    store_or_vectors,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact cosine rerank of ANN candidates (no full scan).

    Accepts either a QuantizedVectorStore or a raw 2-D vector tensor.
    """
    q = F.normalize(query.float().reshape(1, -1), dim=-1)
    if candidate_ids.numel() == 0:
        return torch.empty(0), candidate_ids
    if isinstance(store_or_vectors, QuantizedVectorStore):
        vectors = store_or_vectors.vectors(candidate_ids)
    else:
        vectors = store_or_vectors[candidate_ids].float()
    scores = q @ vectors.T
    top = scores.topk(min(topk, candidate_ids.numel()))
    return top.values[0], candidate_ids[top.indices[0]]


def exact_search(
    query: torch.Tensor, bank: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force cosine search over the whole bank (baseline)."""
    q = F.normalize(query.float().reshape(1, -1), dim=-1)
    scores = q @ bank.float().T
    top = scores.topk(min(topk, bank.shape[0]))
    return top.values[0], top.indices[0]
