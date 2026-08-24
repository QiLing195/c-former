# -*- coding: utf-8 -*-
"""V6.2 ObserverFrame 与确定性访问门控。

核心假设的边界（路线图 §8）：
- 身份解析先于观测点注入——门控只作用于"已解析出的对象是否对该观测点可见"，
  绝不参与身份猜测；
- 观测点是坐标/索引，不是知识副本——对象向量单副本，权限是掩码不是过滤后的新库；
- 权限泄漏率为 0：被拒对象不得以任何 supported 形式暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from cformer_v59 import CandidateStatus


@dataclass(frozen=True)
class ObserverFrame:
    observer_id: str
    allowed_companies: frozenset[str] | None = None   # None = 全部可见
    allowed_regions: frozenset[str] | None = None     # None = 全部可见
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


class ObserverGate:
    """Deterministic visibility mask over resolved object ids."""

    def __init__(
        self,
        *,
        company_of: Callable[[str], str | None],   # object_id -> 公司 | None
        region_of: Callable[[str], str | None],    # object_id -> 区域 | None
    ) -> None:
        self.company_of = company_of
        self.region_of = region_of

    def check(self, frame: ObserverFrame | None, object_id: str) -> AccessDecision:
        if frame is None:
            return AccessDecision(True, "no_observer")
        if frame.allowed_companies is not None:
            company = self.company_of(object_id)
            if company is None or company not in frame.allowed_companies:
                return AccessDecision(False, f"company_not_visible:{company}")
        if frame.allowed_regions is not None:
            region = self.region_of(object_id)
            if region is None or region not in frame.allowed_regions:
                return AccessDecision(False, f"region_not_visible:{region}")
        return AccessDecision(True, "visible")

    @staticmethod
    def denied_status() -> CandidateStatus:
        return CandidateStatus.ACCESS_DENIED
