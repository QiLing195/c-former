# -*- coding: utf-8 -*-
"""服务端身份解析：把「客户端声称的角色」改为「服务端签发的令牌 → 权限级别」。

安全背景（#3 修复）：
  修复前 role 由请求体传入——任何人构造 {"role":"hr"} 即可越权查看管理细则，
  这是演示原型最常见的致命漏洞。
  修复后：客户端只能提供 **API Token**；服务端据此解析**权限级别**，
  请求体里的任何 role 字段一律忽略（防伪造）。

为什么用「级别」而不是角色名：级别（0 最低 → 越高权限越大）是跨数据集通用语义，
  企业制度里 0=员工/1=经理/2=HR，家庭场景 0=孩子/1=家长/2=财务管家——
  同一个 Token 在不同知识库自动映射到对应角色，无需为每个库维护角色名映射。

生产化：把 DEMO_TOKENS 换成 SSO / LDAP / OAuth 校验即可（接口不变）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

# 演示令牌（仅本地演示；生产必须换成真实身份系统校验）
DEMO_TOKENS: dict[str, dict] = {
    "demo-l0-employee": {"level": 0, "label": "普通员工 / 学生 / 孩子"},
    "demo-l1-manager": {"level": 1, "label": "部门经理 / 管理员 / 家长"},
    "demo-l2-hr": {"level": 2, "label": "HR / 财务管家"},
}

TOKEN_ENV = "GOVLAYER_TOKENS"          # 可用 JSON 覆盖演示令牌
DEV_MODE_ENV = "GOVLAYER_DEV_MODE"     # =1 时允许匿名请求并接受请求体 role（仅本地演示）


@dataclass
class Identity:
    level: int
    label: str
    token: str


def _tokens() -> dict[str, dict]:
    raw = os.environ.get(TOKEN_ENV, "")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return DEMO_TOKENS


def dev_mode() -> bool:
    return os.environ.get(DEV_MODE_ENV, "") in ("1", "true", "True")


def resolve_identity(token: str | None, claimed_role: str | None,
                     role_levels: dict[str, int]) -> Identity | None:
    """解析请求身份。

    - 有合法 Token：以 Token 的级别为准（**忽略 claimed_role**，防越权伪造）；
    - 无 Token 且处于 dev 模式：降级为最低级别（演示用，且不接受越权声明）；
    - 无 Token 且非 dev 模式：返回 None（调用方应回 401）。
    """
    if token:
        info = _tokens().get(token)
        if info is not None:
            return Identity(level=int(info["level"]),
                            label=str(info.get("label", f"level-{info['level']}")),
                            token=token)
        return None  # 无效 token，不降级
    if dev_mode():
        # dev 模式：允许匿名，但**级别固定为最低**，不采信 claimed_role
        lowest = min(role_levels.values()) if role_levels else 0
        return Identity(level=lowest, label="匿名（dev 模式·最低权限）", token="")
    return None


def role_for_level(role_levels: dict[str, int], level: int) -> str | None:
    """把权限级别映射回该知识库里的角色名（级别 ≤ 目标级别中最高者）。"""
    candidates = [(lvl, role) for role, lvl in role_levels.items() if lvl <= level]
    if not candidates:
        return None
    return max(candidates)[1]  # 取满足条件的最高级别角色
