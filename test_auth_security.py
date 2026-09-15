# -*- coding: utf-8 -*-
"""#3 安全修复验证：角色伪造防护（服务端令牌决定身份）。

修复前的漏洞：role 由请求体传入 → 构造 {"role":"hr"} 即可越权看管理细则。
修复后：身份来自 X-API-Token（服务端签发），请求体 role 一律忽略。

本脚本直接测服务逻辑（不经 HTTP），验证：
  1. 低权限令牌 + 伪造 role=hr → 仍按低权限处理（不越权）；
  2. 合法 HR 令牌 → 可见管理细则；
  3. 无令牌且非 dev 模式 → 身份解析失败（应 401）；
  4. 无效令牌 → 失败；
  5. 跨数据集级别映射（同一令牌在家庭库映射到对应角色）。

用法：D:/conda/envs/cformer-gpu/python.exe test_auth_security.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "server"))

from auth import DEMO_TOKENS, resolve_identity, role_for_level  # noqa: E402

ROLES = {"employee": 0, "manager": 1, "hr": 2}
HOME_ROLES = {"child": 0, "parent": 1, "finance": 2}

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"OK  {name}")
    else:
        FAIL += 1
        print(f"XX  {name}  {detail}")


def main() -> None:
    os.environ.pop("GOVLAYER_DEV_MODE", None)  # 确保处于生产模式（无匿名降级）

    l0 = "demo-l0-employee"
    l2 = "demo-l2-hr"

    # 1) 低权限令牌 + 伪造 role=hr → 身份级别仍是最低（不越权）
    ident = resolve_identity(l0, "hr", ROLES)
    check("低权限令牌伪造 role=hr 不越权", ident is not None and ident.level == 0,
          f"got level={getattr(ident, 'level', None)}")
    check("伪造后映射回低权限角色", role_for_level(ROLES, ident.level) == "employee")

    # 2) 合法 HR 令牌 → 级别 2 → hr
    ident_hr = resolve_identity(l2, None, ROLES)
    check("合法 HR 令牌得到级别 2", ident_hr is not None and ident_hr.level == 2)
    check("HR 令牌映射到 hr 角色", role_for_level(ROLES, ident_hr.level) == "hr")

    # 3) 无令牌且非 dev 模式 → 解析失败（调用方应 401）
    check("无令牌（生产模式）被拒绝", resolve_identity(None, "hr", ROLES) is None)

    # 4) 无效令牌 → 拒绝（即使带了 role 也不降级）
    check("无效令牌被拒绝", resolve_identity("fake-token", "hr", ROLES) is None)

    # 5) 跨数据集级别映射：同一 HR 令牌在家庭库映射到 finance
    check("级别 → 家庭库角色映射正确",
          role_for_level(HOME_ROLES, ident_hr.level) == "finance")

    # 6) dev 模式：匿名可得**最低**级别（不接受越权声明）
    os.environ["GOVLAYER_DEV_MODE"] = "1"
    ident_dev = resolve_identity(None, "hr", ROLES)
    check("dev 模式匿名仅得最低级别（不采信 role）",
          ident_dev is not None and ident_dev.level == 0)
    os.environ.pop("GOVLAYER_DEV_MODE", None)

    print(f"\n{'-' * 50}\n通过 {PASS} / 失败 {FAIL}")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
