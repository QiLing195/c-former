# C-Former 交接文档（V6.3 更新版 · 已归档）

> ⚠️ **本文件已归档**：完整项目说明与最新成果见 [`README.md`](README.md)。本文件保留历史交接信息。

## 1. 当前状态快照（2026-08 会话线）

| 项 | 状态 |
|---|---|
| 版本 | 会话线 V6.3（内部 V6.x ↔ 测试版 0.6.x；历史 tag `v0.6.1c`） |
| 数据 | AI 模型 273 对象 + 国家 68 + 电影 60 + 制度/流程域（`data/`） |
| 测试 | `D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q` |
| 打包/CI | `pyproject.toml` + `.github/workflows/ci.yml` |
| 冻结基线 | V6.0 编码器（2 层 C-Former，共享 Token Transformer） |

## 2. 关键结论（实验证据，全部真实数据 3 种子）

1. 身份解析（混合架构：精确层+神经层）：name/alias 精确 **100%**，神经层 heldout **99%**（V6.3b 变体增强后）；关系推理归 V6.3 递归层；
2. 观测点：selection 92.2% / invariance 97.2% / permission 100%（零泄漏）；
3. **V6.3 递归层**：AI/电影/国家三域 latest/predecessor/successor 全 100%（V6.3b 数据重排 + V6.3c 多域）；
4. 理解层：33 条盲测 100%，库外版本越界拦截；
5. 跨域：零样本迁移不成立（5.2%）；多域联合训练有效；
6. TTT 查询编码：**负结果**；
7. toB 治理四支柱：权限 0 泄漏 / 三级分级 / 空白识别 100% / 先例闭环（见 TOB_POC_REPORT.md）。

## 3. 复现命令（同 README）

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py --steps 600 --seeds 1 2 3 --d-model 256
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py
D:\conda\envs\cformer-gpu\python.exe eval_employee_rules.py
D:\conda\envs\cformer-gpu\python.exe eval_rule_gap.py
D:\conda\envs\cformer-gpu\python.exe eval_precedent_loop.py
```

## 4. Git 状态与推送

- 本仓库 `E:\deepseek\c-former` 为会话线；V6.5 主线完整备份在 `E:\oprncode\c-former`；
- 日常推送用 `push.bat`；提交信息中文一句话，里程碑才打 tag。
