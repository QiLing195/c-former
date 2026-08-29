# C-Former 会话线交接文档（V6.3 更新版）

> 本文件是历史交接文档的**当前化**版本（原稿为 V6.0 时期，已过期）。完整版本说明见 [`README.md`](README.md)。

## 1. 当前状态快照（2026-08 会话线）

| 项 | 状态 |
|---|---|
| 版本 | 会话线 V6.3（内部 V6.x ↔ 测试版 0.6.x；历史 tag `v0.6.1c`） |
| 数据 | AI 模型 273 对象 + 国家 68 + 电影 60 + 观测点查询集（`data/`） |
| 测试 | `D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q` |
| 打包/CI | `pyproject.toml` + `.github/workflows/ci.yml` |
| 冻结基线 | V6.0 编码器（2 层 C-Former，共享 Token Transformer） |

## 2. 关键结论（实验证据，全部真实数据 3 种子）

1. 身份解析（AI 域，分域评测）：identity_top1 **76.3%**；关系推理（predecessor/latest）33% → 归 V6.3 递归层；
2. 观测点：selection 92.2% / invariance 97.2% / permission 100%（零泄漏，mask_caught 31–85）；
3. **V6.3 递归层**（确定性关系图）：predecessor 100% / 多跳 98.3% / latest 86%，四重控制（cycle/depth/time/version）全部通过；
4. 跨域：零样本迁移不成立（5.2% ≈ 随机）；多域联合训练有效（电影 74.9% / 国家 67.9% / AI 46.5%）；
5. TTT（测试时训练）查询编码：**负结果**，未超过基线；
6. 诚实边界：小对比模型在训练查询上是记忆而非泛化；数据集为检索/常识近似，正式使用前需人工核对。

## 3. 复现命令（同 README）

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py --steps 600 --seeds 1 2 3
D:\conda\envs\cformer-gpu\python.exe build_observer_queries.py
D:\conda\envs\cformer-gpu\python.exe eval_observer_real.py
D:\conda\envs\cformer-gpu\python.exe train_eval_cross.py --steps 300 --seeds 1 2 3
D:\conda\envs\cformer-gpu\python.exe train_ttt_real.py --steps 600 --seeds 1 2 3
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py
```

## 4. Git 状态与推送

- 本仓库 `E:\deepseek\c-former` 为会话线；V6.5 主线完整备份在 `E:\oprncode\c-former`；
- 推送前执行 `repush.bat`（清理嵌套克隆 → 修正提交信息 → 测试门禁 → 推送）；
- 日常推送用 `push.bat`；提交信息中文一句话，里程碑才打 tag。
