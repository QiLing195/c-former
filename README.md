# C-Former（会话线 V6.3 · 真实数据验证）

面向验证的 **Constellation Transformer (C-Former)** 研究原型，核心假设：

> 所有信息存放在共享对象世界中，同一对象可被不同问题、不同观测点复用；观测点是坐标/索引，而不是知识的副本。身份解析先于观测点注入；神经模型只能提出候选，正式身份写入由确定性治理层控制。

## 当前版本（2026-08 会话线）

本仓库保存 **V6.3 会话线**：在真实数据上完成「共享对象库 → 身份解析 → 观测点 → 递归推理」的实证，并记录跨域与 TTT 实验（含负结果）。版本号映射：内部 V6.x ↔ 测试版 0.6.x（历史 tag：`v0.6.1c`）。

### 成果一览（真实数据，3 种子）

| 能力 | 指标 | 报告 |
|---|---|---|
| 身份解析（AI 域，273 对象，d=256 重训） | identity_top1 **75.8%** | [`V62_OBSERVER_REPORT.md`](V62_OBSERVER_REPORT.md) |
| 观测点：不同观测点不同答案 | selection **92.2%** | 同上 |
| 观测点：身份不随观测点漂移 | invariance **97.2%** | 同上 |
| 观测点：可见性零泄漏 | permission **100%**（mask_caught 31–85） | 同上 |
| **V6.3 递归层**（确定性关系图，V6.3b 重排 + V6.3c 多域） | AI：predecessor/多跳/latest 全 **100%**；电影域（15 系列）也全 **100%** | [`V63_RECURSION_REPORT.md`](V63_RECURSION_REPORT.md) |
| 跨域：零样本迁移 | **不成立**（5.2% ≈ 随机） | [`V62_CROSS_DOMAIN_REPORT.md`](V62_CROSS_DOMAIN_REPORT.md) |
| 跨域：多域联合训练 | 电影 74.9% / 国家 67.9% / AI 46.5% | 同上 |
| TTT 查询编码 | **负结果**（未超过基线） | [`TTT_EXPERIMENT_REPORT.md`](TTT_EXPERIMENT_REPORT.md) |

### 诚实边界

- **零样本跨域迁移不成立**：编码器学到的是域内「词汇→身份」映射，需要逐域或多域联合训练；不是零样本开放世界解析器；
- 关系推理（predecessor/latest）由确定性递归层承接（V6.3），身份层不做跨候选比较；
- 数据集为检索与常识近似（AI 273 / 国家 68 / 电影 60 对象），正式使用前需人工核对（候选→审核→verified 流程）；
- 小对比模型在训练查询上是记忆而非泛化（留出集远低于训练集），是 V6.2 推理块与 V6.3 递归层存在的实证理由。

## 目录结构

```text
cformer_v59/  治理层：EvidenceVerifier（分类型 margin）+ CandidateLedger
cformer_v60/  共享 Token Transformer（中文，2 层冻结基线）+ 对比/拒答/歧义损失
cformer_v61/  torch IVF ANN 分层检索（历史基线，含 512K 规模验证）
cformer_v63/  V6.3 递归层：RelationGraph + RecursiveResolver（确定性多跳）
cformer_real/ 真实数据管线：MixedTokenizer / AIModelWorld / TTTResolver
data/         AI 模型(273) · 国家(68) · 电影(60) · 观测点查询 数据集
tests/        单元测试（pytest）
```

## 环境与测试

```powershell
pip install -e .[dev]                      # Python 3.10+ / PyTorch 2.x
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q     # 全量测试
```

GPU 训练/评测使用 `D:\conda\envs\cformer-gpu\python.exe`（RTX 3050 4GB）。CI 见 `.github/workflows/ci.yml`（push/PR 自动跑 pytest，CPU 环境）。

## 复现命令

```powershell
# 身份解析 + 观测点（先训练检查点，再生成观测点查询并评测）
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py --steps 600 --seeds 1 2 3
D:\conda\envs\cformer-gpu\python.exe build_observer_queries.py
D:\conda\envs\cformer-gpu\python.exe eval_observer_real.py

# 跨域实验（零样本 vs 联合训练，轻量快速版）
D:\conda\envs\cformer-gpu\python.exe train_eval_cross.py --steps 300 --seeds 1 2 3

# TTT 对照（负结果复现）
D:\conda\envs\cformer-gpu\python.exe train_ttt_real.py --steps 600 --seeds 1 2 3

# V6.3 递归层（确定性，秒级）
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py

# margin 分布诊断
D:\conda\envs\cformer-gpu\python.exe diag_margins_real.py --checkpoint artifacts/real_checkpoints/real_seed1.pt
```

## 历史版本基线（保留供审计，非当前线）

- **V6.0** 共享 Token Transformer：2/4/6 层消融后冻结 2 层；64K Top-1 100%。[设计](V60_TOKEN_DESIGN.md) · [报告](V60_TOKEN_REPORT.md)
- **V6.0b** 模板外中文盲测集：已知 Top-1 93.75%，发现区域轴近义别名弱点。[盲测](V60B_BLINDSET_REPORT.md) · [校准](V60B_CALIBRATION_REPORT.md)
- **V6.0c** 区域轴增强修复：**负结果**，未晋升。[报告](V60C_REGION_FIX_REPORT.md)
- **V6.1 / V6.1b** torch IVF ANN：Recall@256=100%、512K FP16 64 MiB。[ANN 报告](V61_ANN_REPORT.md) · [规模报告](V61B_SCALE_REPORT.md) · [集成设计](V61C_INTEGRATION_DESIGN.md)
- 逐版目标与质量闸门： [`V60_TO_V65_ROADMAP.md`](V60_TO_V65_ROADMAP.md)

## 工程规范

- 打包：`pyproject.toml`（`pip install -e .[dev]`）；依赖：`requirements.txt`；
- 大文件（模型检查点、原始结果 JSON）在 `artifacts/`，不入库（`.gitignore`）；
- 日常推送：`push.bat`（改一行提交信息后双击）；提交信息中文一句话，里程碑才打 tag。
