# C-Former（测试版 0.6.1c）

面向验证的 **Constellation Transformer (C-Former)** 研究原型，核心假设：

> 所有信息存放在共享对象世界中，同一对象可被不同问题、不同观测点复用；观测点是坐标/索引，而不是知识的副本。身份解析先于观测点注入；神经模型只能提出候选，正式身份写入由确定性治理层控制。

> ⚠️ 当前为**测试版 0.6.1c**（对应内部里程碑 V6.1c），尚在调试，不是正式版本。

## 当前版本范围

当前代码只保留 V6.0 之后的内容（V1–V5.8 的旧实验代码已移除）：

- `cformer_v59` 无编号开放别名解析 + 治理层（verifier / 版本化 ledger）
- `cformer_v60` 共享 Token Transformer（中文，2 层冻结基线）
- `cformer_v60b` 模板外中文盲测集 + 校准集 + 分类型阈值
- `cformer_v60c` 区域轴增强实验（负结果，未晋升）
- `cformer_v61` torch IVF ANN 分层检索（含 512K 规模）

## 环境

Python 3.10+、PyTorch 2.x。依赖见 `pyproject.toml`，先执行：

```powershell
pip install -e .[dev]
```

GPU 训练/评测使用 `D:\conda\envs\cformer-gpu\python.exe`（RTX 3050 4GB）。CI 见 `.github/workflows/ci.yml`。

## 测试

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q
```

V6.0 到 V6.5 的逐版目标、层数、数据、质量闸门、回退条件和最终验收标准见 [`V60_TO_V65_ROADMAP.md`](V60_TO_V65_ROADMAP.md)。

## V6.0：共享 Token Transformer 与四证据融合

V6.0 使用共享 Token Transformer 处理查询及名称、属性、关系、变化四类对象证据，并以严格对称否定集验证词序能力。2/4/6 层消融后，默认配置冻结为 2 层；正式测试覆盖 2K/8K/64K、每级 4 个世界和 3 个训练种子。

```powershell
D:\conda\envs\cformer-gpu\python.exe train_evaluate_v60.py --kinds cformer mlp flat --layers 2 4 --seeds 601 602 603 --scales 2048 8192 65536 --worlds 4 --steps 300 --batch-size 128 --queries 96
D:\conda\envs\cformer-gpu\python.exe aggregate_v60.py
D:\conda\envs\cformer-gpu\python.exe -m pytest -q
```

架构和训练数据见 [`V60_TOKEN_DESIGN.md`](V60_TOKEN_DESIGN.md)，55,296 次查询的结果、失败修正和限制见 [`V60_TOKEN_REPORT.md`](V60_TOKEN_REPORT.md)。

## V6.0b：模板外中文盲测集

V6.0b 建立第一个与训练模板隔离的盲测集：36 条人工撰写查询（新句式、口语、错字、同名不同对象、自然未知、冲突证据），冻结 2 层编码器零调参评测。量化了模板饱和：已知 Top-1 从模板集的 100% 降到 93.75%，主要失败模式是**区域轴近义别名**（冰原/腹地/高原/中央/乡村、河流上段/下段），同时给出第一组风险—覆盖率校准曲线。

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v60b.py
D:\conda\envs\cformer-gpu\python.exe evaluate_v60b.py
D:\conda\envs\cformer-gpu\python.exe calibrate_v60.py
```

设计与约束见 [`V60B_BLINDSET_DESIGN.md`](V60B_BLINDSET_DESIGN.md)，结果见 [`V60B_BLINDSET_REPORT.md`](V60B_BLINDSET_REPORT.md)，阈值校准见 [`V60B_CALIBRATION_REPORT.md`](V60B_CALIBRATION_REPORT.md)。

## V6.0c：区域轴修复尝试（负结果，未晋升）

V6.0c 尝试用数据增强（区域近义硬负例 + 内容字错字）修复 V6.0b 的区域轴混淆，两轮迭代结论为**负结果**：正式 64K Top-1 从 100% 回退到 99.57%、盲测已知无提升。**保留 V6.0 编码器为冻结基线**，转向阈值校准路径。代码与检查点保留供审计。

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v60c.py
D:\conda\envs\cformer-gpu\python.exe train_evaluate_v60c.py --seeds 701 702 703 --steps 500 --batch-size 128
```

失败分析与下一步见 [`V60C_REGION_FIX_REPORT.md`](V60C_REGION_FIX_REPORT.md)。

## V6.1：ANN 分层检索

V6.1 用纯 torch 的 IVF 倒排索引把在线检索从 `O(N)` 全量扫描降为「粗召回 Top-256 + 精确精排」：3 规模 × 4 世界全矩阵 Recall@256 = 100%、ANN 相对精确扫描 Top-1 下降为 0、64K FP16 向量库 8.0 MiB（INT8 4.1 MiB）、向量化后查询 p50≈2.2 ms / p95≈3.2 ms。附墓碑删除、快照回滚与单副本存储。

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v61.py
D:\conda\envs\cformer-gpu\python.exe evaluate_v61.py
D:\conda\envs\cformer-gpu\python.exe bench_v61_latency.py
```

设计见 [`V61_ANN_DESIGN.md`](V61_ANN_DESIGN.md)，结果与升级路径见 [`V61_ANN_REPORT.md`](V61_ANN_REPORT.md)。

### V6.1b：IVF 512K 规模

V6.1b 用 V5.9 编码器的真实向量把 IVF 扩展到 128K/256K/512K：Recall@256 = 100%、Top-1 下降 = 0、512K FP16 64.0 MiB（命中闸门）、向量化 k-means 建索引 0.33 s、查询 p50≈1.5 ms。

```powershell
D:\conda\envs\cformer-gpu\python.exe bench_v61_scale.py --scales 131072 262144 524288
```

规模结果与诚实边界（仍为组合语义空间）见 [`V61B_SCALE_REPORT.md`](V61B_SCALE_REPORT.md)。V6.1c 集成设计见 [`V61C_INTEGRATION_DESIGN.md`](V61C_INTEGRATION_DESIGN.md)。

## 真实语料冒烟线（进行中，未晋升）

`cformer_real/` 是第一条真实语料试验线：以主流 AI 大模型为对象域，`build_ai_models_dataset.py`
展开 200+ 对象四证据数据集（`data/ai_models_dataset.json`，不确定条目标 `needs_review`），
`data/ai_models_blindset.json` 为与生成模板隔离的人工盲测集（口语、别称、错字、描述性指代、
同名歧义、库外未知）。`train_eval_real.py` 按 `md5(text) mod 5` 把 known 查询切成训练/留出集，
留出集不参与训练——修复了早期"训练=评测"的泄漏。

当前诚实基线（3 种子均值，212 对象版本）：known 训练集 Top-1 ≈100%，**留出集 Top-1 6.7%，
盲测已知 Top-1 5.6–13.9%**。结论：小对比模型在训练查询上是记忆而非泛化。

### 泛化消融（2026-08，负结果）

为修复记忆问题尝试了同义改写增强（`cformer_real/augment.py`，词表封闭性有测试保证）与
同系列硬负例（`contrastive_loss` 负例列），2×2 消融 + 600 步组合：

| 配置 | 留出集 Top-1 | 歧义检出 | unknown 拦截 |
|---|---:|---:|---:|
| 基线（原目标） | 6.7% | 81.6% | 83.3% |
| 仅改写 | 6.7% | 82.5% | 100% |
| 仅硬负例 | 6.7% | **30.8%** | 88.9% |
| 改写+硬负例（600步） | **0%** | 55.3% | 55.6% |

**结论（架构级）**：留出集查询是"X 的 Y 系列最新模型是什么？"——解析"最新"需要**跨候选比较**
同系列各对象的发布年份/前代关系；而 dual-encoder 对每个候选独立打分，原理上无法做这种
listwise 推理，只能背下训练时见过的系列。因此任何训练信号调整都无法修复，反而硬负例会
锐化边界、破坏 verifier 的歧义 margin。该负结果从实证上支持路线图中 V6.2"C-Former 世界推理块"
（先聚合候选再做观测点调制/关系传播）的必要性：超级指代类查询必须在推理层解决，
而不是在检索打分层解决。默认参数回退为 steps=400、hard-negatives=0、改写开启；
消融开关保留供复现。

### V6.2 世界推理块最小原型（已实现，缺口关闭）

`cformer_v62/reasoner.py`：确定性跨候选选择器——方向词（最新/最早）→ 词法锚定系列
（查询显式含公司/系列名；神经锚点仅回退）→ 年份极值 + series_index 平局裁决 →
输出带完整轨迹的选择；任何一步无法裁决即回退神经路径。挂接于 pipeline 重排后、
verifier 前，无训练参数。A/B 结果（三种子一致）：**留出集 0%→100%、主数据 known
86.0%→100%**，盲测歧义/unknown 安全指标零损失。设计与失败分析见
[`V62_REASONER_REPORT.md`](V62_REASONER_REPORT.md)。

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v62.py
D:\conda\envs\cformer-gpu\python.exe evaluate_v62.py
```

### V6.2 观测点端到端（闸门全过）

`cformer_v62/observer.py`：确定性可见性掩码（公司/区域两轴），身份解析完成后才注入；
exact/reasoned/ann 三条支持路径统一过闸，拒绝返回 `ACCESS_DENIED` 且不暴露对象。
路线图 §8 五项闸门（3 种子一致）：身份一致性 **100%**、权限泄漏 **0**、跨视角召回 **100%**、
可追溯 **100%**、向量单副本恒等。详见 [`V62_OBSERVER_REPORT.md`](V62_OBSERVER_REPORT.md)。

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v62_observer.py
D:\conda\envs\cformer-gpu\python.exe evaluate_observers.py
```

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_real.py
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py
```

### V6.1c：统一存储与检索链路集成（已完成）

`cformer_v61c/` 把「精确别名 → ANN 粗召回 → 全精度重排 → 分类型 margin 校验 → CandidateLedger」
串成端到端链路，在 212 对象真实库上验证正确性：重排 vs 穷举一致性 100%、ledger 自动 verified=0、
p95 延迟 4.3ms。随后 `calibrate_v61c.py` 在真实库上重校准阈值（发现并确定性拦截**结构性歧义**），
known Top-1 66.7%→**76.3%**、覆盖率 69.3%→**85.1%**，安全指标零损失。
详见 [`V61C_INTEGRATION_REPORT.md`](V61C_INTEGRATION_REPORT.md)。

```powershell
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v61c.py
D:\conda\envs\cformer-gpu\python.exe calibrate_v61c.py
D:\conda\envs\cformer-gpu\python.exe evaluate_v61c.py --minimum-score 0.40 --minimum-coverage 0.60 --known-margin 0.01
```

## 工程规范

- 打包与依赖：`pyproject.toml`（`pip install -e .[dev]`）；
- CI：`.github/workflows/ci.yml` 在 push/PR 时自动运行 `python -m pytest`（CPU 环境，Python 3.10 / 3.12）；
- 版本控制：当前测试版 **0.6.1c**（tag `v0.6.1c`），正式版前不再升主版本；
- 大文件（模型检查点、原始结果 JSON）保留在 `artifacts/`，不进入版本库。

### 如何新增一个版本（V6.2 起）

1. 写 `VXX_DESIGN.md`：目标、架构、测试矩阵、质量闸门与回退条件；
2. 新建 `cformer_vXX/` 包，只实现设计的最小部分，复用前版 verifier/ledger 与数据工具；
3. 写 `tests/test_cformer_vXX.py` 并通过小规模闸门；
4. 写 `train_evaluate_vXX.py` / `evaluate_vXX.py` 与 `aggregate_vXX.py`；
5. 固定 3 个训练种子、每规模 4–5 个世界，原始 JSON 存入 `artifacts/`；
6. 写 `VXX_REPORT.md`（含失败案例与限制），对照质量闸门逐条说明；
7. 提交并打标签：`git add -A && git commit -m "VXX: ..." && git tag -a vXX -m "..."`。
