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

当前诚实基线（3 种子均值，2026-08 检索核对后 212 对象版本，仅 2 条存疑标记）：
known 训练集 Top-1 100%，**留出集 Top-1 6.7%，盲测已知 Top-1 11.1%**
（盲测 mean coverage 0.72）；unknown 拦截率 83.3%、盲测误支持率 16.7%。结论：小对比模型在训练
查询上是记忆而非泛化，真实泛化是下一步要解决的问题；边界拒答行为相对健康。

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_real.py
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py
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
