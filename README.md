# C-Former 最小可运行 Demo

这是一个面向验证的 **Constellation Transformer (C-Former)** 原型。它没有试图一次实现完整的“持续生长知识宇宙”，而是先验证最关键的问题：

> 同一段输入能否在不同“观测点”指令下，稳定地产生不同且正确的结果？

Demo 使用一个合成任务：输入 6 个数字 token，观测点指令分别要求模型读取序列的左侧、中间或右侧 token。模型只从 `[CLS]` 隐藏状态分类，因此必须利用观测点来改变注意力行为。

## 首版包含

- 独立观测点编码器：将 `look left / center / right` 编码为全局观测点向量。
- OP-FiLM 多头注意力：用观测点向量调制每层的 Q/K，且采用零初始化，初始行为等同标准注意力。
- 安全的自我中心投影（可选）：只投影非主星 token，保留主星，并用接近 0 的可学习门控控制强度。
- 视角对比解码：比较默认观测点与自定义观测点 logits。
- 合成训练、验证、保存 checkpoint 和交互推理脚本。

暂不包含动态词表、持久注意力偏置和 MoE 星座库。这些模块应在核心观测点条件化被实验确认后再加入。

## 运行

环境要求：Python 3.10+、PyTorch 2.x。依赖见 `pyproject.toml`，可先执行 `pip install -e .[dev]`。

```powershell
python train_demo.py --steps 600
python infer_demo.py --checkpoint artifacts/cformer_demo.pt --tokens 3 8 1 12 5 9
python evaluate_demo.py --checkpoint artifacts/cformer_demo.pt --samples-per-view 10000
pytest -q
```

训练输出会报告三个观测点各自的准确率，并将模型保存到 `artifacts/cformer_demo.pt`。

交互推理会分别展示左、中、右三个视角的预测，以及启用视角对比解码后的结果。

## 已验证结果

在 Python 3.13、PyTorch 2.9 CPU 环境下，默认配置共 103,376 个参数：

```text
4 tests passed
step=132  accuracy=1.000  L/C/R=1.000/1.000/1.000
step=400  accuracy=1.000  L/C/R=1.000/1.000/1.000
```

启用安全自我中心投影后共 103,378 个参数，在第 150 步也达到了三个视角 100% 准确率。该结果仅证明机制和梯度路径可运行，不代表真实语言任务上的效果。

## 结构

```text
观测点文本 -> ObserverEncoder -> o
                                 |
输入 token -> Embedding --------+-> CFormerBlock x N -> CLS -> logits
                                      | OP-FiLM Q/K
                                      | gated ego projection (optional)
```

核心公式：

```text
gamma_q, beta_q, gamma_k = MLP(o)
Q' = Q * (1 + tanh(gamma_q)) + beta_q
K' = K * (1 + tanh(gamma_k))
```

调制层最后一层使用零初始化，所以训练开始时 `Q'=Q, K'=K`，可从标准 Transformer 平滑学习观测点条件化。

安全投影为：

```text
u = h_primary / (||h_primary|| + eps)
h_i' = h_i - gate * <h_i, u>u,  i != primary
h_primary' = h_primary
```

## 下一步实验

1. 与“观测点 prefix token”基线比较参数量、收敛速度和准确率。
2. 分别关闭 Q 调制、K 调制、对比解码和自我中心投影做消融。
3. 将合成任务换成同一事实的多角色、多立场或多时间点描述数据。
4. 核心机制有效后，再用预留扩展槽位验证弹性嵌入。

完整实测数据和局限分析见 [`TEST_REPORT.md`](TEST_REPORT.md)。

## V2：共享世界多观测点查询

V2 将观测点从“位置选择器”改成共享知识空间的坐标，并把问题与观测点分离：

```powershell
python train_and_evaluate_v2.py --steps 500
```

设计说明见 [`V2_DESIGN.md`](V2_DESIGN.md)，四个固定小世界的对照测试见 [`V2_TEST_REPORT.md`](V2_TEST_REPORT.md)。

## V3：分层记忆规模测试

V3 使用低维全局索引和两阶段精确检索，将世界扩大到 128、512、2,048 条事实：

```powershell
python train_scale_demo.py --steps 3000
```

设计见 [`V3_DESIGN.md`](V3_DESIGN.md)，每个规模 5 个世界的完整结果见 [`V3_TEST_REPORT.md`](V3_TEST_REPORT.md)。

## V4：可靠性与边界

V4 压缩参数并增加五类输出、检索前硬边界和冲突/拒答测试：

```powershell
python train_reliability_demo.py --steps 3000
```

设计与安全责任划分见 [`V4_DESIGN.md`](V4_DESIGN.md)，三个规模、每级五个世界的结果见 [`V4_TEST_REPORT.md`](V4_TEST_REPORT.md)。

## V5：冲突、时间、版本与公平 RAG 对照

V5 新增可信冲突索引，并让 C-Former、Evidence-RAG 和 Dense 使用同一个边界控制器：

```powershell
python evaluate_v5.py --rag artifacts/v5_rag_equal_budget/evidence_rag.pt
```

设计见 [`V5_DESIGN.md`](V5_DESIGN.md)，1,800题测试与三模型对照见 [`V5_TEST_REPORT.md`](V5_TEST_REPORT.md)。

## V5.5：Object、Transformation 与受控模拟

V5.5 在版本化时空框架上增加最小认知层：显式对象身份与生命周期、对象状态、带三值约束的 Transformation、观察者投影和假设沙箱。模拟结论必须由外部观测验证，且世界版本未变化时才能写入正式记忆：

```powershell
python evaluate_v55.py
python -m pytest -q tests/test_cformer_v55.py
```

设计和诚实边界见 [`V55_COGNITIVE_DESIGN.md`](V55_COGNITIVE_DESIGN.md)，实测与消融见 [`V55_COGNITIVE_REPORT.md`](V55_COGNITIVE_REPORT.md)。这一版不宣称已经解决自动语义吸引子、因果发现或通用世界模拟。

## V5.6：神经—结构化联合检索

V5.6 将观测点门控 C-Former 接到 V5.5 控制器之后：结构化层先完成对象身份、权限、时间、版本和冲突判断，神经层只在固定 Top-64 合法 ObjectState/Transformation 中重排。

```powershell
python train_evaluate_v56.py --steps 600 --seeds 301 302 303
python aggregate_v56.py
```

架构见 [`V56_HYBRID_DESIGN.md`](V56_HYBRID_DESIGN.md)，5 种子、每级 5 个世界的结果见 [`V56_HYBRID_REPORT.md`](V56_HYBRID_REPORT.md)。

## V5.7：噪声文本与多跳 Transformation

V5.7 增加 Unicode/标点/空格归一化、字符 n-gram 文本特征、规范对象键、词法 Top-64 硬负例训练和受控多跳 Transformation。文本负责表述匹配，对象键负责身份稳定，确定性控制器继续负责权限、时间、版本、循环和深度终止。

```powershell
python train_evaluate_v57.py --steps 300 --seeds 401 402 403
python -m pytest -q tests/test_cformer_v57.py
```

设计见 [`V57_TEXT_DESIGN.md`](V57_TEXT_DESIGN.md)，3 种子、每级 5 世界的结果和失败修正过程见 [`V57_TEXT_REPORT.md`](V57_TEXT_REPORT.md)。

## V5.8：磁盘优先别名索引与候选沙箱

V5.8 将已验证别名、开放文本候选和观察视角分层存储：SQLite B-tree 做精确别名，FTS5 内容空倒排索引生成候选，int8 对象向量通过 mmap 按行读取，四个视角不再复制对象向量。未知别名只进入候选沙箱，审核前不能写入正式身份。

```powershell
python evaluate_v58_storage.py --scales 32768 65536 131072
python rebenchmark_v58_queries.py
python -m pytest -q tests/test_cformer_v58.py
```

设计见 [`V58_STORAGE_DESIGN.md`](V58_STORAGE_DESIGN.md)，最高 131K 对象/524K 视角候选的磁盘、加载和查询结果见 [`V58_STORAGE_REPORT.md`](V58_STORAGE_REPORT.md)。

## V5.9：无编号开放别名、多证据解析与 CUDA 梯度测试

V5.9 去掉查询中的对象编号和身份键，以名称、属性、关系、变化四类证据解析共享对象；模型只能提出 `supported` 候选，显式审核后才能进入正式别名，且支持版本回滚。CUDA 测试覆盖 2K 到 512K、每级 4 个世界，并与相同步数的 Transformer dual encoder 对照。

```powershell
D:\conda\envs\cformer-gpu\python.exe evaluate_v59.py --stage all --steps 300 --batch-size 256 --queries 80 --worlds 4
D:\conda\envs\cformer-gpu\python.exe evaluate_v59_transformer_scale.py
D:\conda\envs\cformer-gpu\python.exe -m pytest -q
```

架构与身份边界见 [`V59_OPEN_ALIAS_DESIGN.md`](V59_OPEN_ALIAS_DESIGN.md)，完整结果与局限见 [`V59_OPEN_ALIAS_REPORT.md`](V59_OPEN_ALIAS_REPORT.md)。

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
```

设计与约束见 [`V60B_BLINDSET_DESIGN.md`](V60B_BLINDSET_DESIGN.md)，3 种子结果与 13 个失败样本见 [`V60B_BLINDSET_REPORT.md`](V60B_BLINDSET_REPORT.md)。

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

设计见 [`V61_ANN_DESIGN.md`](V61_ANN_DESIGN.md)，结果与 V6.1b（faiss HNSW/IVF-PQ + V5.8 磁盘层、512K+）升级路径见 [`V61_ANN_REPORT.md`](V61_ANN_REPORT.md)。

## 工程规范（V6.0 起）

- 打包与依赖：`pyproject.toml`（`pip install -e .[dev]`）；
- CI：`.github/workflows/ci.yml` 在 push/PR 时自动运行 `python -m pytest`（CPU 环境，Python 3.10 / 3.12）；
- 版本控制：每个冻结版本打 annotated tag（当前基线 `v6.0`）；
- 大文件（模型检查点、原始结果 JSON）保留在 `artifacts/`，不进入版本库。

### 如何新增一个版本（V6.1 起）

1. 写 `VXX_DESIGN.md`：目标、架构、测试矩阵、质量闸门与回退条件；
2. 新建 `cformer_vXX/` 包，只实现设计的最小部分，复用前版 verifier/ledger 与数据工具；
3. 写 `tests/test_cformer_vXX.py` 并通过小规模闸门；
4. 写 `train_evaluate_vXX.py` / `evaluate_vXX.py` 与 `aggregate_vXX.py`；
5. 固定 3 个训练种子、每规模 4–5 个世界，原始 JSON 存入 `artifacts/`；
6. 写 `VXX_REPORT.md`（含失败案例与限制），对照质量闸门逐条说明；
7. 提交并打标签：`git add -A && git commit -m "VXX: ..." && git tag -a vXX -m "..."`。
