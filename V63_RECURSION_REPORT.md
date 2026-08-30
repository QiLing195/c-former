# C-Former V6.3 递归层报告（真实数据 · V6.3b 数据重排版）

## 结论

用确定性递归层（v1，无神经网络）在真实 AI 模型数据集的**前代关系图**上承接 predecessor/latest/多跳推理，**分层设计实证成功**；V6.3b 数据重排后三类指标全部 100%：

| 指标 | V6.3 首版 | **V6.3b（重排后）** | 身份层对照（V62） |
|---|---:|---:|---:|
| latest_accuracy | 86%（43/50） | **100%**（52/52） | 33% |
| predecessor_accuracy | 100%（215/215） | **100%**（181/181） | 33% |
| multi_hop_accuracy | 98.3%（171/174） | **100%**（180/180） | 33% |
| 循环/深度/时间/版本控制探针 | 全部通过 | **全部通过** | — |

**平均 100%**，对比身份解析层直接答这类关系查询仅 33%——**「关系推理归 V6.3、身份解析归 V6.0」的分层设计成立**。

## 1. 实现

- `cformer_v63/recursive.py`：`RelationGraph`（由对象结构化 `predecessor` 字段构建前代→后继图）+ `RecursiveResolver`（latest 沿链走到链尾、predecessor 前代查表、chain 多跳），四重确定性控制：
  - **循环**：visited 集合检测；
  - **深度**：hops > max_depth(4) 拒绝；
  - **时间单调**：后继年份 < 前驱年份 → 拒绝（time_violation）；
  - **版本固定**：world_version 过滤 > 截止年份的后继；
- 数据：`build_ai_models_dataset.py` 给每个对象加结构化 `predecessor`（存对象 ID），主链成员严格时间单调，规格/能力变体 `predecessor=None`（不进链）。

## 2. V6.3b 数据重排（本版修复）

V6.3 首版 latest 86% 的 7 个失败系列，根因两类，本版全部修复：

### 2.1 链长超限（depth_exceeded）→ 拆变体缩主链

`RecursiveResolver.max_steps=16`，Qwen（27 节点）、Llama（20 节点）把尺寸规格全部串成一条链，链长超过上限。

**修复**：尺寸规格（Llama 2 7B/13B/70B、Qwen3 0.6B~235B 等）、多模态（Qwen-VL/Qwen3-VL）、专项能力（Qwen2.5-Coder/Math）拆为**变体**（`predecessor=None`，不进链，不参与 latest/多跳），主链只留代数序节点（Qwen 8、Llama 7）。

### 2.2 年份倒退（time_violation）→ 拆独立系列 / 挪回时间区

变体子线被追加在主链末尾，造成后继年份 < 前驱年份的伪前代关系：

| 系列 | 问题 | 修复 |
|---|---|---|
| Qwen | Qwen3.7-Max(2026) → Qwen-VL(2024) 等 | 变体拆出主链（见 2.1） |
| 豆包 | 豆包 1.5 Pro(2026) → Seed 1.5(2025) | Seed 拆为独立系列（多模态线） |
| OLMo | OLMo2 7B(2025) → Molmo 7B(2024) | Molmo 拆为独立系列 |
| DeepSeek | DeepSeek-Coder-V2(2024) 排在 2025 后 | 挪回 2024 区 |
| Mistral | Mixtral 8x22B(2024) 排在 Small 3.1(2026) 后 | 挪回 2024 区 |
| GLM | GLM-5(2025) 排在 4.7(2026) 后；Z1(2025) 排到 5.3(2026) 后 | Z1 拆变体，主链严格时间序 |

### 2.3 配套修正

- `latest_of_series` 多 head 平局时取 **path 更长者**（主链走到链尾的 path 必然更长），避免变体 head 同年被误选；
- `train_eval_v63.py` 多跳评测从**年份最小的 head**（主链头）出发，避开变体 head。

## 3. 结果解读

1. **latest 100%**：主链严格时间单调后，沿链走到链尾即系列最新，时间控制不再误报；
2. **multi_hop 100%**：主链头出发走 1/2/3 跳全部命中系列成员；
3. **predecessor 100%**：前代精确查表，确定性层零失误（计数从 215 降到 181，因为变体不再生成 predecessor 关系——语义更准）；
4. **四重控制全通过**：人为构造的循环/深度/时间回退/版本超限全部被正确拒绝。

## 4. 对项目的影响

1. **分层设计实证强化**：身份解析层做 name/alias（76.3%），递归层做关系推理（100%）——各层干擅长的事，整体远强于单层硬扛（33%）；
2. **数据标注纪律**：「系列列表 = 一条链」的假设被变体子线打破，时间控制把它暴露出来；V6.3b 确立**主链（代数序）+ 变体（不进链）**双结构，latest/多跳语义与真实"代数演进"一致；
3. **确定性控制的价值**：零 GPU、秒级、100% 可重放路径（每跳记录 path），可审计——正是框架「先身份、后观测点、再推理」的推理层该有的样子。

## 5. 已知改进项

- **身份层重训（已做）**：V6.3b 改了证据文本（变体关系字段从"前一代是 X"改为"是 X 系列规格/能力变体"），身份层已用 d=256 最终配置重训（3 种子 600 步）：**heldout identity_top1 75.8%**（旧 76.3%，-0.5pp 在种子方差内）——**数据重排对身份解析无伤**；ambiguous_detected 100%（旧 99.1%）；
- **多域扩展**：国家（隶属/邻国链）、电影（系列链）的关系图同样适用，v1 先用 AI 域验证；
- **神经递归块**（设计文档中的共享参数模块）：v1 为纯确定性，神经化是后续方向（需对照参数节省 ≥30% 闸门）。

## 6. 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v63.py -q
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py
```

结果：`artifacts/v63_results.json`（latest 1.0 / predecessor 1.0 / multi_hop 1.0）。设计见 [`V63_RECURSION_DESIGN.md`](V63_RECURSION_DESIGN.md)。
