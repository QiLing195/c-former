# C-Former V6.3 递归层报告（真实数据 · V6.3b 重排 + V6.3c 多域）

## 结论

用确定性递归层（v1，无神经网络）在真实数据集的**前代关系图**上承接 predecessor/latest/多跳推理，**分层设计实证成功**；V6.3b 数据重排 + V6.3c 多域扩展（AI/电影/国家三域）后全部 100%：

| 指标 | V6.3 首版（AI） | **V6.3b/c（AI 域）** | **电影域** | **国家域（政体继承）** | 身份层对照 |
|---|---:|---:|---:|---:|---:|
| latest_accuracy | 86%（43/50） | **100%**（52/52） | **100%**（12/12） | **100%**（8/8） | 33% |
| predecessor_accuracy | 100%（215/215） | **100%**（181/181） | **100%**（16/16） | **100%**（8/8） | 33% |
| multi_hop_accuracy | 98.3%（171/174） | **100%**（156/156） | **100%**（36/36） | **100%**（24/24） | 33% |
| 循环/深度/时间/版本控制探针 | 全部通过 | **全部通过** | **全部通过** | **全部通过** | — |

**三域全部 100%**，对比身份解析层直接答这类关系查询仅 33%——**「关系推理归 V6.3、身份解析归 V6.0」的分层设计成立，且跨域可复用**。

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

## 3. V6.3c 多域扩展（本版新增）

### 3.1 电影域建链

`build_movies_dataset.py`：60 部电影中 **15 个显式系列**（漫威宇宙/星球大战/指环王/哈利波特/侏罗纪/速度与激情/黑客帝国/战狼/唐人街/流浪地球/长津湖/变形金刚/蝙蝠侠/教父/吉卜力），系列内**按上映年份排序建 predecessor 链**（前一部 → 后一部）；每个有前代的电影新增 `predecessor` 字段 + 前代推理查询，每系列新增 latest 查询。无显式系列的电影（series=genre）不进链。

### 3.2 评测脚本泛化

`train_eval_v63.py` 新增 `--data` 参数（默认 AI 域），控制探针全部改为自建图（不依赖特定数据集），任何数据集可跑。

### 3.3 修复的工程 bug：RelationGraph 注册顺序

首版 `RelationGraph.__init__` 第一遍建 predecessor 边时要求前驱**已注册**（`pred in self.nodes`），AI 域因对象按链序生成而未暴露；电影域对象按原始列表顺序写入，前驱可能在列表更晚处（如"复仇者联盟"→"美国队长"）→ 前驱关系被静默丢弃，predecessor 掉到 75%。

**修复**：第一遍只注册节点与系列成员，第二遍（全部节点就绪）统一建 predecessors/successors 边。修复后电影域 predecessor **75%→100%**——同时为 AI 域增加了任意对象顺序的鲁棒性。

### 3.4 国家域：政体继承链

`build_countries_recursion.py`：8 条**政体继承链**（前身政体 → 继承国），独立于主国家数据集（不污染跨域实验）：
- 苏联(1922) → 俄罗斯(1991)、捷克斯洛伐克(1918) → 捷克(1993)、南斯拉夫(1918) → 塞尔维亚(2006)、英属印度(1858) → 印度(1947)、奥斯曼帝国(1299) → 土耳其(1923)、荷属东印度(1800) → 印度尼西亚(1945)、波斯(1501) → 伊朗(1935)、锡兰(1948) → 斯里兰卡(1972)；
- 每条链独立 series（"苏联继承"等），链内年份严格单调（前身 → 继承国）；
- 结果：latest 8/8、predecessor 8/8、multi_hop 24/24 全 100%，控制探针全过。

### 3.5 结果

三域合计：latest 72/72、predecessor 205/205、multi_hop 216/216 **全 100%**；AI 域回归不变；全量测试 51 passed。

## 4. 结果解读

1. **latest 100%**：主链严格时间单调后，沿链走到链尾即系列最新，时间控制不再误报；
2. **multi_hop 100%**：主链头出发走 1/2/3 跳全部命中系列成员；
3. **predecessor 100%**：前代精确查表，确定性层零失误（计数从 215 降到 181，因为变体不再生成 predecessor 关系——语义更准）；
4. **四重控制全通过**：人为构造的循环/深度/时间回退/版本超限全部被正确拒绝。

## 5. 对项目的影响

1. **分层设计实证强化**：身份解析层做 name/alias（75.8%），递归层做关系推理（AI/电影双域 100%）——各层干擅长的事，整体远强于单层硬扛（33%）；
2. **数据标注纪律**：「系列列表 = 一条链」的假设被变体子线打破，时间控制把它暴露出来；V6.3b 确立**主链（代数序）+ 变体（不进链）**双结构，latest/多跳语义与真实"代数演进"一致；
3. **跨域可复用**：同一 RelationGraph/RecursiveResolver 无需改动即在电影域（系列续集链）拿到 100%——确定性推理层与数据 schema 解耦；
4. **确定性控制的价值**：零 GPU、秒级、100% 可重放路径（每跳记录 path），可审计——正是框架「先身份、后观测点、再推理」的推理层该有的样子。

## 6. 已知改进项

- **身份层重训（已做）**：V6.3b 改了证据文本（变体关系字段从"前一代是 X"改为"是 X 系列规格/能力变体"），身份层已用 d=256 最终配置重训（3 种子 600 步）：**heldout identity_top1 75.8%**（旧 76.3%，-0.5pp 在种子方差内）——**数据重排对身份解析无伤**；ambiguous_detected 100%（旧 99.1%）；
- **国家域关系图（已做）**：`build_countries_recursion.py` 建 8 条政体继承链（苏联→俄罗斯、英属印度→印度等），三域递归全 100%；
- **神经递归块（已做，架构对照见 §8）**：共享参数递归 vs 直接堆叠，参数节省 ~72%（闸门 ≥30%），且共享递归准确率不降反升。

## 7. 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe build_movies_dataset.py
D:\conda\envs\cformer-gpu\python.exe build_countries_recursion.py
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py                                          # AI 域
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py --data data/movies_dataset.json         # 电影域
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py --data data/countries_recursion.json   # 国家域
D:\conda\envs\cformer-gpu\python.exe train_eval_v63_neural.py                                 # 神经递归对照
```

结果：`artifacts/v63_results.json`（三域 latest/predecessor/multi_hop 全 1.0）、`artifacts/v63_neural_recursion.json`。设计见 [`V63_RECURSION_DESIGN.md`](V63_RECURSION_DESIGN.md)。

## 8. 神经递归块（v2 实验）：共享参数 vs 直接堆叠

`train_eval_v63_neural.py`：实现设计文档 §5 的「参数闸门」对照。任务 = 链尾识别（给定系列成员特征序列，预测最新成员索引，与确定性 latest 同语义），69 条真实链（AI 52 + 电影 12 + 国家 8 中可重建单链系列）。

| 条件 | 架构 | 参数 | train_acc | test_acc | 参数节省 |
|---|---:|---:|---:|---:|---:|
| 随机特征（无语义） | shared | 4,737 | 1.0 | **0.571** | 72.5% |
| 随机特征 | stacked | 17,217 | 1.0 | 0.571 | — |
| **+年份通道（语义）** | shared | 4,769 | 1.0 | **0.929** | 72.4% |
| +年份通道 | stacked | 17,249 | 1.0 | 0.857 | — |

**三个结论**：

1. **参数闸门通过**：两种条件下共享递归均省 **~72%** 参数（闸门 ≥30%），且 test_acc 不低于堆叠（semantic 条件下 92.9% vs 85.7%，共享反而更高——小样本下共享权重提供正则化）；
2. **语义特征必要性（架构级实证）**：随机特征下两架构都只有 57%（只能猜位置），加年份归一化通道后升到 86–93%——「最新」这类跨候选比较必须携带语义线索（年份/前代），否则神经层学不会；这正是确定性 v1 用**年份单调控制**兜底的架构依据，也解释了 V62 泛化消融（dual-encoder 独立打分做不了 listwise）的根因；
3. **共享递归设计成立**：共享参数块重复 hops 次不损失能力（v1 确定性 100% 之外，神经版也验证了共享权重的可行性），为将来软语义递归（模糊关系、自然语言措辞）提供参数高效的基座。
