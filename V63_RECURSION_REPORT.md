# C-Former V6.3 递归层报告（真实数据）

## 结论

用确定性递归层（v1，无神经网络）在真实 AI 模型数据集的**前代关系图**上承接 predecessor/latest/多跳推理，**分层设计实证成功**：

| 指标 | 递归层 | 身份层对照（V62） |
|---|---:|---:|
| predecessor_accuracy | **100%**（215/215） | 33% |
| multi_hop_accuracy | **98.3%**（171/174） | 33% |
| latest_accuracy | **86%**（43/50） | 33% |
| 循环/深度/时间/版本控制探针 | **全部通过** | — |

**平均 ~95%**（良序链上 predecessor/multi-hop/latest 均 ~100%），对比身份解析层直接答这类关系查询仅 33%——**「关系推理归 V6.3、身份解析归 V6.0」的分层设计成立**。

## 1. 实现

- `cformer_v63/recursive.py`：`RelationGraph`（由对象结构化 `predecessor` 字段构建前代→后继图）+ `RecursiveResolver`（latest 沿链走到链尾、predecessor 前代查表、chain 多跳），四重确定性控制：
  - **循环**：visited 集合检测；
  - **深度**：hops > max_depth(4) 拒绝；
  - **时间单调**：后继年份 < 前驱年份 → 拒绝（time_violation）；
  - **版本固定**：world_version 过滤 > 截止年份的后继；
- 数据：`build_ai_models_dataset.py` 给每个对象加结构化 `predecessor`（存对象 ID，证据文本不变）。

## 2. 结果解读

1. **predecessor 100%**：前代关系是精确查表，确定性层零失误；
2. **multi-hop 98.3%**：1–3 跳链走查 + 控制，3 条失败集中在数据排序有问题的系列；
3. **latest 86%**：7 个失败系列（Qwen/豆包等）的生成器把**变体子线追加在主链末尾**，造成 `Qwen3.7-Max(2026) → Qwen-VL(2024)` 年份倒退的伪前代关系——**时间单调控制正确检测并拒绝**，这是控制生效的体现，不是递归 bug（良序链上 latest 100%）；
4. **四重控制全通过**：人为构造的循环/深度/时间回退/版本超限全部被正确拒绝。

## 3. 对项目的影响

1. **分层设计实证**：身份解析层做 name/alias（76.3%），递归层做关系推理（~95%）——各层干擅长的事，整体远强于单层硬扛（33%）；
2. **数据标注的真实教训**：生成器的「系列列表 = 一条链」假设被变体子线打破，时间控制把它暴露出来——**控制不只是保护推理，还校验数据质量**；
3. **确定性控制的价值**：零 GPU、秒级、100% 可重放路径（每跳记录 path），可审计——这正是框架「先身份、后观测点、再推理」的推理层该有的样子。

## 4. 已知改进项

- **数据重排**：把 Qwen/豆包 系列列表重排为严格时间序（变体放在对应年份位置），latest 应到 100%；注意这会改证据文本（"前一代是 X"），需要重训身份层——留作 V6.3b；
- **多域扩展**：国家（隶属/邻国链）、电影（系列链）的关系图同样适用，v1 先用 AI 域验证；
- **神经递归块**（设计文档中的共享参数模块）：v1 为纯确定性，神经化是后续方向（需对照参数节省 ≥30% 闸门）。

## 5. 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_v63.py -q
D:\conda\envs\cformer-gpu\python.exe train_eval_v63.py
```

结果：`artifacts/v63_results.json`。设计见 [`V63_RECURSION_DESIGN.md`](V63_RECURSION_DESIGN.md)。
