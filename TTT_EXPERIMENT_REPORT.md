# TTT-Linear 查询编码对照实验报告（负结果）

## 结论

把 TTT-Linear（Test-Time Training, Sun et al. 2024 思想）作为查询编码器，在真实 AI 模型身份解析任务上与基线 Transformer 查询编码对照。**两个变体均未超过基线，整体为负结果**：

| 配置 | heldout 已知 Top-1 | name | predecessor | latest |
|---|---:|---:|---:|---:|
| 基线（Transformer） | **48.7%** | ~46% | ~62% | ~21% |
| TTT 全梯度（`detach_inner=False`） | 35.6% | ~59.5% ↑ | ~3% ↓↓ | ~46.5% ↑ |
| TTT 稳定配方（`detach_inner=True`） | 25.1% | ~36.6% | ~6% ↓↓ | ~39.5% ↑ |

基线 3 种子聚合自 `artifacts/ai_models_results.json`；TTT 两轮结果见 `artifacts/ttt_results.json`（覆盖写）。

## 实验设计

- 数据/边界损失/train-heldout 分割/评测：与基线 `train_eval_real.py` **完全一致**，只把查询编码器换成 `TTTResolver`（`cformer_real/ttt.py`）；
- TTT 只作用于**查询路径**（测试时自适应表示），候选路径保持共享 Transformer（对齐「身份判定稳定、查询表示可自适应」原则）；
- 两种变体：全梯度（内层 48 步展开参与外层回传）与 detach-inner（论文推荐的稳定配方：内层梯度只更新 W，不回传 k/v 投影）；
- 内层学习率 0.05，查询 batch 64。

## 结果解读

1. **TTT 的自适应表示确实对症「语义匹配」**：latest 类查询在两个变体中都比基线高近一倍（21% → 39–46%），全梯度变体下 name 也提升（46% → 59.5%）——说明「测试时自适应」方向对「没见过的句式」有真效果；
2. **但 predecessor（精确证据查表）在两个变体中都崩塌（62% → 3–6%）**：TTT 的重构式自适应扭曲了精确匹配的表征——这是模式性的，不是单纯实现不稳定；
3. **边界（歧义/未知）也变弱**（84%→44%/36%，67%→33%/40%）；
4. detach-inner 稳定配方反而更差（25.1%）：切掉内层梯度回传后，name 的提升也丢了，只剩 latest 的残余增益。

## 实现边界（诚实声明）

- 本实现是**最小版 TTT-Linear**：手工内层梯度、输出均值池化、单层线性状态；未做逐样本学习率、规范化技巧、多步内层等工程优化；
- 更忠实的 TTT 实现（如 TTT-MLP、规范化 inner loop）可能有不同表现，本报告只对「当前实现下的查询编码」负责。

## 项目层面的判断

1. **TTT 不做为查询编码的替代**：整体负结果 + 精确查表崩塌 + 边界弱化，不值得替换基线 Transformer；
2. **值得记录的方向**：latest 类查询在两个变体中持续提升——「测试时自适应」对**软语义匹配**有效、对**硬精确查表**有害。若未来要做混合架构（对软匹配查询走自适应路径、对硬查表走精确路径），这是依据，但不是现在；
3. **身份层主线仍应是数据/别名/容量**：基线 48.7% 的 heldout 已知是当前最有提升空间的地方，TTT 实验不改变这个判断。

## 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe train_ttt_real.py --steps 600 --seeds 1 2 3
D:\conda\envs\cformer-gpu\python.exe train_ttt_real.py --steps 600 --seeds 1 2 3 --detach-inner
```

代码：`cformer_real/ttt.py`、`train_ttt_real.py`。
