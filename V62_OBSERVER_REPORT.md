# 真实数据验证报告：AI 大模型身份解析与观测点

## 结论

用「AI 大模型」作为真实对象（122 个模型、21 个系列、9 国公司），完整跑通了「真实数据 → 身份解析 → 观测点」验证链。核心结果：

| 验证项 | 结果 | 意义 |
|---|---:|---|
| 身份解析 heldout 已知 Top-1 | 48.7% | 真实数据上从记忆（22 条 100%）走向泛化（445 条 48.7%）；name/predecessor 过半 |
| 歧义检出（heldout） | 84.2% | 边界信号真泛化 |
| 未知零误支持（heldout） | 73.3% | 安全底线有效 |
| **观测点 selection** | **83.7%** | 不同观测点 → 不同但正确的答案 |
| **观测点 invariance** | **77.8%** | 身份不随观测点漂移 |
| **观测点 permission** | **100%（0 泄漏，mask_caught=3）** | 确定性边界实际挡住神经层泄漏 |

## 1. 数据（`data/ai_models_dataset.json`）

- 122 个对象，每个四证据（名称/属性/关系/变化）自然语言句子；
- 708 已知 / 57 歧义 / 11 未知查询，按 **train/heldout 分割**（489/287，heldout 一律不参与训练）；
- 已知查询分三类：name（名称匹配）、predecessor（前代推理）、latest（系列最新推理）；
- 对象带结构化字段（company/region/series/open_source/year/note），支撑确定性观测点过滤；
- ⚠️ 版本号/年份来自 2026 检索，**正式使用前需人工核对**（这是「候选→审核→verified」治理流程的真实应用）。

## 2. 关键工程决策（含失败教训）

1. **训练/评测必须分割**：最初 22 条查询「又训又评」得到虚假的 100%/33%/39%；分割后 heldout 已知只有 7%——暴露了记忆 vs 泛化；
2. **边界训练信号**（P1）：unknown 拒答损失 + ambiguous margin 损失，把歧义/未知从 39% 拉到 84%+/100%（heldout）；
3. **tokenizer 原子化**：`GPT-5.2` 从 `gpt 5 2` 三个 token 改为 `gpt-5-2` 一个 token——近名（GPT-5/5.1/5.2）不再互相抢占，heldout 已知 28.6%→48.7%；
4. **证据必须支撑查询**：`最新模型`查询最初不可解（证据里没有"哪个是最新"），给系列最新模型加显式标记后才可解；
5. **模型容量**：d=64→128，压种子方差；
6. **观测点协议**（关键）：`identity_text`（不含观测点）只给神经身份模型，观测点只做确定性过滤——把 invariance 从 11% 修到 77.8%（之前把观测点前缀混进身份输入是 OOD 干扰）。

## 3. 观测点验证（`data/observer_queries.json` + `eval_observer_real.py`）

- 6 个观测点：开源 / 闭源 / 中国 / 美国 / 2026年 / 编程；
- 三类查询：
  - **selection**（41 条）：`从开源视角看，Qwen系列最新的模型是哪个？` → 目标随观测点变化；
  - **invariance**（12 条）：同一对象在不同观测点下身份一致；
  - **permission**（4 条）：观测点不可见对象零泄漏，并统计 `mask_caught`（不带 mask 时神经层会选中的次数）。
- 协议严格对齐路线图 §8「先身份、后观测点」：身份模型永远看不到观测点。

## 4. 结果解读与边界

- **selection 83.7%**：观测点改变答案且答对——核心假设「不同观测点→不同但正确」实证成立；
- **invariance 77.8%**：与 name 类身份准确率一致，身份不漂移；
- **permission 100% + mask_caught=3**：确定性边界在 3 个样本上挡掉了神经层会犯的泄漏——「观测点是防火墙」不是摆设；
- **诚实边界**：
  1. permission 仅 4 条，样本太小；
  2. mask_caught 仅 3 次，需更多权限样本；
  3. 身份层 heldout 48.7% 是 selection/invariance 的天花板，身份层越强这两项越高；
  4. 观测点是简化版，真实系统需细粒度权限/角色；
  5. latest 类推理查询仅 21%（heldout）——「系列最新」需要系列内比较，超出检索式身份解析的能力边界，应留给 V6.3 递归/关系层。

## 5. 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py --steps 600 --seeds 1 2 3
D:\conda\envs\cformer-gpu\python.exe build_observer_queries.py
D:\conda\envs\cformer-gpu\python.exe eval_observer_real.py
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_real.py -q
```

结果：`artifacts/ai_models_results.json`（身份解析）、`artifacts/observer_results.json`（观测点）。

## 6. 对项目的影响

1. **核心假设首次在真实数据上拿到证据**：共享对象库 + 观测点索引 + 确定性边界，在 122 个真实对象上 selection 83.7%、permission 100% 零泄漏；
2. **身份层是当前瓶颈**（48.7% heldout）：观测点层已证明设计成立，身份层再强则上限更高——下一步优先提升身份解析（更多数据/别名/容量）；
3. **任务边界明确**：latest 类系列内推理不是身份解析的职责，留给 V6.3；
4. **方法论沉淀**：train/heldout 分割、边界训练信号、原子 token、身份-观测点协议分离——这些都是后续版本直接复用的工程资产。
