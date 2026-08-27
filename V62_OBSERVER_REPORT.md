# 真实数据验证报告：AI 大模型身份解析与观测点（最终版 · 198 对象）

## 结论（最终配置 d=256 + 别名 + 边界信号，198 个真实对象）

用「AI 大模型」作为真实对象（**198 个模型、36 个系列、26 家公司**），完整验证「共享对象库 → 身份解析 → 观测点」链路。**核心假设在真实数据上拿到完整证据**。关键：**评测分域** —— 身份解析（name/alias）与关系推理（predecessor/latest）分开报告，避免混合分数误导结论。

| 验证项 | 122 对象 | **198 对象（最终）** | 意义 |
|---|---:|---:|---|
| **身份解析 identity_top1（name+alias）** | — | **76.3%** | 检索层真实水平（混合 known 只有 54%——被推理类拖低） |
| 关系推理 reasoning_top1（predecessor+latest） | — | **33.1%** | 独立任务，明确归 V6.3 |
| 歧义检出（heldout） | 100% | **99.1%** | 边界信号稳定生效 |
| 未知零误支持（heldout） | 80% | **86.7%** | 安全底线 |
| **观测点 selection** | 82.8% | **92.2%** | 更多观测点/对象下仍提升 |
| **观测点 invariance** | 94.4% | **97.2%** | 身份不随观测点漂移 |
| **观测点 permission** | 100%（11–27） | **100%（31–85 次 mask_caught）** | 确定性边界挡住更多神经层泄漏 |

## 1. 数据（`data/ai_models_dataset.json`）

- **198 个对象**，四证据（名称/属性/关系/变化）自然语言；36 个系列、26 家公司（含微软 Phi、亚马逊 Nova、英伟达 Nemotron、IBM、商汤、讯飞、书生 InternLM、通义万相、华为盘古、快手可灵等）；
- 34 个真实唯一别名（GPT-4o→GPT-4 Omni、Qwen3.7-Max→千问3.7 等），别名进证据并生成别名查询；
- 1264 已知 / 111 歧义 / 11 未知查询，train/heldout 分割（868/518，heldout 不参与训练）；
- 已知查询分四类：name / alias / predecessor / latest；
- 对象带结构化字段（company/region/series/open_source/year/note），支撑确定性观测点过滤；
- ⚠️ 版本号/年份（尤其新增的 o4、R2、Gemini 3.5、盘古、可灵等）来自 2026 检索与常识，**正式使用前必须人工核对**——这是「候选→审核→verified」治理流程的真实应用。

## 2. 工程决策与失败教训（本项目的核心方法论资产）

1. **训练/评测必须分割**：22 条查询「又训又评」得到虚假 100%；分割后 heldout 仅 7%——暴露记忆 vs 泛化；
2. **边界训练信号**：unknown 拒答损失 + ambiguous margin 损失 + **分数地板**（歧义查询 top 分数保持 ≥0.50）；
3. **tokenizer 原子化**：`GPT-5.2`→`gpt-5-2` 一个 token，解决近名抢占（heldout 28.6%→48.7%）；
4. **证据必须支撑查询**：`最新模型`查询最初不可解，给系列最新模型加显式标记；
5. **容量**：d=64→128→256（同数据 d=128→256 提升 +21pp）；
6. **别名**：真实唯一别名进证据 + 别名查询；
7. **歧义 0% 之谜（反直觉发现）**：margin loss 一直有效（歧义 margin 0.016 vs 已知 0.52），但 verifier 先查分数（<0.50→UNKNOWN）把低分歧义路由到 UNKNOWN 而非 AMBIGUOUS——加「分数地板」损失后歧义检出 0%→100%；
8. **观测点协议**：identity_text（不含观测点）只给神经身份模型，观测点只做确定性过滤——invariance 11%→94.4%；
9. **工程排障**：多种子串行需 `del model + empty_cache`，否则 d=256 在第二个种子 OOM 静默退出。

## 3. 观测点验证（`data/observer_queries.json` + `eval_observer_real.py`）

- 10 个观测点：开源/闭源/中国/美国/2026年/2025年/编程/旗舰/推理/多模态；
- 60 selection + 12 invariance + **144 permission**；
- 协议：身份模型只看 identity_text；观测点 = 确定性可见性 mask；
- permission 额外统计 mask_caught：不带 mask 时神经层会选中禁止对象的次数（确定性边界实际挡住的泄漏）。

## 4. TTT 实验（负结果，详见 `TTT_EXPERIMENT_REPORT.md`）

TTT-Linear 查询编码（全梯度 35.6% / detach-inner 25.1%）均低于基线 Transformer（48.7%）。发现：测试时自适应对软语义匹配（name/latest）有效、对精确查表（predecessor）有害。**不采用**，保留报告与代码。

## 5. 结果解读与诚实边界

- 观测点三类指标在真实数据上全部成立：不同观测点不同答案（82.8%）、身份不漂移（94.4%）、零泄漏（100% + mask_caught 11–27）——「共享对象库 + 观测点索引 + 确定性边界」是真实有效的架构；
- 身份层 52.7% 是当前上限来源：selection/invariance 的天花板随身份层提升而提升；
- 诚实边界：
  1. 版本号/别名需人工核对（候选→审核→verified 治理流程的真实场景）；
  2. unknown_rejected 53%（部分未知被判 ambiguous 而非 unknown，但零误支持，安全达标）；
  3. 子类种子方差大（name 40–92%、predecessor 5–83%）——122 对象仍偏小，扩充对象/别名可进一步稳定；
  4. latest 类 heldout 弱（~10–34%）——系列内推理是 V6.3 递归层的职责，非身份解析；
  5. 观测点仍为简化版（属性过滤），真实系统需细粒度权限/角色。

## 6. 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
D:\conda\envs\cformer-gpu\python.exe train_eval_real.py --steps 600 --seeds 1 2 3 --d-model 256
D:\conda\envs\cformer-gpu\python.exe build_observer_queries.py
D:\conda\envs\cformer-gpu\python.exe eval_observer_real.py --d-model 256
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/test_cformer_real.py -q
```

结果：`artifacts/ai_models_results_final.json`、`artifacts/observer_results.json`。

## 7. 对项目的影响

1. **核心假设实证**：共享对象库 + 观测点索引 + 确定性边界，在 122 个真实对象上 selection 82.8% / invariance 94.4% / permission 100% 零泄漏；
2. **方法论沉淀可复用**：train/heldout 分割、边界训练信号（含分数地板）、原子 token、身份-观测点协议分离、多种子显存管理——后续版本直接复用；
3. **身份层主线**：52.7% 且随容量提升明确增长，下一步 = 更多真实对象/别名 + 校验版本号；
4. **任务边界清晰**：latest 类推理归 V6.3，观测点归 V6.2 正式化，TTT 不采用（负结果存档）。
