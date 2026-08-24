# C-Former V6.1c 集成报告

日期：2026-08。对应设计：[`V61C_INTEGRATION_DESIGN.md`](V61C_INTEGRATION_DESIGN.md)。
范围：在 212 对象真实 AI 模型库（`data/ai_models_dataset.json`）上，把「精确别名 → ANN 粗召回 →
全精度重排 → 分类型 margin 校验 → CandidateLedger」串成一条可运行链路，验证**正确性而非速度**。

## 1. 实现物

| 组件 | 文件 | 说明 |
|---|---|---|
| `UnifiedObjectStore` | `cformer_v61c/store.py` | SQLite 对象记录 + verified/proposed 别名表 + FTS5 trigram 文档索引；墓碑删除、版本号递增；向量不落库（IVF 单副本） |
| `UnifiedResolutionPipeline` | `cformer_v61c/pipeline.py` | 链路编排；精确命中旁路神经计算；未知短文本按保守启发式进 ledger，绝不自动写入正式身份 |
| 端到端评测 | `evaluate_v61c.py` | 3 种子 × 训练编码器 × 建库建索引 × 全查询走链路 |
| 单元测试 | `tests/test_cformer_v61c.py` | 7 项：旁路、状态三态、墓碑、别名生命周期、提案去重、FTS、小规模 IVF 一致性 |

## 2. 结果（3 种子均值，原始 JSON 见 `artifacts/v61c_results.json`）

```text
主数据 known Top-1（链路 + SUPPORTED 才计分）   66.7%
主数据 supported 覆盖率                        69.3%
精确路径占比                                   0%（查询是句子，别名命中由盲测/单元测试覆盖）
ANN 目标进入候选比例（nprobe=16/16 质心）       100%
链路 vs 穷举重排一致性                          100%
留出集 known Top-1（oracle intent）             6.7%（与泛化消融结论一致）
端到端延迟 p50 / p95                           3.7 / 4.3 ms
ledger 提案 / 自动 verified                     ≥1 / 0
```

盲测集（oracle intent）：known Top-1 8.3–16.7%、ambiguous 检出 66.7%、unknown 拦截 100%、
短别名型 unknown 的 ledger 提案率 ≈17%。

## 3. 对照质量闸门

| 闸门（路线图 §7 / 设计文档） | 结果 |
|---|---|
| 重排相对穷举最终 Top-1 下降 ≤0.5pp | ✅ 0pp（一致性 100%） |
| ANN Recall@256 ≥99.5% | ⚠️ 本规模需 nprobe=全部质心才达 100%；212 对象时 ANN 即廉价全查，**真实 ANN 收益必须在 ≥64K 规模复验**（V6.1b 已在组合空间 512K 证明可行） |
| 高风险错误对象自动验证率 = 0 | ✅ `ledger_auto_verified = 0`，verify 仅 reviewer 可触发 |
| 正式别名不被候选自动污染 | ✅ 提案只进 ledger；`add_verified_alias` 是唯一写入口 |
| 增量失败回滚 | ✅ 墓碑 + 版本号单元测试覆盖 |

## 4. 诚实的限制

1. **known Top-1 66.7% 的主因是 verifier 覆盖率而非检索**：小编码器在真实文本上的 score/margin
   分布弱，约 31% 已知查询被拒答。V6.0b 的分类型校准方法论需要在真实库上重新校准阈值
   （当前 margin_by_type 用的是模板域推荐值 + oracle intent 类型）。
2. **超级指代缺口未变**：留出集"系列最新"类查询仍 6.7%，与泛化消融的架构级结论一致——
   需要 V6.2 世界推理块做跨候选比较。
3. **FTS trigram 对 <3 字中文别名无召回**（如"豆包"），该场景由 ANN 兜底；生产需补二元切分。
4. 规模太小，本测不构成对 ANN/FTS 性能的主张。

## 5. 真实库阈值重校准（2026-08 追加）

`calibrate_v61c.py` 按 V6.0b 方法论在真实库上重新校准：known 查询三分切割（训练/校准/评测互斥），
校准面用训练查询的 T1/T4 改写体（与训练表面形式不相交），合成 24 条库外 unknown 补量，
网格搜索"歧义/unknown 零误支持"硬约束下的最大 known 覆盖率。

**校准中的关键发现——结构性歧义**：裸系列指代（"Kimi 是哪一个模型？"）在嵌入空间天然贴近
旗舰成员（score 0.73、margin 0.13），**score-margin 原理上检测不了这类歧义**。解法是确定性
规则（治理优先于模型）：多成员系列 + 查询无选择标准措辞（最新/旗舰/推理…见
`SELECTION_PHRASES`）→ 直接判 AMBIGUOUS，已实现于 pipeline 并有单元测试。

**残余风险**："Mistral Large 4" 类库外近失版本得 score 0.75 + margin 0.13，任何阈值组合都
拦不住——需要未来的语义新颖性检测，当前靠审核兜底。

### 校准前后对比（同一链路、同一种子组）

| 指标 | 校准前（V6.0b 模板域阈值） | 校准后（真实库阈值） |
|---|---:|---:|
| 主数据 known Top-1 | 66.7% | **76.3%** |
| supported 覆盖率 | 69.3% | **85.1%** |
| 盲测歧义检出 | 83.3% | 83.3%（无损失） |
| 盲测 unknown 拦截 | 100% | 100%（无损失） |
| ledger 自动 verified | 0 | 0 |

最终阈值：`minimum_score=0.40 / minimum_coverage=0.60 / known_margin=0.01 /
安全类 margin=0.08`（跨种子中位数；margin=0.005 还能再+8pp 覆盖但近失风险翻倍，
0.01 是保守折中）。复现：`calibrate_v61c.py` → `evaluate_v61c.py --minimum-score 0.40
--minimum-coverage 0.60 --known-margin 0.01`。

## 6. 下一步

1. V6.2 世界推理块最小原型：跨候选聚合打分，攻"系列最新"超级指代（留出集 6.7% 的架构缺口）；
2. 库外近失版本的语义新颖性检测（"Mistral Large 4"问题）；
3. 数据扩到 1K+ 后复验 ANN 规模闸门与校准稳定性。
