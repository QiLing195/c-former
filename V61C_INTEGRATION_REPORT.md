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

## 5. 下一步

1. 在真实库上重做 score/margin/coverage 校准（复用 `calibrate_v60.py` 方法），恢复 supported 覆盖率；
2. V6.2 世界推理块最小原型：跨候选聚合打分，直接攻"系列最新"类超级指代；
3. 数据规模扩到 1K+ 后在 ≥64K 合成压力规模复验 ANN 闸门。
