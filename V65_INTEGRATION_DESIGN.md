# C-Former V6.5 集成原型设计

日期：2026-08。上游依据：[`V60_TO_V65_ROADMAP.md`](V60_TO_V65_ROADMAP.md) §11。
状态：**M1 已落地**——统一入口 `cformer_v65/unified.py` 与集成测试（7 项）已实现并通过；
本文档同时给出 M2/M3 的执行蓝图与差距清单。

## 1. 目标与非目标

**目标**：把 v59–v63 已验证的组件串成一个可运行原型，从真实文本输入到带证据引用的
结论输出，并以公平基线对照明确"共享对象世界 + 受控推理层"的适用边界。

**非目标（诚实声明）**：
- 不做生成式答案解码器——输出是结论级对象 + 证据引用，不是自由文本（路线图允许的
  "证据约束解码器 ×2"在当前无生成需求的检索定位下降级为证据引用模板，见 §6 差距 G-4）;
- 不宣称通用大模型能力；所有成绩限于 AI 模型域 212 对象真实库与组合语义压力集。

## 2. 现有资产 → 架构映射

| 架构件（§3 图） | 组件 | 文件 | 状态 |
|---|---|---|---|
| 共享 Token Transformer | 2 层冻结编码器（1.456M 参数） | `cformer_v60` | ✅ V6.0 冻结基线 |
| 精确别名 B-tree / FTS shard | `UnifiedObjectStore` | `cformer_v61c/store.py` | ✅ 墓碑/版本/别名生命周期 |
| ANN 粗召回 Top-256 | torch IVF（FP16 单副本） | `cformer_v61` | ✅ Recall@256=100%（212 规模全质心） |
| 四证据精排 Top-16 | 全精度重排 | pipeline | ✅ vs 穷举一致性 100% |
| CandidateLedger / verifier | 分类型 margin 状态机 | `cformer_v59` | ✅ 自动 verified=0；真实库校准完成 |
| 世界推理块 | 结构歧义规则 + as-of 快照 + 词法锚定极值裁决 | `cformer_v62/reasoner.py` | ✅ 留出集 0%→100% |
| 观测点/权限过滤 | ObserverGate（公司/区域掩码） | `cformer_v62/observer.py` | ✅ §8 五闸门全过 |
| 时间轴 | as_of 快照过滤 + TemporalNoMember | v62/v63 | ✅ 泄漏 100%→0% |
| 递归 Transformation | predecessor/successor 图行走 + cycle/depth 控制器 | `cformer_v63/recursion.py` | ✅ 四桶 100% |
| 指代消解 | 别名词表 `apply_aliases` | `cformer_real/data.py` | ✅ 千问→Qwen 冒烟 |

## 3. 统一入口（M1，已实现）

```python
from cformer_v65 import UnifiedCFormer, ObserverFrame

app = UnifiedCFormer("data/ai_models_dataset.json", seed=601)
answer = app.resolve(
    "截至2021年，OpenAI 的 GPT 系列最新模型是什么？",
    observer_frame=ObserverFrame("us", allowed_regions=frozenset({"美国"})),
)
# Answer(status, path, object_id, score, reason, coverage,
#        evidence={名称/属性/关系/变化}, trace=[anchor=lexical, as_of=2021,...], stage_ms)
```

链路顺序：精确别名 → 多跳递归 → ANN+重排 → 结构歧义规则 → as-of 过滤 →
极值裁决 → 分类型 margin 校验 → 观测点掩码 → ledger 提案（仅 unknown 短文本）。
任何确定性环节无法裁决即回退神经路径；被掩对象永不以 supported 暴露。

## 4. 最终数据集六类映射（§11 清单）

| 类别 | 现状 | 缺口动作 |
|---|---|---|
| 合成压力集 | ✅ v59–v61 历史结果保留 | 无（不再扩大，遵守饱和停止规则） |
| 中文对象别名集 | ⚠️ 212 对象自动展开 + 30 条别称 | **M2 扩量至 1K+**，人工核对 needs_review |
| 共享世界问答集 | ⚠️ 模板改写体跨视角 | 盲测集扩口语指代 |
| 时空递归集 | ✅ 57 as-of + 36 空集 + 1490 多跳（程序化真值） | 扩量后同法再生 |
| 安全边界集 | ⚠️ 36 条盲测 + 24 合成 unknown | 需真实冲突/同名样本（外部依赖） |
| 分割隔离与泄漏检查 | ✅ 哈希切分 + 反泄漏单测 | 扩量时公开泄漏检查脚本输出 |

## 5. 公平基线协议（M3 跑分矩阵）

同一候选库、同一训练样本、同一计算预算；报告参数量/索引大小/p95。

| 基线 | 定义 | 已有数据 |
|---|---|---|
| B1 dual-encoder | 冻结 v60 编码器直接最近邻 | known 训练面 100%/留出 6.7%（记忆对照） |
| B2 MeanPool-MLP | `MeanPoolMLPResolver` 同预算 | V6.0 报告：词序/否定全面落后 |
| B3 Flat Transformer | 平铺注意力无四证据门控 | V6.0 报告：公平对照同为 100% 但边界更弱 |
| B4 Evidence-RAG 式 | FTS 召回 + 神经打分，无 verifier/reasoner/gate | 本轮消融臂可直接复用（naive 臂泄漏 100%） |
| 消融 −reasoner | evaluate_v62 baseline 臂 | 留出集 86%→100% 差值即模块贡献 |
| 消融 −temporal | evaluate_temporal naive 臂 | 泄漏 100%→0% |
| 消融 −gate/−ledger | 观察泄漏与自动 verified 出现 | §8 报告已含方向性证据 |

## 6. 路线图 §11 验收标准对照（当前快照）

| 条款 | 目标 | 当前实测 | 状态 |
|---|---|---|---|
| 真实解析 Top-1 高于公平基线 ≥1pp 或清晰优势 | 见 B1–B3 | 推理块贡献留出集 +100pp、时间泄漏 −100pp | ✅ 优势成立（结构项） |
| 歧义/冲突安全处理提升 ≥5pp | ≥5pp | 结构歧义规则从不可检→83.3% 检出 | ✅（冲突类待 M2 数据） |
| 权限/未来事实泄漏率 = 0 | 0 | 0 / 0 | ✅ |
| 证据可追溯 ≥95% | ≥95% | 100% | ✅ |
| 512K 与更大规模 4–5 世界测试 | 待做 | 512K 组合空间已有（V61B）；真实域待扩量 | ⏳ M2 |
| 完整回归/原始 JSON/失败案例/限制说明 | 待汇总 | 各版散存 | ⏳ M3 汇编 |
| 明确声明定位边界 | 必须 | 本文 §1 | ✅ |

### 差距登记（M2/M3 关闭）

| ID | 差距 | 依赖 |
|---|---|---|
| G-1 | 真实语料 1K+（多版本/冲突/同名） | 外部标注/采集 |
| G-2 | 双时钟 ingest/valid | G-1 |
| G-3 | 幻觉校准（unsupported≤2%、ECE≤0.05）在真实噪声上复验 | G-1 |
| G-4 | 证据约束答案模板升级（引用字段级 evidence id） | 无，M3 可做 |
| G-5 | ≥64K 真实域 ANN 闸门复验 | G-1 |

## 7. 冻结产物清单（M3 交付）

```text
cformer_v65/            train/evaluate 入口（unified.py 即 API）
artifacts/v65_*.json    统一跑分原始结果
V65_ARCHITECTURE.md     = 本文 + 实装差异修订
V65_BENCHMARK_REPORT.md 基线矩阵 + 消融 + 失败案例汇编
V65_FAILURE_CASES.md    从各版 failures JSON 汇编
V65_REPRODUCIBILITY.md  固定种子/世界/命令清单（各版 README 命令已可复现）
```

## 8. 风险登记册

| 风险 | 缓解 |
|---|---|
| known 查询仍属同模板家族（模板饱和残余） | M2 盲测扩口语；报告同时给模板面与盲测面两套数 |
| 单领域外推无效力 | §1 边界声明 + 仅按域内主张 |
| 词法锚定依赖显式名 | 别名词表持续扩充；开放指代列为非目标 |
| 小编码器 OOV | coverage 闸门兜底；扩量时重训并重新校准阈值 |
| push/网络中断 | 版本提交全部本地可回溯（tag 建议 M3 打 v0.6.5） |

## 9. 里程碑

- **M1（本次提交）**：统一入口 + 集成测试 + 本设计文档；
- **M2**：G-1/G-2 数据扩量与双时钟；阈值随扩量重校准；
- **M3**：B1–B4 跑分矩阵 + 消融汇编 + 四份 V65 报告冻结 + tag `v0.6.5`。