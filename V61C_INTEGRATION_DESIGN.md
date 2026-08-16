# C-Former V6.1c：量化闸门修复与磁盘层集成设计

## 已完成：INT8 固定 scale（本版本）

V6.1b 报告指出 512K INT8 = 33.0 MiB，比路线图「≤32 MiB」超 1 MiB，原因是每向量额外存 2 B 的 FP16 scale。

本版本给 `QuantizedVectorStore` 增加 `fixed_scale=True`（仅 INT8）：输入向量已 L2 归一化（分量 ∈ [-1,1]），用全局 scale=1/127 量化，**不再存 scale 列**，每对象正好 64 B。

- 512K × 64 B = **32.0 MiB**（正好命中闸门）；
- 余弦保真度 ≥ 0.99（单元测试验证，`tests/test_cformer_v61.py::test_fixed_scale_int8_meets_32mib_gate_with_fidelity`）；
- 向后兼容：默认仍为每向量 scale 模式。

## 待实现：与 V5.8 磁盘层串联

V5.8 `LayeredAliasStore` 的向量是 256 维 `HashedTextEncoder` 文本哈希，V6.1 IVF 用 64 维学习编码器——两个向量空间不同，不能直接拼。

### 统一对象记录

集成需要每个对象同时携带两种表示，且只存一份学习向量：

```text
ObjectRecord:
    object_id           稳定身份
    canonical_name      精确别名 B-tree 键
    document            文本（FTS 倒排 + 文本哈希源）
    aliases             已验证别名（含来源/置信/版本）
    learned_vector      64 维学习向量（FP16/INT8，ANN 用，只存一份）
    version / tombstone 版本化生命周期
```

### 检索链路

```text
查询文本
  → normalize
  → 精确别名 B-tree 命中（created<=v < removed）→ 直接返回
  → 未命中：FTS5 内容空倒排生成候选 shard
  → 候选 shard 内：IVF/精确向量粗召回 Top-256
  → 四证据 C-Former 精排 Top-16
  → EvidenceVerifier（分类型 margin，见 V6.0b 校准）
  → supported / ambiguous / unknown
  → 未知别名 → CandidateLedger propose（审核前不写正式身份）
```

关键原则不变：**精确命中即返回，不浪费神经计算；ANN 只服务未命中；模型只能 supported，外部审核才 verified**。

### 依赖的真实语料（阻塞项）

V6.1b 已确认 IVF 在组合语义空间 100% 无区分度。要验证这条链路是否真的有价值，需要**非组合、含近重复与噪声的真实对象向量**——这是 V6.1c 集成实现前的输入依赖，需要人工标注或真实知识来源，无法用合成数据替代。

## 顺序建议

1. （本版本）INT8 固定 scale → 32 MiB 闸门 ✅；
2. V6.1c 实现：`UnifiedObjectStore`（对象记录 + 精确别名 + 学习向量单副本）+ 上述链路 + 端到端测试；
3. 真实语料盲测：Recall、Top-1、误合并、审核量、延迟、内存全量报告；
4. 之后才进入 V6.2（观测点端到端问答），避免在未经真实数据检验的检索层之上叠加推理层。
