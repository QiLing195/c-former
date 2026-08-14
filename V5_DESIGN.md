# C-Former V5：冲突索引与公平证据基线

## 冲突索引

V5 不再要求神经网络从大量候选中偶然同时检出冲突。可信结构化控制器按以下键组织事实：

```text
Entity + Predicate + Time Range + Version + Source
```

只有满足以下条件才输出 `CONFLICT`：

```text
实体和谓词相同
时间范围重叠
可信来源给出不同值
不存在新版本覆盖关系
```

不同时间的正常变化、显式版本更新和低于可信阈值的来源不会被判为冲突。

## 系统分工

```text
策略控制器：权限和主体认知边界
冲突控制器：时间、版本、来源和冲突
神经模型：普通问题、证据检索和答案
```

冲突控制器同时提供给所有基线，因此三模型的差异只来自普通问题查询能力。

## Evidence-RAG 基线

Evidence-RAG 与 C-Former 完全相同：

- 58,042 参数；
- 两阶段证据监督；
- 24 维索引；
- 共享循环查询层；
- 相同训练步数和硬边界。

唯一差异：

```text
RAG:     query = question + observer_embedding
C-Former query = question + gate(question, observer) * transform(observer)
```

这样可以隔离观测点门控本身的贡献。

