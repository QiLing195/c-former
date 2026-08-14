# C-Former V6.0：Token 语义编码设计

## 目标与边界

V6.0 将 V5.9 中不感知词序的平均池化主路径替换为共享 Token Transformer，验证对象解析能否理解中文词序、否定和多种证据表达。它仍是检索式对象解析器，不是生成式大语言模型，也没有在本阶段加入观测点、时间、空间或递归推理。

本版本保持“先确认身份、后注入观测点”的原则。查询和候选文本都不含对象编号、评测标签或观测点信息；正式身份写入仍由外部 verifier 控制。

## 架构

```text
查询文本 ──字符 Token + 位置编码──┐
                                  ├─共享 Transformer Encoder × L─查询向量(64)
名称证据 ─字符 Token + 位置编码───┤
属性证据 ─字符 Token + 位置编码───┤
关系证据 ─字符 Token + 位置编码───┤─逐字段共享编码─类型嵌入─Softmax 门控─对象向量(64)
变化证据 ─字符 Token + 位置编码───┘

查询向量 × 对象向量库 → 余弦 Top-K → score/margin/coverage verifier
                                  ├─ supported
                                  ├─ ambiguous
                                  └─ unknown
```

默认候选配置：

| 配置项 | 数值 |
|---|---:|
| Token Transformer 层数 | 2（同时测试 4、6 层） |
| `d_model` | 256 |
| 注意力头 | 8 |
| FFN 维度 | 768 |
| 检索向量 | 64 |
| 最大 Token 长度 | 128 |
| 候选证据字段 | 名称、属性、关系、变化，共 4 个 |

查询和四个证据字段共用同一套 embedding 与 Transformer 参数。C-Former 与平铺 Transformer 的区别不在骨干网络，而在候选编码：C-Former 独立编码四个字段，加入证据类型后做门控融合；公平基线将四字段展平为一个长序列。

## 训练内容

数据覆盖四条中文语义轴，每轴 16 种取值，可组合出 65,536 个无编号对象。训练按实体族折叠隔离，并使用世界置换避免对象顺序成为捷径。文本包含：

- 名称、属性、关系、变化四类证据；
- 规范表达与别名表达；
- 常规句式和否定句式；
- 只差一个语义轴的硬负例；
- 缺字段查询、歧义查询和词表外未知查询；
- 全半角、标点与空格归一化检查。

否定硬集特意构造成两条查询具有相同字符多重集合、只改变词序。例如“不是 A，应为 B”和“不是 B，应为 A”不能被平均池化仅靠词频稳定区分。单元测试验证该约束。

优化目标是批内对比损失，并额外加入每条查询的硬负例。训练使用 AdamW、300 步、batch 128、FP16 CUDA；2/4 层各跑 3 个种子，6 层按停止规则只跑 1 个种子。

## 安全边界

- 模型输出只能是候选和相似度，不能自动变成正式身份；
- `EvidenceVerifier` 以分数、第一/第二候选间隔和证据覆盖率给出状态；
- 未知、歧义与已知支持率必须同时报告；
- 观测点不能影响身份向量；
- 本版本仍为受控中文半合成数据，不能外推为开放世界中文理解能力；
- 当前精确向量检索仍为 `O(N)`，64K 以上扩展由 V6.1 的分层索引解决。

## 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe train_evaluate_v60.py --kinds cformer mlp flat --layers 2 4 --seeds 601 602 603 --scales 2048 8192 65536 --worlds 4 --steps 300 --batch-size 128 --queries 96
D:\conda\envs\cformer-gpu\python.exe train_evaluate_v60.py --kinds cformer --layers 6 --seeds 601 --scales 2048 8192 65536 --worlds 4 --steps 300 --batch-size 128 --queries 96
D:\conda\envs\cformer-gpu\python.exe train_evaluate_v60.py --kinds flat --flat-layers 2 --seeds 601 602 603 --scales 2048 8192 65536 --worlds 4 --steps 300 --batch-size 128 --queries 96
D:\conda\envs\cformer-gpu\python.exe aggregate_v60.py
```

