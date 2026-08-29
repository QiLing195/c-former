# 跨域实验报告：零样本迁移 vs 多域联合训练

## 结论

用「AI 大模型」和「国家」两个完全不同的域验证小编码器的跨域能力，结论清晰：

| 训练方式 | AI 域 identity_top1 | 国家域 identity_top1 |
|---|---:|---:|
| **单域训练**（只训 AI，国家对象仅作库内负例） | 42.4% | **5.2%**（随机基线 1.5% 的 3.5 倍） |
| **联合训练**（AI + 国家一起训） | 36.7% | **60.0%** |

1. **零样本跨域迁移不成立**：单域训练后，新域解析 ≈ 随机水平；
2. **多域联合训练有效**：国家域 5.2% → 60%，AI 域保持；
3. **本质**：d=128 小编码器学到的是**域特定的「词汇→身份」映射**，不是域无关的「证据→身份」通用能力；零样本迁移需要更大预训练模型或跨域结构约束——超出本项目范围。

## 实验设计

- 数据：AI 模型域（273 对象）+ 国家域（68 对象，`data/countries_dataset.json`），合并共享 tokenizer + 域标签；
- 训练：小批量（64）、d=128、300 步、3 种子；单域（仅 AI 查询）vs 联合（两域查询）；
- 评测：`evaluate(domain=...)` 分域报告；identity_top1 = name+alias；
- 随机基线：1 / 68 ≈ 1.47%。

## 关键观察

1. 域内性能随模型容量提升（d64 31% → d128 42%），但跨域性能纹丝不动（4.4% → 5.2%）——域内学习不带来跨域迁移；
2. 联合训练时 AI 域略降（42.4% → 36.7%）：固定容量下多域分摊，d=256 联合可能两域都更高；
3. 国家域边界指标：联合训练下未知零误支持 100%、歧义检出 67%——边界机制同样需要该域训练数据。

## 对项目的影响

1. **共享对象世界概念成立**：一个对象库可容纳多域（AI 模型+国家混放无冲突）；
2. **编码器是域特定的**：每个域需要自己的训练数据（或联合训练），**不是零样本开放世界解析器**——这是项目边界的明确声明，避免过度宣传；
3. **第二组数据的真正价值**：不只是"测试泛化"，而是揭示了「需要多域训练」这一架构真相；
4. **未来方向**：若要做真正跨域，需要①更大容量（d=256+ 联合训练复测），②跨域结构约束（如统一的证据槽位语义），③或直接接受"逐域训练"作为项目定位。

## 复现

```powershell
D:\conda\envs\cformer-gpu\python.exe build_countries_dataset.py
D:\conda\envs\cformer-gpu\python.exe build_ai_models_dataset.py
# 单域（AI 训练）：
D:\conda\envs\cformer-gpu\python.exe train_eval_cross.py --steps 300 --seeds 1 2 3 --d-model 128 --train-domain ai
# 联合训练：
D:\conda\envs\cformer-gpu\python.exe train_eval_cross.py --steps 300 --seeds 1 2 3 --d-model 128 --train-domain all
```

结果：`artifacts/cross_domain_d128.json`、`artifacts/cross_domain_joint.json`。
