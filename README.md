# C-Former

> **让企业 AI 从"敢说"变成"说得对、不越权、可追责"。**
> C-Former 是"共享对象世界 + 受控推理层"的检索与身份治理系统——它不是又一个知识库方案，而是给任何 RAG/LLM 装上**确定性治理层**的中间件。

## 为什么存在

企业 AI 落地失败，不是因为模型不够聪明，而是因为**不敢信**：

| 怕什么 | C-Former 怎么解决 |
|---|---|
| 怕泄密——员工问到别人的机密 | **检索前权限 mask**：不可见内容原理上不可达（实测 0 泄漏） |
| 怕乱编——AI 胡诌制度条文 | **制度空白识别**：明文才答，空白诚实升级 HR，绝不编造 |
| 怕无据可查——答错了谁负责 | **全链路留痕**：谁问的、权限判定、检索依据、升级记录全可追溯 |
| 怕"规则在难搞的人手里" | **先例沉淀闭环**：每次人工裁决记录成案例，隐性规则逐步显性化 |

## 核心成果（全部真实数据实测）

| 能力 | 指标 | 报告 |
|---|---|---|
| 身份解析（精确层 + 神经层混合） | name/alias 精确命中 **100%**；神经层 heldout **99%** | [`V62_OBSERVER_REPORT.md`](V62_OBSERVER_REPORT.md) |
| 理解层（QueryUnderstanding） | 33 条盲测（口语改写/网页语境/库外对象）**100%** | [`V63_RECURSION_REPORT.md`](V63_RECURSION_REPORT.md) |
| 递归层（确定性关系图） | AI/电影/国家**三域** 全 **100%** | 同上 |
| **RAG 权限闸门**（检索前过滤） | 敏感问题：无闸门泄漏 67–83% → 闸门后 **0 泄漏** | [`RAG_FUSION_POC.md`](RAG_FUSION_POC.md) |
| **toB 治理四支柱**（真实企业制度） | 权限 0 泄漏 · 三级角色 10/10 · 空白识别 100% · 先例闭环 | [`TOB_POC_REPORT.md`](TOB_POC_REPORT.md) |

**诚实声明**（项目一贯纪律，负结果完整存档）：零样本跨域迁移不成立（实测 5.2% ≈ 随机）；TTT 查询编码为负结果；身份层对"措辞远离训练"的问法泛化有限——这些边界都有报告与数据支撑，不粉饰。

## 架构一览

```
用户问题
  → 理解层（QueryUnderstanding）：意图 + 锚定 + 库外拦截
  → 精确层（PreciseMatch）：对象名/别名 100% 精确匹配
  → 神经层（V6.0 编码器）：描述性指代兜底（heldout 99%）
  → 递归层（V6.3）：latest/predecessor/successor 确定性推理
  → 【toB 治理】权限 mask · 字段级分级 · 空白识别 · 先例沉淀
  → 审计：全链路可重放
```

## 快速开始

```powershell
# 1. 安装（Python 3.10+ / PyTorch 2.x）
pip install -e .[dev]

# 2. 全量测试（31 passed）
python -m pytest tests/ -q

# 3. 身份解析训练与评测（真实 AI 模型 273 对象）
python train_eval_real.py --steps 600 --seeds 1 2 3 --d-model 256

# 4. V6.3 递归层（确定性，秒级）——三域全 100%
python train_eval_v63.py
python train_eval_v63.py --data data/movies_dataset.json
python train_eval_v63.py --data data/countries_recursion.json

# 5. toB 治理验证（真实企业制度）
python eval_employee_rules.py               # 权限闸门
python eval_rule_gap.py                     # 制度空白识别
python eval_precedent_loop.py               # 先例沉淀闭环
```

## 目录导航

```text
cformer_v59/   治理层：EvidenceVerifier + CandidateLedger
cformer_v60/   共享 Token Transformer（身份编码）
cformer_v61/   torch IVF ANN 分层检索
cformer_v63/   理解层 + 精确层 + 递归层（核心）
cformer_real/  真实数据管线
data/          多域数据集（AI/国家/电影/制度/流程）
TOB_POC_REPORT.md   企业 AI 治理落地完整报告（toB 入口）
RAG_FUSION_POC.md   RAG × C-Former 融合 POC
```

## 历史版本与路线图（保留审计）

- V6.0–V6.1 逐版演进、失败案例与方法论：`V60_TOKEN_REPORT.md` · `V60B_BLINDSET_REPORT.md` · `V60C_REGION_FIX_REPORT.md` · `V61_ANN_REPORT.md` 等
- 总体研发计划与质量闸门：[`V60_TO_V65_ROADMAP.md`](V60_TO_V65_ROADMAP.md)
- 版本号映射：内部 V6.x ↔ 测试版 0.6.x

## 工程

- 大文件（检查点、结果 JSON）在 `artifacts/`，不入库
- 提交信息中文一句话，里程碑打 tag
