# C-Former V2 设计

## 目标

V2 验证“所有信息始终存在，观测点是坐标或索引”的模型：

```text
共享世界 M + 观测点 o + 问题 q -> 答案
```

共享世界不随问题或观测点改变，先独立编码并缓存；观测点只影响查询，不屏蔽其他事实。

## 小世界

每个世界包含：

- 4 个人的位置；
- 2 个物品的持有者；
- 4 个人分别对 2 个物品位置的认知。

共 14 条事实。事实顺序在训练时随机打乱，防止模型依赖固定位置。

## 四类问题

1. `true_location_2hop`：先找物品持有者，再找持有者位置；与观测点无关。
2. `observer_belief`：查当前观测者认为物品在哪里；与观测点相关。
3. `object_holder`：查物品持有者；与观测点无关。
4. `outside_person_location`：查询另一个人的位置，确保模型能使用观测点之外的信息。

## 三个对照模型

- `concat_transformer`：世界、观测点和问题每次拼接并整体重编码。
- `shared_question_only`：世界可缓存，但查询不使用观测点。
- `observer_cformer`：世界可缓存，问题与观测点融合，并由观测点关系偏置调制 Cross-Attention。

## Observer-CFormer

```text
M = MemoryEncoder(world)
q = QuestionEncoder(question)
o = ObserverEncoder(observer)
z = q + sigmoid(W[q;o]) * T(o)

score_i = Q(z)K(M_i) / sqrt(d)
        + Q_o(o)K_r(M_i) / sqrt(d)
```

查询器执行两轮 Cross-Attention，允许从共享记忆中进行两跳聚合。`M` 的编码与 `o`、`q` 无关，因此可以服务同一世界的多个问题。

