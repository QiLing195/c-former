# C-Former V4：可靠性与边界设计

## 输出状态

```text
ANSWER
UNKNOWN
CONFLICT
OBSERVER_UNKNOWN
ACCESS_DENIED
```

输出被拆成状态头和答案头。只有状态为 `ANSWER` 时，答案头结果才允许返回。

## 参数压缩

- 事实、问题、观测点共享 token embedding；
- 第一、第二跳共享同一个循环查询块；
- 第一、第二跳共享 24 维索引投影；
- 世界缓存由 48 维精确表示和 24 维索引组成。

V4 C-Former 为 58,042 参数，可靠性 Dense Transformer 为 67,621 参数。

## 边界责任

```text
权限/认知边界控制器 -> Hard Mask + 强制状态
神经检索器             -> 相关证据、UNKNOWN、CONFLICT
答案头                 -> 只在 ANSWER 状态下返回内容
```

`ACCESS_DENIED` 和 `OBSERVER_UNKNOWN` 不依靠概率模型决定。可信策略层根据权限表、时间和主体知识元数据，在神经检索前过滤事实并设置强制状态。观测角度仍使用软调制，但安全边界必须是确定性的。

## 两阶段检索

同一个共享查询块循环使用两次：

```text
z0 = Fuse(question, observer)
e1 = Retrieve(z0, allowed_memory)
z1 = SharedBlock(z0, e1)
e2 = Retrieve(z1, allowed_memory)
z2 = SharedBlock(z1, [e1, e2])
```

状态头同时读取最终查询表示、两跳最高检索分数和候选分差，用于判断证据是否充分或相互冲突。

