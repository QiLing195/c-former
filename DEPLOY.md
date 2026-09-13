# 部署说明（C-Former GovLayer）

把确定性治理层部署为一个可访问的 Web 服务：**多角色提问 → 权限隔离 → 空白诚实升级 → 依据可追溯**。

## 方式一：本地直接运行（开发/演示最快）

```bash
pip install -r requirements-serve.txt
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

可选启用 LLM 自然语言回答（只把**权限过滤后**的可见内容喂给模型）：

```bash
export DEEPSEEK_API_KEY=sk-xxx      # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 方式二：Docker 部署（交付/私有化）

```bash
docker compose up --build -d
# 浏览器打开 http://<服务器IP>:8000
docker compose logs -f        # 看日志
docker compose down           # 停止
```

镜像特点：**不含 torch**（治理层纯标准库）→ 体积小、启动秒级、可在低配服务器/内网离线运行。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/datasets` | 列出可用知识库与角色 |
| POST | `/api/ask` | `{dataset, role, question}` → 答案 + 依据 + 权限判定 |
| GET | `/healthz` | 健康检查（供容器探针） |
| GET | `/` | 前端演示页 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"dataset":"employee_rules","role":"employee","question":"旷工超过多少天会被劝退？"}'
```

同一问题换 `"role":"hr"`，可见内容层级不同——这是治理层的核心能力。

## 演示脚本（面试/客户现场 3 分钟）

1. 打开页面，知识库选 `employee_rules`，角色选 **employee**；
2. 问："旷工超过多少天会被劝退？" → 只见公开条款（扣 50 元），**劝退标准被权限截断**；
3. 角色切到 **hr**，问同一问题 → 完整看到"当月旷工 5 天/全年 7 天劝退"；
4. 问："出差超期了，原定 2 天结果待了 5 天怎么处理？" → 判定 **制度空白**，不给编造答案，
   给出**过往先例**（仅供参考）；
5. 换知识库 `home`（toC 示例），角色 **child** 问"家里储蓄账户有多少钱" → **明确无权限**；
   角色切 **finance** → 可见。

## 接新客户（零代码）

新增一个知识库只需在 `data/` 放一个 `gov_<name>.json`：

```json
{
  "objects": [{"id":"leave","title":"请假","keywords":["请假","病假"],
               "levels": {"0":"公开内容…","1":"经理级内容…","2":"HR级内容…"}}],
  "roles": {"employee":0,"manager":1,"hr":2},
  "probe_rules": [[["超期","延长"],["变更","延长"]]],
  "precedent_cases": [{"case_id":"C-001","topic":"出差超期","question":"…",
                       "ruling":"…","reasoning":"…","approver":"HR","date":"2026-03-12"}]
}
```

重启服务即生效——这就是"私人化定制只换数据集"的落地形态。

## 生产化待补（诚实清单）

- 鉴权与用户体系对接（当前角色由请求参数传入，生产应对接 SSO/RBAC）；
- 知识对象化的自动化工具（当前制度条款需人工+LLM 半自动整理）；
- 审计日志持久化（当前返回在响应里，未落库）；
- 检索升级（当前关键词锚定；大库需接语义检索 + ANN）；
- 并发与压测、监控告警、K8s 编排。
