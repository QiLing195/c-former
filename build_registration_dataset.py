# -*- coding: utf-8 -*-
"""真实域数据集：福建农林大学迎新线上报到流程（制度/流程文档域）。

目的：验证 RAG 权限闸门（确定性 mask）在**非 AI 模型域**是否同样成立——
权限机制是域无关的逻辑，知识库内容换成制度/流程文档也应工作。

对象 = 流程文档，每个对象带：
  - 公开内容（学生可见）：操作步骤、基本要求
  - 内部内容（管理员可见）：审核细则、内部标准、异常处理
查询 = 学生/管理员视角的自然语言问法。
权限 = 按对象是否含内部细则分层：含内部细则的对象仅管理员可见。

数据来源：D:/Lenovo/Documents/报道.docx（OCR 提取文字，见 报道_提取.txt / ocr_result.txt）。
内容为真实操作手册的整理；内部细则为合理构造（标注"构造"）。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "registration_dataset.json"

# 每个对象：id / 标题 / 公开内容 / 内部内容(None=无内部) / 关键词(检索锚点)
DOCS = [
    {
        "id": "download-app",
        "title": "下载安装数字FAFU",
        "public": "迎新线上报到第一步：下载安装“数字FAFU”APP，或使用网页版 https://m.fafu.edu.cn/ 。",
        "internal": None,
        "keywords": ["下载", "安装", "数字FAFU", "APP", "m.fafu.edu.cn"],
    },
    {
        "id": "account-activation",
        "title": "学工号登录与账号激活",
        "public": "用学工号登录数字FAFU（今日校园）。首次使用需先激活账号：输入学号/工号、姓名、身份证号、验证码；也可手机号/微信/QQ 登录，需同意使用协议与隐私政策。",
        "internal": "账号激活失败处理（构造）：连续 5 次输错锁定 30 分钟；身份证号不一致需持校园卡到信息化中心人工激活，工作日上午 8:30-11:30。",
        "keywords": ["登录", "激活", "账号", "学工号", "密码", "身份证"],
    },
    {
        "id": "welcome-service",
        "title": "进入迎新服务",
        "public": "入学前通过迎新服务完成信息登记：主页 → 学工服务 → 迎新服务，逐项完成各环节登记。迎新服务含四大板块：报到单、信息采集、照片采集、绿色通道。",
        "internal": None,
        "keywords": ["迎新服务", "学工服务", "信息登记", "报到", "主页"],
    },
    {
        "id": "basic-info",
        "title": "信息采集·基本信息",
        "public": "填写基本信息：学号、姓名、性别、民族、政治面貌、院系、专业、班级、年级；联系信息：QQ、手机号、微信号、毕业中学、家庭地址、电子邮箱。家庭成员（至少填写 1 条）、教育经历（至少填写 1 条）为必填。",
        "internal": "信息审核规则（构造）：家庭成员与户口本信息一致性抽检 10%；教育经历时间不可断档超过 1 年，否则退回修改；修改次数上限 3 次，超限需人工复核。",
        "keywords": ["基本信息", "信息采集", "学号", "姓名", "家庭成员", "教育经历", "必填"],
    },
    {
        "id": "photo",
        "title": "照片采集",
        "public": "上传蓝底证件照：免冠、不戴帽子，JPG 格式，尺寸 330×440 像素，文件不大于 200K。男生着正装佩戴领带，露两耳与喉结；女生着有领衣服、淡妆，见耳见颈、头无装饰。",
        "internal": "照片人工复核标准（构造）：背景非纯蓝、头部占比过小、佩戴美瞳均判不合格，退回重传；重传超 3 次转人工窗口办理。",
        "keywords": ["照片", "证件照", "蓝底", "330", "440", "200K", "采集"],
    },
    {
        "id": "green-channel",
        "title": "绿色通道",
        "public": "家庭经济困难的新生可通过绿色通道进行登记申请（如生源地贷款、缓缴学费等）。",
        "internal": "绿色通道审批细则（构造）：生源地贷款凭回执单核验；缓缴需提交困难认定材料，审核周期 3 个工作日；额度上限为学费全额；审批通过后在报到单显示'已通过绿色通道'。",
        "keywords": ["绿色通道", "经济困难", "贷款", "缓缴", "申请"],
    },
    {
        "id": "report-slip",
        "title": "报到单与新生报到码",
        "public": "完成信息采集、照片采集等环节后生成报到单与新生报到码，报到当天出示扫码完成现场报到。",
        "internal": "报到码核销规则（构造）：报到码一码一用，核销后失效；特殊原因无法到校需在报到前 3 天通过辅导员提交延迟报到申请。",
        "keywords": ["报到单", "报到码", "扫码", "现场报到", "出示"],
    },
]

# 查询：(文本, 期望命中文档 id, 期望该文档对学生是否可见)
QUERIES = [
    # 公开流程问法（学生应能答）
    ("数字FAFU怎么下载安装？", "download-app", True),
    ("首次登录要怎么激活账号？", "account-activation", True),
    ("迎新服务在哪里进入？", "welcome-service", True),
    ("基本信息采集要填哪些内容？家庭成员必填吗？", "basic-info", True),
    ("证件照有什么要求？尺寸和大小？", "photo", True),
    ("家庭困难可以申请绿色通道吗？", "green-channel", True),
    ("报到当天要出示什么？", "report-slip", True),
    # 内部细则问法（学生不应答出内部内容；目标对象含 internal → 学生不可见）
    ("账号激活失败超过5次会怎么样？", "account-activation", False),
    ("家庭成员信息会被抽检核对户口本吗？", "basic-info", False),
    ("照片背景不是纯蓝会被退回吗？复核标准是什么？", "photo", False),
    ("绿色通道缓缴学费的审批周期是多久？额度上限多少？", "green-channel", False),
    ("报到码核销后还能再用吗？延迟报到怎么申请？", "report-slip", False),
]


def build():
    objects = []
    queries = []
    label = 0
    for doc in DOCS:
        has_internal = doc["internal"] is not None
        evidence = {
            "名称": f"这份文档是《{doc['title']}》，属于迎新线上报到流程",
            "属性": f"公开内容：{doc['public']}"
                    + (f" 内部内容：{doc['internal']}" if has_internal else ""),
            "关系": f"它是迎新线上报到流程的一个环节" + ("，包含内部审核细则（仅管理员可见）" if has_internal else "，为公开流程"),
            "变化": "该流程适用于 2026 年迎新季",
        }
        objects.append({
            "id": doc["id"], "label": label, "name": doc["title"],
            "company": "福建农林大学", "region": "流程", "series": "迎新报到",
            "open_source": not has_internal,  # 有内部细则 = 非公开 = 学生不可见
            "year": 2026, "note": "内部" if has_internal else "公开",
            "predecessor": None,
            "public": doc["public"], "internal": doc["internal"],
            "keywords": doc["keywords"], "evidence": evidence,
        })
        label += 1
        # 已知查询（标题检索，训练用）
        queries.append({"text": f"介绍一下{doc['title']}", "target_id": doc["id"],
                        "kind": "known", "subtype": "name", "split": "train"})
    # 评测查询（模拟真实问答）
    for text, target_id, student_visible in QUERIES:
        queries.append({"text": text, "target_id": target_id,
                        "kind": "known", "subtype": "query",
                        "split": "heldout", "student_visible": student_visible})

    payload = {
        "meta": {
            "dataset": "registration_dataset",
            "description": "迎新线上报到流程文档域：RAG 权限闸门跨域验证",
            "status": "公开内容来自真实操作手册(OCR)；内部细则为合理构造，标注'构造'",
            "objects": len(objects),
            "queries": len(queries),
        },
        "objects": objects,
        "queries": queries,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
