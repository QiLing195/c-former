# -*- coding: utf-8 -*-
"""把既有 toB 数据集统一为 GovLayer 约定的通用格式（objects+roles+probe_rules+cases）。

产出（data/gov_*.json）——任何域（企业制度 / 个人知识）都可用同一个 GovLayer 加载：
  gov_employee_rules.json  员工管理制度（三级角色）
  gov_registration.json    迎新报到流程（两级角色：学生/管理员）

用法：D:/conda/envs/cformer-gpu/python.exe build_gov_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_employee_rules() -> dict:
    """员工管理制度 → GovLayer 格式（levels 分级，probe_rules 空白识别）。"""
    objects = [
        {"id": "work-hours", "title": "作息时间",
         "keywords": ["上班时间", "作息", "几点", "下班", "8:30", "工作时间"],
         "levels": {0: "公司上班时间8:30-12:00，13:00-17:30，一般每天8小时、每周6天。"}},
        {"id": "punch", "title": "考勤打卡",
         "keywords": ["打卡", "视频考勤", "忘打卡", "漏打卡"],
         "levels": {0: "实行视频考勤，每天上下班必须视频登记；忘记打卡需说明情况并留存记录。",
                    1: "忘记打卡的说明记录由部门经理核验；频繁漏打卡需面谈提醒。"}},
        {"id": "leave", "title": "请假",
         "keywords": ["请假", "病假", "事假", "请假条", "休假", "医院证明", "批准", "请假单"],
         "levels": {0: "请假需填请假条注明原因天数，经理签字后休息；病假需县市级医院证明；事假填请假单经批准。",
                    1: "批准权限：一般员工请假3天内直接上级批准，3天以上报本部门经理；部门经理请假需总经理批准。",
                    2: "请假记录作为工资发放依据存放考勤员处；HR 定期抽查请假条与医院证明一致性。"}},
        {"id": "return", "title": "销假续假",
         "keywords": ["销假", "续假", "返岗", "假期结束"],
         "levels": {0: "假期内返回需到考勤员处销假；不及时销假按缺勤/旷工处理；假期结束不能返岗需联系考勤员和主管申请续假。"}},
        {"id": "travel", "title": "出差外勤",
         "keywords": ["出差", "外勤", "调休", "报备"],
         "levels": {0: "出差需出差前报备；外勤为全天在外办事；调休需提交调休申请。"}},
        {"id": "penalty", "title": "考勤处罚",
         "keywords": ["迟到", "早退", "扣", "旷工", "劝退", "处罚", "扣工资", "罚款", "缺勤", "30分钟"],
         "levels": {0: "迟到早退每次扣日工资50元；因公外出或经部门经理书面证明除外。",
                    1: "部门经理对本部门考勤处罚有复核权，可对书面证明情况豁免扣款。",
                    2: "旷工不发薪资津贴并按天处罚；当月旷工5天或全年累计7天予以劝退；迟到超30分钟未到岗按旷工处理。"}},
        {"id": "stats", "title": "考勤统计",
         "keywords": ["考勤统计", "上报", "考勤周期", "考勤表"],
         "levels": {0: "每月一个考勤周期，各部门每月固定日前上报考勤统计表。",
                    2: "考勤统计由部门负责人全权负责，HR 汇总核查。"}},
    ]
    roles = {"employee": 0, "manager": 1, "hr": 2}
    probe_rules = [
        # 注意顺序：更 specific 的规则在前（"超期"先于"出差"，避免被泛规则提前判 covered）
        (["超期", "延长", "变更", "超天"], ["变更", "延长", "超期"]),  # 出差超期→无此词=gap
        (["上班", "作息", "几点"], ["8:30", "上班时间"]),
        (["迟到", "早退", "扣"], ["50", "扣"]),
        (["病假"], ["医院证明"]),
        (["出差", "报备"], ["报备"]),
        (["打卡", "忘打卡"], ["视频", "说明"]),
    ]
    precedent_cases = [
        {"case_id": "C-2026-001", "topic": "出差超期",
         "question": "出差原定2天结果待了5天，超出报备天数怎么处理？",
         "context": "部门A员工市场活动出差，客户临时加需求延长3天",
         "ruling": "按实际出差核销，补交变更说明经部门经理签字，超3天报总经理知情。",
         "reasoning": "制度未规定出差超期，参照请假续假精神：超3天需上级批准。",
         "approver": "HR经理（王）", "department": "人力资源部",
         "date": "2026-03-12", "status": "closed", "reference_only": True},
    ]
    return {"dataset": "employee_rules", "description": "员工管理制度（真实来源）",
            "objects": objects, "roles": roles,
            "probe_rules": probe_rules, "precedent_cases": precedent_cases}


def build_registration() -> dict:
    """迎新报到流程 → GovLayer 格式（学生/管理员两级）。"""
    objects = [
        {"id": "download-app", "title": "下载安装数字FAFU",
         "keywords": ["下载", "安装", "数字FAFU", "APP"],
         "levels": {0: "下载安装数字FAFU APP，或使用网页版 m.fafu.edu.cn 。"}},
        {"id": "account-activation", "title": "账号激活",
         "keywords": ["登录", "激活", "账号", "学工号", "身份证"],
         "levels": {0: "用学工号登录；首次使用需激活：输入学号/工号、姓名、身份证号、验证码。",
                    1: "激活失败处理：连续5次输错锁定30分钟；身份证不一致需到信息化中心人工激活。"}},
        {"id": "welcome-service", "title": "迎新服务",
         "keywords": ["迎新服务", "学工服务", "信息登记"],
         "levels": {0: "主页→学工服务→迎新服务，含报到单/信息采集/照片采集/绿色通道四大板块。"}},
        {"id": "basic-info", "title": "基本信息采集",
         "keywords": ["基本信息", "信息采集", "家庭成员", "教育经历", "必填"],
         "levels": {0: "填写学号/姓名/性别/民族/院系/专业/班级；家庭成员至少1条、教育经历至少1条为必填。",
                    1: "审核规则：家庭成员与户口本一致性抽检10%；教育经历断档超1年退回修改。"}},
        {"id": "photo", "title": "照片采集",
         "keywords": ["照片", "证件照", "蓝底", "330", "440", "200K"],
         "levels": {0: "蓝底免冠证件照，JPG，330×440像素，不大于200K。",
                    1: "复核标准：背景非纯蓝/头部占比过小/戴美瞳判不合格退回。"}},
        {"id": "green-channel", "title": "绿色通道",
         "keywords": ["绿色通道", "经济困难", "贷款", "缓缴"],
         "levels": {0: "家庭经济困难新生可通过绿色通道申请（生源地贷款、缓缴学费等）。",
                    1: "审批细则：贷款凭回执核验；缓缴审核周期3个工作日；额度上限为学费全额。"}},
        {"id": "report-slip", "title": "报到单与报到码",
         "keywords": ["报到单", "报到码", "扫码"],
         "levels": {0: "完成采集后生成报到单与新生报到码，报到当天扫码完成现场报到。"}},
    ]
    roles = {"student": 0, "admin": 1}
    probe_rules = [
        (["下载", "安装"], ["下载", "APP"]),
        (["激活"], ["激活"]),
        (["报到", "出示"], ["报到"]),
    ]
    return {"dataset": "registration", "description": "迎新线上报到流程",
            "objects": objects, "roles": roles,
            "probe_rules": probe_rules, "precedent_cases": []}


def main() -> None:
    for name, builder in (("gov_employee_rules.json", build_employee_rules),
                          ("gov_registration.json", build_registration)):
        payload = builder()
        out = ROOT / "data" / name
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out} (objects={len(payload['objects'])}, roles={list(payload['roles'])})")


if __name__ == "__main__":
    main()
