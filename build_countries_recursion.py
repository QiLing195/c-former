# -*- coding: utf-8 -*-
"""国家域递归数据集：政体继承链（前身政体 → 继承国），供 V6.3 递归层跨域验证。

与主数据集 data/countries_dataset.json（68 对象，跨域实验用）分离，
避免污染跨域实验的可复现性。本数据集只含「有明确政体继承关系」的对象，
每条链独立 series（如「苏联继承」），链内按成立年份排序：前身 → 继承国。

链设计（历史常识近似，需人工核对）：
- 苏联(1922) → 俄罗斯(1991)：苏联解体，俄罗斯为继承国
- 捷克斯洛伐克(1918) → 捷克(1993)：天鹅绒分离，捷克为继承国
- 南斯拉夫(1918) → 塞尔维亚(2006)：联邦解体，塞尔维亚为继承国
- 英属印度(1858) → 印度(1947)：非殖民化独立
- 奥斯曼帝国(1299) → 土耳其(1923)：帝国灭亡，土耳其共和国继承
- 荷属东印度(1800) → 印度尼西亚(1945)：非殖民化独立
- 波斯(1501) → 伊朗(1935)：更名延续（萨法维王朝起）
- 锡兰(1948) → 斯里兰卡(1972)：更名延续

用法：
    D:/conda/envs/cformer-gpu/python.exe build_countries_recursion.py
输出：data/countries_recursion.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "countries_recursion.json"

# (对象名, [别名], 大洲, 首都, 语言, 成立年份, 继承自[前身对象名或 None])
CHAIN = [
    ("苏联", ["USSR", "苏维埃社会主义共和国联盟"], "欧洲", "莫斯科", "俄语", "1922", None),
    ("俄罗斯", ["Russia", "俄罗斯联邦"], "欧洲", "莫斯科", "俄语", "1991", "苏联"),
    ("捷克斯洛伐克", ["Czechoslovakia"], "欧洲", "布拉格", "捷克语", "1918", None),
    ("捷克", ["Czech Republic"], "欧洲", "布拉格", "捷克语", "1993", "捷克斯洛伐克"),
    ("南斯拉夫", ["Yugoslavia"], "欧洲", "贝尔格莱德", "塞尔维亚-克罗地亚语", "1918", None),
    ("塞尔维亚", ["Serbia"], "欧洲", "贝尔格莱德", "塞尔维亚语", "2006", "南斯拉夫"),
    ("英属印度", ["British Raj", "British India"], "亚洲", "德里", "英语", "1858", None),
    ("印度", ["India", "印度共和国"], "亚洲", "新德里", "印地语", "1947", "英属印度"),
    ("奥斯曼帝国", ["Ottoman Empire"], "亚洲", "伊斯坦布尔", "土耳其语", "1299", None),
    ("土耳其", ["Turkey", "土耳其共和国"], "亚洲", "安卡拉", "土耳其语", "1923", "奥斯曼帝国"),
    ("荷属东印度", ["Dutch East Indies"], "亚洲", "巴达维亚", "荷兰语", "1800", None),
    ("印度尼西亚", ["Indonesia", "印尼"], "亚洲", "雅加达", "印尼语", "1945", "荷属东印度"),
    ("波斯", ["Persia"], "亚洲", "德黑兰", "波斯语", "1501", None),
    ("伊朗", ["Iran", "伊朗伊斯兰共和国"], "亚洲", "德黑兰", "波斯语", "1935", "波斯"),
    ("锡兰", ["Ceylon"], "亚洲", "科伦坡", "僧伽罗语", "1948", None),
    ("斯里兰卡", ["Sri Lanka"], "亚洲", "科伦坡", "僧伽罗语", "1972", "锡兰"),
]


def build():
    objects = []
    queries = []
    label = 0
    for name, aliases, continent, capital, language, year, predecessor in CHAIN:
        # 链 = 同一「继承系列」：前身与继承国共享 series 名（以继承国为准，链首为前身）
        series = f"{name}继承" if predecessor is None else f"{predecessor}继承"
        evidence = {
            "名称": f"这个国家的全称是 {name}"
                   + (f"，也常被称作{'、'.join(aliases)}" if aliases else ""),
            "属性": f"它位于{continent}，主要语言是{language}",
            "关系": f"它的首都是{capital}"
                    + (f"，前身政体是{predecessor}" if predecessor else "，是该系列的最初政体")
                    + ("，是该系列现行继承国" if predecessor else ""),
            "变化": f"它于 {year} 年成立或确立现行政体",
        }
        object_id = name.lower().replace(" ", "-")
        prev_id = predecessor.lower().replace(" ", "-") if predecessor else None
        objects.append({
            "id": object_id, "label": label, "name": name, "company": "国家",
            "region": continent, "series": series, "open_source": False,
            "year": int(year), "note": "", "predecessor": prev_id, "evidence": evidence,
        })
        label += 1

        queries.append({"text": f"介绍一下{name}这个国家", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
        queries.append({"text": f"我想了解{name}这个国家", "target_id": object_id, "kind": "known", "subtype": "name", "split": "heldout"})
        for alias in aliases[:2]:
            queries.append({"text": f"介绍一下{alias}", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
            queries.append({"text": f"{alias}是哪个国家？", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "heldout"})

        if predecessor:
            prev_id_q = predecessor.lower().replace(" ", "-")
            queries.append({"text": f"哪个国家的前身政体是{predecessor}？", "target_id": object_id, "kind": "known", "subtype": "predecessor", "split": "train"})
            queries.append({"text": f"{predecessor}的继承国是哪个？", "target_id": object_id, "kind": "known", "subtype": "predecessor", "split": "heldout"})

    # 每链「现行继承国」latest 查询
    for name, aliases, continent, capital, language, year, predecessor in CHAIN:
        if predecessor is None:
            continue
        series = f"{predecessor}继承"
        latest_id = name.lower().replace(" ", "-")
        queries.append({"text": f"{predecessor}的现行继承国是哪个？", "target_id": latest_id, "kind": "known", "subtype": "latest", "split": "train"})
        queries.append({"text": f"{series}现在由哪个国家继承？", "target_id": latest_id, "kind": "known", "subtype": "latest", "split": "heldout"})

    # 歧义与未知
    queries.extend([
        {"text": "哪个欧洲国家说俄语？", "target_id": None, "kind": "ambiguous", "split": "train"},
        {"text": "哪个亚洲国家的首都是德黑兰？", "target_id": None, "kind": "ambiguous", "split": "heldout"},
        {"text": "神圣罗马帝国的继承国是哪个？", "target_id": None, "kind": "unknown", "split": "train"},
        {"text": "拜占庭帝国什么时候灭亡的？", "target_id": None, "kind": "unknown", "split": "heldout"},
    ])

    payload = {
        "meta": {
            "dataset": "countries_recursion_dataset",
            "description": "国家域政体继承链数据集：V6.3 递归层跨域验证用，与主国家数据集分离",
            "status": "自动生成；政体继承关系为历史常识近似，需人工核对",
            "objects": len(objects),
            "queries": {
                "known": sum(1 for q in queries if q["kind"] == "known"),
                "ambiguous": sum(1 for q in queries if q["kind"] == "ambiguous"),
                "unknown": sum(1 for q in queries if q["kind"] == "unknown"),
                "train": sum(1 for q in queries if q.get("split", "train") == "train"),
                "heldout": sum(1 for q in queries if q.get("split", "heldout") == "heldout"),
            },
        },
        "objects": objects,
        "queries": queries,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
