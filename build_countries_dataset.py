# -*- coding: utf-8 -*-
"""第二域数据集：国家（countries），用于跨域迁移测试。

与 AI 模型数据集同一 schema（四证据 + train/heldout 查询），内容完全不同。
用法：
    D:/conda/envs/cformer-gpu/python.exe build_countries_dataset.py
输出：data/countries_dataset.json
注意：首都/语言/成立年份为常识性近似，正式使用前需人工核对。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "countries_dataset.json"

# (中文名, [别名], 大洲, 首都, 语言, 成立年份)
COUNTRIES = [
    ("中国", ["China", "PRC", "中华人民共和国"], "亚洲", "北京", "汉语", "1949"),
    ("美国", ["USA", "美利坚", "United States"], "北美洲", "华盛顿", "英语", "1776"),
    ("英国", ["UK", "大不列颠", "United Kingdom"], "欧洲", "伦敦", "英语", "1707"),
    ("法国", ["France", "法兰西"], "欧洲", "巴黎", "法语", "1958"),
    ("德国", ["Germany", "德意志"], "欧洲", "柏林", "德语", "1990"),
    ("日本", ["Japan", "日本国"], "亚洲", "东京", "日语", "1947"),
    ("俄罗斯", ["Russia", "俄罗斯联邦"], "欧洲", "莫斯科", "俄语", "1991"),
    ("印度", ["India", "印度共和国"], "亚洲", "新德里", "印地语", "1947"),
    ("巴西", ["Brazil", "巴西联邦共和国"], "南美洲", "巴西利亚", "葡萄牙语", "1822"),
    ("加拿大", ["Canada"], "北美洲", "渥太华", "英语", "1867"),
    ("澳大利亚", ["Australia", "澳洲"], "大洋洲", "堪培拉", "英语", "1901"),
    ("韩国", ["Korea", "大韩民国"], "亚洲", "首尔", "韩语", "1948"),
    ("意大利", ["Italy", "意大利共和国"], "欧洲", "罗马", "意大利语", "1946"),
    ("西班牙", ["Spain", "西班牙王国"], "欧洲", "马德里", "西班牙语", "1978"),
    ("墨西哥", ["Mexico", "墨西哥合众国"], "北美洲", "墨西哥城", "西班牙语", "1821"),
    ("印度尼西亚", ["Indonesia", "印尼"], "亚洲", "雅加达", "印尼语", "1945"),
    ("荷兰", ["Netherlands", "尼德兰"], "欧洲", "阿姆斯特丹", "荷兰语", "1815"),
    ("瑞典", ["Sweden"], "欧洲", "斯德哥尔摩", "瑞典语", "1523"),
    ("瑞士", ["Switzerland"], "欧洲", "伯尔尼", "德语", "1848"),
    ("波兰", ["Poland", "波兰共和国"], "欧洲", "华沙", "波兰语", "1918"),
    ("比利时", ["Belgium"], "欧洲", "布鲁塞尔", "荷兰语", "1830"),
    ("奥地利", ["Austria"], "欧洲", "维也纳", "德语", "1955"),
    ("挪威", ["Norway"], "欧洲", "奥斯陆", "挪威语", "1905"),
    ("丹麦", ["Denmark"], "欧洲", "哥本哈根", "丹麦语", "1849"),
    ("芬兰", ["Finland"], "欧洲", "赫尔辛基", "芬兰语", "1917"),
    ("爱尔兰", ["Ireland"], "欧洲", "都柏林", "英语", "1949"),
    ("葡萄牙", ["Portugal"], "欧洲", "里斯本", "葡萄牙语", "1974"),
    ("希腊", ["Greece"], "欧洲", "雅典", "希腊语", "1974"),
    ("土耳其", ["Turkey", "土耳其共和国"], "亚洲", "安卡拉", "土耳其语", "1923"),
    ("沙特阿拉伯", ["Saudi Arabia", "沙特"], "亚洲", "利雅得", "阿拉伯语", "1932"),
    ("以色列", ["Israel"], "亚洲", "耶路撒冷", "希伯来语", "1948"),
    ("伊朗", ["Iran", "伊朗伊斯兰共和国"], "亚洲", "德黑兰", "波斯语", "1979"),
    ("泰国", ["Thailand", "泰国"], "亚洲", "曼谷", "泰语", "1932"),
    ("越南", ["Vietnam"], "亚洲", "河内", "越南语", "1945"),
    ("菲律宾", ["Philippines"], "亚洲", "马尼拉", "菲律宾语", "1946"),
    ("马来西亚", ["Malaysia"], "亚洲", "吉隆坡", "马来语", "1957"),
    ("新加坡", ["Singapore", "狮城"], "亚洲", "新加坡", "英语", "1965"),
    ("巴基斯坦", ["Pakistan"], "亚洲", "伊斯兰堡", "乌尔都语", "1947"),
    ("孟加拉国", ["Bangladesh"], "亚洲", "达卡", "孟加拉语", "1971"),
    ("埃及", ["Egypt", "阿拉伯埃及共和国"], "非洲", "开罗", "阿拉伯语", "1952"),
    ("尼日利亚", ["Nigeria"], "非洲", "阿布贾", "英语", "1960"),
    ("南非", ["South Africa", "南非共和国"], "非洲", "比勒陀利亚", "英语", "1910"),
    ("肯尼亚", ["Kenya"], "非洲", "内罗毕", "斯瓦希里语", "1963"),
    ("摩洛哥", ["Morocco"], "非洲", "拉巴特", "阿拉伯语", "1956"),
    ("埃塞俄比亚", ["Ethiopia"], "非洲", "亚的斯亚贝巴", "阿姆哈拉语", "1941"),
    ("加纳", ["Ghana"], "非洲", "阿克拉", "英语", "1957"),
    ("阿根廷", ["Argentina"], "南美洲", "布宜诺斯艾利斯", "西班牙语", "1816"),
    ("智利", ["Chile"], "南美洲", "圣地亚哥", "西班牙语", "1818"),
    ("秘鲁", ["Peru"], "南美洲", "利马", "西班牙语", "1821"),
    ("哥伦比亚", ["Colombia"], "南美洲", "波哥大", "西班牙语", "1810"),
    ("委内瑞拉", ["Venezuela"], "南美洲", "加拉加斯", "西班牙语", "1811"),
    ("新西兰", ["New Zealand"], "大洋洲", "惠灵顿", "英语", "1907"),
    ("蒙古", ["Mongolia"], "亚洲", "乌兰巴托", "蒙古语", "1924"),
    ("尼泊尔", ["Nepal"], "亚洲", "加德满都", "尼泊尔语", "2008"),
    ("斯里兰卡", ["Sri Lanka"], "亚洲", "科伦坡", "僧伽罗语", "1948"),
    ("哈萨克斯坦", ["Kazakhstan"], "亚洲", "阿斯塔纳", "哈萨克语", "1991"),
    ("乌克兰", ["Ukraine"], "欧洲", "基辅", "乌克兰语", "1991"),
    ("捷克", ["Czech Republic"], "欧洲", "布拉格", "捷克语", "1993"),
    ("匈牙利", ["Hungary"], "欧洲", "布达佩斯", "匈牙利语", "1989"),
    ("罗马尼亚", ["Romania"], "欧洲", "布加勒斯特", "罗马尼亚语", "1878"),
    ("保加利亚", ["Bulgaria"], "欧洲", "索菲亚", "保加利亚语", "1908"),
    ("塞尔维亚", ["Serbia"], "欧洲", "贝尔格莱德", "塞尔维亚语", "2006"),
    ("克罗地亚", ["Croatia"], "欧洲", "萨格勒布", "克罗地亚语", "1991"),
    ("冰岛", ["Iceland"], "欧洲", "雷克雅未克", "冰岛语", "1944"),
    ("卢森堡", ["Luxembourg"], "欧洲", "卢森堡市", "卢森堡语", "1890"),
    ("古巴", ["Cuba"], "北美洲", "哈瓦那", "西班牙语", "1902"),
    ("牙买加", ["Jamaica"], "北美洲", "金斯敦", "英语", "1962"),
    ("巴拿马", ["Panama"], "北美洲", "巴拿马城", "西班牙语", "1903"),
]


def build():
    objects = []
    queries = []
    label = 0
    for name, aliases, continent, capital, language, year in COUNTRIES:
        evidence = {
            "名称": f"这个国家的全称是 {name}"
                   + (f"，也常被称作{'、'.join(aliases)}" if aliases else ""),
            "属性": f"它位于{continent}，主要语言是{language}",
            "关系": f"它的首都是{capital}，属于{continent}地区",
            "变化": f"它于 {year} 年成立或确立现行政体",
        }
        object_id = name.lower().replace(" ", "-")
        objects.append({
            "id": object_id, "label": label, "name": name, "company": "国家",
            "region": continent, "series": continent, "open_source": False,
            "year": int(year), "note": "", "evidence": evidence,
        })
        label += 1

        queries.append({"text": f"介绍一下{name}这个国家", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
        queries.append({"text": f"{name}的首都是哪里？", "target_id": object_id, "kind": "known", "subtype": "attribute", "split": "train"})
        queries.append({"text": f"{name}属于哪个大洲？", "target_id": object_id, "kind": "known", "subtype": "attribute", "split": "train"})
        queries.append({"text": f"我想了解{name}这个国家", "target_id": object_id, "kind": "known", "subtype": "name", "split": "heldout"})
        queries.append({"text": f"{name}的官方语言是什么？", "target_id": object_id, "kind": "known", "subtype": "attribute", "split": "heldout"})

        for alias in aliases[:2]:
            queries.append({"text": f"介绍一下{alias}", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
            queries.append({"text": f"帮我查{alias}的信息", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
            queries.append({"text": f"{alias}是哪个国家？", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "heldout"})

    # 歧义（共享语言/大洲）与未知
    queries.extend([
        {"text": "哪个欧洲国家的首都是柏林？", "target_id": None, "kind": "ambiguous", "split": "heldout"},
        {"text": "官方语言是英语的国家有哪些？", "target_id": None, "kind": "ambiguous", "split": "train"},
        {"text": "哪个亚洲国家说阿拉伯语？", "target_id": None, "kind": "ambiguous", "split": "train"},
        {"text": "火星联邦是什么时候成立的？", "target_id": None, "kind": "unknown", "split": "heldout"},
        {"text": "亚特兰蒂斯岛的首都是哪里？", "target_id": None, "kind": "unknown", "split": "train"},
    ])

    payload = {
        "meta": {
            "dataset": "countries_dataset",
            "description": "国家第二域数据集：跨域迁移测试用，与 AI 模型数据集同 schema 但内容完全无关",
            "status": "自动生成；首都/语言/成立年份为常识性近似，需人工核对",
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
