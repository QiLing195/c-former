# -*- coding: utf-8 -*-
"""第三域数据集：电影（movies），用于多域联合训练与复测。

与 AI 模型/国家数据集同一 schema（四证据 + train/heldout 查询）。
用法：
    D:/conda/envs/cformer-gpu/python.exe build_movies_dataset.py
输出：data/movies_dataset.json
注意：类型/导演/年份为常识性信息，正式使用前需人工核对。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "movies_dataset.json"

# (中文名, [别名], 类型, 导演, 系列, 上映年份)
MOVIES = [
    ("阿凡达", ["Avatar", "阿凡达2"], "科幻", "詹姆斯·卡梅隆", "", "2009"),
    ("泰坦尼克号", ["Titanic"], "爱情", "詹姆斯·卡梅隆", "", "1997"),
    ("复仇者联盟", ["Avengers"], "科幻", "乔斯·韦登", "漫威宇宙", "2012"),
    ("复仇者联盟4：终局之战", ["Avengers: Endgame"], "科幻", "罗素兄弟", "漫威宇宙", "2019"),
    ("美国队长", ["Captain America"], "动作", "乔·庄斯顿", "漫威宇宙", "2011"),
    ("钢铁侠", ["Iron Man"], "科幻", "乔恩·费儒", "漫威宇宙", "2008"),
    ("蜘蛛侠：英雄远征", ["Spider-Man: Far From Home"], "科幻", "乔·沃茨", "漫威宇宙", "2019"),
    ("星球大战：新希望", ["Star Wars", "星球大战"], "科幻", "乔治·卢卡斯", "星球大战系列", "1977"),
    ("星球大战：帝国反击战", ["The Empire Strikes Back"], "科幻", "伊尔文·克什纳", "星球大战系列", "1980"),
    ("变形金刚", ["Transformers"], "动作", "迈克尔·贝", "变形金刚系列", "2007"),
    ("速度与激情", ["Fast & Furious"], "动作", "罗伯·科恩", "速度与激情系列", "2001"),
    ("速度与激情8", ["The Fate of the Furious"], "动作", "F·加里·格雷", "速度与激情系列", "2017"),
    ("黑客帝国", ["The Matrix"], "科幻", "沃卓斯基姐妹", "黑客帝国系列", "1999"),
    ("黑客帝国2：重装上阵", ["The Matrix Reloaded"], "科幻", "沃卓斯基姐妹", "黑客帝国系列", "2003"),
    ("盗梦空间", ["Inception"], "科幻", "克里斯托弗·诺兰", "", "2010"),
    ("星际穿越", ["Interstellar"], "科幻", "克里斯托弗·诺兰", "", "2014"),
    ("蝙蝠侠：黑暗骑士", ["The Dark Knight", "黑暗骑士"], "动作", "克里斯托弗·诺兰", "蝙蝠侠系列", "2008"),
    ("敦刻尔克", ["Dunkirk"], "战争", "克里斯托弗·诺兰", "", "2017"),
    ("指环王：护戒使者", ["The Lord of the Rings", "指环王"], "奇幻", "彼得·杰克逊", "指环王系列", "2001"),
    ("指环王：王者归来", ["The Return of the King"], "奇幻", "彼得·杰克逊", "指环王系列", "2003"),
    ("霍比特人", ["The Hobbit"], "奇幻", "彼得·杰克逊", "指环王系列", "2012"),
    ("哈利·波特与魔法石", ["Harry Potter", "哈利波特"], "奇幻", "克里斯·哥伦布", "哈利波特系列", "2001"),
    ("哈利·波特与死亡圣器", ["Harry Potter and the Deathly Hallows"], "奇幻", "大卫·叶茨", "哈利波特系列", "2010"),
    ("侏罗纪公园", ["Jurassic Park"], "科幻", "史蒂文·斯皮尔伯格", "侏罗纪系列", "1993"),
    ("侏罗纪世界", ["Jurassic World"], "科幻", "科林·特雷沃罗", "侏罗纪系列", "2015"),
    ("辛德勒的名单", ["Schindler's List"], "剧情", "史蒂文·斯皮尔伯格", "", "1993"),
    ("ET外星人", ["E.T."], "科幻", "史蒂文·斯皮尔伯格", "", "1982"),
    ("拯救大兵瑞恩", ["Saving Private Ryan"], "战争", "史蒂文·斯皮尔伯格", "", "1998"),
    ("阿甘正传", ["Forrest Gump"], "剧情", "罗伯特·泽米吉斯", "", "1994"),
    ("肖申克的救赎", ["The Shawshank Redemption"], "剧情", "弗兰克·达拉邦特", "", "1994"),
    ("教父", ["The Godfather"], "剧情", "弗朗西斯·福特·科波拉", "教父系列", "1972"),
    ("低俗小说", ["Pulp Fiction"], "剧情", "昆汀·塔伦蒂诺", "", "1994"),
    ("被解救的姜戈", ["Django Unchained"], "剧情", "昆汀·塔伦蒂诺", "", "2012"),
    ("千与千寻", ["Spirited Away"], "动画", "宫崎骏", "吉卜力", "2001"),
    ("龙猫", ["My Neighbor Totoro"], "动画", "宫崎骏", "吉卜力", "1988"),
    ("你的名字", ["Your Name"], "动画", "新海诚", "", "2016"),
    ("天气之子", ["Weathering with You"], "动画", "新海诚", "", "2019"),
    ("流浪地球", ["The Wandering Earth"], "科幻", "郭帆", "流浪地球系列", "2019"),
    ("流浪地球2", ["The Wandering Earth 2"], "科幻", "郭帆", "流浪地球系列", "2023"),
    ("战狼2", ["Wolf Warrior 2"], "动作", "吴京", "战狼系列", "2017"),
    ("战狼", ["Wolf Warrior"], "动作", "吴京", "战狼系列", "2015"),
    ("红海行动", ["Operation Red Sea"], "动作", "林超贤", "", "2018"),
    ("唐人街探案", ["Detective Chinatown"], "喜剧", "陈思诚", "唐人街系列", "2015"),
    ("唐人街探案3", ["Detective Chinatown 3"], "喜剧", "陈思诚", "唐人街系列", "2021"),
    ("哪吒之魔童降世", ["Ne Zha"], "动画", "饺子", "", "2019"),
    ("姜子牙", ["Jiang Ziya"], "动画", "程腾", "", "2020"),
    ("长津湖", ["The Battle at Lake Changjin"], "战争", "陈凯歌", "长津湖系列", "2021"),
    ("长津湖之水门桥", ["Water Gate Bridge"], "战争", "陈凯歌", "长津湖系列", "2022"),
    ("满江红", ["Full River Red"], "悬疑", "张艺谋", "", "2023"),
    ("悬崖之上", ["Cliff Walkers"], "悬疑", "张艺谋", "", "2021"),
    ("英雄", ["Hero"], "动作", "张艺谋", "", "2002"),
    ("霸王别姬", ["Farewell My Concubine"], "剧情", "陈凯歌", "", "1993"),
    ("卧虎藏龙", ["Crouching Tiger, Hidden Dragon"], "武侠", "李安", "", "2000"),
    ("少年派的奇幻漂流", ["Life of Pi"], "剧情", "李安", "", "2012"),
    ("断背山", ["Brokeback Mountain"], "剧情", "李安", "", "2005"),
    ("让子弹飞", ["Let the Bullets Fly"], "剧情", "姜文", "", "2010"),
    ("邪不压正", ["Hidden Man"], "动作", "姜文", "", "2018"),
    ("疯狂的石头", ["Crazy Stone"], "喜剧", "宁浩", "", "2006"),
    ("疯狂的外星人", ["Crazy Alien"], "喜剧", "宁浩", "", "2019"),
    ("我不是药神", ["Dying to Survive"], "剧情", "文牧野", "", "2018"),
]


def build():
    objects = []
    queries = []
    label = 0
    # 显式系列（MOVIES 元组第 5 字段非空）→ 组内按上映年份排序，建前代链
    series_groups: dict[str, list[tuple[int, str]]] = {}
    for name, _aliases, _genre, _director, series, year in MOVIES:
        if series:
            series_groups.setdefault(series, []).append((int(year), name))
    predecessor_of: dict[str, str] = {}
    for series, members in series_groups.items():
        members.sort()  # 按年份升序
        for index, (_year, name) in enumerate(members):
            if index > 0:
                predecessor_of[name] = members[index - 1][1]

    for name, aliases, genre, director, series, year in MOVIES:
        prev_name = predecessor_of.get(name)
        evidence = {
            "名称": f"这部电影的全称是《{name}》"
                   + (f"，也常被称作{'、'.join(aliases)}" if aliases else ""),
            "属性": f"它是一部{genre}类型电影，于 {year} 年上映",
            "关系": f"它的导演是{director}"
                    + (f"，属于{series}" if series else "")
                    + (f"，前一部是《{prev_name}》" if prev_name else ""),
            "变化": f"它于 {year} 年上映，是当年的热门影片",
        }
        object_id = name.lower().replace(" ", "-").replace("：", "-")
        prev_id = prev_name.lower().replace(" ", "-").replace("：", "-") if prev_name else None
        objects.append({
            "id": object_id, "label": label, "name": name, "company": "电影",
            "region": genre, "series": series or genre, "open_source": False,
            "year": int(year), "note": genre, "predecessor": prev_id, "evidence": evidence,
        })
        label += 1

        queries.append({"text": f"介绍一下电影《{name}》", "target_id": object_id, "kind": "known", "subtype": "name", "split": "train"})
        queries.append({"text": f"《{name}》的导演是谁？", "target_id": object_id, "kind": "known", "subtype": "attribute", "split": "train"})
        queries.append({"text": f"《{name}》是什么类型？", "target_id": object_id, "kind": "known", "subtype": "attribute", "split": "train"})
        queries.append({"text": f"我想了解电影《{name}》", "target_id": object_id, "kind": "known", "subtype": "name", "split": "heldout"})
        queries.append({"text": f"《{name}》是哪一年上映的？", "target_id": object_id, "kind": "known", "subtype": "attribute", "split": "heldout"})

        for alias in aliases[:2]:
            queries.append({"text": f"介绍一下{alias}这部电影", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "train"})
            queries.append({"text": f"{alias}是部什么电影？", "target_id": object_id, "kind": "known", "subtype": "alias", "split": "heldout"})

        if prev_name:
            queries.append({"text": f"在{series}系列中，《{prev_name}》的续集是哪部？", "target_id": object_id, "kind": "known", "subtype": "predecessor", "split": "train"})
            queries.append({"text": f"哪部电影的前一部是《{prev_name}》？", "target_id": object_id, "kind": "known", "subtype": "predecessor", "split": "heldout"})

    # 系列「最新一部」推理查询（显式系列才有意义）
    for series, members in series_groups.items():
        members.sort()
        latest_name = members[-1][1]
        latest_id = latest_name.lower().replace(" ", "-").replace("：", "-")
        queries.append({"text": f"{series}系列最新的电影是哪部？", "target_id": latest_id, "kind": "known", "subtype": "latest", "split": "train"})
        queries.append({"text": f"{series}系列最近上映的是什么？", "target_id": latest_id, "kind": "known", "subtype": "latest", "split": "heldout"})

    # 歧义（同导演/同系列）与未知
    queries.extend([
        {"text": "克里斯托弗·诺兰导演的科幻片有哪些？", "target_id": None, "kind": "ambiguous", "split": "train"},
        {"text": "漫威宇宙系列里有哪些电影？", "target_id": None, "kind": "ambiguous", "split": "train"},
        {"text": "宫崎骏的动画电影有哪些？", "target_id": None, "kind": "ambiguous", "split": "heldout"},
        {"text": "《泰坦尼克号2》是什么时候上映的？", "target_id": None, "kind": "unknown", "split": "heldout"},
        {"text": "《西游记之大圣归来2》的导演是谁？", "target_id": None, "kind": "unknown", "split": "train"},
    ])

    payload = {
        "meta": {
            "dataset": "movies_dataset",
            "description": "电影第三域数据集：多域联合训练与复测用，与 AI/国家数据集同 schema",
            "status": "自动生成；类型/导演/年份为常识性信息，需人工核对",
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
