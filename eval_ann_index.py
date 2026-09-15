# -*- coding: utf-8 -*-
"""#4 规模化检索实测：ANN 忠实度、延迟、n_probe 校准、零跨级泄漏。

用法：
    python eval_ann_index.py                         # 1k/10k/50k，哈希编码器
    python eval_ann_index.py --cluster-multiplier 4  # 质心×4（实测这个才是可用配置）
    set GOVLAYER_ONNX_MODEL=models\\bge-small-zh-v1.5-int8
    python eval_ann_index.py                         # 真实语义编码器

测三件不同的事，严禁混为一谈：
  A. 忠实度@5（ANN vs 暴力精确）—— 只反映近似检索有没有漏命中，与编码器语义无关。
     用 48 条**保真度问题**统计，样本量足够；7 条问题算 recall 的粒度是 1/35，等于没有统计意义。
  B. 针命中@5（真实制度问题能否召回对应条款）—— 只有真实语义编码器才算数。
     同时打印"暴力精确检索的针命中"作为对照：ANN 低于它才是索引的问题，
     两者都低就是编码器语义弱，与索引无关。
  C. 跨级泄漏 —— 级别 L 的查询不得返回"最低可见级别 > L"的对象。

会打印向量空间的簇结构诊断（separation），用于判断 IVF 是否适用：
  separation 接近 0 → 球面近似均匀分布、没有可聚的簇 → IVF 召回差属数据性质问题。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cformer_v63.ann_index import PermissionPartitionedIndex  # noqa: E402
from cformer_v63.embedding import build_encoder              # noqa: E402

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
DEFAULT_PROBES = [1, 2, 4, 8, 16, 32, 64]
FIDELITY_N = 48          # 保真度问题条数（样本量，别再用 7 条糊弄）

# 手写的真实制度问题（用于语义质量对照，对应 gov_employee_rules.json 的条款）
SEMANTIC_QUERIES: list[tuple[str, str]] = [
    ("员工连续旷工多久会被劝退？", "penalty"),
    ("上班时间是几点到几点？", "work-hours"),
    ("忘记打卡怎么办？", "punch"),
    ("请假需要什么材料和审批？", "leave"),
    ("假期结束不能回岗怎么办？", "return"),
    ("出差需要提前报备吗？", "travel"),
    ("迟到早退怎么扣钱？", "penalty"),
]

SUFFIXES = ["怎么办", "的规定", "如何申请", "标准是什么", "具体流程", "有什么要求"]

# 合成干扰条款的词池（只为把库撑大，不参与召回评估）
POOL = ["考勤", "请假", "报销", "加班", "培训", "绩效", "薪酬", "社保", "公积金", "迟到",
        "早退", "调休", "出差", "补贴", "晋升", "考核", "合同", "试用期", "转正", "离职",
        "交接", "年假", "婚假", "产假", "病假", "事假", "打卡", "外勤", "值班", "轮岗",
        "岗位", "职级", "奖惩", "申诉", "审批", "流程", "规范", "细则", "附件", "备案"]


def load_needles() -> list[dict]:
    """真实制度条款作为"针"（另加 1 条只在高密级存在的秘密条款）。

    泄漏判据用 min_level（在 run_size 里对**整个语料**计算，含合成干扰条款）
    而不是 max_level：一个对象可以既在 level 0 有公开条款、又在 level 2 有管理细则
    （如"考勤处罚"），此时 level 0 命中它是正确的。
    只有"最低级别都高于我"的对象被我命中，才算泄漏。
    """
    spec = json.loads((DATA / "gov_employee_rules.json").read_text(encoding="utf-8"))
    objects = list(spec["objects"])
    objects.append({
        "id": "secret-salary",
        "title": "高管薪酬明细",
        "keywords": ["薪酬", "工资", "年薪", "高管"],
        "levels": {"2": "高管薪酬明细仅 HR 与总经理可见，含各岗位年薪档位与期权授予记录。"},
    })
    return objects


def build_fidelity_queries(needles: list[dict], n: int = FIDELITY_N,
                           seed: int = 1) -> list[str]:
    """从真实条款的关键词生成 n 条保真度问题（样本量足够，才谈得上 recall）。"""
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n):
        obj = rng.choice(needles)
        kws = obj.get("keywords") or ["制度"]
        k = rng.sample(kws, min(len(kws), rng.randint(1, 3)))
        out.append("".join(k) + rng.choice(SUFFIXES))
    return out


def make_corpus(needles: list[dict], size: int, rng: random.Random) -> list[dict]:
    """needles + 合成干扰条款，凑到 size 条。干扰条款绝大多数是最低密级。"""
    corpus = list(needles)
    i = 0
    while len(corpus) < size:
        level = 2 if i % 137 == 0 else (1 if i % 17 == 0 else 0)
        words = [rng.choice(POOL) for _ in range(rng.randint(6, 14))]
        text = "".join(words)
        corpus.append({
            "id": f"syn-{i}",
            "title": text[:6],
            "keywords": words[:4],
            "levels": {str(level): text},
        })
        i += 1
    return corpus


def exact_fidelity(index: PermissionPartitionedIndex, level: int, top_k: int,
                   queries: list[str]) -> tuple[dict[str, list[str]], float, int]:
    """算一次暴力精确基准（整档规模只算一次）。"""
    ref: dict[str, list[str]] = {}
    times = []
    for q in queries:
        t0 = time.perf_counter()
        ref[q] = [oid for oid, _, _ in index.exact_search(q, level, top_k=top_k)]
        times.append((time.perf_counter() - t0) * 1000)
    return ref, statistics.mean(times), index.allowed_entries(level)


def measure(index: PermissionPartitionedIndex, level: int, top_k: int, n_probe: int,
            min_level: dict[str, int], ref: dict[str, list[str]],
            exact_needle: int) -> dict:
    """忠实度（多问题）+ 针命中（对照暴力）+ 延迟 + 泄漏。"""
    recalls, ann_ms, scanned, leaks = [], [], [], []
    for q, truth in ref.items():
        t0 = time.perf_counter()
        got = [oid for oid, _, _ in index.search(q, level, top_k=top_k, n_probe=n_probe)]
        ann_ms.append((time.perf_counter() - t0) * 1000)
        scanned.append(index.last_scanned)
        recalls.append(len(set(got) & set(truth)) / max(1, len(truth)))
        for oid in got:
            if min_level.get(oid, 0) > level:
                leaks.append((q, oid, min_level.get(oid, 0), level))

    needle_hits = 0
    for question, expected in SEMANTIC_QUERIES:
        ids = [oid for oid, _, _ in index.search(question, level, top_k=top_k,
                                                n_probe=n_probe)]
        if expected in ids:
            needle_hits += 1

    return {
        "n_probe": n_probe,
        "ann_recall_at_k": round(statistics.mean(recalls), 4),
        "n_recall_queries": len(recalls),
        "needle_hit_at_k": f"{needle_hits}/{len(SEMANTIC_QUERIES)}",
        "exact_needle_hit_at_k": f"{exact_needle}/{len(SEMANTIC_QUERIES)}",
        "ann_ms_avg": round(statistics.mean(ann_ms), 3),
        "ann_scanned_avg": int(statistics.mean(scanned)),
        "leaks": leaks,
    }


def cluster_structure(index: PermissionPartitionedIndex, level: int,
                      sample: int = 2000, seed: int = 0) -> dict | None:
    """量化向量空间有没有可聚的簇结构。

    separation 接近 0 → 向量在球面上近似均匀分布，k-means 分出的簇没有意义，
    此时 IVF 召回差是**数据性质**导致的，换编码器才是解法。
    """
    part = index.partitions.get(level)
    if part is None or part.vecs is None or part.centroids is None:
        return None
    n = len(part.entries)
    if n < 10:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    vecs = part.vecs[idx]
    nearest = (vecs @ part.centroids.T).max(axis=1).mean()

    half = len(vecs) // 2
    if half == 0:
        return None
    random_pair = float((vecs[:half] * vecs[half:2 * half]).sum(axis=1).mean())
    return {
        "n_sampled": int(len(vecs)),
        "nearest_centroid_sim_mean": round(float(nearest), 4),
        "random_pair_sim_mean": round(random_pair, 4),
        "separation": round(float(nearest) - random_pair, 4),
    }


def run_size(needles: list[dict], size: int, encoder, n_probes: list[int], seed: int,
             top_k: int, cluster_multiplier: float, target_recall: float,
             fidelity_n: int) -> dict:
    rng = random.Random(seed)
    corpus = make_corpus(needles, size, rng)

    # 泄漏基准必须覆盖**整个语料**（含合成干扰条款）。
    # 只覆盖真实条款的话，一个 level 2 的合成干扰项被 level 0 命中就会被漏判。
    corpus_min_level = {o["id"]: min(int(k) for k in o["levels"]) for o in corpus}

    index = PermissionPartitionedIndex(encoder, n_probe=1, seed=seed,
                                       cluster_multiplier=cluster_multiplier)
    build_t0 = time.perf_counter()
    index.build(corpus)
    build_s = time.perf_counter() - build_t0

    fid_queries = build_fidelity_queries(needles, n=fidelity_n, seed=seed + 1)
    # 必须去重！随机拼关键词会产生重复问题串，而下面的暴力基准是 dict（重复 key 会被折叠），
    # 不去重就会出现"measure 按去重后算、calibrate 按含重复算"两个不同分母，
    # 于是打印的召回与判定所依据的召回不一致——专门制造"看着达标其实没达标"的假象。
    fid_queries = list(dict.fromkeys(fid_queries))
    ref, exact_ms, exact_scanned = exact_fidelity(index, level=0, top_k=top_k,
                                                  queries=fid_queries)
    exact_needle = sum(
        1 for question, expected in SEMANTIC_QUERIES
        if expected in [oid for oid, _, _ in index.exact_search(question, 0, top_k=top_k)])

    rows = []
    for n_probe in n_probes:
        row = measure(index, level=0, top_k=top_k, n_probe=n_probe,
                      min_level=corpus_min_level, ref=ref, exact_needle=exact_needle)
        row["corpus_size"] = size
        row["exact_ms_avg"] = round(exact_ms, 3)
        row["exact_scanned_avg"] = exact_scanned
        row["scan_ratio"] = round(row["ann_scanned_avg"] / max(1, exact_scanned), 4)
        rows.append(row)

    # n_probe 校准：找达到目标召回的最小 n_probe（复用已算好的暴力基准）
    calibration = index.calibrate_n_probe(fid_queries, level=0,
                                         target_recall=target_recall, top_k=top_k,
                                         reference=ref)
    cal_row = None
    if calibration.get("reached"):
        cal_row = measure(index, level=0, top_k=top_k, n_probe=index.n_probe,
                          min_level=corpus_min_level, ref=ref, exact_needle=exact_needle)
        cal_row["scan_ratio"] = round(cal_row["ann_scanned_avg"] / max(1, exact_scanned), 4)
        cal_row["speedup"] = round(exact_ms / max(1e-9, cal_row["ann_ms_avg"]), 1)
        # 一致性校验：判定所依据的召回（calibration["recall"]）必须与事后独立重测一致。
        # 不一致说明评测脚本内部有分歧，绝不能当成"达标"写进报告。
        cal_row["recall_matches_gate"] = (
            round(cal_row["ann_recall_at_k"], 4) == round(calibration["recall"], 4))

    stats = index.stats()
    return {
        "corpus_size": size,
        "build_seconds": round(build_s, 2),
        "entries_per_level": stats["per_level"],
        "centroids_per_level": stats["centroids_per_level"],
        "exact_ms_avg": round(exact_ms, 3),
        "exact_scanned_avg": exact_scanned,
        "exact_needle_hit": f"{exact_needle}/{len(SEMANTIC_QUERIES)}",
        "n_recall_queries": len(fid_queries),
        "cluster_structure_level0": cluster_structure(index, 0),
        "max_probe_level0": (len(index.partitions[0].centroids)
                             if index.partitions.get(0) is not None
                             and index.partitions[0].centroids is not None else 1),
        "calibration": calibration,
        "calibrated_row": cal_row,
        "target_recall": target_recall,
        "rows": rows,
        "secret_leaked": [
            (q, lvl) for q, _ in SEMANTIC_QUERIES for lvl in (0, 1)
            if "secret-salary" in [oid for oid, _, _ in index.search(q, lvl, top_k=10)]
        ],
        "total_leaks": sum(len(r["leaks"]) for r in rows)
        + sum(len(r["leaks"]) for r in ([cal_row] if cal_row else [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--n-probes", type=int, nargs="+", default=DEFAULT_PROBES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--encoder", choices=["auto", "onnx", "hashing"], default="auto")
    parser.add_argument("--cluster-multiplier", type=float, default=1.0)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--fidelity-queries", type=int, default=FIDELITY_N)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    encoder = build_encoder(prefer=args.encoder)
    semantic = not encoder.name.startswith("hashing")
    needles = load_needles()

    print(f"\n编码器：{encoder.name}（dim={encoder.dim}）")
    print(f"真实条款 {len(needles)} 条（含 1 条仅 level 2 的秘密条款）")
    print(f"忠实度统计生成 {args.fidelity_queries} 条问题（去重后条数见每档输出）"
          f" | 语义对照用 {len(SEMANTIC_QUERIES)} 条手写问题")
    print(f"规模 {args.sizes} | n_probe 扫 {args.n_probes} | 质心系数 {args.cluster_multiplier} "
          f"| 目标召回 {args.target_recall}\n")

    report = {"encoder": encoder.name, "semantic_encoder": semantic,
              "cluster_multiplier": args.cluster_multiplier,
              "target_recall": args.target_recall,
              "fidelity_queries": args.fidelity_queries,
              "sizes": args.sizes, "runs": []}

    header = (f"{'规模':>7} | {'n_probe':>7} | {'忠实度@5':>9} | {'针命中':>7} | "
              f"{'ANN ms':>8} | {'暴力 ms':>8} | {'扫描比':>7} | 泄漏")
    print(header)
    print("-" * len(header))

    for size in args.sizes:
        run = run_size(needles, size, encoder, args.n_probes, args.seed, args.top_k,
                       args.cluster_multiplier, args.target_recall, args.fidelity_queries)
        report["runs"].append(run)

        for row in run["rows"]:
            print(f"{size:>7} | {row['n_probe']:>7} | {row['ann_recall_at_k']:>9.4f} | "
                  f"{row['needle_hit_at_k']:>7} | {row['ann_ms_avg']:>8.2f} | "
                  f"{row['exact_ms_avg']:>8.2f} | {row['scan_ratio']:>7.3f} | "
                  f"{len(row['leaks'])}")

        print(f"        构建 {run['build_seconds']}s | 各级条目 {run['entries_per_level']} | "
              f"质心 {run['centroids_per_level']}")
        print(f"        暴力精确检索的针命中：{run['exact_needle_hit']}"
              f"（ANN 低于它 = 索引漏命中；两者都低 = 编码器语义弱，与索引无关）")
        cs = run["cluster_structure_level0"]
        if cs:
            print(f"        簇结构(level0)：最近质心相似度 {cs['nearest_centroid_sim_mean']} "
                  f"vs 随机点对 {cs['random_pair_sim_mean']} → separation {cs['separation']}"
                  f"{'  ← 几乎没有簇结构，IVF 不适用' if cs['separation'] < 0.02 else ''}")
        cal, crow = run["calibration"], run["calibrated_row"]
        print(f"        忠实度问题数（去重后）：{run['n_recall_queries']}")
        if cal.get("reached") and crow:
            flag = "" if crow.get("recall_matches_gate") else \
                "  ⚠️ 判定值与重测值不一致，勿采信"
            print(f"        ✅ 校准：目标召回 {args.target_recall} → n_probe={cal['n_probe']}"
                  f"（level0 共 {run['max_probe_level0']} 个质心）"
                  f"判定召回 {cal['recall']} / 重测 {crow['ann_recall_at_k']}，"
                  f"{crow['ann_ms_avg']}ms vs 暴力 {run['exact_ms_avg']}ms"
                  f"（{crow['speedup']}×），扫描比 {crow['scan_ratio']}{flag}")
        else:
            print(f"        ⚠️ 校准失败：探遍 level0 全部 {run['max_probe_level0']} 个质心"
                  f"也达不到 {args.target_recall}（最好 {cal.get('recall')}）"
                  f" → 需调大 --cluster-multiplier，或该规模直接用暴力精确检索")

    total_leaks = sum(r["total_leaks"] for r in report["runs"])
    report["total_leaks"] = total_leaks

    print("\n" + "=" * 78)
    if not semantic:
        print("⚠️ 本次用的是哈希编码器（无语义能力）：")
        print("   · 「忠实度@5」有效——衡量 ANN 相对暴力检索有没有漏命中；")
        print("   · 「针命中」无意义（低是编码器的问题，不是索引的问题），")
        print("     要测真实语义召回请设 GOVLAYER_ONNX_MODEL 后重跑；")
        print("   · 合成干扰条款来自 40 词的小词表 → 人为制造强簇，**会高估 IVF 效果**，")
        print("     真实语料的簇结构更弱，实测数字应以真实数据复核。")
    print(f"跨级泄漏总数：{total_leaks}  {'✅ 零泄漏' if total_leaks == 0 else '❌ 有泄漏，必须修'}")
    print("=" * 78)

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "ann_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out}")

    return 1 if total_leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
