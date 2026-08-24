# -*- coding: utf-8 -*-
"""V6.5-M2 扩量工具：标注表生成与数据集校验。

generate：从现有数据集产出人工扩量模板——
    expansion_seed.csv   存量对象逐行展开（含 needs_review 标记，供人工核对修正）
    expansion_new.csv    新增对象空白表头 + 两行示例（含同名不同对象约定）
    expansion_conflicts.jsonl  多源冲突条目模板

validate：对标准 schema 数据集执行纪律检查（可作数据投放闸门）——
    R1 必填字段与四证据非空     R5 系列链完整（非首成员必须引用在册前代）
    R2 变化字段年份可解析       R6 别名表面冲突（同一别名指向多对象）
    R3 编号泄漏                 R7 needs_review 占比报告
    R4 跨公司同名提醒

用法：
    python -m cformer_v65.expansion generate --data data/ai_models_dataset.json --out data/expansion
    python -m cformer_v65.expansion validate --data data/ai_models_dataset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = ("名称", "属性", "关系", "变化")
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
LEAK_PATTERNS = (
    re.compile(r"对象\s*\d+"),
    re.compile(r"label", re.IGNORECASE),
    re.compile(r"编号[:：]?\s*\d+"),
)

HEADERS = [
    "id", "name", "company", "region", "open_source", "series", "series_index",
    "release_year", "aliases", "needs_review", "notes", *FIELDS,
]


def _alias_variants(name: str) -> list[str]:
    return [name, name.replace(" ", "")] if " " in name else [name]


def seed_rows(raw: dict) -> list[dict]:
    rows = []
    for obj in raw["objects"]:
        meta = obj.get("meta") or {}
        evidence = obj["evidence"]
        year_match = YEAR_PATTERN.search(evidence["变化"])
        rows.append({
            "id": obj["id"], "name": obj["name"],
            "company": meta.get("company", ""), "region": meta.get("region", ""),
            "open_source": meta.get("open_source", ""),
            "series": meta.get("series", ""), "series_index": meta.get("series_index", ""),
            "release_year": year_match.group() if year_match else "",
            "aliases": "|".join(_alias_variants(obj["name"])),
            "needs_review": obj.get("needs_review", ""), "notes": "",
            **{field: evidence[field] for field in FIELDS},
        })
    return rows


_NEW_EXAMPLES = [
    {
        "id": "example-model-one", "name": "示例模型 One", "company": "示例公司",
        "region": "中国", "open_source": "True", "series": "示例系列",
        "series_index": "0", "release_year": "2024",
        "aliases": "示例模型One|例一", "needs_review": "",
        "notes": "首成员：关系写「是该系列早期版本」",
        "名称": "这个模型的全称是 示例模型 One，属于 示例 系列",
        "属性": "它由 示例公司 开发，是开源模型",
        "关系": "它在 示例 系列中，是该系列早期版本",
        "变化": "它于 2024 年发布",
    },
    {
        "id": "example-model-two", "name": "示例模型 Two", "company": "示例公司",
        "region": "中国", "open_source": "True", "series": "示例系列",
        "series_index": "1", "release_year": "2025",
        "aliases": "示例模型Two", "needs_review": "",
        "notes": "非首成员：关系必须引用在册前代全名；同名不同对象在 notes 注明",
        "名称": "这个模型的全称是 示例模型 Two，属于 示例 系列",
        "属性": "它由 示例公司 开发，是开源模型，轻量",
        "关系": "它在 示例 系列中，前一代是 示例模型 One",
        "变化": "它于 2025 年发布",
    },
]

_CONFLICT_TEMPLATE = [{
    "subject_id": "<object_id>",
    "claim_a": {"source": "<来源A>", "text": "<表述A>", "as_of": "<YYYY或*>"},
    "claim_b": {"source": "<来源B>", "text": "<矛盾表述B>", "as_of": "<YYYY或*>"},
    "resolution": "pending",
}]


def write_templates(out_dir: Path, raw: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    seed_path = out_dir / "expansion_seed.csv"
    with seed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(seed_rows(raw))
    written.append(seed_path)

    new_path = out_dir / "expansion_new.csv"
    with new_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(_NEW_EXAMPLES)
    written.append(new_path)

    conflicts_path = out_dir / "expansion_conflicts.jsonl"
    conflicts_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in _CONFLICT_TEMPLATE),
        encoding="utf-8",
    )
    written.append(conflicts_path)
    return written

# -- validate -----------------------------------------------------------------

def validate(raw: dict, *, max_needs_review_ratio: float = 0.10) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    objects: list[dict] = raw.get("objects", [])
    by_name: dict[str, list[str]] = {}
    alias_owner: dict[str, str] = {}
    series_groups: dict[tuple, list[dict]] = {}

    for obj in objects:
        oid = obj.get("id", "<missing>")
        evidence = obj.get("evidence") or {}

        # R1 必填与四证据非空
        for key in ("id", "name"):
            if not obj.get(key):
                errors.append(f"R1 {oid}: missing {key}")
        for field in FIELDS:
            if not str(evidence.get(field, "")).strip():
                errors.append(f"R1 {oid}: empty evidence.{field}")

        # R2 年份可解析
        if not YEAR_PATTERN.search(str(evidence.get("变化", ""))):
            errors.append(f"R2 {oid}: 变化 lacks parseable year")

        # R3 编号/标签泄漏
        for field in FIELDS:
            text = str(evidence.get(field, ""))
            for pattern in LEAK_PATTERNS:
                if pattern.search(text):
                    errors.append(f"R3 {oid}: leakage pattern in {field}: {pattern.pattern}")

        name = obj.get("name", "")
        meta = obj.get("meta") or {}
        company = meta.get("company")
        by_name.setdefault(name, []).append(oid)

        # 别名表面冲突（含去空格变体）
        for surface in _alias_variants(name):
            normalized = surface.lower()
            if normalized in alias_owner and alias_owner[normalized] != oid:
                errors.append(
                    f"R6 alias '{surface}' maps to both "
                    f"{alias_owner[normalized]} and {oid}"
                )
            else:
                alias_owner[normalized] = oid

        if company:
            series_groups.setdefault(
                (company, meta.get("series")), []
            ).append({"oid": oid, "name": name,
                      "index": meta.get("series_index", 0),
                      "relation": str(evidence.get("关系", ""))})

    # R4 跨公司同名提醒
    for name, owners in by_name.items():
        companies = {
            (obj.get("meta") or {}).get("company")
            for obj in objects if obj.get("name") == name
        }
        if len(companies) > 1:
            warnings.append(f"R4 same name across companies: {name} -> {sorted(filter(None, companies))}")

    # R5 系列链完整：非首成员的关系必须引用同系列在册成员全名（大小写不敏感）
    for (company, series), members in series_groups.items():
        ordered = sorted(members, key=lambda m: m["index"])
        known_names = [normalize_name(m["name"]) for m in ordered]
        for position, member in enumerate(ordered):
            relation_lower = member["relation"].lower()
            if "前一代是" not in relation_lower:
                if position != 0:
                    errors.append(
                        f"R5 {member['oid']}: non-first member lacks 前一代 reference"
                    )
                continue
            referenced = any(
                name and (f"前一代是 {name}" in relation_lower
                          or f"前一代是{name}" in relation_lower)
                for name in known_names if name != normalize_name(member["name"])
            )
            if not referenced:
                errors.append(
                    f"R5 {member['oid']}: predecessor name not an in-series member"
                )

    needs_review = sum(1 for obj in objects if obj.get("needs_review"))
    ratio = needs_review / max(1, len(objects))
    if ratio > max_needs_review_ratio:
        warnings.append(
            f"R7 needs_review ratio {ratio:.2%} exceeds {max_needs_review_ratio:.0%}"
        )

    return {
        "objects": len(objects),
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "needs_review": needs_review,
            "needs_review_ratio": round(ratio, 4),
            "series_count": len(series_groups),
        },
    }


def normalize_name(text: str) -> str:
    return " ".join(text.split()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--data", type=Path, required=True)
    generate_parser.add_argument("--out", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path, default=None)
    validate_parser.add_argument("--max-needs-review-ratio", type=float, default=0.10)

    args = parser.parse_args()
    raw = json.loads(args.data.read_text(encoding="utf-8"))

    if args.command == "generate":
        written = write_templates(args.out, raw)
        print(json.dumps({"written": [str(path) for path in written]}, ensure_ascii=False))
        return

    report = validate(raw, max_needs_review_ratio=args.max_needs_review_ratio)
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
