import json

from cformer_v65.expansion import validate


def _base_raw() -> dict:
    return {
        "objects": [
            {
                "id": "m-1", "name": "M1", "needs_review": False,
                "meta": {"company": "C", "series": "S", "series_index": 0},
                "evidence": {
                    "名称": "全称 M1，属于 S 系列",
                    "属性": "由 C 开发",
                    "关系": "它在 S 系列中，是该系列早期版本",
                    "变化": "它于 2020 年发布",
                },
            },
            {
                "id": "m-2", "name": "M2", "needs_review": False,
                "meta": {"company": "C", "series": "S", "series_index": 1},
                "evidence": {
                    "名称": "全称 M2，属于 S 系列",
                    "属性": "由 C 开发",
                    "关系": "它在 S 系列中，前一代是 M1",
                    "变化": "它于 2021 年发布",
                },
            },
        ],
        "queries": [],
    }


def test_clean_dataset_passes_with_zero_errors() -> None:
    report = validate(_base_raw())
    assert report["errors"] == []
    assert report["stats"]["series_count"] == 1


def test_planted_errors_are_caught() -> None:
    raw = _base_raw()
    raw["objects"][1]["evidence"]["变化"] = "发布日期不详"          # R2
    raw["objects"][1]["evidence"]["属性"] = "它是对象7的变体"        # R3
    raw["objects"][1]["evidence"]["关系"] = "它在 S 系列中，前一代是 幽灵"  # R5
    report = validate(raw)
    codes = {error.split()[0] for error in report["errors"]}
    assert {"R2", "R3", "R5"} <= codes


def test_alias_surface_conflict_flagged() -> None:
    raw = _base_raw()
    raw["objects"][1]["name"] = "M 1"   # 去空格后与 M1 的变体表面冲突
    report = validate(raw)
    assert any(error.startswith("R6") for error in report["errors"])
