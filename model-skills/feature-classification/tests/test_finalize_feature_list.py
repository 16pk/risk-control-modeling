# -*- coding: utf-8 -*-
"""finalize_feature_list.py 单测: 固化权威清单 + decided_by + 校验。"""
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import pandas as pd

from finalize_feature_list import finalize, validate_names


def make_scan_json(tmp_path, columns=None):
    """构造探查版 feature-classification.json(覆盖三类)。"""
    columns = columns or {
        "fser_date": {"category": "non_feature", "reason": "日期列", "dtype": "str", "null_ratio": 0.0},
        "if_tf": {"category": "non_feature", "reason": "纯标识列(if_*)", "dtype": "int64", "null_ratio": 0.0},
        "dpd30_1c": {"category": "ambiguous", "reason": "疑似标签列(其他口径)", "dtype": "float64", "null_ratio": 0.1},
        "i_30": {"category": "ambiguous", "reason": "匿名编码列(无业务词)", "dtype": "float64", "null_ratio": 0.3},
        "fals_d15_cell_nbank_else_orgnum": {"category": "feature", "reason": "默认保留", "dtype": "float64", "null_ratio": 0.2},
        "ym_tag": {"category": "feature", "reason": "默认保留", "dtype": "int64", "null_ratio": 0.0},
    }
    rec = {
        "schema_version": 1, "generated_as": "scan", "rulebook": "v0",
        "id_col": "fuid", "dt_col": "ftrans_date", "label_col": "dpd30_3c",
        "counts": {"non_feature": 2, "ambiguous": 2, "feature": 2},
        "groups": [],
        "columns": columns,
    }
    path = tmp_path / "feature-classification.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_classification(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- 固化: exclude ----

def test_finalize_exclude_sets_user_and_writes_csv(tmp_path):
    cls_path = make_scan_json(tmp_path)
    out = tmp_path / "out"
    summary = finalize(str(cls_path), str(out), exclude=["fser_date", "if_tf", "dpd30_1c"],
                       confirmed_at="2026-08-20T11:46+08:00")

    assert summary["n_excluded"] == 3
    assert summary["n_features"] == 3  # i_30(ambiguous) + 2 feature

    rec = read_classification(cls_path)
    assert rec["generated_as"] == "final"
    assert rec["user_confirmed_exclude"] == ["fser_date", "if_tf", "dpd30_1c"]
    assert rec["user_confirmed_at"] == "2026-08-20T11:46+08:00"
    # decided_by 固化
    assert rec["columns"]["fser_date"]["decided_by"] == "user"
    assert rec["columns"]["fser_date"]["reason"] == "用户确认非特征"
    assert rec["columns"]["if_tf"]["decided_by"] == "user"
    assert rec["columns"]["dpd30_1c"]["decided_by"] == "user"  # ambiguous 用户剔除 → user
    # 未确认的规则判定
    assert rec["columns"]["i_30"]["decided_by"] == "rule"
    assert rec["columns"]["ym_tag"]["decided_by"] == "rule"
    # 计数: counts 保留初判, current_counts 固化后
    assert rec["counts"]["non_feature"] == 2
    assert rec["current_counts"]["non_feature"] == 3
    assert rec["current_counts"]["feature"] == 2

    fl = pd.read_csv(out / "feature-list.csv")
    assert list(fl["feature_name"]) == ["i_30", "fals_d15_cell_nbank_else_orgnum", "ym_tag"]


# ---- 固化: keep 恢复规则误判 ----

def test_finalize_keep_restores_rule_misjudge(tmp_path):
    cls_path = make_scan_json(tmp_path)
    summary = finalize(str(cls_path), str(tmp_path / "out"), exclude=[], keep=["if_tf", "dpd30_1c"],
                       confirmed_at="2026-08-20T00:00+08:00")
    rec = read_classification(cls_path)
    # if_tf 从 non_feature 恢复为 feature
    assert rec["columns"]["if_tf"]["category"] == "feature"
    assert rec["columns"]["if_tf"]["decided_by"] == "user"
    assert rec["columns"]["if_tf"]["reason"] == "用户确认保留"
    # ambiguous 也可固化保留
    assert rec["columns"]["dpd30_1c"]["category"] == "ambiguous"
    assert rec["columns"]["dpd30_1c"]["decided_by"] == "user"
    # 原保留 4 (ambiguous 2 + feature 2) + if_tf 恢复 1 = 5; dpd30_1c 本就是 ambiguous 保留
    assert summary["n_features"] == 5
    assert summary["n_excluded"] == 0


# ---- 校验 ----

def test_validate_names_unknown_raises(tmp_path):
    cls_path = make_scan_json(tmp_path)
    try:
        finalize(str(cls_path), str(tmp_path / "out"), exclude=["not_a_column"])
        assert False, "应抛 ValueError"
    except ValueError as e:
        assert "not_a_column" in str(e)