# -*- coding: utf-8 -*-
"""replace_invalid 哨兵值替换单测。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from replace_invalid import DEFAULT_INVALID_VALUES, parse_invalid_values, replace_invalid_values


def _df():
    return pd.DataFrame({
        "fuid": ["u1", "u2", "u3", "u4"],
        "f_p_date": ["20250101", "20250101", "20250102", "20250102"],
        "label": [0, 1, 1, 0],
        "f0": [1.0, -1.0, -999.0, 5.0],
        "f1": [-2.0, 3.0, 4.0, -99.0],
        "f2": [10, 20, 30, 40],
    })


def test_default_invalid_values():
    assert DEFAULT_INVALID_VALUES == [-1, -2, -9, -99, -999, -9999, -99999]


def test_replace_hits_sentinel_to_nan():
    df = _df()
    features = ["f0", "f1", "f2"]
    out, report = replace_invalid_values(df, features, DEFAULT_INVALID_VALUES, label_col="label")

    # f0 命中 -1 与 -999
    assert pd.isna(out.loc[1, "f0"])
    assert pd.isna(out.loc[2, "f0"])
    assert out.loc[0, "f0"] == 1.0
    assert out.loc[3, "f0"] == 5.0

    # f1 命中 -2 与 -99
    assert pd.isna(out.loc[0, "f1"])
    assert pd.isna(out.loc[3, "f1"])

    # f2 未命中
    assert out["f2"].tolist() == [10, 20, 30, 40]

    # 报告含两行
    assert set(report["feature"]) == {"f0", "f1"}


def test_replace_skips_non_feature_cols():
    """label / id / dt 列不参与替换, 即使取值为哨兵值。"""
    df = pd.DataFrame({
        "fuid": ["u1", "u2"],
        "f_p_date": ["20250101", "20250101"],
        "label": [-1, 1],  # label 里的 -1 不应被替换
        "f0": [-1.0, 5.0],
    })
    out, _ = replace_invalid_values(df, ["f0"], DEFAULT_INVALID_VALUES, label_col="label")
    assert out["label"].tolist() == [-1, 1]  # label 原样保留
    assert pd.isna(out.loc[0, "f0"])


def test_replace_empty_invalid_values_noop():
    df = _df()
    out, report = replace_invalid_values(df, ["f0", "f1"], [], label_col="label")
    assert out.equals(df)
    assert report.empty


def test_replace_non_numeric_skipped():
    df = pd.DataFrame({
        "fuid": ["u1", "u2"],
        "f_p_date": ["20250101", "20250101"],
        "label": [0, 1],
        "cat": ["-1", "x"],  # object dtype, 不参与数值哨兵替换
    })
    out, report = replace_invalid_values(df, ["cat"], DEFAULT_INVALID_VALUES, label_col="label")
    assert out["cat"].tolist() == ["-1", "x"]
    assert report.empty


def test_replace_report_fields():
    df = _df()
    _, report = replace_invalid_values(df, ["f0", "f1"], DEFAULT_INVALID_VALUES, label_col="label")
    f0_row = report[report["feature"] == "f0"].iloc[0]
    assert f0_row["n_hit"] == 2
    assert "f0" in report["feature"].tolist()
    assert {"feature", "hit_values", "n_hit", "hit_ratio"} <= set(report.columns)


def test_parse_invalid_values_cli_overrides():
    assert parse_invalid_values(None, "-1,-2,-999") == [-1.0, -2.0, -999.0]
    assert parse_invalid_values([-1, -2], None) == [-1.0, -2.0]
    assert parse_invalid_values(None, "") == []
    # 非数值项被忽略
    assert parse_invalid_values(None, "-1,abc") == [-1.0]
