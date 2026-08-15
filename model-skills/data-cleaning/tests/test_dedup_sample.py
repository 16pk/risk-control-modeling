# -*- coding: utf-8 -*-
"""dedup_sample 按用户+日期去重单测。"""
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from dedup_sample import dedup_by_user_date


def test_keeps_label_non_null_row():
    """同 (user, date) 多行时保留 label 非空行。"""
    df = pd.DataFrame({
        "fuid": ["u1", "u1", "u2"],
        "f_p_date": ["20250101", "20250101", "20250102"],
        "label": [None, 1, 0],
        "f0": [1.0, 2.0, 3.0],
    })
    out, report = dedup_by_user_date(df, "fuid", "f_p_date", "label_col" if False else "label")
    assert len(out) == 2
    # u1 保留 label=1 那行(f0=2.0)
    kept = out[out["fuid"] == "u1"].iloc[0]
    assert kept["label"] == 1
    assert kept["f0"] == 2.0
    assert report == {"n_before": 3, "n_after": 2, "n_removed": 1}


def test_all_null_keeps_first():
    """组内 label 全空时保留首行。"""
    df = pd.DataFrame({
        "fuid": ["u1", "u1"],
        "f_p_date": ["20250101", "20250101"],
        "label": [None, None],
        "f0": [1.0, 2.0],
    })
    out, _ = dedup_by_user_date(df, "fuid", "f_p_date", "label")
    assert len(out) == 1
    assert out.iloc[0]["f0"] == 1.0


def test_no_label_col_keeps_first():
    """未提供 label_col 时直接 drop_duplicates 保首行。"""
    df = pd.DataFrame({
        "fuid": ["u1", "u1", "u2"],
        "f_p_date": ["20250101", "20250101", "20250102"],
        "f0": [1.0, 2.0, 3.0],
    })
    out, report = dedup_by_user_date(df, "fuid", "f_p_date")
    assert len(out) == 2
    assert report["n_removed"] == 1


def test_no_duplicates():
    df = pd.DataFrame({
        "fuid": ["u1", "u2"],
        "f_p_date": ["20250101", "20250102"],
        "label": [0, 1],
    })
    out, report = dedup_by_user_date(df, "fuid", "f_p_date", "label")
    assert len(out) == 2
    assert report == {"n_before": 2, "n_after": 2, "n_removed": 0}


def test_distinct_dates_not_deduped():
    """同一用户不同日期不去重。"""
    df = pd.DataFrame({
        "fuid": ["u1", "u1"],
        "f_p_date": ["20250101", "20250102"],
        "label": [1, 0],
    })
    out, _ = dedup_by_user_date(df, "fuid", "f_p_date", "label")
    assert len(out) == 2
