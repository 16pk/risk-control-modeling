# -*- coding: utf-8 -*-
"""clean_data 入口在无 label 场景下的集成测试。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from clean_data import clean_data  # noqa: E402
from dedup_sample import dedup_by_user_date  # noqa: E402


def _make_sample(tmp_path, with_label: bool = True) -> Path:
    rows = {
        "fuid": ["u1", "u1", "u2"],
        "f_p_date": ["20250101", "20250101", "20250102"],
        "f0": [-1.0, 2.0, 3.0],
        "f1": [10, 20, 30],
    }
    if with_label:
        rows["label"] = [None, 1, 0]
    df = pd.DataFrame(rows)
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p


def test_clean_without_label_col(tmp_path):
    """数据含 label 列但不传 label_col: 全链路可跑通, 去重保首行, 哨兵值仍替换。

    未声明 label 时该列被视为普通特征(排除列表只含 id/dt), 不会误伤主键/哨兵替换逻辑。
    """
    src = _make_sample(tmp_path, with_label=True)
    summary = clean_data(
        input_path=str(src),
        session_dir=str(tmp_path / "session"),
        id_col="fuid",
        dt_col="f_p_date",
        label_col=None,
        auto_confirm=True,
    )
    assert summary["aborted"] is False
    out = pd.read_parquet(summary["sample_parquet"])
    # u1 组内无 label 择优, 保首行; 该行 f0 原值为哨兵 -1, 已被替换为 NaN
    assert len(out) == 2
    u1 = out.loc[out["fuid"] == "u1"].iloc[0]
    assert pd.isna(u1["f0"])
    # 未声明的 label 列被当作特征列派生(排除列表仅 id/dt)
    assert summary["features"] == ["f0", "f1", "label"]


def test_clean_without_label_column_in_data(tmp_path):
    """数据本身无 label 列: 不传 label_col 可正常清洗, feature-list 不含 label。"""
    src = _make_sample(tmp_path, with_label=False)
    summary = clean_data(
        input_path=str(src),
        session_dir=str(tmp_path / "session"),
        id_col="fuid",
        dt_col="f_p_date",
        auto_confirm=True,
    )
    assert summary["aborted"] is False
    out = pd.read_parquet(summary["sample_parquet"])
    assert "label" not in out.columns
    assert len(out) == 2
    assert summary["features"] == ["f0", "f1"]


def test_clean_with_label_col_still_works(tmp_path):
    """回归: 传 label_col 时组内仍优先保留 label 非空行, label 不进特征清单。"""
    src = _make_sample(tmp_path, with_label=True)
    summary = clean_data(
        input_path=str(src),
        session_dir=str(tmp_path / "session"),
        id_col="fuid",
        dt_col="f_p_date",
        label_col="label",
        auto_confirm=True,
    )
    out = pd.read_parquet(summary["sample_parquet"])
    kept = out.loc[out["fuid"] == "u1"].iloc[0]
    assert kept["label"] == 1
    assert kept["f0"] == 2.0
    assert summary["features"] == ["f0", "f1"]