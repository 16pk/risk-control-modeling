# -*- coding: utf-8 -*-
"""classify_features.py 单测: 探查扫描三分类 + 通配符分组 + 报告渲染。"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import pandas as pd

from classify_features import scan_features, render_report, _group_columns


def make_df():
    """构造含 id/dt/label + 三类特征的样本。"""
    return pd.DataFrame({
        "fuid": [1, 2, 3],
        "ftrans_date": ["2026-08-01"] * 3,
        "dpd30_3c": [0.0, 1.0, 0.0],
        # non_feature 候选
        "fser_date": ["2026-08-01"] * 3,
        "ftrans_time": ["12:00"] * 3,
        "sx_order_id": ["a", "b", "c"],
        "if_tf": [0, 1, 0],
        "dpd30_1c": [0.0, 1.0, 0.0],
        "fpd7_sx30": [0.0, 1.0, None],
        # ambiguous
        "i_30": [0.1, 0.2, 0.3],
        "m_1221": [0.5, 0.4, 0.3],
        "score_80002": [100.0, 200.0, 300.0],
        # feature
        "f1_loan_cnt": [1, 2, 3],
        "f2_loan_amt": [100.0, 200.0, 300.0],
        "ym_tag": [0, 1, 1],
    })


def test_scan_counts_and_columns():
    df = make_df()
    scan = scan_features(df, "fuid", "ftrans_date", "dpd30_3c")

    assert scan["counts"]["non_feature"] == 6
    assert scan["counts"]["ambiguous"] == 3
    assert scan["counts"]["feature"] == 3
    # 逐列档案字段齐全
    assert set(scan["columns"]["i_30"]) == {"category", "reason", "dtype", "null_ratio"}
    assert scan["columns"]["fser_date"]["category"] == "non_feature"
    assert scan["columns"]["ym_tag"]["category"] == "feature"


def test_scan_excludes_key_cols():
    df = make_df()
    scan = scan_features(df, "fuid", "ftrans_date", "dpd30_3c")
    assert "fuid" not in scan["columns"]
    assert "ftrans_date" not in scan["columns"]
    assert "dpd30_3c" not in scan["columns"]


def test_groups_fold_prefix_and_single_kept():
    df = make_df()
    scan = scan_features(df, "fuid", "ftrans_date", "dpd30_3c")
    groups = {g["group"]: g for g in scan["groups"]}

    # f1_* / f2_* 各 1 列: 不折叠, 展示全名
    assert groups["f1_loan_cnt"]["n"] == 1
    assert not groups["f1_loan_cnt"]["mixed"]
    # 单列不折叠为通配符
    assert "f1_*" not in groups


def test_group_mixed_flag():
    """同前缀下类别不一致 → mixed=True: i_* 组(ambiguous i_30 + feature i_length 风格)。"""
    df = make_df()
    df["i_length_last_v4_loan_consumfin_180day"] = [0.1, 0.2, 0.3]
    scan = scan_features(df, "fuid", "ftrans_date", "dpd30_3c")
    groups = {g["group"]: g for g in scan["groups"]}
    assert groups["i_*"]["mixed"] is True
    assert set(groups["i_*"]["categories"]) == {"ambiguous", "feature"}


def test_render_report_contains_block_sections():
    scan = scan_features(make_df(), "fuid", "ftrans_date", "dpd30_3c")
    report = render_report(scan)
    assert "non_feature 候选" in report
    assert "ambiguous" in report
    assert "feature (默认保留)" in report
    assert "类别计数" in report


def test_min_group_parameter():
    """min_group=1 时单列也折叠; 前缀须纯字母开头的合法 token(f1_* 数字开头不折叠)。"""
    df = make_df()
    scan = scan_features(df, "fuid", "ftrans_date", "dpd30_3c", min_group=1)
    groups = {g["group"] for g in scan["groups"]}
    assert "ym_*" in groups  # 单列纯字母前缀 + min_group=1 折叠
    assert "f1_loan_cnt" in groups  # 数字前缀不与 f1_* 折叠(原型一致)