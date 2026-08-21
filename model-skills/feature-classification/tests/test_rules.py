# -*- coding: utf-8 -*-
"""rules.py 单测: 语义三分类规则库 v0 的逐规则覆盖。"""
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from rules import classify_column, DEFAULT_LABEL_PREFIXES, DEFAULT_NON_FEATURE_PATTERNS


def _s(values):
    """构造测试 Series(数值/非数值)。"""
    return pd.Series(values)


# ---- 用户红线: fpd*/dpd* 标签列, 置顶 ----

def test_label_prefix_redline():
    cat, reason = classify_column("fpd7_sx30", _s([0.0, 1.0, None]))
    assert cat == "non_feature" and "标签列" in reason


def test_label_prefix_redline_dpd():
    cat, reason = classify_column("dpd30_1c", _s([0.0, 1.0, None]))
    assert cat == "non_feature" and "标签列" in reason


def test_label_prefix_redline_overrides_specific_pattern():
    """红线置顶: fpd/dpd 即使近似其他模式也不被抢匹配。"""
    cat, _ = classify_column("dpd30_6c", _s([0.0, 1.0]))
    assert cat == "non_feature"


# ---- 默认规则: 日期/时间戳/订单号/ID/序号/分区列 ----

def test_date_col():
    cat, reason = classify_column("fser_date", _s(["2025-08-01"]))
    assert cat == "non_feature" and "日期" in reason


def test_dt_frame_date_pattern():
    assert classify_column("f_p_date", _s(["20260801"]))[0] == "non_feature"


def test_time_col():
    cat, reason = classify_column("ftrans_time", _s(["12:00:00"]))
    assert cat == "non_feature" and "时间戳" in reason


def test_time_prefix_pattern():
    assert classify_column("time_month", _s([1]))[0] == "non_feature"


def test_order_id():
    cat, reason = classify_column("sx_order_id", _s(["x1"]))
    assert cat == "non_feature" and "订单号" in reason


def test_orderid_suffix():
    assert classify_column("jy_orderid", _s(["x1"]))[0] == "non_feature"


def test_id_suffix():
    assert classify_column("user_id", _s([1]))[0] == "non_feature"


def test_uid_suffix():
    assert classify_column("apply_uid", _s([1]))[0] == "non_feature"


def test_rn_seq_no_suffix():
    assert classify_column("row_rn", _s([1]))[0] == "non_feature"
    assert classify_column("my_seq", _s([1]))[0] == "non_feature"
    assert classify_column("order_no", _s(["x"]))[0] == "non_feature"


def test_f_p_partition():
    assert classify_column("f_p_month", _s(["202608"]))[0] == "non_feature"


# ---- 标识前缀启发式 ----

def test_ident_prefix_binary_numeric_nontfeature():
    cat, reason = classify_column("if_tf", _s([0, 1, 0, 1]))
    assert cat == "non_feature" and "纯标识列" in reason


def test_ident_prefix_non_binary_ambiguous():
    cat, reason = classify_column("is_active", _s([0, 1, 2, 3]))
    assert cat == "ambiguous" and "值域待确认" in reason


def test_ident_prefix_non_numeric_ambiguous():
    cat, _ = classify_column("flag_col", _s(["a", "b"]))
    assert cat == "ambiguous"


# ---- 疑似序号列 / 匿名编码列 ----

def test_fst_last_rn_nonfeature_candidate():
    """fst/last_rn 被序号列规则命中 → non_feature 候选(可能是排名特征, 须用户确认)。"""
    cat, reason = classify_column("fst_rn", _s([1, 2, 3]))
    assert cat == "non_feature" and "序号" in reason
    assert classify_column("last_rn", _s([1, 2, 3]))[0] == "non_feature"


def test_anonymous_encoded_ambiguous():
    cat, reason = classify_column("i_30", _s([0.1, 0.2]))
    assert cat == "ambiguous" and "匿名编码列" in reason


def test_anonymous_encoded_m_series():
    cat, reason = classify_column("m_1221", _s([0.1]))
    assert cat == "ambiguous" and "匿名编码列" in reason


def test_scored_anonymous_kept_ambiguous_not_removed():
    """规则局限: score_80002 被划 ambiguous 而非剔除(安全)。"""
    assert classify_column("score_80002", _s([0.5]))[0] == "ambiguous"


# ---- 默认保留 ----

def test_business_word_feature_kept():
    cat, reason = classify_column("fals_d15_cell_nbank_else_orgnum", _s([0.1, None]))
    assert cat == "feature" and "默认保留" in reason


def test_business_word_i_prefix_kept():
    """同前缀但带业务词的列不归匿名: i_length_last_v4_loan_consumfin_180day。"""
    cat, _ = classify_column("i_length_last_v4_loan_consumfin_180day", _s([0.1]))
    assert cat == "feature"


def test_tag_cols_kept():
    assert classify_column("ym_tag", _s([0, 1]))[0] == "feature"


# ---- 用户自定义规则 / 参数覆盖 ----

def test_extra_pattern_apply():
    cat, reason = classify_column(
        "v12_fpd7_v2", _s([0.1]), extra_patterns=[r"fpd\d+"]
    )
    assert cat == "non_feature" and "用户自定义规则" in reason


def test_custom_label_prefixes_override():
    cat, _ = classify_column(
        "zpd30", _s([0.1]), label_prefixes=("zpd",)
    )
    assert cat == "non_feature"


def test_empty_label_prefixes_disables_redline():
    cat, _ = classify_column(
        "fpd7_sx30", _s([0.0, 1.0]), label_prefixes=()
    )
    # fpd* 不在默认规则里, 变成默认保留
    assert cat == "feature"