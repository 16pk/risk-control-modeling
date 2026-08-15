# -*- coding: utf-8 -*-
"""derive_feature_list 派生特征清单单测(纯函数 derive_features / filter_by_list)。"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from derive_feature_list import derive_features, filter_by_list


def test_derive_excludes_non_feature_cols():
    """排除 id/dt/label 列后只剩特征列, 保序。"""
    all_cols = ["user_no", "pday", "label", "f0", "f1", "f2"]
    feats = derive_features(all_cols, ["user_no", "pday", "label"])
    assert feats == ["f0", "f1", "f2"]


def test_derive_keeps_order_and_dedup():
    all_cols = ["f0", "f1", "f0", "f2"]
    feats = derive_features(all_cols, [])
    assert feats == ["f0", "f1", "f2"]


def test_derive_empty_exclude():
    assert derive_features(["a", "b", "c"], []) == ["a", "b", "c"]


def test_derive_exclude_missing_cols_noop():
    assert derive_features(["f0", "f1"], ["nope", "user_no"]) == ["f0", "f1"]


def test_derive_exclude_all():
    assert derive_features(["user_no", "pday"], ["user_no", "pday"]) == []


# ---- filter_by_list 单测 ----

def test_filter_keeps_intersection_in_allow_list_order():
    derived = ["f2", "f0", "f1"]
    allow = ["f0", "f1", "f2"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1", "f2"]
    assert missing == []


def test_filter_drops_allow_features_not_in_sample():
    derived = ["f0", "f1"]
    allow = ["f0", "f1", "f2", "f3"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1"]
    assert missing == ["f2", "f3"]


def test_filter_drops_derived_features_not_in_allow():
    derived = ["f0", "f1", "extra_in_sample"]
    allow = ["f0", "f1"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1"]
    assert missing == []


def test_filter_allow_dedup():
    derived = ["f0", "f1", "f2"]
    allow = ["f0", "f1", "f0", "f2", "f1"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1", "f2"]
    assert missing == []


def test_filter_empty_allow():
    derived = ["f0", "f1"]
    kept, missing = filter_by_list(derived, [])
    assert kept == []
    assert missing == []


def test_filter_no_overlap():
    derived = ["a", "b"]
    allow = ["x", "y"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == []
    assert missing == ["x", "y"]
