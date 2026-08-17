"""统一指标库 metrics.py 测试（v2.0 新增）。

验证 AUC/KS/Gini/PSI/IV/分类指标/分桶 与既有各 skill 口径一致。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from metrics import (
    calc_auc,
    calc_gini,
    calc_iv,
    calc_ks,
    calc_psi,
    classification_metrics,
    decile_buckets,
    psi_from_series,
)


def test_calc_auc_perfect_separation():
    """完美区分：正样本分数全高于负样本 → AUC=1.0。"""
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert calc_auc(scores, labels) == pytest.approx(1.0)


def test_calc_auc_random():
    """随机分数 → AUC≈0.5。"""
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, size=2000)
    assert calc_auc(scores, labels) == pytest.approx(0.5, abs=0.05)


def test_calc_auc_nan_filtered():
    """NaN 自动剔除。"""
    scores = np.array([0.1, np.nan, 0.9, 0.95, np.nan])
    labels = np.array([0, 1, 1, 1, 0])
    # 剔除 NaN 后: 正样本(0.9,0.95) 全高于负样本(0.1) → AUC=1.0
    assert calc_auc(scores, labels) == pytest.approx(1.0)


def test_calc_auc_single_class_returns_none():
    assert calc_auc([0.1, 0.2], [0, 0]) is None


def test_calc_ks_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert calc_ks(scores, labels) == pytest.approx(1.0)


def test_calc_ks_direction_invariant():
    """KS 与分数方向无关（正反序 KS 相同）。"""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    labels = np.array([0, 1, 0, 1, 0, 1, 1])
    assert calc_ks(scores, labels) == pytest.approx(
        calc_ks(-scores, labels), abs=1e-6
    )


def test_calc_gini_from_auc():
    auc = calc_auc([0.1, 0.2, 0.9], [0, 0, 1])
    assert calc_gini([0.1, 0.2, 0.9], [0, 0, 1]) == pytest.approx(abs(2 * auc - 1))


def test_calc_psi_identical_zero():
    """完全同分布 → PSI=0。"""
    x = np.array([0.2, 0.3, 0.5])
    assert calc_psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_calc_psi_shifted_positive():
    """分布漂移 → PSI>0。"""
    base = np.array([0.7, 0.2, 0.1])
    actual = np.array([0.1, 0.2, 0.7])
    assert calc_psi(actual, base) > 0.1


def test_psi_from_series_distinct():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 500)
    actual = rng.normal(3, 1, 500)
    psi = psi_from_series(base, actual, n_bins=10)
    assert psi is not None and psi > 0.1


def test_psi_from_series_too_small_none():
    assert psi_from_series([1.0, 2.0], [1.0, 2.0], n_bins=10) is None


def test_calc_iv_predictive():
    """强预测力特征 → IV 明显大于 0。"""
    rng = np.random.default_rng(2)
    x = rng.normal(size=1000)
    y = (x > 0).astype(int)  # 完全可分的单调关系
    iv = calc_iv(x, y)
    assert iv is not None and iv > 0.5


def test_calc_iv_noise_near_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    y = rng.integers(0, 2, size=2000)
    iv = calc_iv(x, y)
    assert iv is not None and iv < 0.2


def test_calc_iv_single_class_none():
    assert calc_iv([1.0, 2.0, 3.0], [0, 0, 0]) is None


def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.6])
    m = classification_metrics(y_true, y_score, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_decile_buckets_10_rows():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "score": rng.random(1000),
        "label": rng.integers(0, 2, size=1000),
    })
    buckets = decile_buckets(df, "score", n_bins=10)
    assert len(buckets) == 10
    # decile 10 = 最高分档，score_min 应大于 decile 1
    assert buckets[0]["decile"] == 10
    assert buckets[0]["score_min"] >= buckets[-1]["score_max"]
    assert buckets[0]["count"] == pytest.approx(100, abs=5)


def test_decile_buckets_biz_cols():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "score": rng.random(500),
        "label": rng.integers(0, 2, size=500),
        "age": rng.integers(18, 80, size=500).astype(float),
    })
    buckets = decile_buckets(df, "score", biz_cols=["age"], n_bins=10)
    assert "age" in buckets[0]
