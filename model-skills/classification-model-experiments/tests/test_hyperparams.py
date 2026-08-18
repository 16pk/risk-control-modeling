# -*- coding: utf-8 -*-
"""hyperparams.py：M/S 公式推导与边界。"""
import numpy as np
import pytest

from hyperparams import derive_params, optuna_anchors


def test_lgb_params_formula():
    p = derive_params("lgb", n_samples=100_000, n_features=50)
    assert p["n_estimators"] == 1000
    assert p["learning_rate"] == 0.04
    assert p["seed"] == 42
    assert p["early_stopping"] == 100
    # num_leaves = min(31, 2^(50/10)) = min(31, 32) = 31
    assert p["num_leaves"] == 31
    assert p["max_depth"] == -1
    # min_child_samples = max(20, 0.002*100000) = 200
    assert p["min_child_samples"] == 200
    assert p["min_sum_hessian_in_leaf"] == 1e-3
    # colsample = clip(max((31*2)/50, 0.5), 1.0) = clip(1.24, 0.5, 1.0) = 1.0
    assert abs(p["feature_fraction"] - 1.0) < 1e-9
    assert p["bagging_fraction"] == 0.6


def test_lgb_num_leaves_cap():
    # S 很大 → 2^(S/10) 巨大 → cap 31
    p = derive_params("lgb", n_samples=1000, n_features=200)
    assert p["num_leaves"] == 31


def test_lgb_min_child_samples_floor():
    p = derive_params("lgb", n_samples=100, n_features=10)
    assert p["min_child_samples"] == 20  # max(20, 0.002*100=0.2)


def test_xgb_params():
    p = derive_params("xgb", n_samples=50_000, n_features=30)
    assert p["max_depth"] == 4
    assert p["min_child_weight"] == 1e-3
    assert p["subsample"] == 0.6
    assert p["tree_method"] == "hist"
    num_leaves = min(31, int(2 ** (30 / 10)))  # min(31, 8) = 8
    expected_colsample = max((num_leaves * 2) / 30, 0.5)
    assert abs(p["colsample_bytree"] - expected_colsample) < 1e-9


def test_unsupported_algo():
    with pytest.raises(ValueError):
        derive_params("dnn", 1000, 10)


def test_optuna_anchors_lgb():
    p = derive_params("lgb", 100_000, 50)
    sp = optuna_anchors("lgb", p)
    assert sp["learning_rate"] == (0.04 * 0.5, 0.04 * 1.5)
    assert sp["num_leaves"] == (31 - 8, 31 + 8)
    assert sp["min_child_samples"] == (int(200 * 0.6), int(200 * 1.4))


def test_optuna_anchors_xgb():
    p = derive_params("xgb", 100_000, 50)
    sp = optuna_anchors("xgb", p)
    assert sp["max_depth"] == (3, 6)
    assert sp["min_child_weight"] == (1e-4, 1e-2)