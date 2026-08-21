# -*- coding: utf-8 -*-
"""diagnose_winner.py / recommend_winner.py：五状态诊断、lgb/xgb 策略表、Optuna 锚点调整。"""
import numpy as np
import pytest

from diagnose_winner import (EARLY_STOP_RATIO, GAP_OVERFIT, GAP_UNDERFIT,
                             PSI_WARN, TRAIN_LOW_AUC, Diagnosis, diagnose_winner)
from recommend_winner import (BOUNDS_LGB, BOUNDS_XGB, adjust_optuna_anchors,
                              recommend_params)


# ---------------- diagnose_winner 五状态 ----------------

def _metrics(train_auc=0.85, val_auc=0.80, oot_auc=0.81):
    return {"train": {"auc": train_auc}, "val": {"auc": val_auc},
            "oot": {"auc": oot_auc}}


def test_well_fit():
    d = diagnose_winner(_metrics(), {"n_estimators": 1000}, 300, "lgb")
    assert d.status == "well_fit"
    assert d.signals["train_auc"] == 0.85
    assert d.signals["train_oot_gap"] == pytest.approx(0.04)
    assert d.as_dict()["status"] == "well_fit"


def test_overfit():
    d = diagnose_winner(_metrics(train_auc=0.95, oot_auc=0.80), {"n_estimators": 1000}, 300, "xgb")
    assert d.status == "overfit"
    assert d.signals["train_oot_gap"] == pytest.approx(0.15)


def test_underfit_by_gap():
    d = diagnose_winner(_metrics(train_auc=0.80, oot_auc=0.80), {"n_estimators": 1000}, 300, "lgb")
    assert d.status == "underfit"  # gap < GAP_UNDERFIT


def test_underfit_by_train_low():
    d = diagnose_winner(_metrics(train_auc=0.65, oot_auc=0.62), {"n_estimators": 1000}, 300, "lgb")
    assert d.status == "underfit"  # train_auc < TRAIN_LOW_AUC


def test_underconverged_xgb():
    d = diagnose_winner(_metrics(), {"n_estimators": 1000}, 980, "xgb")
    assert d.status == "underconverged"  # 980/1000 = 0.98 >= 0.95
    assert d.signals["early_stop_ratio"] == pytest.approx(0.98)


def test_underconverged_lgb():
    d = diagnose_winner(_metrics(), {"n_estimators": 1000}, 960, "lgb")
    assert d.status == "underconverged"


def test_unstable_psi():
    d = diagnose_winner(_metrics(), {"n_estimators": 1000}, 300, "lgb", new_psi=0.25)
    assert d.status == "unstable_psi"
    assert d.signals["new_psi"] == pytest.approx(0.25)


def test_priority_overfit_underconverged():
    """优先级 overfit > underconverged"""
    d = diagnose_winner(_metrics(train_auc=0.96, oot_auc=0.80),
                        {"n_estimators": 1000}, 990, "lgb")
    assert d.status == "overfit"  # overfit 先命中


def test_no_oot_metrics_ok():
    """OOT 缺失时诊断退化（无 gap 信号，不崩）。"""
    d = diagnose_winner({"train": {"auc": 0.85}, "val": {"auc": 0.80}},
                        {"n_estimators": 1000}, 300, "lgb")
    assert d.status == "well_fit"


def test_unsupported_algo():
    with pytest.raises(ValueError):
        diagnose_winner(_metrics(), {"n_estimators": 1000}, 300, "dnn")


# ---------------- recommend_params（策略表 + BOUNDS） ----------------

def test_recommend_lgb_underfit():
    base = {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.04,
            "n_estimators": 1000, "bagging_fraction": 0.6, "feature_fraction": 0.8}
    d = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend_params(base, d, "lgb")
    assert r["num_leaves"] == 46  # 31*1.5=46.5→47? 实际 round(46.5)=47
    assert r["min_child_samples"] == 50  # 100/2
    assert r["learning_rate"] == 0.02
    assert r["n_estimators"] == 1500


def test_recommend_lgb_underfit_bounds():
    base = {"num_leaves": 200, "min_child_samples": 5, "learning_rate": 0.04,
            "n_estimators": 1500, "bagging_fraction": 0.6, "feature_fraction": 0.8}
    d = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend_params(base, d, "lgb")
    assert r["num_leaves"] == 255  # 200*1.5=300 → clip 到 255
    assert r["min_child_samples"] == 5  # 5/2=2.5 → max(5, ...)=5
    assert r["n_estimators"] == 2000  # 1500*1.5=2250 → clip 2000


def test_recommend_lgb_overfit():
    base = {"num_leaves": 31, "min_child_samples": 100, "reg_alpha": 0.0, "reg_lambda": 1.0}
    d = Diagnosis(status="overfit", reasons=[], signals={})
    r = recommend_params(base, d, "lgb")
    assert r["num_leaves"] == 25  # round(31*0.8)=25
    assert r["min_child_samples"] == 150
    assert r["reg_alpha"] == 0.0  # 0.0*2


def test_recommend_xgb_overfit():
    base = {"max_depth": 6, "min_child_weight": 10, "reg_lambda": 1.0}
    d = Diagnosis(status="overfit", reasons=[], signals={})
    r = recommend_params(base, d, "xgb")
    assert r["max_depth"] == 5
    assert r["reg_lambda"] == 2.0
    assert r["min_child_weight"] == 20


def test_recommend_underconverged():
    base = {"n_estimators": 1200}
    d = Diagnosis(status="underconverged", reasons=[], signals={})
    assert recommend_params(base, d, "lgb")["n_estimators"] == 2000  # clip 2400→2000
    assert recommend_params(base, d, "xgb")["n_estimators"] == 2000


def test_recommend_unstable_psi():
    base = {"subsample": 0.6, "colsample_bytree": 0.6}
    d = Diagnosis(status="unstable_psi", reasons=[], signals={})
    r = recommend_params(base, d, "xgb")
    assert r["subsample"] == 0.5
    assert r["colsample_bytree"] == 0.5
    base2 = {"bagging_fraction": 0.6, "feature_fraction": 0.6}
    r2 = recommend_params(base2, d, "lgb")
    assert r2["bagging_fraction"] == 0.5
    assert r2["feature_fraction"] == 0.5


def test_recommend_well_fit_no_change():
    base = {"num_leaves": 31, "learning_rate": 0.04}
    d = Diagnosis(status="well_fit", reasons=[], signals={})
    r = recommend_params(base, d, "lgb")
    assert r == base


# ---------------- adjust_optuna_anchors ----------------

def _anchors_lgb():
    return {"learning_rate": (0.02, 0.06), "num_leaves": (23, 39),
            "min_child_samples": (12, 28), "feature_fraction": (0.5, 0.9),
            "bagging_fraction": (0.5, 0.9), "n_estimators": 1000, "early_stopping": 100}


def _anchors_xgb():
    return {"learning_rate": (0.02, 0.06), "max_depth": (3, 6),
            "min_child_weight": (1e-4, 1e-2), "colsample_bytree": (0.5, 0.9),
            "subsample": (0.5, 0.9), "n_estimators": 1000, "early_stopping": 100}


def test_adjust_overfit_lgb():
    a = adjust_optuna_anchors(_anchors_lgb(), Diagnosis(status="overfit"), "lgb")
    assert a["num_leaves"] == (23, 24)  # 上限收窄到 24
    # 相对化（v2.5）：下界 ×1.5，不再绝对区间收窄 (5,60)（防样本量锚点脱节静默失效）
    assert a["min_child_samples"] == (18, 28)  # 12*1.5=18
    assert a["bagging_fraction"] == (0.5, 0.8)  # 采样率上界收窄


def test_adjust_overfit_xgb():
    a = adjust_optuna_anchors(_anchors_xgb(), Diagnosis(status="overfit"), "xgb")
    assert a["max_depth"] == (3, 5)  # 上限收窄到 5
    assert a["subsample"] == (0.5, 0.8)
    # 相对化（v2.5）：下界 ×10（1e-3 量级 → 1e-2 量级），上限保持
    assert a["min_child_weight"] == (1e-3, 1e-2)


def test_adjust_overfit_lgb_anchor_after_clip():
    """治本①+②联动：锚点 clip 进经验域后收窄仍有交集（26万样本场景）。"""
    # 样本量驱动锚点 (120, 280)（0.002*26万=520 → clip 200 → (120,280)），overfit 下界相对化 ×1.5
    a = adjust_optuna_anchors(
        {"learning_rate": (0.02, 0.06), "num_leaves": (23, 39),
         "min_child_samples": (120, 280), "feature_fraction": (0.5, 0.9),
         "bagging_fraction": (0.5, 0.9), "n_estimators": 1000, "early_stopping": 100},
        Diagnosis(status="overfit"), "lgb")
    assert a["min_child_samples"] == (180, 280)  # 120*1.5=180，仍有交集且非空
    assert a["min_child_samples"][0] <= a["min_child_samples"][1]


def test_adjust_underfit():
    a = adjust_optuna_anchors(_anchors_lgb(), Diagnosis(status="underfit"), "lgb")
    assert a["num_leaves"] == (23, 48)  # 上限放宽到 48
    assert a["learning_rate"] == (0.02, 0.06)  # (0.02,0.06) ∩ (0.005,0.1) 不变


def test_adjust_underconverged():
    a = adjust_optuna_anchors(_anchors_lgb(), Diagnosis(status="underconverged"), "lgb")
    assert a["n_estimators"] == 1500  # 1000*1.5
    a_xgb = adjust_optuna_anchors(_anchors_xgb(), Diagnosis(status="underconverged"), "xgb")
    assert a_xgb["n_estimators"] == 1500


def test_adjust_unstable_psi():
    a = adjust_optuna_anchors(_anchors_lgb(), Diagnosis(status="unstable_psi"), "lgb")
    assert a["bagging_fraction"] == (0.5, 0.7)
    assert a["feature_fraction"] == (0.5, 0.7)
    a_xgb = adjust_optuna_anchors(_anchors_xgb(), Diagnosis(status="unstable_psi"), "xgb")
    assert a_xgb["subsample"] == (0.5, 0.7)
    assert a_xgb["colsample_bytree"] == (0.5, 0.7)


def test_adjust_well_fit_no_change():
    a = adjust_optuna_anchors(_anchors_lgb(), Diagnosis(status="well_fit"), "lgb")
    assert a == _anchors_lgb()