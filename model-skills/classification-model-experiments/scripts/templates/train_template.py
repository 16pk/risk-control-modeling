#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分类模型训练权威模板（sklearn 风格、高参数化）——供 experiments 每格快照复制。

模板契约（plan §5.2.5）：
  - 权威模板 = scripts/templates/train_template.py；每格实验开始时快照复制进
    <exp_dir>/scripts/train.py 并记 code_sha256 + template_version 到 manifest。
  - **自包含**：本文件不依赖 skill 目录内其他模块（bootstrap/algo_factory/hyperparams 等），
    复现 = 重跑实验目录 scripts/train.py 即可完整重训。
  - 默认全格同代码（可比）；AI 需定制某格时逐格 fork 修改（manifest 记 code_modified=true）。

调用接口（由 run_single_experiment.py 消费）：
  train(X_train, y_train, w_train, X_val, y_val, X_oot, y_oot, algo, params, feature_names, seed)

纪律：
  - 统一 fit(X, y, sample_weight)；val 早停（100 轮）；OOT 仅评估（禁早停/禁进训练/禁统计）
  - scale_pos_weight = "auto" → 按训练段 neg/pos 自动算
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# M/S 公式推导超参（与 hyperparams.py 同源；模板自包含故内嵌一份）
# ---------------------------------------------------------------------------
def derive_params(algo: str, n_samples: int, n_features: int) -> Dict:
    """按用户推荐公式计算基线超参（plan §5.2.6）。"""
    n_features = max(int(n_features), 1)
    n_samples = max(int(n_samples), 1)
    # num_leaves 必须 > 1（LightGBM 参数合法域）；col_sample 合法域 (0,1]
    num_leaves = max(2, min(31, int(2 ** (n_features / 10))))
    col_factor = min(max((num_leaves * 2) / n_features, 0.5), 1.0)
    base: Dict = {
        "metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.04,
        "seed": 42,
        "scale_pos_weight": "auto",
        "early_stopping": 100,
    }
    if algo == "lgb":
        base.update({
            "objective": "binary",  # lgb objective
            "num_leaves": num_leaves,
            "max_depth": -1,
            "verbosity": -1,  # lgb 允许 -1（静默）
            "min_child_samples": max(20, int(0.002 * n_samples)),
            "min_sum_hessian_in_leaf": 1e-3,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "feature_fraction": col_factor,
        })
    elif algo == "xgb":
        base.update({
            "objective": "binary:logistic",  # xgb objective 命名不同
            "max_depth": 4,
            "min_child_weight": 1e-3,
            "verbosity": 0,  # xgb 合法域 [0,3]
            "subsample": 0.6,
            "colsample_bytree": col_factor,
            "tree_method": "hist",
        })
    else:
        raise ValueError(f"unsupported algo: {algo}")
    return base


# ---------------------------------------------------------------------------
# 统一 estimator 接口（算法无关）
# ---------------------------------------------------------------------------
def build_estimator(algo: str, params: Dict):
    fit_params = {k: v for k, v in params.items() if k not in ("early_stopping", "scale_pos_weight")}
    if algo == "lgb":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**fit_params)
    if algo == "xgb":
        import xgboost as xgb

        return xgb.XGBClassifier(**fit_params)
    raise ValueError(f"unsupported algo: {algo}")


def predict_proba(estimator, X) -> np.ndarray:
    p = np.asarray(estimator.predict_proba(X), dtype=float)
    return p[:, 1] if p.ndim == 2 else p.ravel()


def feature_importances(estimator, algo: str, feature_names: List[str]) -> pd.DataFrame:
    """统一特征重要性 DataFrame（feature, total_gain, split_count, gain_pct）。"""
    n = len(feature_names)
    total_gain = np.zeros(n, dtype=float)
    split_count = np.zeros(n, dtype=int)
    if algo == "lgb":
        booster = estimator.booster_
        names = list(booster.feature_name())
        gain_map = dict(zip(names, booster.feature_importance(importance_type="gain")))
        split_map = dict(zip(names, booster.feature_importance(importance_type="split")))
    elif algo == "xgb":
        booster = estimator.get_booster()
        gain_map = booster.get_score(importance_type="gain")
        split_map = booster.get_score(importance_type="weight")
    else:
        raise ValueError(f"unsupported algo: {algo}")
    for i, f in enumerate(feature_names):
        total_gain[i] = float(gain_map.get(f, 0.0))
        split_count[i] = int(split_map.get(f, 0))
    df = pd.DataFrame({"feature": feature_names, "total_gain": total_gain,
                       "split_count": split_count})
    s = float(df["total_gain"].sum())
    df["gain_pct"] = (df["total_gain"] / s * 100.0).round(4) if s > 0 else 0.0
    return df.sort_values("total_gain", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 训练主函数
# ---------------------------------------------------------------------------
def _resolve_scale_pos_weight(params: Dict, y_train) -> float:
    spw = params.get("scale_pos_weight", "auto")
    if isinstance(spw, (int, float)) and spw != "auto":
        return float(spw)
    y = np.asarray(y_train, dtype=float)
    pos = float((y > 0).sum())
    neg = float((y <= 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


def fit_model(estimator, algo: str, X_train, y_train, X_val, y_val,
              sample_weight=None, early_stopping: int = 100):
    """统一 fit（sklearn 兼容，val 早停；OOT 不参与）。"""
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=float).ravel()
    if algo == "lgb":
        import lightgbm as lgb

        estimator.fit(X_train, y_train, sample_weight=sample_weight,
                      eval_set=[(X_val, y_val)],
                      eval_metric="auc",
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                                 lgb.log_evaluation(0)])
    elif algo == "xgb":
        try:
            estimator.set_params(early_stopping_rounds=early_stopping, eval_metric="auc")
        except Exception:
            pass
        estimator.fit(X_train, y_train, sample_weight=sample_weight,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
    else:
        raise ValueError(f"unsupported algo: {algo}")
    return estimator


def train(
    X_train: pd.DataFrame,
    y_train,
    w_train: Optional[np.ndarray],
    X_val: pd.DataFrame,
    y_val,
    X_oot: pd.DataFrame,
    y_oot,
    algo: str,
    params: Dict,
    feature_names: List[str],
    seed: int = 42,
) -> Dict:
    """训练并返回结果 dict（model / preds / importance / best_iter / train_time 等）。

    OOT 纪律：X_oot / y_oot 仅用于最终评估，不参与拟合与早停。
    """
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=float).ravel()
    y_oot = np.asarray(y_oot, dtype=float).ravel()

    fit_params = dict(params)
    early_stopping = int(fit_params.pop("early_stopping", 100))
    fit_params["scale_pos_weight"] = _resolve_scale_pos_weight(fit_params, y_train)

    model = build_estimator(algo, fit_params)

    t0 = time.time()
    fit_model(model, algo, X_train, y_train, X_val, y_val,
              sample_weight=w_train, early_stopping=early_stopping)
    train_time = time.time() - t0

    if algo == "lgb":
        best_iter = int(getattr(model, "best_iteration_", model.n_estimators))
        early_stopped = bool(getattr(model, "best_iteration_", None) is not None)
    elif algo == "xgb":
        best_iter = int(getattr(model, "best_iteration", model.n_estimators))
        early_stopped = bool(getattr(model, "best_iteration", None) is not None)
    else:
        best_iter = model.n_estimators
        early_stopped = False

    preds = {
        "train": predict_proba(model, X_train),
        "val": predict_proba(model, X_val),
        "oot": predict_proba(model, X_oot),
    }
    importance = feature_importances(model, algo, feature_names)

    return {
        "model": model,
        "preds": preds,
        "importance": importance,
        "best_iter": best_iter,
        "early_stopped": early_stopped,
        "train_time": train_time,
        "scale_pos_weight": float(fit_params["scale_pos_weight"]),
        "algo": algo,
        "params": fit_params,
        "feature_names": list(feature_names),
    }


if __name__ == "__main__":
    sys.stderr.write("[train_template] 模板不可直接执行；由 run_single_experiment 调用。\n")
    sys.exit(1)