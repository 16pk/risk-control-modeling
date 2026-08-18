# -*- coding: utf-8 -*-
"""算法无关 sklearn 兼容工厂（统一 fit(X,y,sample_weight) / predict_proba / feature_importances_ 接口）。

lgb = LGBMClassifier、xgb = XGBClassifier；未来加 dnn/lr 只扩展本文件与 hyperparams.py。
其余模块一律不感知具体算法。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def build_estimator(algo: str, params: Dict[str, object]):
    """构造 estimator。剥离训练流程占位参数（early_stopping / scale_pos_weight）。

    Args:
        algo: "lgb" | "xgb"
        params: derive_params 输出（含 early_stopping / scale_pos_weight 占位）

    Returns:
        未 fit 的 estimator（LGBMClassifier / XGBClassifier）。
    """
    fit_params = {k: v for k, v in params.items() if k not in ("early_stopping", "scale_pos_weight")}
    if algo == "lgb":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**fit_params)
    if algo == "xgb":
        import xgboost as xgb

        return xgb.XGBClassifier(**fit_params)
    raise ValueError(f"unsupported algo: {algo}")


def feature_importances(estimator, algo: str, feature_names: List[str]) -> pd.DataFrame:
    """提取特征重要性，统一为 (feature, total_gain, split_count) DataFrame。

    - lgb: booster_.feature_importance(gain) / (split)
    - xgb: booster.get_score(importance_type="gain") / ("weight")，缺失填 0
    """
    import numpy as np

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

    df = pd.DataFrame({
        "feature": feature_names,
        "total_gain": total_gain,
        "split_count": split_count,
    })
    s = float(df["total_gain"].sum())
    df["gain_pct"] = (df["total_gain"] / s * 100.0).round(4) if s > 0 else 0.0
    return df.sort_values("total_gain", ascending=False).reset_index(drop=True)


def predict_proba(estimator, X) -> object:
    """统一 predict_proba 接口，返回违约概率列（shape=(n,)，[0,1]）。"""
    import numpy as np

    p = np.asarray(estimator.predict_proba(X), dtype=float)
    return p[:, 1] if p.ndim == 2 else p.ravel()


def fit_model(estimator, algo: str, X_train, y_train, X_val, y_val,
              sample_weight=None, early_stopping: int = 100, seed: int = 42):
    """统一 fit 接口（sklearn 兼容，val 早停）。

    - lgb：fit 用 callbacks=[early_stopping(early_stopping)]（构造器 verbosity=-1）
    - xgb（3.x sklearn API）：早停参数走构造器 set_params(early_stopping_rounds)，
      fit 只传 eval_set + verbose
    """
    import numpy as np

    y_train = np.asarray(y_train, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=float).ravel()
    if algo == "lgb":
        import lightgbm as lgb

        callbacks = [lgb.early_stopping(early_stopping, verbose=False),
                     lgb.log_evaluation(0)]
        estimator.fit(X_train, y_train, sample_weight=sample_weight,
                      eval_set=[(X_val, y_val)],
                      eval_metric="auc",
                      callbacks=callbacks)
    elif algo == "xgb":
        try:
            estimator.set_params(early_stopping_rounds=early_stopping,
                                 eval_metric="auc")
        except Exception:
            pass  # 旧版或参数不可用时退化为全量训练
        estimator.fit(X_train, y_train, sample_weight=sample_weight,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
    else:
        raise ValueError(f"unsupported algo: {algo}")
    return estimator
