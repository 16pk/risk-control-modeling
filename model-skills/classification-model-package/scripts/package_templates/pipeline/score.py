#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定版模型打分: 特征严格校验 + 按 feature_names 重排 + 推理 → score。自包含, 零专家包依赖。

模型加载（与专家包 model-scoring 口径一致, 仅支持主链路产物 lgb/xgb）:
- xgb: model.json → xgboost.Booster（历史路径）
- lgb / xgb(pkl): model.pkl → joblib.load（LGBMClassifier / XGBClassifier, 二维概率取第 1 列）
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_meta(assets_dir) -> dict:
    meta_path = assets_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"[ERROR] 缺少 assets/model_meta.json: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_file(assets_dir, algo: str):
    if algo == "xgb" and (assets_dir / "model.json").exists():
        return assets_dir / "model.json"
    return assets_dir / "model.pkl"


class _XgbScorer:
    """xgb Booster 包装, 暴露 predict_proba(df)。"""

    def __init__(self, booster):
        self._booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        dmat = xgb.DMatrix(X)
        return self._booster.predict(dmat)


def load_predictor(algo: str, assets_dir):
    model_file = _model_file(assets_dir, algo)
    if not model_file.exists():
        raise SystemExit(f"[ERROR] 模型文件不存在: {model_file}")
    if algo == "xgb" and (assets_dir / "model.json").exists():
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(model_file))
        return _XgbScorer(booster)
    if algo in ("lgb", "xgb"):
        try:
            import joblib
            obj = joblib.load(model_file)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"[ERROR] joblib 加载 model.pkl 失败 ({algo}): {e}")
        if hasattr(obj, "predict_proba"):
            return obj
        return _XgbScorer(obj)
    raise SystemExit(f"[ERROR] 未知 algo={algo!r}（本交付包仅支持 lgb/xgb）")


def apply_score(df: pd.DataFrame, feature_names: list, algo: str, assets_dir,
                score_col: str = "score") -> Tuple[pd.DataFrame, str]:
    """特征严格校验 → 重排 → 推理 → 透传非特征列 + score。返回 (输出 df, score 列名)。"""
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise SystemExit(
            f"[ERROR] 输入数据缺失 {len(missing)} 个特征（定版模型 feature_names）:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )
    X = df[feature_names]
    predictor = load_predictor(algo, assets_dir)
    proba = np.asarray(predictor.predict_proba(X), dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]
    proba = proba.ravel()
    if len(proba) != len(df):
        raise SystemExit(f"[ERROR] 推理输出长度 {len(proba)} 与输入 {len(df)} 不一致")
    out = df[[c for c in df.columns if c not in feature_names]].copy()
    out[score_col] = proba
    return out, score_col