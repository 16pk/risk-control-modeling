#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FICO 转分（纯应用模式）: 概率分 → 校准 → 标准分 bscore。自包含, 零专家包依赖。

打包时 session 存在 fico/coef.json 才包含本模块与资产; 运行期不拟合,
直接吃 assets/coef.json 的 coef/intc 对 score 转 bscore（生产数据可能无 label）。

公式（与专家包 score-to-fico 口径一致）:
  odds          = ln(p/(1-p))                                 # 概率裁剪 1e-6
  logistic_prob = sigmoid(coef * odds + intc)
  bscore        = 400 - 35/ln2 * ln(logistic_prob/(1-logistic_prob))   # 约 [400,780], 分高险低
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

BSCORE_SANE_MIN, BSCORE_SANE_MAX = 400.0, 780.0  # 合理区间, 显著越界仅 WARN
PROB_EPS = 1e-6


def load_coef(assets_dir) -> Optional[dict]:
    """读 assets/coef.json; 不存在 → None（不含 FICO 模块）。"""
    p = assets_dir / "coef.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def calc_odds(prob) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p / (1 - p))


def logistic_prob(odds, coef, intc) -> np.ndarray:
    z = np.clip(coef * np.asarray(odds, dtype=float) + intc, -30, 30)
    return np.exp(z) / (1 + np.exp(z))


def calc_bscore(logistic_prob) -> np.ndarray:
    p = np.clip(np.asarray(logistic_prob, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return 400.0 - 35.0 / np.log(2.0) * np.log(p / (1 - p))


def apply_fico(df: pd.DataFrame, prob_col: str, coef: dict) -> tuple[pd.DataFrame, dict]:
    """对 score 列做 校准 + 转分, 同表追加 odds/logistic_prob/bscore; 返回 (df, fico-summary)。"""
    c = float(coef.get("coef"))
    ic = float(coef.get("intc"))
    if c <= 0:
        print("[WARN] coef<=0: 概率与真实逾期方向相反或量级异常, 请检查概率列是否为违约概率")
    odds = calc_odds(df[prob_col].values)
    lprob = logistic_prob(odds, c, ic)
    bscore = calc_bscore(lprob)
    out = df.copy()
    out["odds"] = odds
    out["logistic_prob"] = lprob
    out["bscore"] = bscore
    # 越界仅 WARN 不中止
    bad = int((~np.isfinite(bscore)).sum())
    lo = int((bscore < BSCORE_SANE_MIN).sum())
    hi = int((bscore > BSCORE_SANE_MAX).sum())
    if bad:
        print(f"[WARN] bscore 非有限值 {bad} 个")
    if lo or hi:
        print(f"[WARN] bscore 越界 {lo} 个(<{BSCORE_SANE_MIN}) / {hi} 个(>{BSCORE_SANE_MAX}) — "
              f"正常区间 [{BSCORE_SANE_MIN}, {BSCORE_SANE_MAX}]")
    summary = {
        "method": "LR_calibration(应用) + FICO_mapping",
        "params": {"coef": c, "intc": ic},
        "range": "[400, 780] (分高险低)",
        "n": int(len(out)),
        "bscore_min": round(float(bscore.min()), 2),
        "bscore_max": round(float(bscore.max()), 2),
        "bscore_mean": round(float(bscore.mean()), 2),
        "n_out_of_range": {"below": lo, "above": hi, "non_finite": bad},
    }
    return out, summary