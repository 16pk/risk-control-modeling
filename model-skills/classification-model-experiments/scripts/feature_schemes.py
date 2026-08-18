# -*- coding: utf-8 -*-
"""特征方案（算法无关）：all(安全过滤) / importance 95% 截断 / iv-psi 直算 / 对抗剔除。

严格正交（plan §2.2 修改 4）：
  - importance 特征方案 = 取**同样本方案 all 格**（{sample_scheme}-all）实验的
    total_gain 累积 95% 截断；依赖为同算法内 sample_scheme 维度的 DAG。
  - iv-psi 单格直算，无依赖（PSI>0.2 / IV<0.015 / 缺失>0.95，OOT 参与 PSI 统计例外②）。
  - 对抗剔除特征 = 对抗分类器 feature importance top-K（由 adversarial 产出）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  注入 _modelevo-shared（metrics）
from metrics import calc_iv, psi_from_series


def all_features(features: List[str]) -> List[str]:
    """all 特征方案：安全过滤后的全量特征。"""
    return list(features)


def importance_features(features: List[str], importance_df: Optional[pd.DataFrame],
                        pct: float = 95.0) -> List[str]:
    """按 total_gain 累积贡献截断到 pct%（默认 95）。

    importance_df 需含 feature / total_gain 列（algo_factory.feature_importances 输出）。
    仅从给定 features 中选（all 格可能有额外列）。
    """
    if importance_df is None or importance_df.empty or "total_gain" not in importance_df.columns:
        return list(features)
    df = importance_df[importance_df["feature"].isin(features)].copy()
    if df.empty:
        return list(features)
    df = df.sort_values("total_gain", ascending=False)
    total = float(df["total_gain"].sum())
    if total <= 0:
        return list(features)
    cum = np.cumsum(df["total_gain"].to_numpy()) / total * 100.0
    n_keep = int(np.searchsorted(cum, pct, side="right")) + 1
    n_keep = max(1, min(n_keep, len(df)))
    return list(df["feature"].head(n_keep))


def iv_psi_features(dev: pd.DataFrame, oot: pd.DataFrame, features: List[str],
                    label_col: str,
                    psi_threshold: float = 0.2,
                    iv_threshold: float = 0.015,
                    missing_threshold: float = 0.95) -> Tuple[List[str], Dict]:
    """IV-PSI 单格直算筛选。

    规则（plan §2.1 C6 放松阈值）：
      - 剔除 missing_rate >= missing_threshold（缺失>0.95）
      - 剔除 IV < iv_threshold 或 IV=NaN（IV<0.015）
      - 剔除 PSI(dev→oot) > psi_threshold（PSI>0.2，OOT 参与 PSI 统计 = 红线例外②）

    Returns:
        (kept, detail)；detail 含每特征 {missing_rate, iv, psi_oot} 供 manifest 追溯。
    """
    label = pd.to_numeric(dev[label_col], errors="coerce")
    kept: List[str] = []
    detail: Dict[str, Dict] = {}
    for f in features:
        if f not in dev.columns:
            kept.append(f)
            detail[f] = {"missing_column": True}
            continue
        dev_s = pd.to_numeric(dev[f], errors="coerce")
        missing_rate = float(dev_s.isna().mean())
        iv = calc_iv(dev_s, label, n_bins=10)
        psi = None
        if f in oot.columns:
            oot_s = pd.to_numeric(oot[f], errors="coerce")
            psi = psi_from_series(dev_s, oot_s, n_bins=10)
        detail[f] = {
            "missing_rate": round(missing_rate, 6),
            "iv": round(float(iv), 6) if iv is not None else None,
            "psi_oot": round(float(psi), 6) if psi is not None else None,
        }
        if missing_rate >= missing_threshold:
            continue
        if iv is not None and iv < iv_threshold:
            continue
        if psi is not None and psi > psi_threshold:
            continue
        kept.append(f)
    return kept, detail


def adversarial_features(features: List[str], top_k: int,
                         adv_importance: Optional[pd.DataFrame]) -> List[str]:
    """对抗剔除特征：取对抗分类器 top-K 判别特征剔除。"""
    if adv_importance is None or adv_importance.empty or "feature" not in adv_importance.columns:
        return list(features)
    adv_top = list(adv_importance["feature"].head(top_k))
    return [f for f in features if f not in set(adv_top)]