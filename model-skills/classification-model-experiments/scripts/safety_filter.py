# -*- coding: utf-8 -*-
"""安全过滤（内聚本模块，不 import training 的 boundary_filter）。

只做安全过滤，不做 IV/PSI 指标筛选（训练过程不通过 IV/PSI 筛特征）：
  1. 常量特征（nunique <= 1 或全 NaN）
  2. 泄漏特征（与 label 完全一致 / 相关系数 >= leak_threshold）
  3. ID 特征（id_col 精确排除）
  4. 全缺失特征（缺失率 >= missing_threshold，默认 0.95）
  5. 数据安全红线（复用 config_io.check_sensitive 拦截身份证/手机号列名）

数据直算（filter_boundary_features_from_df），语义与 training 的 boundary_filter 对齐但独立实现，
禁跨 skill import。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def filter_boundary_features(
    df: pd.DataFrame,
    features: List[str],
    id_col: Optional[str] = "fuid",
    label_col: Optional[str] = None,
    missing_threshold: float = 0.95,
    leak_threshold: float = 0.99,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """安全过滤，返回 (kept_features, dropped)。

    dropped 为 (feature, reason) 列表，reason ∈ {id_col, missing_column, sensitive_column,
    constant, high_missing, leak_equal, leak_high_corr}。
    """
    kept: List[str] = []
    dropped: List[Tuple[str, str]] = []
    label = None
    if label_col and label_col in df.columns:
        label = pd.to_numeric(df[label_col], errors="coerce")

    id_set = {id_col} if id_col else set()

    for f in features:
        if f in id_set:
            dropped.append((f, "id_col"))
            continue
        if f not in df.columns:
            dropped.append((f, "missing_column"))
            continue
        # 数据安全红线：列名不得为身份证/手机号模式（config_io.check_sensitive 兜底）
        try:
            from config_io import check_sensitive

            check_sensitive(f)
        except Exception:
            dropped.append((f, "sensitive_column"))
            continue

        series = pd.to_numeric(df[f], errors="coerce")
        valid = series.dropna()
        # 1) 常量：有效值唯一 <= 1 或全 NaN
        if len(valid) == 0 or valid.nunique() <= 1:
            dropped.append((f, "constant"))
            continue
        # 4) 全缺失：缺失率 >= 阈值
        missing_rate = float(series.isna().mean()) if len(series) else 1.0
        if missing_rate >= missing_threshold:
            dropped.append((f, "high_missing"))
            continue
        # 2) 泄漏：与 label 完全一致 / 相关 >= 阈值
        if label is not None:
            mask = series.notna() & label.notna()
            if mask.sum() > 0:
                lv = series[mask]
                lv2 = label[mask]
                if len(lv2.unique()) == 2 and (lv == lv2).all():
                    dropped.append((f, "leak_equal"))
                    continue
                if lv.nunique() == 1:
                    continue
                try:
                    corr = float(np.corrcoef(lv, lv2)[0, 1])
                    if not np.isnan(corr) and abs(corr) >= leak_threshold:
                        dropped.append((f, "leak_high_corr"))
                        continue
                except Exception:
                    pass
        kept.append(f)
    return kept, dropped


def filter_boundary_features_from_df(
    df: pd.DataFrame,
    id_col: Optional[str] = "fuid",
    label_col: Optional[str] = None,
    missing_threshold: float = 0.95,
    leak_threshold: float = 0.99,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """数据直算入口：features 取自 df 中除 id/label/日期外的全部列。"""
    exclude = {c for c in (id_col, label_col) if c}
    features = [c for c in df.columns if c not in exclude]
    return filter_boundary_features(df, features, id_col=id_col, label_col=label_col,
                                    missing_threshold=missing_threshold, leak_threshold=leak_threshold)