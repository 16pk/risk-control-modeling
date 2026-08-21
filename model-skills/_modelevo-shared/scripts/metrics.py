# -*- coding: utf-8 -*-
"""统一指标计算库（v2.1 精简重构新增）。

统一全仓库 PSI / KS / AUC / Gini / IV / 分类指标 / 分桶排序性 的实现，
供 credit-data-analysis / classification-model-experiments / credit-model-report
等 skill 经 `_bootstrap.py` 注入复用。

口径说明（与既有各 skill 对齐）：
- AUC：Mann-Whitney U（秩和）法，并列用平均秩，与 sklearn.roc_auc_score 一致；
  返回方向性 AUC（分数越高越接近 label=1）。
- KS：行业标准口径 KS = max|累计坏样本率 − 累计好样本率|（等价 max|TPR−FPR|），
  只在每个不同分数处评估（并列不拆分），与 sklearn.roc_curve 一致。
- Gini：2·AUC − 1（取绝对值，避免方向性）。
- PSI：Σ(aᵢ−bᵢ)·ln((aᵢ+ε)/(bᵢ+ε))，a=actual、b=base，占比归一化，ε 平滑防 log(0)。
- IV：Σ(good_pct−bad_pct)·WOE，等频分箱 + 缺失独立分桶。
- 分桶：N 等频降序（decile=N 为最高分档）。

输入约定（AUC/KS 通用）：scores、labels 为等长一维数值序列，NaN 自动剔除；
labels 视为「> 0 即坏样本」，兼容 0/1 与原始逾期天数标签。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================
# AUC / KS / Gini
# ============================================================
def _average_ranks(values_sorted: np.ndarray) -> np.ndarray:
    """对已升序排序的数组返回 1-based 平均秩（并列取平均），与 scipy 'average' 口径一致。"""
    uniq, inverse, counts = np.unique(values_sorted, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])  # 每个 unique 值的 0-based 起始位置
    avg_per_unique = (starts + 1 + starts + counts) / 2.0    # 1-based 平均秩
    return avg_per_unique[inverse]


def _clean_pair(scores, labels):
    """剔除 NaN 后返回 (scores, labels, bad_mask)；无有效样本时返回 None。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~(np.isnan(scores) | np.isnan(labels))
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) == 0:
        return None, None, None
    bad = (labels > 0).astype(float)
    return scores, labels, bad


def calc_auc(scores, labels) -> Optional[float]:
    """计算 AUC（秩和法，与 sklearn.roc_auc_score 一致）。

    Returns:
        方向性 AUC；单类样本或无有效样本时返回 None。
    """
    scores, _, bad = _clean_pair(scores, labels)
    if scores is None:
        return None
    n_pos = int(bad.sum())
    n_neg = len(bad) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = _average_ranks(scores[order])
    sum_pos_ranks = ranks[bad[order] == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calc_ks(scores, labels) -> Optional[float]:
    """计算 KS（max|累计坏样本率 − 累计好样本率|）。

    Returns:
        KS 值 [0,1]；单类或无有效样本时返回 None。
    """
    scores, _, bad = _clean_pair(scores, labels)
    if scores is None:
        return None
    n = len(bad)
    n_bad = int(bad.sum())
    n_good = n - n_bad
    if n_bad == 0 or n_good == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    bad_sorted = bad[order]
    cum_bad = np.cumsum(bad_sorted)
    cum_good = np.cumsum(1.0 - bad_sorted)
    # 只在每个不同分数处评估（并列不拆分），与 sklearn roc_curve 口径一致
    keep = np.ones(n, dtype=bool)
    keep[:-1] = scores_sorted[1:] != scores_sorted[:-1]
    tpr = np.concatenate([[0.0], cum_bad[keep] / n_bad])
    fpr = np.concatenate([[0.0], cum_good[keep] / n_good])
    return float(np.max(np.abs(tpr - fpr)))


def calc_gini(scores, labels) -> Optional[float]:
    """计算 Gini = |2·AUC − 1|。"""
    auc = calc_auc(scores, labels)
    if auc is None:
        return None
    return abs(2.0 * auc - 1.0)


# ============================================================
# PSI
# ============================================================
def calc_psi(actual_pcts, base_pcts, epsilon: float = 1e-10) -> float:
    """计算分布 PSI（输入已归一化占比）。

    Parameters
    ----------
    actual_pcts : array-like
        实际分布各桶占比。
    base_pcts : array-like
        基准分布各桶占比，与 actual_pcts 等长。
    epsilon : float
        平滑项，避免占比为 0 时除零 / log(0)。

    Returns
    -------
    float
        PSI 值，非负。
    """
    actual_pcts = np.asarray(actual_pcts, dtype=float)
    base_pcts = np.asarray(base_pcts, dtype=float)
    return float(
        np.sum((actual_pcts - base_pcts) * np.log((actual_pcts + epsilon) / (base_pcts + epsilon)))
    )


def psi_from_series(base_series, actual_series, n_bins: int = 10) -> Optional[float]:
    """按等频分箱（基准分布边界）计算两组一维序列的 PSI。

    与 credit-data-analysis 原 calc_psi 口径一致：
    基于 base 的等频边界切分，缺失剔除，<20 有效样本返回 None。

    Returns:
        PSI 值；样本不足或分箱退化时返回 None。
    """
    base = pd.Series(base_series).dropna()
    actual = pd.Series(actual_series).dropna()
    if len(base) < n_bins * 2 or len(actual) < n_bins * 2:
        return None
    try:
        _, edges = pd.qcut(base, n_bins, duplicates="drop", retbins=True)
    except (ValueError, IndexError):
        return None
    if edges is None or len(edges) < 2:
        return None
    base_dist = pd.cut(base, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    actual_dist = pd.cut(actual, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    all_idx = base_dist.index.union(actual_dist.index)
    base_dist = base_dist.reindex(all_idx, fill_value=0.0)
    actual_dist = actual_dist.reindex(all_idx, fill_value=0.0)
    return calc_psi(actual_dist.values, base_dist.values)


# ============================================================
# IV
# ============================================================
def _get_bins(series: pd.Series, n_bins: int):
    """等频分箱边界；样本不足或退化返回 None。"""
    s = series.dropna()
    if len(s) < n_bins * 2 or s.nunique() < 2:
        return None
    try:
        _, edges = pd.qcut(s, n_bins, duplicates="drop", retbins=True)
        if len(edges) < 2:
            return None
        return edges
    except (ValueError, IndexError):
        return None


def calc_iv(series, label, n_bins: int = 10) -> Optional[float]:
    """计算单变量 IV（等频分箱 + 缺失独立处理）。

    Returns:
        IV 值；样本不足或单类标签时返回 None。
    """
    series = pd.Series(series)
    label = pd.Series(label)
    mask = series.notna() & label.notna()
    x = series[mask]
    y = label[mask]
    if len(x) < n_bins * 2 or y.nunique() < 2:
        return None
    edges = _get_bins(x, n_bins)
    if edges is None:
        return None
    binned = pd.cut(x, edges, include_lowest=True)
    grouped_good = (y == 0).groupby(binned, observed=True).sum()
    grouped_bad = (y == 1).groupby(binned, observed=True).sum()
    total_good = grouped_good.sum()
    total_bad = grouped_bad.sum()
    if total_good == 0 or total_bad == 0:
        return None
    good_pct = (grouped_good / total_good).clip(lower=1e-4)
    bad_pct = (grouped_bad / total_bad).clip(lower=1e-4)
    woe = np.log(good_pct / bad_pct)
    return float(np.sum((good_pct - bad_pct) * woe))


# ============================================================
# 分类指标 + 分桶排序性
# ============================================================
def classification_metrics(labels, scores, threshold: float = 0.5) -> Dict[str, Optional[float]]:
    """准确率 / 精确率 / 召回率 / F1（以 threshold 为阈值）。

    Returns:
        dict：accuracy / precision / recall / f1，无有效样本时各值为 None。
    """
    y_true = np.asarray(labels, dtype=float)
    y_score = np.asarray(scores, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    y_pred = (y_score >= threshold).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) > 0 else None
    return {
        "accuracy": round(float(accuracy), 6) if accuracy is not None else None,
        "precision": round(float(precision), 6) if precision is not None else None,
        "recall": round(float(recall), 6) if recall is not None else None,
        "f1": round(float(f1), 6) if f1 is not None else None,
    }


def decile_buckets(sub: pd.DataFrame, score_col: str, biz_cols: Optional[Sequence[str]] = None,
                   n_bins: int = 10) -> List[Dict[str, Any]]:
    """N 等频降序分桶（decile=N 为最高分档）。

    Returns:
        list[dict]：每桶含 decile/count/score_min/score_max/label_rate/lift/recall/cum_recall
        + biz_cols 均值。与 evaluation skill 原 decile_buckets 输出结构一致。
    """
    biz_cols = list(biz_cols or [])
    s = sub.sort_values(score_col, ascending=False).copy()
    s["decile"] = pd.cut(range(len(s)), bins=n_bins, labels=False) + 1
    s["decile"] = n_bins + 1 - s["decile"]  # n_bins=最高分
    overall_lr = float(s["label"].mean()) if "label" in s.columns else 0.0
    total_pos = int(s["label"].sum()) if "label" in s.columns else 0
    cum_pos = 0
    result: List[Dict[str, Any]] = []
    for d in range(n_bins, 0, -1):
        b = s[s["decile"] == d]
        if len(b) == 0:
            continue
        bucket_pos = int(b["label"].sum()) if "label" in s.columns else 0
        cum_pos += bucket_pos
        bucket_lr = float(b["label"].mean()) if "label" in s.columns else None
        row: Dict[str, Any] = {
            "decile": d,
            "count": len(b),
            "score_min": round(float(b[score_col].min()), 4),
            "score_max": round(float(b[score_col].max()), 4),
            "label_rate": round(bucket_lr, 6) if bucket_lr is not None else None,
            "lift": round(bucket_lr / overall_lr, 4) if overall_lr > 0 and bucket_lr is not None else None,
            "recall": round(bucket_pos / total_pos, 4) if total_pos > 0 else None,
            "cum_recall": round(cum_pos / total_pos, 4) if total_pos > 0 else None,
        }
        for bc in biz_cols:
            row[bc] = round(float(b[bc].mean()), 4) if bc in b.columns else None
        result.append(row)
    return result
