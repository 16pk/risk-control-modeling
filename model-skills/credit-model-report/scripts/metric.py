"""模型评估指标计算函数。

口径说明：
- KS：行业标准口径 KS = max|累计坏样本率 − 累计好样本率|（等价于 max|TPR − FPR|），
  与 sklearn.metrics.roc_curve 推出的 KS 一致；并列分数只在每个不同分数处评估，
  避免并列样本被人为拆分而高估 KS。
- AUC：Mann-Whitney U（秩和）法，并列分数用平均秩，结果与 sklearn.roc_auc_score
  完全一致。返回方向性 AUC（分数越高越接近 label=1）；对“分数越高风险越低”的评分，
  若只关心区分度大小可取 max(auc, 1 - auc)。
- PSI：PSI = Σ (aᵢ - bᵢ) · ln((aᵢ + ε) / (bᵢ + ε))，其中 a 为实际分布各桶占比、
  b 为基准分布各桶占比（均需归一化到 1），ε 为平滑项。

输入约定（KS / AUC 通用）：
- scores、labels 为等长的一维数值序列，NaN 自动剔除。
- labels 视为「> 0 即坏样本」，兼容 0/1 标签与原始逾期天数标签。
"""

import numpy as np


def calc_psi(actual_pcts, base_pcts, epsilon=1e-10):
    """计算评分分布 PSI。

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
    assert np.isclose(actual_pcts.sum(), 1.0, atol=1e-6) and \
           np.isclose(base_pcts.sum(), 1.0, atol=1e-6), \
        "calc_psi: 各分布占比须归一化到 1(当前 actual=%.6f, base=%.6f)" % (
            actual_pcts.sum(), base_pcts.sum())
    return float(
        np.sum((actual_pcts - base_pcts) * np.log((actual_pcts + epsilon) / (base_pcts + epsilon)))
    )


def _average_ranks(values_sorted):
    """对已升序排序的数组返回 1-based 平均秩（并列取平均），与 scipy 'average' 口径一致。"""
    uniq, inverse, counts = np.unique(values_sorted, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])  # 每个 unique 值的 0-based 起始位置
    avg_per_unique = (starts + 1 + starts + counts) / 2.0    # 1-based 平均秩
    return avg_per_unique[inverse]


def calc_auc(scores, labels):
    """计算 AUC（Mann-Whitney U / 秩和法，与 sklearn.roc_auc_score 一致）。

    Parameters
    ----------
    scores : array-like
        模型评分数组。
    labels : array-like
        标签数组，> 0 视为坏样本（label=1），否则好样本（label=0）。

    Returns
    -------
    float
        方向性 AUC（分数越高越接近 label=1）；单类样本时返回 nan。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~(np.isnan(scores) | np.isnan(labels))
    scores = scores[mask]
    labels = labels[mask]

    bad = (labels > 0).astype(float)
    n_pos = bad.sum()
    n_neg = len(bad) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    bad_sorted = bad[order]

    ranks = _average_ranks(scores_sorted)
    sum_pos_ranks = ranks[bad_sorted == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calc_ks(scores, labels):
    """计算 KS（行业标准口径：max|累计坏样本率 − 累计好样本率|）。

    Parameters
    ----------
    scores : array-like
        模型评分数组。
    labels : array-like
        标签数组，> 0 视为坏样本，否则好样本。

    Returns
    -------
    float
        KS 值，取值 [0, 1]；单类样本或无样本时返回 nan。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~(np.isnan(scores) | np.isnan(labels))
    scores = scores[mask]
    labels = labels[mask]

    n = len(labels)
    if n == 0:
        return float("nan")
    bad = (labels > 0).astype(float)
    n_bad = bad.sum()
    n_good = n - n_bad
    if n_bad == 0 or n_good == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    bad_sorted = bad[order]

    cum_bad = np.cumsum(bad_sorted)
    cum_good = np.cumsum(1.0 - bad_sorted)
    # 只在每个不同分数处评估（处理并列分数），与 sklearn roc_curve 口径一致
    keep = np.ones(n, dtype=bool)
    keep[:-1] = scores_sorted[1:] != scores_sorted[:-1]

    tpr = np.concatenate([[0.0], cum_bad[keep] / n_bad])
    fpr = np.concatenate([[0.0], cum_good[keep] / n_good])
    return float(np.max(np.abs(tpr - fpr)))
