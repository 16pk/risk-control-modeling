# -*- coding: utf-8 -*-
"""样本方案（算法无关）：full / recent-N / 线性时间加权 / 对抗剔除。

开发池 = model.split 的 train + test 两档合并；每格施加样本方案后由
run_single_experiment 做 seed=42 分层随机 70/30 切 train/val；OOT 纯榜单。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from date_utils import month_prefix


def _month_series(df: pd.DataFrame, dt_col: str) -> pd.Series:
    """日期列 → 月份标签（YYYYMM）序列，解析失败置 None。"""
    raw = df[dt_col].astype(str).str.strip()
    out = []
    for v in raw:
        try:
            out.append(month_prefix(v))
        except Exception:
            out.append(None)
    return pd.Series(out, index=df.index)


def full_scheme(dev: pd.DataFrame) -> Dict:
    """全量方案：无过滤、无加权。"""
    return {
        "sample_scheme": "full",
        "filter": np.ones(len(dev), dtype=bool),
        "weight": np.ones(len(dev), dtype=float),
        "meta": {"desc": "全量开发池样本"},
    }


def recent_n_scheme(dev: pd.DataFrame, dt_col: str, n_months: int) -> Dict:
    """最近 N 个月方案：保留月份在全局最近 n_months 内的样本。

    Returns:
        {"filter": bool array, "weight": ones, "meta": {n_months, months, desc}}
    """
    months = _month_series(dev, dt_col)
    valid = months.dropna()
    if len(valid) == 0:
        raise ValueError("日期列无法解析月份，无法执行 recent-N 方案")
    month_vals = sorted(set(str(m) for m in valid.unique()))
    keep_months = set(month_vals[-n_months:])
    filt = months.astype(str).isin(keep_months).to_numpy()
    return {
        "sample_scheme": "recent%d" % n_months,
        "filter": filt,
        "weight": np.ones(len(dev), dtype=float),
        "meta": {"n_months": n_months, "months": sorted(keep_months),
                 "desc": "训练窗口内最近 %d 个月样本" % n_months},
    }


def linear_time_weight_scheme(dev: pd.DataFrame, dt_col: str) -> Dict:
    """线性时间衰减加权（最近月 1.0 → 最远月 0.2）：w = 0.8*(t-t_min)/(t_max-t_min)+0.2。"""
    months = _month_series(dev, dt_col)
    w = np.ones(len(dev), dtype=float)
    valid_mask = months.notna()
    if valid_mask.sum() > 0:
        uniq = sorted(set(str(m) for m in months[valid_mask]))
        t_map = {u: i for i, u in enumerate(uniq)}
        t_arr = np.array([t_map.get(str(m), np.nan) for m in months], dtype=float)
        t_min, t_max = 0.0, float(len(uniq) - 1)
        span = t_max - t_min
        if span > 0:
            w = np.where(np.isnan(t_arr), 1.0, 0.8 * (t_arr - t_min) / span + 0.2)
    return {
        "sample_scheme": "timeweight",
        "filter": np.ones(len(dev), dtype=bool),
        "weight": w,
        "meta": {"desc": "线性时间衰减加权(最近月1.0→最远月0.2)", "w_min": 0.2, "w_max": 1.0},
    }


def adversarial_filter_scheme(dev: pd.DataFrame, drop_mask: np.ndarray,
                              meta: Optional[Dict] = None) -> Dict:
    """对抗剔除样本方案：drop_mask 为要剔除的样本布尔数组。"""
    return {
        "sample_scheme": "adversarial",
        "filter": ~drop_mask,
        "weight": np.ones(len(dev), dtype=float),
        "meta": meta or {"desc": "对抗验证剔除分布差异最大样本"},
    }


def apply_sample_scheme(scheme: Dict, dev: pd.DataFrame) -> pd.DataFrame:
    """按方案施加过滤（返回过滤后 DataFrame；权重由调用方单独取）。"""
    return dev[np.asarray(scheme["filter"], dtype=bool)].copy()


# ---------------------------------------------------------------------------
# 方案组数自决（plan §2.1 C3：组数由 AI 按样本情况动态自决并记录理由）
# ---------------------------------------------------------------------------
def decide_sample_schemes(dev: pd.DataFrame, dt_col: str,
                          label_col: str) -> List[Dict]:
    """动态自决样本方案清单并附理由。

    规则（可解释、可复核）：
      - full：永远 1 组（同时是 baseline 与 importance 依赖锚点）
      - recent-N：按开发池月份总数 m 决定最近 N ∈ {max(3, m//2)}（仅 1 组，收敛格数）
      - timeweight：月份数 >= 3 时启用（否则信息量不足）
      - 异常（日期解析失败 / 单月）：仅 full

    Returns:
        list[dict]：{name: "full|recentN|timeweight", n_months: N|None, reason: str}
    """
    months = _month_series(dev, dt_col).dropna()
    n_month = len(set(str(m) for m in months))
    reasons = ["全量开发池样本，作为 baseline 与 importance 依赖锚点"]
    plans: List[Dict] = [{"name": "full", "n_months": None,
                          "reason": reasons[0]}]
    if n_month >= 2:
        n_recent = max(3, n_month // 2)
        plans.append({"name": "recent%d" % n_recent, "n_months": n_recent,
                      "reason": "开发池共 %d 个月，取最近 %d 个月验证窗口衰减" % (n_month, n_recent)})
    else:
        reasons.append("月份数 %d < 2，跳过 recent-N 与时间加权方案" % n_month)
    if n_month >= 3:
        plans.append({"name": "timeweight", "n_months": None,
                      "reason": "开发池共 %d 个月（>=3），启用线性时间衰减加权（最近月1.0→最远月0.2）" % n_month})
    plans[0]["reason"] = "；".join(reasons)
    return plans