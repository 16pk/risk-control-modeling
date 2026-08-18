# -*- coding: utf-8 -*-
"""sample_schemes.py：recent-N 切窗/线性加权/开发池合并/组数自决。"""
import numpy as np
import pandas as pd
import pytest

import sample_schemes as ss


def _dev(n_per_month=100, months=6, label_col="label"):
    """构造跨 6 个月的开发池。"""
    rows = []
    for mi, m in enumerate(range(1, months + 1)):
        for i in range(n_per_month):
            rows.append({"fuid": f"u{mi}_{i}", "f_p_date": f"2026-{m:02d}-15",
                         "f1": i % 2, "label": i % 5 == 0})
    return pd.DataFrame(rows)


def test_full_scheme():
    dev = _dev()
    s = ss.full_scheme(dev)
    assert s["filter"].sum() == len(dev)
    assert np.allclose(s["weight"], 1.0)


def test_recent_n_scheme():
    dev = _dev(months=6)
    s = ss.recent_n_scheme(dev, "f_p_date", 3)
    kept = dev[s["filter"]]
    assert len(kept) == 300  # 只有最近 3 个月
    assert kept["f_p_date"].str[:7].nunique() == 3


def test_recent_n_bad_dates():
    # 全坏日期 → 抛 ValueError
    dev = _dev(months=3)
    dev["f_p_date"] = "bad-date"
    with pytest.raises(ValueError):
        ss.recent_n_scheme(dev, "f_p_date", 2)
    # 部分坏日期 → 不抛；坏日期样本被排除，最近 2 个月保留 200
    dev2 = _dev(months=3)
    dev2.loc[0, "f_p_date"] = "bad-date"
    s = ss.recent_n_scheme(dev2, "f_p_date", 2)
    assert s["filter"].sum() == 200


def test_linear_time_weight():
    dev = _dev(months=4)
    s = ss.linear_time_weight_scheme(dev, "f_p_date")
    w = s["weight"]
    by_month = dev.assign(w=w).groupby("f_p_date")["w"].mean()
    # 最远月(1月) 0.2，最近月(4月) 1.0，中间线性
    assert abs(by_month.iloc[0] - 0.2) < 1e-6
    assert abs(by_month.iloc[-1] - 1.0) < 1e-6
    # 单调不减
    assert np.all(np.diff(by_month.values) >= 0)


def test_linear_time_weight_single_month():
    dev = _dev(months=1)
    s = ss.linear_time_weight_scheme(dev, "f_p_date")
    assert np.allclose(s["weight"], 1.0)


def test_adversarial_filter_scheme():
    dev = _dev()
    drop = np.zeros(len(dev), dtype=bool)
    drop[:10] = True
    s = ss.adversarial_filter_scheme(dev, drop)
    assert s["filter"].sum() == len(dev) - 10


def test_apply_sample_scheme():
    dev = _dev()
    s = ss.recent_n_scheme(dev, "f_p_date", 2)
    out = ss.apply_sample_scheme(s, dev)
    assert len(out) == 200


def test_decide_sample_schemes():
    dev = _dev(months=6)
    plans = ss.decide_sample_schemes(dev, "f_p_date", "label")
    names = [p["name"] for p in plans]
    assert "full" in names
    assert "timeweight" in names
    recent = [p for p in plans if p["name"].startswith("recent")]
    assert len(recent) == 1
    assert recent[0]["n_months"] == 3  # max(3, 6//2)


def test_decide_sample_schemes_single_month():
    dev = _dev(months=1)
    plans = ss.decide_sample_schemes(dev, "f_p_date", "label")
    names = [p["name"] for p in plans]
    assert names == ["full"]