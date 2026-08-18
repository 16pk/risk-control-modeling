# -*- coding: utf-8 -*-
"""feature_schemes.py：importance 截断/iv-psi 直算/安全过滤。"""
import numpy as np
import pandas as pd
import pytest

from feature_schemes import adversarial_features, importance_features, iv_psi_features


@pytest.fixture
def dev():
    rng = np.random.RandomState(42)
    n = 2000
    df = pd.DataFrame({
        "f_good": rng.rand(n) + (np.arange(n) % 2),       # 有区分度
        "f_iv_low": rng.rand(n),                          # IV 低
        "f_psi_high": np.where(np.arange(n) < 1000, rng.rand(n), 50.0 + rng.rand(n)),  # 前后分布不同
        "f_missing": np.where(rng.rand(n) < 0.97, np.nan, rng.rand(n)),  # 缺失 97%
        "label": (rng.rand(n) < 0.2).astype(int),
    })
    return df


def test_importance_features_95():
    imp = pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "total_gain": [60.0, 30.0, 8.0, 2.0],
    })
    kept = importance_features(["a", "b", "c", "d"], imp, pct=95.0)
    # 累积 60+30+8=98% >= 95% → 保留 a,b,c
    assert kept == ["a", "b", "c"]


def test_importance_features_empty():
    assert importance_features(["a", "b"], None) == ["a", "b"]
    assert importance_features(["a", "b"], pd.DataFrame()) == ["a", "b"]


def test_iv_psi_filters(dev):
    # oot：f_psi_high 分布整体漂移到 50+ → dev→oot PSI 高 → 剔除
    oot = dev.copy()
    oot["f_psi_high"] = 50.0 + dev["f_psi_high"]
    kept, detail = iv_psi_features(dev, oot, list(dev.columns), "label")
    # f_missing 缺失率 0.97 >= 0.95 剔除
    assert "f_missing" not in kept
    # f_psi_high dev→oot 分布差异 → PSI 高剔除
    assert "f_psi_high" not in kept
    # f_good 保留；f_iv_low 纯噪声 IV≈0 < 0.015 → 剔除
    assert "f_good" in kept
    assert "f_iv_low" not in kept
    assert set(detail.keys()) == set(dev.columns)


def test_iv_psi_keeps_good(dev):
    # 同分布 oot（PSI 小）：f_good 保留；f_iv_low 因 IV<0.015 仍剔除
    dev2 = dev.copy().sample(frac=1.0, random_state=1)
    kept, _ = iv_psi_features(dev, dev2, ["f_good", "f_iv_low"], "label")
    assert "f_good" in kept
    assert "f_iv_low" not in kept


def test_adversarial_features():
    adv = pd.DataFrame({"feature": ["a", "b", "c"], "total_gain": [10, 5, 1]})
    # top_k=2 → 剔除 a,b（对抗判别力最强）→ 保留 c,d,e
    kept = adversarial_features(["a", "b", "c", "d", "e"], top_k=2, adv_importance=adv)
    assert kept == ["c", "d", "e"]
    # top_k=3 → 剔除 a,b,c
    kept2 = adversarial_features(["a", "b", "c", "d", "e"], top_k=3, adv_importance=adv)
    assert kept2 == ["d", "e"]


def test_adversarial_features_none():
    assert adversarial_features(["a", "b"], 2, None) == ["a", "b"]