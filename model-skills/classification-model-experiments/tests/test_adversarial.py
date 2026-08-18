# -*- coding: utf-8 -*-
"""adversarial.py：对抗分类器/剔除幅度推荐（不实际训练，数据合成轻量）。"""
import numpy as np
import pandas as pd
import pytest

import adversarial


def test_recommend_drop_tiers():
    # <0.6 不剔除
    assert adversarial.recommend_drop(np.zeros(10), 0.55)["recommended_sample_drop_pct"] == 0.0
    # [0.6,0.7) → 5%
    assert adversarial.recommend_drop(np.zeros(10), 0.65)["recommended_sample_drop_pct"] == 0.05
    # [0.7,0.8) → 8%
    assert adversarial.recommend_drop(np.zeros(10), 0.75)["recommended_sample_drop_pct"] == 0.08
    # >=0.8 → 12%
    assert adversarial.recommend_drop(np.zeros(10), 0.85)["recommended_sample_drop_pct"] == 0.12


def test_compute_drop_masks():
    proba = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.05])
    imp = pd.DataFrame({"feature": ["a", "b", "c"], "total_gain": [9, 2, 1]})
    masks = adversarial.compute_drop_masks(proba, drop_pct=0.5, importance_df=imp,
                                           top_k=2, features=["a", "b", "c", "d"])
    # 50% of 6 = 3 个样本剔除
    assert masks["sample_drop_n"] == 3
    assert masks["sample_drop_mask"][0] and masks["sample_drop_mask"][1]
    assert masks["feature_drop_list"] == ["a", "b"]
    assert masks["feature_drop_n"] == 2


def test_compute_drop_masks_zero():
    proba = np.zeros(10)
    masks = adversarial.compute_drop_masks(proba, 0.0, None, 0, ["a", "b"])
    assert masks["sample_drop_n"] == 0
    assert masks["feature_drop_n"] == 0
    assert masks["feature_drop_list"] == []


def test_save_adversarial_meta(tmp_path):
    adversarial.save_adversarial_meta(str(tmp_path), {"oot_auc": 0.8})
    assert (tmp_path / "adversarial_meta.json").exists()


def test_train_adversarial_smoke():
    """轻量冒烟：小数据上训练对抗分类器（不依赖大样本）。"""
    rng = np.random.RandomState(0)
    dev = pd.DataFrame({"f1": rng.rand(300), "f2": rng.rand(300)})
    oot = pd.DataFrame({"f1": rng.rand(300) + 0.8, "f2": rng.rand(300)})
    model, imp, oot_auc = adversarial.train_adversarial(dev, oot, ["f1", "f2"], seed=42)
    assert 0.0 < oot_auc <= 1.0
    assert sorted(imp["feature"].tolist()) == ["f1", "f2"]