# -*- coding: utf-8 -*-
"""fill_report.py 对 experiments 矩阵转正 run 的适配（v2.3）。

构造最小 session 夹具:
  - inside_runs/new-models/{algo}-v1/config.json(produced_by=skills/model-experiments,
    source_exp=lgb-full-all-v1, 顶层 metrics{oot_auc,val_auc}, features 列表)
  - inside_runs/experiments/lgb-full-all-v1/data/{train,val,oot}.parquet + feature_importance.csv
验证:
  - §IV 能从 experiments 源格 data/ 重建切分信息(test=val 映射)
  - §V 能兜底读 experiments 源格 feature_importance.csv
  - §VI 能显示 experiments 型 run 的 oot_auc/val_auc/n_feat 与"experiments 矩阵转正"标记
  - §VII 为纯占位(comparison 模块已于 v2.7 移除)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import fill_report


SESSION_ROOT = "_report_sessions"


@pytest.fixture(autouse=True)
def _chdir(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    yield
    if os.path.isdir(SESSION_ROOT):
        try:
            os.rmdir(SESSION_ROOT)
        except OSError:
            pass  # 非空则留待清理


def _make_exp_session(session_id):
    """构造 experiments 型转正 run 的最小 session,返回 session_dir。"""
    sdir = os.path.join(SESSION_ROOT, session_id)
    new_models = os.path.join(sdir, "new-models")
    exp_dir = os.path.join(sdir, "experiments", "lgb-full-all-v1")
    data_dir = os.path.join(exp_dir, "data")
    os.makedirs(os.path.join(new_models, "lgb-v1"))
    os.makedirs(os.path.join(new_models, "lgb-v2"))
    os.makedirs(data_dir)
    os.makedirs(os.path.join(sdir, "sample-features", "data-cleaning"))

    rng = np.random.default_rng(0)
    for name, n in [("train", 500), ("val", 200), ("oot", 300)]:
        arr = rng.normal(size=(n, 3))
        df = pd.DataFrame(arr, columns=["f1", "f2", "f3"])
        df["label"] = (df["f1"] + rng.normal(0, 0.5, n) > 0).astype(int)
        df["fuid"] = np.arange(n)
        df["f_p_date"] = "2026-01-01"
        df.to_parquet(os.path.join(data_dir, f"{name}.parquet"))

    fi = pd.DataFrame({"feature": ["f1", "f2", "f3"], "total_gain": [0.6, 0.3, 0.1]})
    fi.to_csv(os.path.join(exp_dir, "feature_importance.csv"), index=False)

    cfg = {
        "produced_by": "skills/model-experiments",
        "run_name": "lgb-v2",
        "algo": "lgb",
        "source_exp": "lgb-full-all-v1",
        "sample_scheme": "recent_12m",
        "feat_scheme": "iv_psi",
        "is_tuned": True,
        "optimistic_bias": True,
        "params": {"n_estimators": 500},
        "features": ["f1", "f2", "f3"],
        "metrics": {"oot_auc": 0.78, "val_auc": 0.80},
    }
    # 最新 run lgb-v2 用 experiments 型 config; 旧 run lgb-v1 用 training 型(兼容性验证)
    with open(os.path.join(new_models, "lgb-v2", "config.json"), "w") as f:
        json.dump(cfg, f)
    old_cfg = {
        "produced_by": "skills/model-training",
        "algo": "xgb",
        "runtime": {"metrics": {"train": {"auc": 0.85}, "val": {"auc": 0.80},
                                "oot": {"auc": 0.76}},
                    "n_features": 3},
    }
    with open(os.path.join(new_models, "lgb-v1", "config.json"), "w") as f:
        json.dump(old_cfg, f)
    return Path(sdir)


def test_section_iv_experiments_split_rebuild():
    sdir = _make_exp_session("s1")
    out = fill_report.build_section_iv(sdir)
    assert "experiments" in out
    assert "train" in out and "oot" in out
    # val 映射到 test 展示
    assert "| test |" in out
    assert "experiments 矩阵源格" in out


def test_section_v_experiments_feature_importance():
    sdir = _make_exp_session("s1")
    fi = fill_report._latest_run_feature_importance(sdir)
    assert fi is not None and fi.name == "feature_importance.csv"
    out = fill_report.build_section_v(sdir)
    assert "f1" in out and "0.6" in out


def test_section_vi_experiments_metrics():
    sdir = _make_exp_session("s1")
    out = fill_report.build_section_vi(sdir)
    assert "lgb-v2" in out
    assert "experiments 矩阵转正" in out
    assert "0.78" in out  # oot_auc
    assert "0.80" in out  # val_auc
    assert "lgb-v1" in out  # 旧 run 仍显示
    assert "baseline" in out  # training 型分类未退化


def test_section_vii_placeholder():
    sdir = _make_exp_session("s1")
    out = fill_report.build_section_vii(sdir)
    assert "leaderboard" in out
    assert "占位" in out


def test_classify_run_experiments():
    cfg = {"produced_by": "skills/model-experiments", "is_tuned": True,
           "optimistic_bias": True, "sample_scheme": "recent_12m",
           "feat_scheme": "iv_psi"}
    s = fill_report._classify_run(cfg)
    assert "experiments 矩阵转正" in s
    assert "+ Optuna" in s
    assert "乐观偏差候选" in s
    assert "recent_12m × iv_psi" in s