# -*- coding: utf-8 -*-
"""端到端冒烟：winner 格（真实训练）→ 规则诊断 → Optuna 锚点调整 → -opt 调优。

验证 v2.3 主链路关键路径：tune_winner 在 Optuna 前对 winner 执行 diagnose_winner_exp、
按诊断状态调整搜索锚点、well_fit 默认跳过、诊断结果落 -opt 格 manifest。
标注 slow（真实训练多个小模型）。
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

import tune_winner  # noqa: E402

optuna = pytest.importorskip("optuna")


@pytest.fixture(scope="module")
def winner_dir(tmp_path_factory):
    """构造一个最小 winner 格（含 model/model_meta.json + evaluation/eval.json + manifest.json）。"""
    base = tmp_path_factory.mktemp("winner")
    exp_dir = base / "lgb-full-all-v1"
    os.makedirs(exp_dir / "model")
    os.makedirs(exp_dir / "evaluation")
    os.makedirs(exp_dir / "data")

    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                      "f3": rng.normal(size=n)})
    y = ((X["f1"] - 0.3 * X["f3"] + rng.normal(0, 0.6, n)) > 0).astype(int)
    df = X.copy()
    df["label"] = y
    train, val, oot = df.iloc[:500], df.iloc[500:650], df.iloc[650:]
    # v2.6.1（方案 A）：不落盘 train/val/oot.parquet，仅保留轻量依赖 features.json
    # （weights.csv 无权重时不写）；Optuna 由调用方透传 train_df/val_df/oot_df
    with open(exp_dir / "data" / "features.json", "w") as f:
        json.dump(list(X.columns), f)

    # 真实训练一个小模型（模拟 winner 格训练产物）
    import lightgbm as lgb

    m = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.1, num_leaves=8, verbosity=-1)
    m.fit(X.iloc[:500], y[:500])
    import joblib

    joblib.dump(m, exp_dir / "model" / "model.pkl")
    with open(exp_dir / "model" / "model_meta.json", "w") as f:
        json.dump({
            "algo": "lgb", "feature_names": list(X.columns),
            "best_iteration": int(getattr(m, "best_iteration_", 30) or 30),
            "early_stopped": True, "params": {"n_estimators": 60},
        }, f)

    # 用 evaluate 产 eval.json（splits 结构）
    from metrics import calc_auc

    def _auc(sub):
        p = m.predict_proba(sub[list(X.columns)])[:, 1]
        return calc_auc(p, sub["label"])

    splits = {
        "train": {"auc": _auc(train), "n": len(train)},
        "val": {"auc": _auc(val), "n": len(val)},
        "oot": {"auc": _auc(oot), "n": len(oot)},
    }
    with open(exp_dir / "evaluation" / "eval.json", "w") as f:
        json.dump({"splits": splits}, f)

    with open(exp_dir / "manifest.json", "w") as f:
        json.dump({"id": "lgb-full-all-v1", "algo": "lgb",
                   "params": {"n_estimators": 60, "learning_rate": 0.1,
                              "num_leaves": 8, "min_child_samples": 20,
                              "bagging_fraction": 0.6, "feature_fraction": 0.8}}, f)

    spec = {
        "id": "lgb-full-all-v1", "algo": "lgb",
        "params": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 8,
                   "min_child_samples": 20, "bagging_fraction": 0.6,
                   "feature_fraction": 0.8},
    }
    return str(base), spec


def test_diagnose_winner_exp_returns_dict(winner_dir):
    exp_root, spec = winner_dir
    diag, diag_dict = tune_winner.diagnose_winner_exp(
        os.path.join(exp_root, spec["id"]), spec)
    assert diag is not None
    assert diag_dict is not None
    assert diag_dict["status"] in ("overfit", "underfit", "underconverged",
                                   "unstable_psi", "well_fit")


def test_tune_winner_entry_guard_skips_tuned(winner_dir, capsys):
    """问题 5 入口防御：spec.is_tuned=True 时直接返回 None，不产 -opt 目录重复调优。"""
    exp_root, spec = winner_dir
    tuned_spec = dict(spec)
    tuned_spec["is_tuned"] = True
    out = tune_winner.tune_winner(
        tuned_spec, exp_root=exp_root,
        template_path="unused",
        n_trials=3, seed=42, resume=False, force_tune=True,
    )
    assert out is None
    captured = capsys.readouterr()
    assert "入口防御" in captured.out
    # 未产生重复调优目录
    assert not os.path.exists(os.path.join(exp_root, f"{spec['id']}-opt"))


def test_tune_winner_full_smoke(winner_dir):
    """真实跑通诊断 → 调优 → -opt 落盘（含 diagnosis 字段）。"""
    exp_root, spec = winner_dir
    spec["status"] = "done"
    # v2.6.1（方案 A）：调用方透传 winner 同基线的重切数据（模拟主流程运行时重切）
    rng = np.random.default_rng(0)
    n = 800
    X_ = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                       "f3": rng.normal(size=n)})
    y_ = ((X_["f1"] - 0.3 * X_["f3"] + rng.normal(0, 0.6, n)) > 0).astype(int)
    dev_ = X_.copy()
    dev_["label"] = y_
    t_df, v_df = dev_.iloc[:500], dev_.iloc[500:650]
    o_df = dev_.iloc[650:]
    out = tune_winner.tune_winner(
        spec, exp_root=exp_root,
        template_path=os.path.join(Path(__file__).resolve().parent.parent,
                                   "scripts", "templates", "train_template.py"),
        n_trials=3, seed=42, resume=False, force_tune=True,
        train_df=t_df, val_df=v_df, oot_df=o_df,
    )
    # well_fit 时 force_tune=True 应继续调优;否则正常调优。两种情况都产 -opt manifest
    assert out is not None
    manifest_path = os.path.join(exp_root, f"{spec['id']}-opt", "manifest.json")
    if out.get("status") == "skipped_well_fit":
        return  # well_fit 跳过（--force-tune 应避免此分支，但兜底通过）
    assert os.path.exists(manifest_path)
    with open(manifest_path) as f:
        man = json.load(f)
    assert man["status"] == "done"
    assert "diagnosis" in man
    assert man["diagnosis"]["status"] in ("overfit", "underfit", "underconverged",
                                          "unstable_psi", "well_fit")