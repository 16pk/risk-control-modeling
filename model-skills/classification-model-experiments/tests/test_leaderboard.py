# -*- coding: utf-8 -*-
"""leaderboard.py：排序/乐观偏差标注/失败清单/code hash 比对（skip 训练）。"""
import json
import os

import pandas as pd
import pytest

import leaderboard as lb


def _make_exp(exp_root, spec, payload=None, importance=True):
    """造一个假实验目录（写 eval.json + feature_importance.csv）。"""
    exp_dir = os.path.join(exp_root, spec["id"])
    os.makedirs(os.path.join(exp_dir, "evaluation"), exist_ok=True)
    if payload is None:
        payload = {
            "splits": {
                "oot": {"auc": 0.7, "n": 100},
                "val": {"auc": 0.75, "n": 300},
            },
            "optimistic_bias": False,
        }
    with open(os.path.join(exp_dir, "evaluation", "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    if importance:
        pd.DataFrame({"feature": ["a"], "total_gain": [1.0]}).to_csv(
            os.path.join(exp_dir, "feature_importance.csv"), index=False)


def _specs():
    return [
        {"id": "lgb-full-all-v1", "algo": "lgb", "sample_scheme": "full", "feat_scheme": "all",
         "status": "done", "n_features": 5, "n_samples": 100},
        {"id": "lgb-full-importance-v1", "algo": "lgb", "sample_scheme": "full",
         "feat_scheme": "importance", "status": "done", "n_features": 3, "n_samples": 100},
        {"id": "xgb-full-all-v1", "algo": "xgb", "sample_scheme": "full", "feat_scheme": "all",
         "status": "done", "n_features": 5, "n_samples": 100},
        {"id": "lgb-adversarial-adversarial-v1", "algo": "lgb", "sample_scheme": "adversarial",
         "feat_scheme": "adversarial", "status": "done", "n_features": 4, "n_samples": 90,
         "optimistic": True,
         "plan": {"desc": "adv"}},
    ]


def test_collect_and_sort(tmp_path):
    exp_root = str(tmp_path)
    specs = _specs()
    _make_exp(exp_root, specs[0])
    _make_exp(exp_root, specs[1])  # 依赖 importance 格（重要性文件存在）
    _make_exp(exp_root, specs[2])
    _make_exp(exp_root, specs[3], payload={
        "splits": {"oot": {"auc": 0.85, "n": 50}, "val": {"auc": 0.8, "n": 100}},
        "optimistic_bias": True,
    })
    # 一个 failed
    specs.append({"id": "lgb-recent4-iv-psi-v1", "algo": "lgb", "sample_scheme": "recent4",
                  "feat_scheme": "iv-psi", "status": "failed", "fail_reason": "xx",
                  "n_features": 0, "n_samples": 0})

    rows = lb.collect_results(exp_root, specs)
    rows = lb.sort_rows(rows)
    # 第一个是最优（乐观偏差 0.85 最高）
    assert rows[0]["id"] == "lgb-adversarial-adversarial-v1"
    assert rows[0]["optimistic_bias"] is True
    # failed 排最后
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["fail_reason"] == "xx"
    # 乐观偏差标注
    top = lb.top_k(rows, 5)
    flags = {r["id"]: r["optimistic_bias"] for r in top}
    assert flags["lgb-adversarial-adversarial-v1"] is True

    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    assert "⚠ 乐观偏差" in md
    assert "lgb-adversarial-adversarial-v1" in md


def test_leaderboard_md_failed_section(tmp_path):
    exp_root = str(tmp_path)
    specs = _specs()[:1]
    specs.append({"id": "xgb-full-iv-psi-v1", "algo": "xgb", "sample_scheme": "full",
                  "feat_scheme": "iv-psi", "status": "failed", "fail_reason": "optuna_missing",
                  "n_features": 0, "n_samples": 0})
    _make_exp(exp_root, specs[0])
    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    assert "## 失败清单" in md
    assert "optuna_missing" in md


def test_code_hash_reproducible(tmp_path):
    """code hash 比对：同一文件两次 hash 一致、内容变更后 hash 变化（可复现性核心）。"""
    f = tmp_path / "train.py"
    f.write_text("print(1)\n", encoding="utf-8")

    import hashlib

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fp:
            h.update(fp.read())  # 二进制读
        return h.hexdigest()

    h1 = sha(str(f))
    assert h1 == sha(str(f))  # 幂等
    f.write_text("print(2)\n", encoding="utf-8")
    assert sha(str(f)) != h1  # 变更 → hash 变化