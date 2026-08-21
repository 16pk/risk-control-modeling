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


def test_is_tuned_flag_propagates_and_marked(tmp_path):
    """v2.5：is_tuned 从 spec 透传到 leaderboard 行 + md 表格标注 tuned。"""
    exp_root = str(tmp_path)
    specs = _specs()[:1]
    specs.append({"id": "lgb-full-all-v1-opt", "algo": "lgb", "sample_scheme": "full",
                  "feat_scheme": "all", "status": "done", "is_tuned": True,
                  "n_features": 5, "n_samples": 100})
    _make_exp(exp_root, specs[0])
    _make_exp(exp_root, specs[1])
    rows = lb.collect_results(exp_root, specs)
    by_id = {r["id"]: r for r in rows}
    assert by_id["lgb-full-all-v1"]["is_tuned"] is False
    assert by_id["lgb-full-all-v1-opt"]["is_tuned"] is True
    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    assert "lgb-full-all-v1-opt" in md
    assert "/tuned" in md  # 表格标注


def _make_opt_exp(exp_root, spec):
    """造一个带 diagnosis + optuna 的 -opt 格（写 eval.json + manifest.json）。"""
    exp_dir = os.path.join(exp_root, spec["id"])
    os.makedirs(os.path.join(exp_dir, "evaluation"), exist_ok=True)
    payload = {
        "splits": {"oot": {"auc": 0.82, "n": 100}, "val": {"auc": 0.8, "n": 300}},
        "optimistic_bias": False,
    }
    with open(os.path.join(exp_dir, "evaluation", "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    manifest = {
        "id": spec["id"], "algo": "lgb", "status": "done", "is_tuned": True,
        "diagnosis": {
            "status": "overfit",
            "reasons": ["train-val gap 0.08 > 0.05", "PSI 0.12 > 0.10"],
            "signals": {"train_auc": 0.95, "val_auc": 0.87, "gap": 0.08},
        },
        "optuna": {
            "n_trials": 25, "seed": 42, "target": "val_auc",
            "best_value": 0.8123, "best_params": {"learning_rate": 0.05, "num_leaves": 24},
            "search_space": {"learning_rate": [0.03, 0.08], "num_leaves": [16, 40]},
        },
    }
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)


def test_diag_optuna_columns_and_detail(tmp_path):
    """v2.5：-opt 格诊断/调优落入表格两列 + 明细小节；普通格显示 -。"""
    exp_root = str(tmp_path)
    specs = _specs()[:1]
    # -opt 格：spec 自带 diagnosis/optuna（run_experiments 追加路径）
    specs.append({"id": "lgb-full-all-v1-opt", "algo": "lgb", "sample_scheme": "full",
                  "feat_scheme": "all", "status": "done", "is_tuned": True,
                  "n_features": 5, "n_samples": 100,
                  "diagnosis": {"status": "overfit", "reasons": ["gap>0.05"],
                                "signals": {"gap": 0.08}},
                  "optuna": {"n_trials": 25, "best_value": 0.8123,
                             "best_params": {"num_leaves": 24},
                             "search_space": {"num_leaves": [16, 40]}}})
    _make_exp(exp_root, specs[0])
    _make_exp(exp_root, specs[1])

    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    # 表格两列：普通格 -，-opt 格状态/调优
    assert "| 诊断 | 调优 |" in md
    assert "| - | - |" in md  # 普通格
    assert "| overfit | tuned(25t, best_val 0.8123) |" in md
    # 明细小节
    assert "## 诊断与调优明细" in md
    assert "`lgb-full-all-v1-opt`" in md
    assert "诊断: `overfit` — gap>0.05；信号: gap=0.0800" in md
    assert "Optuna: trials=25 best_val_auc=0.8123" in md
    assert "best_params: num_leaves=24" in md
    assert "search_space: num_leaves=(16, 40)" in md


def test_diag_manifest_fallback(tmp_path):
    """v2.5：spec 无 diagnosis/optuna 时，从 -opt 格 manifest.json 兜底读取。"""
    exp_root = str(tmp_path)
    specs = [{"id": "xgb-full-all-v1-opt", "algo": "xgb", "sample_scheme": "full",
              "feat_scheme": "all", "status": "done", "is_tuned": True,
              "n_features": 5, "n_samples": 100}]  # spec 不带诊断字段
    _make_opt_exp(exp_root, specs[0])

    rows = lb.collect_results(exp_root, specs)
    assert rows[0]["diagnosis"]["status"] == "overfit"
    assert rows[0]["optuna"]["n_trials"] == 25
    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    assert "## 诊断与调优明细" in md
    assert "overfit" in md


def test_well_fit_skipped_tuning_marked(tmp_path):
    """v2.5：well_fit 跳过调优（有诊断无 optuna）→ 表格标注跳过，明细标注。"""
    exp_root = str(tmp_path)
    specs = [{"id": "lgb-full-all-v1-opt", "algo": "lgb", "sample_scheme": "full",
              "feat_scheme": "all", "status": "done", "is_tuned": False,
              "n_features": 5, "n_samples": 100,
              "diagnosis": {"status": "well_fit", "reasons": ["指标在合理区间"],
                            "signals": {"gap": 0.02}}}]
    _make_exp(exp_root, specs[0])
    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    assert "well_fit(跳过调优)" in md
    assert "跳过(well_fit)" in md
    assert "（well_fit 跳过调优）" in md


def test_no_tune_no_diag_section(tmp_path):
    """v2.5：--no-tune（无诊断/调优）→ 无明细小节，表格列全 -。"""
    exp_root = str(tmp_path)
    specs = _specs()[:1]
    _make_exp(exp_root, specs[0])
    md_path = lb.write_leaderboard(exp_root, specs)
    md = open(md_path, encoding="utf-8").read()
    assert "## 诊断与调优明细" not in md
    assert "| - | - |" in md


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