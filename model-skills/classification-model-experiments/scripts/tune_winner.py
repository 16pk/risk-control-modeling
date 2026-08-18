# -*- coding: utf-8 -*-
"""winner Optuna 邻域调优产 -opt run（plan §2.1 D2 / §5.2.9）。

- 每算法 OOT AUC 最优 1 组进调优；TPE seed=42；目标 = val AUC；100 轮早停；n_trials 默认 25。
- 搜索空间 = 以 winner 格 M/S 推导超参为锚点收窄邻域（hyperparams.optuna_anchors）。
- **-opt 格复用 winner 的 data/ 快照（train/val/oot 同基线），不重新切分**；
  产物规范与单格完全一致（manifest 记 is_tuned / base_exp / optuna）。
- Optuna 缺失时清晰报错并跳过（相关测试 skipif）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import algo_factory
from evaluate import evaluate, write_eval
from hyperparams import optuna_anchors
from leaderboard import collect_results


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _best_per_algo(rows: List[Dict]) -> Dict[str, Dict]:
    """每算法 OOT AUC 最优（done 且 oot_auc 非空）。"""
    best: Dict[str, Dict] = {}
    for r in rows:
        if r["status"] != "done" or r["oot_auc"] is None:
            continue
        cur = best.get(r["algo"])
        if cur is None or r["oot_auc"] > cur["oot_auc"]:
            best[r["algo"]] = r
    return best


def _suggest_params(algo: str, anchor: Dict, trial) -> Dict:
    sp = optuna_anchors(algo, anchor)
    p = dict(anchor)
    p["learning_rate"] = trial.suggest_float("learning_rate", *sp["learning_rate"])
    p["n_estimators"] = int(sp["n_estimators"])
    if algo == "lgb":
        p["num_leaves"] = trial.suggest_int("num_leaves", *sp["num_leaves"])
        p["min_child_samples"] = trial.suggest_int("min_child_samples", *sp["min_child_samples"])
        p["feature_fraction"] = trial.suggest_float("feature_fraction", *sp["feature_fraction"])
        p["bagging_fraction"] = trial.suggest_float("bagging_fraction", *sp["bagging_fraction"])
    elif algo == "xgb":
        p["max_depth"] = trial.suggest_int("max_depth", *sp["max_depth"])
        p["min_child_weight"] = trial.suggest_float("min_child_weight", *sp["min_child_weight"], log=True)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", *sp["colsample_bytree"])
        p["subsample"] = trial.suggest_float("subsample", *sp["subsample"])
    return p


def _load_winner_inputs(win_dir: str):
    """读 winner 格 data/ 快照: (train_df, val_df, oot_df, features, weights)。"""
    win_data = os.path.join(win_dir, "data")
    if not os.path.isdir(win_data):
        return None
    try:
        train_df = pd.read_parquet(os.path.join(win_data, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(win_data, "val.parquet"))
        oot_df = pd.read_parquet(os.path.join(win_data, "oot.parquet"))
        with open(os.path.join(win_data, "features.json"), "r", encoding="utf-8") as f:
            features = json.load(f)
        weights = None
        wpath = os.path.join(win_data, "weights.csv")
        if os.path.exists(wpath):
            weights = pd.read_csv(wpath)["weight"].to_numpy()
        return train_df, val_df, oot_df, features, weights
    except Exception:
        return None


def tune_winner(
    spec: Dict,
    *,
    exp_root: str,
    template_path: str,
    n_trials: int = 25,
    seed: int = 42,
    resume: bool = False,
) -> Optional[Dict]:
    """对单算法 winner 格做 Optuna 邻域调优，产 <winner_id>-opt 格。

    Returns:
        -opt spec（status=done/failed）或 None（Optuna 缺失 / winner 输入缺失）。
    """
    tune_id = f"{spec['id']}-opt"
    exp_dir = os.path.join(exp_root, tune_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    if resume and os.path.exists(os.path.join(exp_dir, "manifest.json")):
        print(f"[tune] {tune_id} 已存在，跳过（resume）")
        with open(os.path.join(exp_dir, "manifest.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        import optuna
    except ImportError:
        print(f'[tune] Optuna 未安装，跳过调优 {tune_id}（pip install --user "optuna<4"）')
        return None

    loaded = _load_winner_inputs(os.path.join(exp_root, spec["id"]))
    if loaded is None:
        print(f"[tune] winner 格数据快照缺失: {spec['id']}")
        return None
    train_df, val_df, oot_df, features, weights = loaded
    label_col_in = None  # 从快照文件无法直接得 label 列名；由调用方透传或推断
    algo = spec["algo"]

    def _num(df: pd.DataFrame) -> pd.DataFrame:
        return df[features].apply(pd.to_numeric, errors="coerce")

    anchor = dict(spec.get("params") or {})
    if "label" not in train_df.columns:
        # 主流程保证 data/ 快照含 label 列（run_single_experiment 存了原始 train_df）
        print(f"[tune] winner 快照缺 label 列: {spec['id']}")
        return None

    def objective(trial):
        params = _suggest_params(algo, anchor, trial)
        try:
            model = algo_factory.build_estimator(algo, params)
            algo_factory.fit_model(
                model, algo, _num(train_df), pd.to_numeric(train_df["label"], errors="coerce"),
                _num(val_df), pd.to_numeric(val_df["label"], errors="coerce"),
                sample_weight=weights, early_stopping=100)
            from metrics import calc_auc

            auc = calc_auc(algo_factory.predict_proba(model, _num(val_df)),
                           pd.to_numeric(val_df["label"], errors="coerce"))
            return auc if auc is not None else 0.5
        except Exception as e:
            sys.stderr.write(f"[tune] trial 失败: {e}\n")
            return 0.5

    try:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception as e:
        print(f"[tune] Optuna 优化失败: {e}")
        return None

    best_params = dict(anchor)
    best_params.update(study.best_params)
    best_params.update({"early_stopping": 100, "scale_pos_weight": "auto", "n_estimators": 1000})

    # 代码快照（模板复制 + hash）
    scripts_dir = os.path.join(exp_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    target = os.path.join(scripts_dir, "train.py")
    if not os.path.exists(target):
        shutil.copyfile(template_path, target)
    code_sha = _sha256(target)

    # 用 best_params 训练（import 快照代码，保证复现）
    try:
        sys.path.insert(0, scripts_dir)
        sys.modules.pop("train", None)
        import train as train_mod

        params_for_train = {k: v for k, v in best_params.items()
                            if k not in ("early_stopping", "scale_pos_weight")}
        params_for_train["scale_pos_weight"] = "auto"
        t0 = time.time()
        model = algo_factory.build_estimator(algo, params_for_train)
        algo_factory.fit_model(
            model, algo, _num(train_df), pd.to_numeric(train_df["label"], errors="coerce"),
            _num(val_df), pd.to_numeric(val_df["label"], errors="coerce"),
            sample_weight=weights, early_stopping=100)
        train_time = time.time() - t0
        preds = {
            "train": algo_factory.predict_proba(model, _num(train_df)),
            "val": algo_factory.predict_proba(model, _num(val_df)),
            "oot": algo_factory.predict_proba(model, _num(oot_df)),
        }
        importance = algo_factory.feature_importances(model, algo, features)
        best_iter = int(getattr(model, "best_iteration_" if algo == "lgb" else "best_iteration",
                                model.n_estimators))
    except Exception as e:
        print(f"[tune] best_params 训练失败: {e}")
        return None
    finally:
        try:
            sys.path.remove(scripts_dir)
        except ValueError:
            pass

    # 评估四档
    all_X = pd.concat([_num(train_df), _num(val_df), _num(oot_df)], axis=0)
    all_y = np.concatenate([
        pd.to_numeric(train_df["label"], errors="coerce").to_numpy(),
        pd.to_numeric(val_df["label"], errors="coerce").to_numpy(),
        pd.to_numeric(oot_df["label"], errors="coerce").to_numpy()])
    all_pred = np.concatenate([preds["train"], preds["val"], preds["oot"]])
    scores = {"train": pd.Series(preds["train"]), "val": pd.Series(preds["val"]),
              "oot": pd.Series(preds["oot"]), "all": pd.Series(all_pred)}
    labels = {"train": pd.Series(pd.to_numeric(train_df["label"], errors="coerce")),
              "val": pd.Series(pd.to_numeric(val_df["label"], errors="coerce")),
              "oot": pd.Series(pd.to_numeric(oot_df["label"], errors="coerce")),
              "all": pd.Series(all_y)}
    payload = evaluate(scores, labels, algo=algo, features=features, params=best_params,
                       optimistic_bias=False)
    write_eval(payload, os.path.join(exp_dir, "evaluation"), tune_id)

    # 落盘 model + importance
    model_dir = os.path.join(exp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    import joblib

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    meta = {
        "algo": algo, "feature_names": features, "params": best_params,
        "best_iteration": best_iter,
        "early_stopped": bool(getattr(model, "best_iteration_" if algo == "lgb" else "best_iteration", None) is not None),
        "train_time_sec": round(train_time, 3), "seed": seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "is_tuned": True,
    }
    with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    importance.to_csv(os.path.join(exp_dir, "feature_importance.csv"), index=False)

    # manifest
    tune_spec = dict(spec)
    tune_spec["id"] = tune_id
    tune_spec["params"] = best_params
    tune_spec["plan"] = {**(spec.get("plan") or {}), "base_exp": spec["id"]}
    tune_spec["is_tuned"] = True
    tune_spec["code_sha256"] = code_sha
    tune_spec["template_version"] = "v1"
    tune_spec["code_modified"] = False
    tune_spec["status"] = "done"
    tune_spec["fail_reason"] = None
    tune_spec["optuna"] = {
        "n_trials": n_trials, "seed": seed, "target": "val_auc",
        "best_value": float(study.best_value), "best_params": best_params,
    }
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(tune_spec, f, ensure_ascii=False, indent=2)
    with open(os.path.join(exp_dir, "logs", "run.log"), "w", encoding="utf-8") as f:
        f.write("[%s] optuna tuned: %s trials=%d best_val_auc=%.4f\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), tune_id, n_trials, study.best_value))
    print(f"[tune] {tune_id} 完成 best_val_auc={study.best_value:.4f}")
    return tune_spec