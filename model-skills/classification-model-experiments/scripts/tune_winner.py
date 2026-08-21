# -*- coding: utf-8 -*-
"""winner 规则诊断 + Optuna 邻域调优产 -opt run（plan §2.1 D2 / §5.2.9 / v2.3）。

- 每算法 OOT AUC 最优 1 组进调优；TPE seed=42；目标 = val AUC；100 轮早停；n_trials 默认 25。
- **v2.3 起 Optuna 前先执行 winner 规则诊断**（diagnose_winner）：
  按状态（overfit/underfit/underconverged/unstable_psi/well_fit）调用
  recommend_winner.adjust_optuna_anchors 调整搜索锚点；well_fit 默认跳过 Optuna 直接复用 winner
  （--force-tune 覆盖）；诊断结果落 -opt 格 manifest.json["diagnosis"] 并在日志展示。
- 搜索空间 = 以 winner 格 M/S 推导超参为锚点收窄邻域（hyperparams.optuna_anchors）±诊断调整。
- **v2.6.1（方案 A）：-opt 格不再读取 winner 的 data/train|val|oot.parquet**（该快照已不在每格落盘）；
  train/val/oot 由主流程（run_experiments.main）运行时重切并透传，天然与 winner 同基线；
  data/ 仅保留 features.json + params.json + weights.csv 作为 Optuna 依赖。
- 产物规范与单格完全一致（manifest 记 is_tuned / base_exp / optuna / diagnosis）。
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
from diagnose_winner import diagnose_winner
from evaluate import evaluate, write_eval
from hyperparams import optuna_anchors
from leaderboard import collect_results
from recommend_winner import adjust_optuna_anchors


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


def _load_winner_inputs_legacy(win_dir: str):
    """兼容旧目录：读 winner 格 data/ 完整快照（train/val/oot.parquet + features/weights）。

    仅当主流程未透传 train_df 时回退使用（v2.6.1 之前生成的实验格）。
    """
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


def _load_winner_inputs(win_dir: str):
    """读 winner 格 data/ 依赖: (features, weights)。

    v2.6.1（方案 A）：train/val/oot 各格不再落盘，由主流程运行时重切透传；
    本函数只读 features.json（特征名）+ weights.csv（样本权重，可选）两个轻量依赖。
    """
    win_data = os.path.join(win_dir, "data")
    if not os.path.isdir(win_data):
        return None
    try:
        with open(os.path.join(win_data, "features.json"), "r", encoding="utf-8") as f:
            features = json.load(f)
        weights = None
        wpath = os.path.join(win_data, "weights.csv")
        if os.path.exists(wpath):
            weights = pd.read_csv(wpath)["weight"].to_numpy()
        return features, weights
    except Exception:
        return None


def _suggest_params(algo: str, anchor: Dict, search_space: Dict, trial) -> Dict:
    """按(诊断调整后的)搜索空间采样超参;search_space 由 optuna_anchors ± 诊断调整得到。"""
    sp = search_space
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


def _load_winner_meta(win_dir: str) -> Optional[Dict]:
    """读 winner 格 model/model_meta.json: 返回 dict 或 None。"""
    meta_path = os.path.join(win_dir, "model", "model_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_winner_eval(win_dir: str) -> Optional[Dict]:
    """读 winner 格 evaluation/eval.json: 返回 payload 或 None。"""
    eval_path = os.path.join(win_dir, "evaluation", "eval.json")
    if not os.path.exists(eval_path):
        return None
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def diagnose_winner_exp(win_dir: str, spec: Dict):
    """对 winner 格执行规则诊断（输入全部来自实验格自身产物）。

    Returns:
        (Diagnosis, Diagnosis.as_dict()) 二元组；输入缺失时返回 (None, None)。
    """
    meta = _load_winner_meta(win_dir)
    eval_payload = _load_winner_eval(win_dir)
    if meta is None or eval_payload is None:
        return None, None
    algo = spec.get("algo", "lgb")
    metrics = eval_payload.get("splits", {})
    used_params = dict(spec.get("params") or {})
    best_iteration = meta.get("best_iteration")
    psi_oot = None
    # 对抗/IV-PSI 例外格自带 psi_oot；普通格由主流程视需要补算后覆盖（此处默认 None）
    oot_metrics = metrics.get("oot") or {}
    if oot_metrics.get("psi_oot") is not None:
        psi_oot = oot_metrics.get("psi_oot")
    from diagnose_winner import Diagnosis

    diag = diagnose_winner(
        metrics=metrics, used_params=used_params,
        best_iteration=best_iteration, algo=algo, new_psi=psi_oot,
    )
    return diag, diag.as_dict()


def tune_winner(
    spec: Dict,
    *,
    exp_root: str,
    template_path: str,
    n_trials: int = 25,
    seed: int = 42,
    resume: bool = False,
    force_tune: bool = False,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    oot_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """对单算法 winner 格做规则诊断 + Optuna 邻域调优，产 <winner_id>-opt 格。

    v2.3 起流程：Optuna 前先对 winner 执行规则诊断（diagnose_winner）→
    按诊断状态调整搜索锚点（overfit/underfit/underconverged/unstable_psi/well_fit）；
    well_fit 默认跳过 Optuna 直接复用 winner（force_tune=True 强制调优）；
    诊断结果落 -opt 格 manifest.json["diagnosis"]。

    v2.6.1（方案 A）：data 快照不再落盘。train_df/val_df/oot_df 由主流程（run_experiments.main）
    传入 winner 格同基线的已重切数据；若未传入则回退读取 winner 的 data/ 快照（兼容旧目录）。

    Returns:
        -opt spec（status=done/failed）或 None（Optuna 缺失 / winner 输入缺失 / well_fit 跳过 /
        入口防御触发）。
    """
    # 入口防御（三保险）：winner 已是 -opt 格则直接返回，防任何调用路径重复调优
    if spec.get("is_tuned"):
        print(f"[tune] {spec['id']} 已是调优格（is_tuned），入口防御跳过")
        return None
    tune_id = f"{spec['id']}-opt"
    exp_dir = os.path.join(exp_root, tune_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    if resume and os.path.exists(os.path.join(exp_dir, "manifest.json")):
        print(f"[tune] {tune_id} 已存在，跳过（resume）")
        with open(os.path.join(exp_dir, "manifest.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    algo = spec.get("algo", "lgb")

    # v2.3：规则诊断（输入全部来自 winner 格自身产物）
    diag, diagnosis = diagnose_winner_exp(os.path.join(exp_root, spec["id"]), spec)
    if diagnosis is None:
        print(f"[diag] {spec['id']} 诊断输入缺失（model_meta/eval 不存在），跳过诊断")

    # well_fit 默认跳过调优，直接复用 winner（force_tune 覆盖）
    if diagnosis is not None and diagnosis.get("status") == "well_fit" and not force_tune:
        print(f"[diag] {spec['id']} → well_fit（指标在合理区间），默认跳过 Optuna 调优"
              f"（--force-tune 强制）")
        return {
            "id": tune_id, "algo": algo, "status": "skipped_well_fit",
            "is_tuned": False, "skipped_reason": "well_fit",
            "diagnosis": diagnosis, "base_exp": spec["id"],
        }

    try:
        import optuna
    except ImportError:
        print(f'[tune] Optuna 未安装，跳过调优 {tune_id}（pip install --user "optuna<4"）')
        return None

    loaded = None
    if train_df is None:
        # 兼容旧目录：回退读 winner 格 data/ 快照（train/val/oot.parquet）
        loaded = _load_winner_inputs_legacy(os.path.join(exp_root, spec["id"]))
        if loaded is None:
            print(f"[tune] winner 格数据快照缺失: {spec['id']}")
            return None
        train_df, val_df, oot_df, features, weights = loaded
    else:
        deps = _load_winner_inputs(os.path.join(exp_root, spec["id"]))
        if deps is None:
            print(f"[tune] winner 格 data/ 依赖缺失（features.json/weights.csv）: {spec['id']}")
            return None
        features, weights = deps
    # 由主流程透传的 train_df 已含 label 列；旧快照也统一重命名过

    def _num(df: pd.DataFrame) -> pd.DataFrame:
        return df[features].apply(pd.to_numeric, errors="coerce")

    anchor = dict(spec.get("params") or {})
    if "label" not in train_df.columns:
        # 主流程保证 data/ 快照含 label 列（run_single_experiment 存了原始 train_df）
        print(f"[tune] winner 快照缺 label 列: {spec['id']}")
        return None

    # v2.3：按诊断状态调整 Optuna 搜索锚点
    diag_status = (diagnosis or {}).get("status") or "well_fit"
    anchors = optuna_anchors(algo, anchor)
    if diag is not None and diag_status != "well_fit":
        anchors = adjust_optuna_anchors(anchors, diag, algo)
        print(f"[diag] {spec['id']} → {diag_status}，Optuna 锚点已按诊断调整")
    if diag_status == "well_fit":
        print(f"[diag] {spec['id']} → well_fit，按默认锚点调优（--force-tune 生效）")

    def objective(trial):
        params = _suggest_params(algo, anchor, anchors, trial)
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
    if diagnosis is not None:
        tune_spec["diagnosis"] = diagnosis
    tune_spec["optuna"] = {
        "n_trials": n_trials, "seed": seed, "target": "val_auc",
        "best_value": float(study.best_value), "best_params": best_params,
        "search_space": anchors,  # 记录诊断调整后的搜索空间（含 n_estimators/early_stopping 等固定值）
    }
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(tune_spec, f, ensure_ascii=False, indent=2)
    with open(os.path.join(exp_dir, "logs", "run.log"), "w", encoding="utf-8") as f:
        f.write("[%s] optuna tuned: %s trials=%d best_val_auc=%.4f\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), tune_id, n_trials, study.best_value))
    print(f"[tune] {tune_id} 完成 best_val_auc={study.best_value:.4f}")
    return tune_spec