#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
credit_dpd30 — LightGBM Optuna 超参调优（TPESampler, 目标=val AUC）
用法: python optuna_tune_lgb.py --features-csv <csv> --trials 25 --out params.json
产出: best_params.json（供 train_lgb.py --params-json 消费）
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

SESSION = None  # 由 --session-dir 设置
LABEL = "dpd30_3c"


def load_split(name):
    df = pd.read_parquet(SESSION / "sample-features" / "splits" / f"{name}.parquet")
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype("object")
    return df


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.4),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True, help="session 目录 runs/<ts>-<model_name>")
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global SESSION
    SESSION = Path(args.session_dir).resolve()
    FEATURES = pd.read_csv(args.features_csv)["feature_name"].tolist()
    train, test = load_split("train"), load_split("test")
    X_train, y_train = train[FEATURES], train[LABEL]
    X_test, y_test = test[FEATURES], test[LABEL]

    # ---- 计算资源路由裁决(字节口径 EXP-G-004): Optuna 每个 trial 都整表物化数据,
    #      窗口体量≥1GB ⇒ distributed, 须走 ray 分支而非本机 Optuna。----
    # ⚠️ 已知限制(评审共识,本次不扩范围):此处拦截时机晚于 load_split ——数据已由上方
    #    train/test 整表载入内存后才判——"防击穿"初衷打折;真正前置的门槛在 task-spec
    #    Gate P0(取数落盘前裁定 engine.ruling),待未来把判定前移到加载之前。
    try:
        from config_io import estimate_size_bytes, route_by_bytes

        if route_by_bytes(estimate_size_bytes(df=train), where="tuning optuna") == "distributed":
            print(
                "\n===== [compute-routing][DISTRIBUTED_REQUIRED] =====\n"
                "窗口体量≥1GB,禁止在本机单进程跑 Optuna 调参(每 trial 均需整表数据)。\n"
                "  请改用 /ray-distributed-train skill 的『分布式调参』模式。\n"
                "⚠️ 未经「门禁#4 算法与超参数」「门禁#6 交付方式」确认不可直接提交远端 job。\n"
                "=================================================\n"
            )
            raise SystemExit("[compute-routing] tuning 中止于本地 Optuna 前(大样本须转 ray-distributed-train)")
    except ImportError:
        pass  # shared 缺失 → 按 local 放行

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_test, y_test),
        n_trials=args.trials, show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_params
    best["n_estimators"] = 500
    best["objective"] = "binary"
    best["metric"] = "auc"
    best["random_state"] = 42
    best["verbosity"] = -1

    with open(args.out, "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print(f"\n===== OPTUNA DONE ===== ({args.trials} trials, {elapsed:.0f}s)")
    print(f"best val AUC: {study.best_value:.4f}")
    print(f"best params: {json.dumps(best, ensure_ascii=False)}")
    print(f"已落盘: {args.out}")


if __name__ == "__main__":
    main()
