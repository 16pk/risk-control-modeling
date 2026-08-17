#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
credit_dpd30 — LightGBM 通用训练脚本（自定义路径，套框架 8 阶段产物规范）
用法:
  python train_lgb.py --run-label feat-v1 --features-csv <csv> [--params-json <json>]
产物: new-models/lgb-{run_label}/{config,features,model,evaluation,predictions,explainability,logs}
纪律: val 段早停（OOT 仅评估）+ 自动 scale_pos_weight（不欠采样）
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SESSION = None  # 由 --session-dir 设置
# v2.1: eval_single.py 已从 classification-model-evaluation 迁入本 skill scripts/
EVAL_SINGLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_single.py")

LABEL = "dpd30_3c"
DT = "fsx_time"
ID = "fuid"

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.02,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.15,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True, help="session 目录 runs/<ts>-<model_name>")
    ap.add_argument("--run-label", required=True, help="如 feat-v1 / tuned-v1")
    ap.add_argument("--features-csv", required=True, help="特征清单 csv (feature_name 列)")
    ap.add_argument("--params-json", default=None, help="超参 json 覆盖默认")
    ap.add_argument("--early-stopping", type=int, default=100)
    args = ap.parse_args()

    global SESSION
    SESSION = Path(args.session_dir).resolve()
    run_label = args.run_label
    RUN_DIR = SESSION / "new-models" / f"lgb-{run_label}"
    for d in ["config", "features", "model", "evaluation", "predictions", "explainability", "logs"]:
        (RUN_DIR / d).mkdir(parents=True, exist_ok=True)

    # 特征清单
    FEATURES = pd.read_csv(args.features_csv)["feature_name"].tolist()

    # 超参
    params = dict(DEFAULT_PARAMS)
    if args.params_json:
        over = json.load(open(args.params_json))
        params.update({k: v for k, v in over.items() if k != "scale_pos_weight"})
    EARLY = args.early_stopping

    # 加载三档
    def load_split(name):
        df = pd.read_parquet(SESSION / "sample-features" / "splits" / f"{name}.parquet")
        for c in df.columns:
            if df[c].dtype == "object" or str(df[c].dtype).startswith("string"):
                df[c] = df[c].astype("object")
        return df

    train = load_split("train")
    test = load_split("test")
    oot = load_split("oot")
    for name, df in [("train", train), ("test", test), ("oot", oot)]:
        miss = [f for f in FEATURES if f not in df.columns]
        assert not miss, f"{name} 缺特征列: {miss}"

    neg, pos = (train[LABEL] == 0).sum(), (train[LABEL] == 1).sum()
    scale_pos_weight = neg / pos
    params = {**params, "scale_pos_weight": scale_pos_weight}
    print(f"[{run_label}] features={len(FEATURES)} n={len(train)} scale_pos_weight={scale_pos_weight:.2f}")

    X_train, y_train = train[FEATURES], train[LABEL]
    X_test, y_test = test[FEATURES], test[LABEL]
    X_oot, y_oot = oot[FEATURES], oot[LABEL]

    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = getattr(model, "best_iteration_", model.n_estimators)
    train_time = time.time() - t0
    print(f"[{run_label}] done, best_iteration={best_iter}, time={train_time:.1f}s")

    # predictions
    for name, X, df in [("train", X_train, train), ("test", X_test, test), ("oot", X_oot, oot)]:
        proba = model.predict_proba(X)[:, 1]
        out = pd.DataFrame({ID: df[ID].values, "label": df[LABEL].values, "score": proba})
        out["bucket"] = pd.qcut(proba, 10, labels=False, duplicates="drop") + 1
        out.to_parquet(RUN_DIR / "predictions" / f"{name}_predictions.parquet", index=False)

    # 特征重要性（total_gain + split）
    imp = pd.DataFrame({
        "feature": model.feature_name_,
        "total_gain": model.booster_.feature_importance(importance_type="gain"),
        "split_count": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("total_gain", ascending=False)
    imp["gain_pct"] = (imp["total_gain"] / imp["total_gain"].sum() * 100).round(4)
    imp.to_csv(RUN_DIR / "explainability" / "feature-importance-total_gain.csv", index=False)

    # 落盘
    config = {
        "model_name": "credit_dpd30", "algo": "lgb", "run_label": run_label,
        "label_col": LABEL, "dt_col": DT, "id_cols": [ID],
        "params": params, "best_iteration": int(best_iter),
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "features": FEATURES, "n_features": len(FEATURES),
        "split": "dev random stratified 7:3 (seed=42) + oot merge(2025-08~10)",
        "train_time_sec": round(train_time, 2),
        "produced_by": "custom-lightgbm-credit_dpd30",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RUN_DIR / "config" / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (RUN_DIR / "config" / "config.json.runtime").write_text(
        json.dumps({"change": f"LightGBM {run_label}", "algo": "lgb"}, indent=2, ensure_ascii=False))
    (RUN_DIR / "features" / "features.json").write_text(json.dumps(FEATURES, indent=2, ensure_ascii=False))
    pd.DataFrame({"feature_name": FEATURES}).to_csv(RUN_DIR / "features" / "features.csv", index=False)
    joblib.dump(model, RUN_DIR / "model" / "model.pkl")
    (RUN_DIR / "model" / "model.json").write_text(json.dumps(
        {"algo": "lgb", "best_iteration": int(best_iter), "params": params,
         "feature_names": FEATURES, "has_scorecard": False}, indent=2, ensure_ascii=False))
    (RUN_DIR / "logs" / "run.log").write_text(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] lgb-{run_label} trained, best_iter={best_iter}, "
        f"time={train_time:.1f}s, features={len(FEATURES)}, es={EARLY}\n")

    # 标准评估三件套
    subprocess.run([
        sys.executable, EVAL_SINGLE,
        "--input-dir", str(RUN_DIR / "predictions"),
        "--label-col", "label", "--score-col", "score",
        "--output-dir", str(RUN_DIR / "evaluation"),
        "--name", f"lgb-{run_label}", "--model-type", "lgb",
    ], check=True)

    # 汇总指标
    metrics = {}
    for split in ["train", "test", "oot"]:
        d = json.load(open(RUN_DIR / "evaluation" / f"lgb-{run_label}_{split}_predictions_eval.json"))
        seg = d["metric_by_segment"]["全量"]
        metrics[split] = {"auc": seg["auc"], "ks": seg["ks"], "n": seg["count"], "label_rate": seg["label_rate"]}

    # 单 run report
    md = f"""# lgb-{run_label} 模型报告 — credit_dpd30

> 算法: LightGBM ｜ 标签: dpd30_3c ｜ 特征: {len(FEATURES)} ｜ 训练时间: {train_time:.1f}s

## 超参
{json.dumps(params, indent=2, ensure_ascii=False)}

## 三档评估
| 档 | 样本量 | bad rate | AUC | KS |
|----|--------|----------|-----|-----|
| train | {metrics['train']['n']:,} | {metrics['train']['label_rate']*100:.2f}% | {metrics['train']['auc']:.4f} | {metrics['train']['ks']:.4f} |
| val (test) | {metrics['test']['n']:,} | {metrics['test']['label_rate']*100:.2f}% | {metrics['test']['auc']:.4f} | {metrics['test']['ks']:.4f} |
| OOT | {metrics['oot']['n']:,} | {metrics['oot']['label_rate']*100:.2f}% | {metrics['oot']['auc']:.4f} | {metrics['oot']['ks']:.4f} |

## 特征重要性 Top10
{json.dumps([{'feature': r['feature'], 'total_gain': int(r['total_gain']), 'gain_pct': float(r['gain_pct'])} for _, r in imp.head(10).iterrows()], ensure_ascii=False, indent=1)}
"""
    (RUN_DIR / "report.md").write_text(md, encoding="utf-8")

    print("\n===== SUMMARY =====")
    for split, m in metrics.items():
        print(f"{split:6s} AUC={m['auc']:.4f} KS={m['ks']:.4f}")
    print(f"模型: {RUN_DIR / 'model' / 'model.pkl'}")


if __name__ == "__main__":
    main()
