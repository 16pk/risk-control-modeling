#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""model-scoring: 用定版模型对清洗后数据跑推理, 输出违约概率分 score。

仅支持在建模 pipeline 内调用(classification-model-development Stage 5), 置于
「模型定版」之后、「score-to-fico」之前。输入为 data-cleaning 清洗后的数据文件
(含特征列 + 非特征列[id/date/label 等])。

职责边界:
- 只做「模型推理 → 违约概率分 score」, 不做校准、不转 FICO(转分交给 score-to-fico)。
- 定版判定是编排层职责, 本脚本只收 --model-path(模型文件或 model/ 目录), 不做定版校验。

模型加载(按算法分流):
- xgb:  model.json → xgboost.Booster 直接加载(最稳健的 xgb 推理路径);
       feature_names 取自 model_meta.json(与 dnn/lr 统一从该文件读, 避免依赖
       XgbFitter.save_model/load 对 meta 文件名的两套命名差异)。
- dnn:  model.pkl → pickle.load 得 DnnPredictor(需 trainers.train_dnn 可 import,
       其 predict_proba 内部完成 缺失填充+标准化+MLP 前向)。
- lr:   model.pkl → pickle.load 得 LrPredictor(需 trainers.train_lr 可 import,
       其 predict_proba 内部完成 WoE 编码 + LR)。

特征对齐(安全红线):
- 读 model_meta.json 的 feature_names, 严格校验输入数据含全部特征(缺失报错列出),
  并按 feature_names 顺序重排后喂模型, 避免列序错位打错分。

输出(减少存储成本):
- 透传所有非特征列(id/date/label 等) + score 列, 不含原特征列。

用法:
  python score_data.py \
      --model-path <model_dir|model_file> \
      --data <清洗后 parquet/csv> \
      --out <score.parquet> \
      [--score-col score] [--algo xgb|dnn|lr]

依赖: pandas / numpy / pyarrow; 按算法额外需要 xgboost(xgb) 或 torch(dnn)
      / scikit-learn(lr)。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径定位
# ---------------------------------------------------------------------------
def _locate_training_scripts() -> Path:
    """定位 classification-model-training/scripts(与本 skill 同属 model-skills/)。

    dnn/lr 模型是 pickle 的 predictor 对象, unpickle 时需 trainers.train_dnn /
    trainers.train_lr 可 import, 故须把训练脚本目录注入 sys.path。
    """
    here = Path(__file__).resolve()          # .../model-scoring/scripts/score_data.py
    model_skills = here.parents[2]           # .../model-skills/
    training_scripts = model_skills / "classification-model-training" / "scripts"
    if not training_scripts.is_dir():
        raise SystemExit(
            f"[ERROR] 未找到分类训练脚本目录: {training_scripts}\n"
            "model-scoring 依赖 classification-model-training 的 engines/trainers 加载 dnn/lr 模型。"
        )
    return training_scripts


# ---------------------------------------------------------------------------
# 模型定位与元信息
# ---------------------------------------------------------------------------
def resolve_model_dir(model_path: str) -> Path:
    """把 --model-path 规整为 model/ 目录(传文件则取其父目录)。"""
    p = Path(model_path).expanduser()
    if not p.exists():
        raise SystemExit(f"[ERROR] 模型路径不存在: {p}")
    return p if p.is_dir() else p.parent


def read_model_meta(model_dir: Path) -> dict:
    """读 model_dir/model_meta.json(三种算法均含 feature_names)。"""
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"[ERROR] 缺少 model_meta.json: {meta_path}\n"
            "定版模型的 model/ 目录应含 model_meta.json(由 model-training 落盘)。"
        )
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_algo(model_dir: Path, meta: dict, algo_override: Optional[str]) -> str:
    """判定算法: model.json → xgb; model.pkl → meta.algo(dnn/lr)。

    显式 --algo 优先; 无法判定时报错提示。
    """
    if algo_override:
        return algo_override.lower()
    if (model_dir / "model.json").exists():
        return "xgb"
    if (model_dir / "model.pkl").exists():
        algo = (meta.get("algo") or "").lower()
        if algo in ("dnn", "lr"):
            return algo
        raise SystemExit(
            "[ERROR] model.pkl 存在但 model_meta.json 缺 algo 字段, 无法判定 dnn/lr。"
            "请用 --algo dnn|--algo lr 显式指定。"
        )
    raise SystemExit(
        f"[ERROR] 目录中未找到 model.json / model.pkl: {model_dir}"
    )


def _model_file(model_dir: Path, algo: str) -> Path:
    return model_dir / ("model.json" if algo == "xgb" else "model.pkl")


class _XgbScorer:
    """xgboost.Booster 的轻量包装, 暴露与 DnnPredictor/LrPredictor 一致的 predict_proba(df)。

    用原生 Booster 而非 XgbFitter, 规避 save_model/load 对 meta 文件名的两套命名差异;
    feature_names 已由上游(read_model_meta)按 model_meta.json 提供。
    """

    def __init__(self, booster):
        self._booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        dmat = xgb.DMatrix(X)  # 传 DataFrame 保留特征名, 与训练时的 feature_names 对齐
        return self._booster.predict(dmat)


def load_predictor(algo: str, model_dir: Path):
    """按算法加载定版模型, 返回带 predict_proba(df[features]) 接口的预测器。"""
    model_file = _model_file(model_dir, algo)
    if not model_file.exists():
        raise SystemExit(f"[ERROR] 模型文件不存在: {model_file}")

    if algo == "xgb":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(model_file))
        return _XgbScorer(booster)

    # dnn / lr: 注入训练脚本目录后 pickle.load(predictor 内部已打包预处理逻辑)
    training = _locate_training_scripts()
    if str(training) not in sys.path:
        sys.path.insert(0, str(training))
    if algo == "dnn":
        from trainers.train_dnn import DnnPredictor  # noqa: F401  确保类可 import
    elif algo == "lr":
        from trainers.train_lr import LrPredictor  # noqa: F401  确保类可 import
    else:
        raise SystemExit(f"[ERROR] 未知 algo={algo!r}, 仅支持 xgb|dnn|lr")

    with model_file.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 数据加载与推理
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise SystemExit(f"[ERROR] 仅支持 parquet/csv 输入: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="定版模型打分(model-scoring)")
    parser.add_argument("--model-path", required=True,
                        help="定版模型文件(model.json/model.pkl)或 model/ 目录")
    parser.add_argument("--data", required=True, help="清洗后数据文件(parquet/csv)")
    parser.add_argument("--out", required=True, help="打分输出 parquet 路径")
    parser.add_argument("--score-col", default="score", help="输出概率分列名(默认 score)")
    parser.add_argument("--algo", default=None, help="算法覆盖: xgb|dnn|lr(默认自动判定)")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    meta = read_model_meta(model_dir)
    algo = infer_algo(model_dir, meta, args.algo)
    feature_names = list(meta.get("feature_names") or [])
    if not feature_names:
        raise SystemExit(
            f"[ERROR] model_meta.json 缺 feature_names: {model_dir / 'model_meta.json'}"
        )
    print(f"[SCORING] algo={algo} | 模型目录={model_dir} | 特征数={len(feature_names)}")

    df = load_data(args.data)
    print(f"[SCORING] 输入数据: {args.data} | shape={df.shape}")

    # 1) 特征严格校验: 缺失则报错并列出
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise SystemExit(
            f"[ERROR] 输入数据缺失 {len(missing)} 个特征(定版模型 feature_names):\n"
            + "\n".join(f"  - {f}" for f in missing)
        )

    # 2) 按 feature_names 顺序重排对齐(安全红线: 避免列序错位打错分)
    X = df[feature_names]

    # 3) 推理得违约概率分
    predictor = load_predictor(algo, model_dir)
    score = np.asarray(predictor.predict_proba(X), dtype=float).ravel()
    if len(score) != len(df):
        raise SystemExit(
            f"[ERROR] 推理输出长度 {len(score)} 与输入 {len(df)} 不一致, 请检查模型/数据对齐。"
        )

    # 4) 输出 = 透传所有非特征列 + score 列(减少存储成本, 不含原特征列)
    non_feature_cols = [c for c in df.columns if c not in feature_names]
    out = df[non_feature_cols].copy()
    out[args.score_col] = score

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out.to_csv(out_path, index=False)
    else:
        out.to_parquet(out_path, index=False)

    print(f"[SAVE] 输出: {out_path} | shape={out.shape}")
    print(f"[SAVE] 透传非特征列 {len(non_feature_cols)} 个: {non_feature_cols}")
    print(f"[SAVE] 概率分列: {args.score_col} | 范围=[{float(score.min()):.6f}, {float(score.max()):.6f}]")
    print("[DONE] 定版模型打分完成(未校准、未转 FICO)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
