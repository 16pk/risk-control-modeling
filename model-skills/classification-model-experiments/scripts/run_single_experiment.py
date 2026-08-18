# -*- coding: utf-8 -*-
"""单格执行器：模板快照+hash、训练、评估、manifest、失败容错（plan §2.2 修改 5 / F1）。

每格实验目录：
  <exp_dir>/
  ├── manifest.json            # 全超参/方案/seed/依赖源/code_sha256/template_version/code_modified/status
  ├── model/model.pkl + model_meta.json
  ├── evaluation/eval.{json,md}
  ├── feature_importance.csv
  ├── scripts/train.py         # 训练代码快照（train_template.py 副本 + code_sha256）
  ├── logs/run.log
  └── data/                    # 训练输入快照（复现用：train/val/oot.parquet + features/params/weights.json）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import _bootstrap  # noqa: F401
import algo_factory
from evaluate import evaluate, write_eval
from hyperparams import derive_params
from plan_matrix import get_spec, update_spec
import sample_schemes as ss


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_dev(dev: pd.DataFrame, label_col: str, seed: int = 42,
              train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """开发池 → seed=42 分层随机 70/30 切 train/val（plan §2.1 D1）。"""
    y = pd.to_numeric(dev[label_col], errors="coerce")
    stratify = y.fillna(-1).astype(int).to_numpy()
    uniq, counts = np.unique(stratify, return_counts=True)
    if len(uniq) < 2 or (counts == 1).any():
        # 单类时退化随机切分
        train, val = train_test_split(dev, test_size=1 - train_ratio, random_state=seed)
    else:
        train, val = train_test_split(
            dev, test_size=1 - train_ratio, random_state=seed, stratify=stratify)
    return train, val


def load_training_code(exp_dir: str, template_path: str,
                       template_version: str = "v1") -> Tuple[str, str, bool]:
    """把权威模板快照复制进 <exp_dir>/scripts/train.py，返回 (code_sha256, template_version, code_modified)。

    已存在且未标记 modified 时沿用（断点续跑不重复复制）。
    """
    scripts_dir = os.path.join(exp_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    target = os.path.join(scripts_dir, "train.py")
    if os.path.exists(target):
        # 已存在：检查是否与模板一致（一致 → 未修改）
        import filecmp

        modified = not filecmp.cmp(target, template_path, shallow=False)
        return _sha256(target), template_version, modified
    shutil.copyfile(template_path, target)
    return _sha256(target), template_version, False


def _save_inputs(exp_dir: str, train: pd.DataFrame, val: pd.DataFrame,
                 oot: pd.DataFrame, features: List[str], params: Dict,
                 weight: Optional[np.ndarray], label_col: str) -> None:
    """训练输入快照落盘（复现用）。

    label 列统一重命名为 `label`（tune_winner 按固定列名消费快照，同一基线可比）。
    """
    data_dir = os.path.join(exp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, df_ in (("train", train), ("val", val), ("oot", oot)):
        snap = df_.rename(columns={label_col: "label"}) if label_col != "label" else df_.copy()
        snap.to_parquet(os.path.join(data_dir, f"{name}.parquet"), index=False)
    with open(os.path.join(data_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2, default=str)
    if weight is not None:
        pd.Series(weight).to_csv(os.path.join(data_dir, "weights.csv"), index=False, header=["weight"])


def run_experiment(
    spec: Dict,
    *,
    dev: pd.DataFrame,
    oot: pd.DataFrame,
    label_col: str,
    dt_col: str,
    id_col: str,
    base_features: List[str],
    exp_root: str,
    template_path: str,
    sample_scheme: Optional[Dict] = None,
    feat_scheme: str = "all",
    feat_override: Optional[List[str]] = None,   # 对抗特征剔除后的最终特征列表
    importance_source: Optional[Dict] = None,   # {"exp_dir": ..., "importance_df": ...}
    iv_psi_detail: Optional[Dict] = None,
    optimistic_bias: bool = False,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
) -> Dict:
    """执行单格实验。

    Args:
        spec: plan_matrix.build_experiment_spec 输出
        dev: 开发池 DataFrame（train+test 合并）
        oot: OOT DataFrame
        sample_scheme: 施加的样本方案 dict（full/recentN/timeweight/adversarial）
        feat_scheme: "all" | "importance" | "iv-psi" | "adversarial"
        feat_override: 对抗格传入对抗特征剔除后的最终特征列表（feat_scheme=adversarial 用）
        importance_source: 依赖源（importance 特征方案 = 同样本 all 格）
        optimistic_bias: 对抗/IV-PSI 例外格 True

    Returns:
        更新后的 spec（status=done/failed）。
    """
    log = logger or logging.getLogger("exp")
    exp_dir = os.path.join(exp_root, spec["id"])
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    def _fail(reason: str) -> None:
        spec["status"] = "failed"
        spec["fail_reason"] = reason
        with open(os.path.join(exp_dir, "logs", "run.log"), "a", encoding="utf-8") as f:
            f.write("[%s] FAILED: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), reason))
        return spec

    # 断点续跑：已 done 直接返回
    if resume and spec.get("status") == "done" and os.path.exists(os.path.join(exp_dir, "manifest.json")):
        log.info("[exp] %s 已 done，跳过（resume）", spec["id"])
        return spec

    log.info("[exp] %s 开始（sample=%s feat=%s wave=%s）",
             spec["id"], spec["sample_scheme"], feat_scheme, spec["wave"])

    # 1) 样本方案
    try:
        if sample_scheme is None:
            sample_scheme = ss.full_scheme(dev)
        filtered = ss.apply_sample_scheme(sample_scheme, dev)
        weight = np.asarray(sample_scheme["weight"], dtype=float)
        if len(filtered) == 0:
            return _fail("样本方案过滤后为空")
    except Exception as e:
        return _fail("样本方案失败: %r" % e)

    # 2) 特征方案
    try:
        if feat_scheme == "importance":
            imp_df = importance_source.get("importance_df") if importance_source else None
            from feature_schemes import importance_features

            features = importance_features(base_features, imp_df, pct=95.0)
            if not features:
                return _fail("importance 截断后特征为空")
        elif feat_scheme == "iv-psi":
            from feature_schemes import iv_psi_features

            features, iv_detail = iv_psi_features(filtered, oot, base_features, label_col)
            if iv_psi_detail is not None:
                iv_psi_detail["detail"] = iv_detail
            if not features:
                return _fail("iv-psi 筛选后特征为空")
        elif feat_scheme == "adversarial":
            if feat_override is None or not feat_override:
                return _fail("对抗特征方案缺 feat_override（对抗剔除后特征列表）")
            features = list(feat_override)
        else:
            features = list(base_features)
    except Exception as e:
        return _fail("特征方案失败: %r" % e)

    # 3) 切分 train/val（seed=42 分层 70/30）
    try:
        train_df, val_df = split_dev(filtered, label_col, seed=42)
        # 权重按 train 段对齐
        w_train = weight[train_df.index.to_numpy()]
        # 数值化
        def _num(df: pd.DataFrame) -> pd.DataFrame:
            return df[features].apply(pd.to_numeric, errors="coerce")

        X_train, X_val = _num(train_df), _num(val_df)
        y_train = pd.to_numeric(train_df[label_col], errors="coerce").to_numpy()
        y_val = pd.to_numeric(val_df[label_col], errors="coerce").to_numpy()
        X_oot = _num(oot)
        y_oot = pd.to_numeric(oot[label_col], errors="coerce").to_numpy()
    except Exception as e:
        return _fail("切分失败: %r" % e)

    # 4) 超参推导（M/S 每格独立计算）
    try:
        params = derive_params(spec["algo"], len(train_df), len(features))
        spec["n_samples"] = len(train_df)
        spec["n_features"] = len(features)
        spec["params"] = params
    except Exception as e:
        return _fail("超参推导失败: %r" % e)

    # 5) 训练代码快照
    try:
        code_sha, tpl_ver, code_modified = load_training_code(exp_dir, template_path)
        spec["code_sha256"] = code_sha
        spec["template_version"] = tpl_ver
        spec["code_modified"] = code_modified
    except Exception as e:
        return _fail("训练代码快照失败: %r" % e)

    # 6) 训练（import 快照代码，保证复现=重跑实验目录代码）
    try:
        sys.path.insert(0, os.path.join(exp_dir, "scripts"))
        import train as train_mod

        result = train_mod.train(
            X_train, y_train, w_train, X_val, y_val, X_oot, y_oot,
            spec["algo"], params, features, seed=42)
    except Exception as e:
        return _fail("训练失败: %r" % e)

    # 7) 评估（四档）
    try:
        all_X = pd.concat([X_train, X_val, X_oot], axis=0)
        all_y = np.concatenate([y_train, y_val, y_oot])
        all_pred = np.concatenate([result["preds"]["train"], result["preds"]["val"],
                                   result["preds"]["oot"]])
        scores = {
            "train": pd.Series(result["preds"]["train"]),
            "val": pd.Series(result["preds"]["val"]),
            "oot": pd.Series(result["preds"]["oot"]),
            "all": pd.Series(all_pred),
        }
        labels = {
            "train": pd.Series(y_train),
            "val": pd.Series(y_val),
            "oot": pd.Series(y_oot),
            "all": pd.Series(all_y),
        }
        oot_psi_base = None
        if optimistic_bias:
            oot_psi_base = pd.Series(result["preds"]["val"])
        payload = evaluate(
            scores, labels, algo=spec["algo"], features=features,
            params=params, optimistic_bias=optimistic_bias,
            oot_psi_base=oot_psi_base,
            iv_features=(features if feat_scheme == "iv-psi" else None),
            iv_train_df=train_df if feat_scheme == "iv-psi" else None,
        )
        write_eval(payload, os.path.join(exp_dir, "evaluation"), spec["id"])
    except Exception as e:
        return _fail("评估失败: %r" % e)

    # 8) 落盘模型 + 重要性 + 输入快照
    try:
        model_dir = os.path.join(exp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        import joblib

        joblib.dump(result["model"], os.path.join(model_dir, "model.pkl"))
        meta = {
            "algo": result["algo"],
            "feature_names": features,
            "params": result["params"],
            "best_iteration": result["best_iter"],
            "early_stopped": result["early_stopped"],
            "scale_pos_weight": result["scale_pos_weight"],
            "train_time_sec": round(result["train_time"], 3),
            "seed": 42,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        result["importance"].to_csv(os.path.join(exp_dir, "feature_importance.csv"), index=False)
        _save_inputs(exp_dir, train_df, val_df, oot, features, result["params"], w_train, label_col)
    except Exception as e:
        return _fail("落盘失败: %r" % e)

    # 9) manifest 落盘 + 状态
    try:
        with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return _fail("manifest 落盘失败: %r" % e)
    spec["status"] = "done"
    oot_auc = payload["splits"].get("oot", {}).get("auc")
    log.info("[exp] %s 完成 oot_auc=%s", spec["id"], oot_auc)
    return spec