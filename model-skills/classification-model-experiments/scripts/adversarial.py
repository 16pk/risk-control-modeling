# -*- coding: utf-8 -*-
"""对抗验证（算法无关）：lgb train-vs-oot 分类器，双产出（样本剔除 + 特征剔除）。

plan §2.1 E1 / C7 / C8 / D4：
  - 分类器：lgb 小模型（num_leaves=31 / lr=0.05 / 100 轮早停 / train vs oot）
  - 特征剔除依据：total_gain（对抗 importance top-K，剔除最能区分两期分布的漂移特征）
  - 样本剔除依据：predict_proba（被判为 oot 的概率，**剔除最不像 OOT 的低分样本**，
    保留与未来分布接近的高分样本，使训练分布贴近 OOT）
  - 剔除幅度：AI 运行时评估推荐（按 AUC 与分位数），由主流程与用户确认后执行
  - 早停/评估：合并集按 7:3 分层切 train/val（seed=42），val 作早停与 AUC 评估
    （不复用训练集，保证早停有效 + 每轮评估成本降至约 30%）

红线例外①（仅本模块授权）：OOT 可参与对抗分类器训练与样本/特征筛选统计；
禁早停集 / 禁进训练集 / 禁结构超参选择。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from algo_factory import feature_importances


def train_adversarial(dev: pd.DataFrame, oot: pd.DataFrame, features: List[str],
                      seed: int = 42, log: Optional[logging.Logger] = None) -> Tuple[object, pd.DataFrame, float]:
    """训练 train-vs-oot 对抗分类器。

    合并集（dev=0/oot=1）按 seed=42 分层 7:3 切 train/val：val 同时作早停集与 AUC 评估集。
    不复用训练集做 eval_set（否则早停永不触发 + 每轮对全量算 AUC，成为性能瓶颈）。

    Args:
        dev: 开发池 DataFrame（train+test 合并）
        oot: OOT DataFrame
        features: 参与对抗的特征列表
        seed: 随机种子
        log: 可选 logger（打训练进度与时长）

    Returns:
        (model, importance_df, oot_auc)
        - model: LGBMClassifier（train=0/oot=1）
        - importance_df: feature_importances 输出（total_gain 降序）
        - oot_auc: 对抗分类器在 val 上判 oot 的 AUC（衡量可分性，越高分布差异越大）
    """
    import time

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    n_dev, n_oot = len(dev), len(oot)
    y = np.array([0] * n_dev + [1] * n_oot, dtype=int)
    cols = [f for f in features if f in dev.columns and f in oot.columns]
    X = pd.concat([dev[cols], oot[cols]], axis=0).reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce")

    # 分层 7:3 切 train/val（val 早停 + 评估，评估量从全量降至约 30%）
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    if log is not None:
        log.info("[对抗] 合并集 %d 行（dev=%d/oot=%d），切 train=%d/val=%d，特征 %d",
                 len(X), n_dev, n_oot, len(X_tr), len(X_va), len(cols))

    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=1000,
        max_depth=-1, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    if log is not None:
        log.info("[对抗] 训练完成：%d 轮（早停）/ 耗时 %.1fs", model.best_iteration_, time.time() - t0)
    proba = model.predict_proba(X_va)[:, 1]
    oot_auc = float(roc_auc_score(y_va, proba))
    imp = feature_importances(model, "lgb", cols)
    return model, imp, oot_auc


def recommend_drop(proba_dev: np.ndarray, oot_auc: float,
                   base_pct: float = 0.05) -> Dict:
    """按对抗 AUC 动态推荐样本剔除幅度（AI 评估，供用户确认）。

    规则（可解释）：
      - AUC < 0.6：分布差异小，不剔除（0%）
      - AUC in [0.6, 0.7)：剔除 top 5%
      - AUC in [0.7, 0.8)：剔除 top 8%
      - AUC >= 0.8：剔除 top 12%
    """
    if oot_auc < 0.6:
        drop_pct = 0.0
    elif oot_auc < 0.7:
        drop_pct = base_pct
    elif oot_auc < 0.8:
        drop_pct = 0.08
    else:
        drop_pct = 0.12
    return {
        "oot_auc": round(oot_auc, 4),
        "recommended_sample_drop_pct": drop_pct,
        "desc": "对抗 AUC=%.3f → 推荐剔除样本 %.0f%%" % (oot_auc, drop_pct * 100),
    }


def compute_drop_masks(proba_dev: np.ndarray, drop_pct: float,
                       importance_df: pd.DataFrame, top_k: int,
                       features: List[str]) -> Dict:
    """按确认幅度计算双产出。

    样本剔除方向：剔除 proba 最低（最不像 OOT）的样本，保留与未来分布接近的高分样本，
    使训练分布贴近 OOT（与特征剔除方向一致）。

    Returns:
        {"sample_drop_mask": bool (dev 侧), "feature_drop_list": [feat],
         "sample_drop_n": int, "feature_drop_n": int}
    """
    n_dev = len(proba_dev)
    k = int(round(n_dev * drop_pct))
    sample_drop = np.zeros(n_dev, dtype=bool)
    if k > 0 and n_dev > 0:
        idx = np.argsort(proba_dev)[:k]
        sample_drop[idx] = True
    adv_top: List[str] = []
    if importance_df is not None and not importance_df.empty and top_k > 0:
        adv_top = list(importance_df["feature"].head(top_k))
    return {
        "sample_drop_mask": sample_drop,
        "feature_drop_list": adv_top,
        "sample_drop_n": int(sample_drop.sum()),
        "feature_drop_n": len(adv_top),
    }


def save_adversarial_meta(out_dir: str, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "adversarial_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)