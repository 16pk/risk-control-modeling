# -*- coding: utf-8 -*-
"""M/S 公式推导超参（lgb / xgb 两侧，每组实验独立计算）。

用户推荐公式（plan §5.2.6 / 记忆 2026-08-18）：
  - n_estimators=1000，100 轮早停（train 拟合、val 早停），learning_rate=0.04
  - lgb：num_leaves=min(31, 2^(S/10))、max_depth=-1、min_child_samples=max(20, 0.002*M)、
         min_sum_hessian_in_leaf=1e-3、bagging_fraction=0.6、
         feature_fraction=max((num_leaves*2)/S, 0.5)
  - xgb：max_depth=4、min_child_weight=1e-3、subsample=0.6、
         colsample_bytree=max((num_leaves*2)/S, 0.5)
  - 共用：scale_pos_weight=neg/pos 自动（占位 "auto"，由训练模板按数据算）、seed=42

仅本文件与 algo_factory.py 与算法相关；其余模块算法无关。
"""
from __future__ import annotations

from typing import Dict


def derive_params(algo: str, n_samples: int, n_features: int) -> Dict[str, object]:
    """按用户推荐公式计算基线超参。

    Args:
        algo: "lgb" | "xgb"
        n_samples: 样本数 M（施加样本方案后的开发池训练段）
        n_features: 特征维度 S（安全过滤后）

    Returns:
        estimator 构造参数 dict（含 early_stopping / scale_pos_weight 占位，
        由训练模板消费：early_stopping 作早停轮数、scale_pos_weight="auto" 时按数据算）。
    """
    n_features = max(int(n_features), 1)
    n_samples = max(int(n_samples), 1)
    # num_leaves 必须 > 1（LightGBM 参数合法域），S 很小时 2^(S/10) 可能 < 2 → 保底下限
    num_leaves = max(2, min(31, int(2 ** (n_features / 10))))
    # colsample_bytree / feature_fraction 合法域 (0,1]，clip 到 [0.5, 1.0]
    col_factor = min(max((num_leaves * 2) / n_features, 0.5), 1.0)
    base: Dict[str, object] = {
        "metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.04,
        "seed": 42,
        "scale_pos_weight": "auto",
        "early_stopping": 100,
    }
    if algo == "lgb":
        base.update({
            "objective": "binary",  # lgb objective
            "num_leaves": num_leaves,
            "max_depth": -1,
            "verbosity": -1,  # lgb 允许 -1（静默）
            "min_child_samples": max(20, int(0.002 * n_samples)),
            "min_sum_hessian_in_leaf": 1e-3,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "feature_fraction": col_factor,
        })
    elif algo == "xgb":
        base.update({
            "objective": "binary:logistic",  # xgb objective 命名不同
            "max_depth": 4,
            "min_child_weight": 1e-3,
            "verbosity": 0,  # xgb 合法域 [0,3]
            "subsample": 0.6,
            "colsample_bytree": col_factor,
            "tree_method": "hist",
        })
    else:
        raise ValueError(f"unsupported algo: {algo}")
    return base


def validate_anchors(space: Dict[str, object], name: str = "search_space") -> None:
    """fail-fast：断言搜索空间所有区间键 low <= high，非法即 RuntimeError。

    防两类脱节静默失效：① 样本量公式锚点 vs 经验域无交集；② 未来新增收窄分支引入反转。
    在 optuna_anchors / adjust_optuna_anchors 出口统一调用，让非法区间在启动即暴露
    （而非 Optuna 运行中崩溃或收窄静默失效）。
    """
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if lo > hi:
                raise RuntimeError(
                    f"{name} 非法区间 {k}: ({lo}, {hi}) low > high；"
                    "请检查锚点生成/收窄逻辑（样本量公式 vs 经验域脱节）"
                )
    return space


def optuna_anchors(algo: str, params: Dict[str, object]) -> Dict[str, object]:
    """以 winner 格 M/S 推导超参为锚点，收窄 Optuna 邻域搜索空间。

    返回搜索空间定义（供 tune_winner 使用）：
      lr ∈ [锚点lr*0.5, 锚点lr*1.5]；num_leaves ∈ [锚点-8, 锚点+8] 下限 8；
      min_child_samples ∈ [max(clip(锚点,20,200)*0.6,5), clip(锚点,20,200)*1.4]（lgb，先 clip 进经验域
      20~200 再生成邻域，防 0.002×M 公式在 26 万样本下算出 520 这类远超经验域的值，导致与
      overfit 收窄 (5,60) 无交集而静默失效）；
      min_child_weight 保持 1e-3 附近 (1e-4, 1e-2)（overfit 语义由 adjust_optuna_anchors 相对化处理）。

    出口经 validate_anchors fail-fast 校验。
    """
    lr = float(params.get("learning_rate", 0.04))
    num_leaves = int(params.get("num_leaves", 31))
    space: Dict[str, object] = {
        "learning_rate": (max(lr * 0.5, 0.005), lr * 1.5),
        "n_estimators": 1000,
        "early_stopping": 100,
    }
    if algo == "lgb":
        mcs = int(params.get("min_child_samples", 20))
        # ① 锚点约束：公式值先 clip 进经验域 [20, 200] 再生成邻域（v2.5 治本）
        mcs_clip = max(20, min(200, mcs))
        space["num_leaves"] = (max(num_leaves - 8, 8), num_leaves + 8)
        space["min_child_samples"] = (max(int(mcs_clip * 0.6), 5), int(mcs_clip * 1.4))
        space["feature_fraction"] = (0.5, 0.9)
        space["bagging_fraction"] = (0.5, 0.9)
    elif algo == "xgb":
        space["max_depth"] = (3, 6)
        space["min_child_weight"] = (1e-4, 1e-2)
        space["colsample_bytree"] = (0.5, 0.9)
        space["subsample"] = (0.5, 0.9)
    validate_anchors(space, "optuna_anchors")
    return space
