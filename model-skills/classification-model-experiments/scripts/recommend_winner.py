# -*- coding: utf-8 -*-
"""winner 规则推荐 + Optuna 锚点调整（移植自原 classification-model-tuning/scripts/recommend_params.py，
仅保留 lgb/xgb；新增 lgb 策略表与 adjust_optuna_anchors）。

两部分职责：
  1. recommend_params(params, diagnosis, algo)：按诊断状态给出"规则推荐超参"（受 BOUNDS 约束），
     供 tune_winner 展示给用户/作为 Optuna 锚点参考。
  2. adjust_optuna_anchors(anchors, diagnosis, algo)：按诊断状态收窄/放宽 Optuna 搜索空间，
     让调优聚焦在问题方向。

lgb 策略表（按 lgb 参数域等价推导，各受 BOUNDS_LGB 约束）：
  - underfit      : num_leaves×1.5, min_child_samples/2, learning_rate/2, n_estimators×1.5
  - overfit       : num_leaves×0.8, min_child_samples×1.5, reg_alpha×2
  - underconverged: n_estimators×2（≤2000）
  - unstable_psi  : bagging_fraction−0.1, feature_fraction−0.1（≥0.5）
  - well_fit      : 不动

xgb 策略表（与原 tuning 逐字对齐）：
  - underfit      : max_depth+1, min_child_weight/2, learning_rate/2, n_estimators×1.5
  - overfit       : max_depth−1, reg_lambda×2, min_child_weight×2
  - underconverged: n_estimators×2（≤2000）
  - unstable_psi  : subsample−0.1, colsample_bytree−0.1（≥0.5）
  - well_fit      : 不动
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from diagnose_winner import Diagnosis

# 边界约束(经验性,防止"加容量"加飞) —— xgb 与原 tuning 逐字对齐
BOUNDS_XGB = {
    "max_depth": (3, 10),
    "min_child_weight": (1, 200),
    "learning_rate": (0.005, 0.3),
    "n_estimators": (100, 2000),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.1, 10.0),
}

# lgb 参数域对应的边界约束(经验性, 与 lgb 合法域一致)
BOUNDS_LGB = {
    "num_leaves": (8, 255),
    "min_child_samples": (5, 1000),
    "learning_rate": (0.005, 0.3),
    "n_estimators": (100, 2000),
    "bagging_fraction": (0.5, 1.0),
    "feature_fraction": (0.5, 1.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.1, 10.0),
}

BOUNDS = BOUNDS_LGB  # 兼容别名(主链路默认 lgb)


def _clip(bounds: Dict[str, tuple], name: str, val: float) -> float:
    """按 bounds 限制取值;名字不在表中原样返回。"""
    if name not in bounds:
        return val
    lo, hi = bounds[name]
    return max(lo, min(hi, val))


def _ci(bounds: Dict[str, tuple], name: str, val: float) -> int:
    """clip 后取 int。"""
    return int(round(_clip(bounds, name, val)))


def _recommend_xgb(baseline_params: Dict[str, Any], diagnosis: Diagnosis) -> Dict[str, Any]:
    new = dict(baseline_params)
    status = diagnosis.status
    B = BOUNDS_XGB
    if status == "underfit":
        new["max_depth"] = _ci(B, "max_depth", new.get("max_depth", 4) + 1)
        new["min_child_weight"] = max(
            1, _ci(B, "min_child_weight", new.get("min_child_weight", 50) / 2)
        )
        new["learning_rate"] = round(
            _clip(B, "learning_rate", new.get("learning_rate", 0.04) / 2), 4
        )
        new["n_estimators"] = _ci(B, "n_estimators", new.get("n_estimators", 1000) * 1.5)
    elif status == "overfit":
        new["max_depth"] = _ci(B, "max_depth", new.get("max_depth", 4) - 1)
        new["reg_lambda"] = round(
            _clip(B, "reg_lambda", new.get("reg_lambda", 1.0) * 2), 4
        )
        new["min_child_weight"] = _ci(
            B, "min_child_weight", new.get("min_child_weight", 50) * 2
        )
    elif status == "underconverged":
        new["n_estimators"] = _ci(B, "n_estimators", new.get("n_estimators", 1000) * 2)
    elif status == "unstable_psi":
        new["subsample"] = round(
            _clip(B, "subsample", new.get("subsample", 0.6) - 0.1), 3
        )
        new["colsample_bytree"] = round(
            _clip(B, "colsample_bytree", new.get("colsample_bytree", 0.6) - 0.1), 3
        )
    # well_fit: 不动
    return new


def _recommend_lgb(baseline_params: Dict[str, Any], diagnosis: Diagnosis) -> Dict[str, Any]:
    new = dict(baseline_params)
    status = diagnosis.status
    B = BOUNDS_LGB
    if status == "underfit":
        new["num_leaves"] = _ci(B, "num_leaves", new.get("num_leaves", 31) * 1.5)
        new["min_child_samples"] = max(
            5, _ci(B, "min_child_samples", new.get("min_child_samples", 20) / 2)
        )
        new["learning_rate"] = round(
            _clip(B, "learning_rate", new.get("learning_rate", 0.04) / 2), 4
        )
        new["n_estimators"] = _ci(B, "n_estimators", new.get("n_estimators", 1000) * 1.5)
    elif status == "overfit":
        new["num_leaves"] = _ci(B, "num_leaves", new.get("num_leaves", 31) * 0.8)
        new["min_child_samples"] = _ci(
            B, "min_child_samples", new.get("min_child_samples", 20) * 1.5
        )
        new["reg_alpha"] = round(
            _clip(B, "reg_alpha", new.get("reg_alpha", 0.0) * 2), 4
        )
    elif status == "underconverged":
        new["n_estimators"] = _ci(B, "n_estimators", new.get("n_estimators", 1000) * 2)
    elif status == "unstable_psi":
        new["bagging_fraction"] = round(
            _clip(B, "bagging_fraction", new.get("bagging_fraction", 0.6) - 0.1), 3
        )
        new["feature_fraction"] = round(
            _clip(B, "feature_fraction", new.get("feature_fraction", 0.6) - 0.1), 3
        )
    # well_fit: 不动
    return new


def recommend_params(
    baseline_params: Dict[str, Any],
    diagnosis: Diagnosis,
    algo: str = "lgb",
) -> Dict[str, Any]:
    """根据 Diagnosis 调整 baseline_params,返回规则推荐超参 dict。

    Args:
        baseline_params: winner 训练用的完整超参(manifest.json.params)
        diagnosis: 诊断结果
        algo: 'lgb' | 'xgb'

    Returns:
        新超参 dict(原 dict 浅拷贝后被修改的若干键)
    """
    algo = (algo or "lgb").lower()
    if algo == "xgb":
        return _recommend_xgb(baseline_params, diagnosis)
    if algo == "lgb":
        return _recommend_lgb(baseline_params, diagnosis)
    raise ValueError(f"unknown algo={algo!r}, 仅支持 lgb|xgb")


def _tighten_bounds(anchors: Dict[str, Tuple[Any, Any]],
                    names: Dict[str, Tuple[Any, Any]]) -> None:
    """就地收窄 anchors 中指定键的搜索区间(取与给定边界的交集)。

    交集为空（原始锚点下界 > 调整上界 或 原始上界 < 调整下界）时保留原区间，
    避免产生 low>high 的非法 Optuna 搜索区间。
    """
    for name, (lo, hi) in names.items():
        if name in anchors and isinstance(anchors[name], tuple) and len(anchors[name]) == 2:
            a_lo, a_hi = anchors[name]
            new_lo, new_hi = max(a_lo, lo), min(a_hi, hi)
            if new_lo <= new_hi:
                anchors[name] = (new_lo, new_hi)


def _set_interval(anchors: Dict[str, Tuple[Any, Any]], name: str,
                  lo: Any, hi: Any) -> None:
    """设置 anchors[name] 区间；lo>hi 时保留原区间（避免 Optuna low>high 非法区间）。"""
    if name in anchors and lo <= hi:
        anchors[name] = (lo, hi)


def adjust_optuna_anchors(
    anchors: Dict[str, Tuple[Any, Any]],
    diagnosis: Diagnosis,
    algo: str = "lgb",
) -> Dict[str, Any]:
    """按诊断状态调整 Optuna 搜索空间(原 anchors 浅拷贝,不原地改)。

    方向:
      - overfit       : 收窄容量上限 + 提高正则下界 → max_depth/num_leaves 上限收窄、
                        min_child_weight/min_child_samples 下界抬高、reg_alpha/reg_lambda 下界抬高
      - underfit      : 放宽容量上限 + 降低 lr 下界 → max_depth/num_leaves 上限放宽、
                        learning_rate 下界降低
      - underconverged: 拉高 n_estimators 固定值(搜索空间该项改为固定 n_estimators)
      - unstable_psi  : 收窄采样率上界到 ≤0.7 → subsample/bagging_fraction 与
                        colsample_bytree/feature_fraction 上界收窄
      - well_fit      : 不动

    v2.5 治本（问题 3）：「数值域跨数量级」参数（min_child_samples / min_child_weight）的 overfit
    收窄改为**相对锚点**（抬高下界比例：lgb ×1.5 / xgb ×10），不再用绝对区间（绝对区间与
    样本量驱动锚点域可能无交集 → 交集为空保留原区间 → 收窄静默失效）；容量参数
    （max_depth/num_leaves）与正则/采样参数保留绝对收窄（域小、与经验域天然重叠）。
    出口经 validate_anchors fail-fast 校验（防未来新增分支引入 low>high 反转）。

    Returns:
        调整后的 anchors dict
    """
    anchors = dict(anchors)
    if algo == "xgb":
        if diagnosis.status == "overfit":
            _tighten_bounds(anchors, {"max_depth": (3, 5),
                                      "reg_alpha": (0.1, 5.0), "reg_lambda": (0.5, 10.0)})
            # min_child_weight 相对化：锚点下界 ×10（1e-3 量级 → 1e-2 量级），上限保持
            if "min_child_weight" in anchors and isinstance(anchors["min_child_weight"], tuple):
                mlo, mhi = anchors["min_child_weight"]
                new_lo = mlo * 10.0
                if new_lo <= mhi:
                    anchors["min_child_weight"] = (new_lo, mhi)
            if "subsample" in anchors:
                _set_interval(anchors, "subsample", 0.5, min(anchors["subsample"][1], 0.8))
        elif diagnosis.status == "underfit":
            _tighten_bounds(anchors, {"max_depth": (4, 8)})
            if "learning_rate" in anchors:
                lo, hi = anchors["learning_rate"]
                _set_interval(anchors, "learning_rate", max(lo, 0.005), min(hi, 0.1))
        elif diagnosis.status == "underconverged":
            n_est = int(anchors.get("n_estimators", 1000))
            anchors["n_estimators"] = min(int(n_est * 1.5), 2000)
        elif diagnosis.status == "unstable_psi":
            for k in ("subsample", "colsample_bytree"):
                if k in anchors and isinstance(anchors[k], tuple) and len(anchors[k]) == 2:
                    lo, hi = anchors[k]
                    _set_interval(anchors, k, lo, min(hi, 0.7))
    elif algo == "lgb":
        if diagnosis.status == "overfit":
            _tighten_bounds(anchors, {"num_leaves": (8, 24),
                                      "reg_alpha": (0.1, 5.0), "reg_lambda": (0.5, 10.0)})
            # min_child_samples 相对化：锚点下界 ×1.5（空交集保留原区间的问题已由锚点 clip + 相对化解决）
            if "min_child_samples" in anchors and isinstance(anchors["min_child_samples"], tuple):
                mlo, mhi = anchors["min_child_samples"]
                new_lo = max(int(mlo * 1.5), 5)
                if new_lo <= mhi:
                    anchors["min_child_samples"] = (new_lo, mhi)
            if "bagging_fraction" in anchors:
                _set_interval(anchors, "bagging_fraction", 0.5, min(anchors["bagging_fraction"][1], 0.8))
        elif diagnosis.status == "underfit":
            if "num_leaves" in anchors:
                anchor_lo, anchor_hi = anchors["num_leaves"]
                anchors["num_leaves"] = (anchor_lo, max(anchor_hi, 48))
            if "learning_rate" in anchors:
                lo, hi = anchors["learning_rate"]
                _set_interval(anchors, "learning_rate", max(lo, 0.005), min(hi, 0.1))
        elif diagnosis.status == "underconverged":
            n_est = int(anchors.get("n_estimators", 1000))
            anchors["n_estimators"] = min(int(n_est * 1.5), 2000)
        elif diagnosis.status == "unstable_psi":
            for k in ("bagging_fraction", "feature_fraction"):
                if k in anchors and isinstance(anchors[k], tuple) and len(anchors[k]) == 2:
                    lo, hi = anchors[k]
                    _set_interval(anchors, k, lo, min(hi, 0.7))
    # well_fit: 不动
    try:
        from hyperparams import validate_anchors

        validate_anchors(anchors, "adjust_optuna_anchors")
    except ImportError:
        pass  # 独立 import 场景（无 hyperparams 时）跳过校验
    return anchors