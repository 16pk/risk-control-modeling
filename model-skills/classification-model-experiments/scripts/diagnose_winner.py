# -*- coding: utf-8 -*-
"""winner 规则诊断（移植自 classification-model-tuning/scripts/diagnose.py，仅保留 lgb/xgb）。

主链路（pipelines）：矩阵实验 → leaderboard 评选出每算法 winner → 本模块规则诊断 →
根据诊断调整 Optuna 搜索锚点（recommend_winner.adjust_optuna_anchors）→ 邻域调优。

诊断输入全部来自实验格自身产物（禁跨 skill import 纪律）：
  - metrics        ← <exp_dir>/evaluation/eval.json 的 splits{"train","val","oot"}（各含 auc/ks）
  - used_params    ← <exp_dir>/manifest.json 的 params（训练超参，derive_params 输出）
  - best_iteration ← <exp_dir>/model/model_meta.json 的 best_iteration
  - psi_oot        ← 对抗/IV-PSI 例外格 eval.json 自带 psi_oot；普通格由调用方按
                      winner 格产物补算（metrics.psi_from_series，train→oot）或传 None

阈值与原 tuning 完全一致：overfit gap>0.05 / underfit gap<0.005 或 train_auc<0.70 /
unstable_psi>0.10 / underconverged best_iteration/n_estimators ≥ 0.95。
状态优先级 overfit > underfit > underconverged > unstable_psi > well_fit。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 阈值: 与全局红线一致(PSI>0.10 警戒);其他来自调参经验(与原 tuning 模块同源)
GAP_OVERFIT = 0.05         # train_auc - oot_auc 超过此值视为过拟合
GAP_UNDERFIT = 0.005       # train_auc - oot_auc 低于此值 → 欠拟合(独立触发, 不要求 train_auc 低)
TRAIN_LOW_AUC = 0.70       # train_auc 低于此值视为欠拟合 (信号叠加)
PSI_WARN = 0.10            # 与全局红线一致
EARLY_STOP_RATIO = 0.95    # best_iteration / n_estimators 比例 (lgb/xgb 共用)

SUPPORTED_ALGOS = ("lgb", "xgb")
STATUSES = ("overfit", "underfit", "underconverged", "unstable_psi", "well_fit")


@dataclass(frozen=True)
class Diagnosis:
    """诊断结论(主要状态 + 多条触发原因 + 中间信号)。"""

    status: str                        # overfit | underfit | underconverged | unstable_psi | well_fit
    reasons: List[str] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """用于落入 -opt 格 manifest.json 的 diagnosis 字段。"""
        return {"status": self.status, "reasons": list(self.reasons),
                "signals": dict(self.signals)}


def _safe_get(d: Dict[str, Any], *keys, default: Optional[float] = None) -> Optional[float]:
    """逐层取值,中间缺失返回默认。"""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _check_underconverged_tree(
    used_params: Dict[str, Any], best_iteration: Optional[int]
) -> Optional[str]:
    """lgb/xgb: best_iteration / n_estimators >= 0.95 → 未收敛(共用规则)。

    experiments 的 model_meta.json.best_iteration 由 train_template/tune_winner 统一落盘,
    lgb 与 xgb 均为 best_iteration 语义(best_iteration_ / best_iteration)。
    """
    n_estimators = used_params.get("n_estimators")
    if best_iteration is None or not n_estimators:
        return None
    ratio = best_iteration / n_estimators
    if ratio >= EARLY_STOP_RATIO:
        return (
            f"best_iteration={best_iteration} / n_estimators={n_estimators} "
            f"= {ratio:.2%} ≥ {EARLY_STOP_RATIO:.0%}(未收敛)"
        )
    return None


def diagnose_winner(
    metrics: Dict[str, Dict[str, float]],
    used_params: Dict[str, Any],
    best_iteration: Optional[int],
    algo: str,
    new_psi: Optional[float] = None,
) -> Diagnosis:
    """规则诊断。优先级 overfit > underfit > underconverged > unstable_psi > well_fit
    (同一 winner 可能多条触发,status 取最先命中的,其他作为 reasons)。

    Args:
        metrics: eval.json 的 splits 子集,形如 {"train": {"auc": ...}, "val": {...}, "oot": {...}}
        used_params: winner 训练超参(manifest.json.params); underconverged 检查读 n_estimators
        best_iteration: early stopping 命中的最佳 iteration(model_meta.json.best_iteration)
        algo: 'lgb' | 'xgb'
        new_psi: train→oot 分数 PSI;对抗/IV-PSI 例外格传 eval.json 的 psi_oot,
            普通格由调用方按 winner 格补算（eval.json 已含 psi_oot 时直接传）或传 None

    Returns:
        Diagnosis 实例
    """
    algo = (algo or "xgb").lower()
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"unsupported algo={algo!r}（诊断仅支持 lgb|xgb）")

    train_auc = _safe_get(metrics, "train", "auc")
    oot_auc = _safe_get(metrics, "oot", "auc")

    signals: Dict[str, float] = {}
    if train_auc is not None:
        signals["train_auc"] = float(train_auc)
    if oot_auc is not None:
        signals["oot_auc"] = float(oot_auc)
    if train_auc is not None and oot_auc is not None:
        signals["train_oot_gap"] = float(train_auc - oot_auc)
    if new_psi is not None:
        signals["new_psi"] = float(new_psi)
    # 早期收敛信号(lgb/xgb 共用 ratio)
    if best_iteration is not None and used_params.get("n_estimators"):
        signals["early_stop_ratio"] = float(best_iteration) / float(used_params["n_estimators"])

    reasons: List[str] = []
    statuses_hit: List[str] = []

    # overfit / underfit (algo-agnostic, 只看 AUC gap)
    if train_auc is not None and oot_auc is not None:
        gap = train_auc - oot_auc
        if gap > GAP_OVERFIT:
            statuses_hit.append("overfit")
            reasons.append(f"train-oot AUC gap={gap:.4f} > {GAP_OVERFIT}(过拟合)")
        elif gap < GAP_UNDERFIT or train_auc < TRAIN_LOW_AUC:
            statuses_hit.append("underfit")
            if gap < GAP_UNDERFIT and train_auc < TRAIN_LOW_AUC:
                reasons.append(
                    f"train AUC={train_auc:.4f} < {TRAIN_LOW_AUC} 且 train-oot gap={gap:.4f} "
                    f"< {GAP_UNDERFIT}(欠拟合)"
                )
            elif gap < GAP_UNDERFIT:
                reasons.append(
                    f"train-oot gap={gap:.4f} < {GAP_UNDERFIT}"
                    f"(欠拟合: 训练都没学进去)"
                )
            else:
                reasons.append(f"train AUC={train_auc:.4f} < {TRAIN_LOW_AUC}(欠拟合)")

    # underconverged (lgb/xgb 共用同一树模型动力学规则)
    under_msg = _check_underconverged_tree(used_params, best_iteration)
    if under_msg:
        statuses_hit.append("underconverged")
        reasons.append(under_msg)

    # unstable_psi (algo-agnostic)
    if new_psi is not None and new_psi > PSI_WARN:
        statuses_hit.append("unstable_psi")
        reasons.append(f"train→oot PSI={new_psi:.4f} > {PSI_WARN}(分数不稳定 [PSI_WARN])")

    if not statuses_hit:
        return Diagnosis(status="well_fit", reasons=["指标在合理区间"], signals=signals)
    return Diagnosis(status=statuses_hit[0], reasons=reasons, signals=signals)