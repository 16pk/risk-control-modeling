# -*- coding: utf-8 -*-
"""矩阵规划器：AI 自决组数+理由、实验清单（plan §5.1）、断点状态。

ExperimentSpec 契约（manifest 落盘核心，plan §7）：
  {
    "id": "lgb-full-all-v1", "algo": "lgb", "wave": 1, "version": "v1",
    "sample_scheme": "full", "feat_scheme": "all",
    "depends_on": None,                    # importance 格 = 同(algo,sample_scheme) 的 all 格
    "n_samples": 0, "n_features": 0, "params": {}, "seed": 42,
    "code_sha256": "", "template_version": "v1", "code_modified": False,
    "status": "pending|done|failed", "fail_reason": None,
  }
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from hyperparams import derive_params


def exp_id(algo: str, sample_scheme: str, feat_scheme: str, version: str = "v1") -> str:
    """目录命名：{algo}-{sample_scheme}-{feat_scheme}-v{N}（plan §2.1 A2）。"""
    return f"{algo}-{sample_scheme}-{feat_scheme}-{version}"


def build_experiment_spec(algo: str, sample_scheme: str, feat_scheme: str, wave: int,
                          version: str = "v1", depends_on: Optional[str] = None,
                          n_samples: int = 0, n_features: int = 0,
                          params: Optional[Dict] = None,
                          plan: Optional[Dict] = None) -> Dict:
    """构造实验格 spec（未跑：status=pending，code_sha256 空待训练时落）。"""
    spec = {
        "id": exp_id(algo, sample_scheme, feat_scheme, version),
        "algo": algo,
        "wave": wave,
        "version": version,
        "sample_scheme": sample_scheme,
        "feat_scheme": feat_scheme,
        "depends_on": depends_on,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "params": params or {},
        "seed": 42,
        "code_sha256": "",
        "template_version": "v1",
        "code_modified": False,
        "status": "pending",
        "fail_reason": None,
        "plan": plan or {},
    }
    return spec


def build_matrix(algos: List[str], sample_plans: List[Dict],
                 oot_available: bool = True,
                 max_experiments: int = 12) -> List[Dict]:
    """构建实验矩阵（波1 all 格 + 波2 importance/iv-psi 格 + 对抗格）。

    Args:
        algos: ["lgb", "xgb"]
        sample_plans: sample_schemes.decide_sample_schemes 输出
        oot_available: OOT 是否有样本（无 OOT 无对抗/iv-psi 格，仍跑 all/importance）
        max_experiments: 单算法格数上限（含 -opt；默认 12）

    Returns:
        specs 列表（含对抗样本格 spec；对抗特征格与 iv-psi 由主流程决定是否并入）。
    """
    specs: List[Dict] = []
    for algo in algos:
        # 波1：每个样本方案 1 个 all 格（lgb-full-all-v1 兼 baseline）
        for sp in sample_plans:
            specs.append(build_experiment_spec(
                algo, sp["name"], "all", wave=1,
                n_samples=0, n_features=0, plan={"sample_plan_reason": sp.get("reason", "")}))

        # 波2：importance（依赖同样本 all 格）+ iv-psi（单格直算）
        for sp in sample_plans:
            sample_name = sp["name"]
            dep = exp_id(algo, sample_name, "all", "v1")
            specs.append(build_experiment_spec(
                algo, sample_name, "importance", wave=2, depends_on=dep,
                plan={"sample_plan_reason": sp.get("reason", "")}))
        if oot_available:
            for sp in sample_plans:
                specs.append(build_experiment_spec(
                    algo, sp["name"], "iv-psi", wave=2,
                    plan={"sample_plan_reason": sp.get("reason", "")}))

        # 对抗格（独立 1 格，不与其余方案交叉；仅 lgb 主跑，xgb 侧不重复）
        # 双产出合并应用：剔除样本（train-vs-oot 概率高分）+ 剔除特征（对抗 total_gain top-K）
        if algo == "lgb" and oot_available:
            specs.append(build_experiment_spec(
                algo, "adversarial", "adversarial", wave=3,
                plan={"desc": "对抗验证：lgb train-vs-oot 分类器，双产出(样本剔除+特征剔除)合并应用"}))

        # 预算校验：单算法格数 <= max_experiments
        n = sum(1 for s in specs if s["algo"] == algo)
        if n > max_experiments:
            raise ValueError(
                f"[plan] 单算法 {algo} 实验格数 {n} 超出上限 {max_experiments}，"
                f"请收敛样本/特征方案组数（--max-experiments-per-algo 可调高）")
    return specs


def load_state(plan_file: str) -> Optional[List[Dict]]:
    """读断点状态（matrix-plan.json 内嵌 specs）。文件不存在返回 None。"""
    if not os.path.exists(plan_file):
        return None
    try:
        with open(plan_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("experiments", [])
    except Exception:
        return None


def save_state(plan_file: str, specs: List[Dict], reasons: List[str]) -> None:
    """落 matrix-plan.json（含 specs 断点状态）+ matrix-plan.md（人读）。"""
    os.makedirs(os.path.dirname(plan_file), exist_ok=True)
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump({"experiments": specs, "planning_reasons": reasons},
                  f, ensure_ascii=False, indent=2)
    _write_plan_md(os.path.join(os.path.dirname(plan_file), "matrix-plan.md"), specs, reasons)


def _write_plan_md(path: str, specs: List[Dict], reasons: List[str]) -> None:
    lines = ["# 实验矩阵规划", ""]
    if reasons:
        lines.append("## 规划理由（组数自决）")
        lines.append("")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")
    lines.append("## 实验清单")
    lines.append("")
    lines.append("| id | algo | wave | sample | feat | depends_on | status |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in specs:
        lines.append("| {id} | {algo} | {wave} | {sample} | {feat} | {dep} | {status} |".format(
            id=s["id"], algo=s["algo"], wave=s["wave"],
            sample=s["sample_scheme"], feat=s["feat_scheme"],
            dep=s["depends_on"] or "-", status=s["status"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def update_spec(specs: List[Dict], exp_id_: str, **updates) -> None:
    for s in specs:
        if s["id"] == exp_id_:
            s.update(updates)
            return
    raise KeyError(f"experiment spec not found: {exp_id_}")


def get_spec(specs: List[Dict], exp_id_: str) -> Optional[Dict]:
    for s in specs:
        if s["id"] == exp_id_:
            return s
    return None