# -*- coding: utf-8 -*-
"""leaderboard：全实验 OOT AUC 排序总表 + 乐观偏差标注 + 诊断/调优列与明细 + 失败清单（plan §2.1 F2）。

输出 leaderboard.md / leaderboard.xlsx（缺 openpyxl 时仅 md）。
v2.5：表格新增「诊断」「调优」两列 + 「诊断与调优明细」小节（来源 = -opt 格 spec 的
diagnosis/optuna 字段，manifest.json 兜底）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def _load_diag_optuna(exp_dir: str, spec: Dict):
    """读格的诊断/调优信息：spec 优先（run_experiments 追加的 -opt spec 自带），
    manifest.json 兜底（断点/外部调用）。返回 (diagnosis, optuna)，可为 None。"""
    diag = spec.get("diagnosis")
    opt = spec.get("optuna")
    if diag is not None and opt is not None:
        return diag, opt
    mf = os.path.join(exp_dir, "manifest.json")
    if not os.path.exists(mf):
        return diag, opt
    try:
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (diag if diag is not None else data.get("diagnosis"),
                opt if opt is not None else data.get("optuna"))
    except Exception:
        return diag, opt


def _diag_str(r: Dict) -> str:
    """表格「诊断」列：-opt 格显示五状态；well_fit 未调优显示 well_fit(跳过调优)；其余 -。"""
    diag = r.get("diagnosis") or {}
    status = diag.get("status") or "-"
    if status == "well_fit" and not r.get("optuna"):
        return "well_fit(跳过调优)"
    return status


def _tune_str(r: Dict) -> str:
    """表格「调优」列：调优完成显示 trials+best_val；well_fit 未调优显示 跳过；其余 -。"""
    opt = r.get("optuna")
    if not opt:
        if (r.get("diagnosis") or {}).get("status") == "well_fit":
            return "跳过(well_fit)"
        return "-"
    return "tuned(%dt, best_val %.4f)" % (opt.get("n_trials", "?"),
                                          opt.get("best_value") or 0.0)


def collect_results(exp_root: str, specs: List[Dict]) -> List[Dict]:
    """扫描每格 manifest + eval.json，汇总 leaderboard 行。"""
    rows: List[Dict] = []
    for s in specs:
        exp_dir = os.path.join(exp_root, s["id"])
        diag, opt = _load_diag_optuna(exp_dir, s)
        if s["status"] == "failed":
            rows.append({
                "id": s["id"], "algo": s["algo"], "sample_scheme": s["sample_scheme"],
                "feat_scheme": s["feat_scheme"], "status": "failed",
                "fail_reason": s.get("fail_reason"), "oot_auc": None, "val_auc": None,
                "n_features": s.get("n_features"), "n_samples": s.get("n_samples"),
                "optimistic_bias": False, "is_tuned": bool(s.get("is_tuned")),
                "diagnosis": diag, "optuna": opt,
            })
            continue
        eval_json = os.path.join(exp_dir, "evaluation", "eval.json")
        if not os.path.exists(eval_json):
            rows.append({
                "id": s["id"], "algo": s["algo"], "sample_scheme": s["sample_scheme"],
                "feat_scheme": s["feat_scheme"], "status": "missing_eval", "fail_reason": None,
                "oot_auc": None, "val_auc": None,
                "n_features": s.get("n_features"), "n_samples": s.get("n_samples"),
                "optimistic_bias": False, "is_tuned": bool(s.get("is_tuned")),
                "diagnosis": diag, "optuna": opt,
            })
            continue
        with open(eval_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        splits = payload.get("splits", {})
        oot = splits.get("oot", {})
        val = splits.get("val", {})
        rows.append({
            "id": s["id"], "algo": s["algo"], "sample_scheme": s["sample_scheme"],
            "feat_scheme": s["feat_scheme"], "status": "done",
            "fail_reason": None,
            "oot_auc": oot.get("auc"), "val_auc": val.get("auc"),
            "oot_n": oot.get("n"), "val_n": val.get("n"),
            "n_features": s.get("n_features"), "n_samples": s.get("n_samples"),
            "optimistic_bias": bool(payload.get("optimistic_bias", False)),
            "is_tuned": bool(s.get("is_tuned")),
            "diagnosis": diag, "optuna": opt,
        })
    return rows


def sort_rows(rows: List[Dict]) -> List[Dict]:
    """按 OOT AUC 降序（None 排末尾），同值按 id。"""
    return sorted(rows, key=lambda r: (r["oot_auc"] is None, -(r["oot_auc"] or -1), r["id"]))


def top_k(rows: List[Dict], k: int = 10) -> List[Dict]:
    return [r for r in rows if r["status"] == "done"][:k]


def _format_signals(diag: Dict) -> str:
    """明细小节：把 signals 转成易读文本（过滤 None/非标量）。"""
    signals = diag.get("signals") or {}
    parts = []
    for k, v in signals.items():
        if v is None or isinstance(v, (dict, list)):
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def _detail_section(rows: List[Dict]) -> List[str]:
    """「诊断与调优明细」小节：仅对带 diagnosis/optuna 的格输出。"""
    lines: List[str] = []
    tuned_rows = [r for r in rows if (r.get("diagnosis") or r.get("optuna"))]
    if not tuned_rows:
        return lines
    lines.append("")
    lines.append("## 诊断与调优明细")
    lines.append("")
    for r in tuned_rows:
        diag = r.get("diagnosis") or {}
        opt = r.get("optuna") or {}
        lines.append(f"- **`{r['id']}`**")
        if diag:
            status = diag.get("status")
            reason = "；".join(diag.get("reasons") or []) or "-"
            sig = _format_signals(diag)
            skip = "（well_fit 跳过调优）" if status == "well_fit" and not opt else ""
            lines.append(f"  - 诊断: `{status}`{skip} — {reason}"
                         + (f"；信号: {sig}" if sig else ""))
        if opt:
            bp = opt.get("best_params") or {}
            params = "、".join(f"{k}={v}" for k, v in sorted(bp.items())
                               if k not in ("early_stopping", "scale_pos_weight"))
            lines.append("  - Optuna: trials=%d best_val_auc=%.4f" % (
                opt.get("n_trials", "?"), opt.get("best_value") or 0.0))
            if params:
                lines.append(f"    - best_params: {params}")
            ss = opt.get("search_space") or {}
            lines.append("    - search_space: %s" % ("，".join(
                f"{k}={tuple(v) if isinstance(v, list) else v}" for k, v in sorted(ss.items()))))
    return lines


def write_leaderboard(exp_root: str, specs: List[Dict],
                      out_prefix: Optional[str] = None) -> str:
    """写 leaderboard.md + leaderboard.xlsx（缺 openpyxl 仅 md），返回 md 路径。"""
    rows = collect_results(exp_root, specs)
    rows = sort_rows(rows)
    md_path = os.path.join(exp_root, out_prefix or "leaderboard.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    lines = ["# 实验 Leaderboard（OOT AUC 降序）", ""]
    header = ("| rank | id | algo | sample | feat | OOT AUC | val AUC | n_feat | n_train | 标注 | 诊断 | 调优 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for i, r in enumerate(rows, 1):
        note = "⚠ 乐观偏差" if r["optimistic_bias"] else "-"
        status_note = "FAILED" if r["status"] == "failed" else ""
        tuned_note = "tuned" if r.get("is_tuned") else ""
        lines.append("| {i} | {id} | {algo} | {sample} | {feat} | {oot} | {val} | {nf} | {ns} | {note}{status}{tuned} | {diag} | {tune} |".format(
            i=i, id=r["id"], algo=r["algo"], sample=r["sample_scheme"], feat=r["feat_scheme"],
            oot="-" if r["oot_auc"] is None else "%.4f" % r["oot_auc"],
            val="-" if r["val_auc"] is None else "%.4f" % r["val_auc"],
            nf=r["n_features"], ns=r["n_samples"], note=note, status=status_note,
            tuned=("/tuned" if tuned_note else ""),
            diag=_diag_str(r), tune=_tune_str(r)))
    # 失败清单
    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        lines.append("")
        lines.append("## 失败清单")
        lines.append("")
        for r in failed:
            lines.append("- `{id}`: {reason}".format(id=r["id"], reason=r["fail_reason"]))
    # 诊断与调优明细（v2.5）
    lines.extend(_detail_section(rows))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # xlsx（可选；新增 诊断/调优 两列文本）
    try:
        import pandas as pd

        pdf = pd.DataFrame([{
            **{k: v for k, v in r.items() if k not in ("optimistic_bias", "diagnosis", "optuna")},
            "诊断": _diag_str(r), "调优": _tune_str(r),
        } for r in rows])
        xlsx_path = os.path.join(exp_root, (out_prefix or "leaderboard").rsplit(".", 1)[0] + ".xlsx")
        if not out_prefix:
            xlsx_path = os.path.join(exp_root, "leaderboard.xlsx")
        pdf.to_excel(xlsx_path, index=False)
    except Exception:
        pass
    return md_path