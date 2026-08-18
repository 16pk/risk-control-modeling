# -*- coding: utf-8 -*-
"""leaderboard：全实验 OOT AUC 排序总表 + 乐观偏差标注 + 失败清单（plan §2.1 F2）。

输出 leaderboard.md / leaderboard.xlsx（缺 openpyxl 时仅 md）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def collect_results(exp_root: str, specs: List[Dict]) -> List[Dict]:
    """扫描每格 manifest + eval.json，汇总 leaderboard 行。"""
    rows: List[Dict] = []
    for s in specs:
        if s["status"] == "failed":
            rows.append({
                "id": s["id"], "algo": s["algo"], "sample_scheme": s["sample_scheme"],
                "feat_scheme": s["feat_scheme"], "status": "failed",
                "fail_reason": s.get("fail_reason"), "oot_auc": None, "val_auc": None,
                "n_features": s.get("n_features"), "n_samples": s.get("n_samples"),
                "optimistic_bias": False,
            })
            continue
        exp_dir = os.path.join(exp_root, s["id"])
        eval_json = os.path.join(exp_dir, "evaluation", "eval.json")
        if not os.path.exists(eval_json):
            rows.append({
                "id": s["id"], "algo": s["algo"], "sample_scheme": s["sample_scheme"],
                "feat_scheme": s["feat_scheme"], "status": "missing_eval", "fail_reason": None,
                "oot_auc": None, "val_auc": None,
                "n_features": s.get("n_features"), "n_samples": s.get("n_samples"),
                "optimistic_bias": False,
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
        })
    return rows


def sort_rows(rows: List[Dict]) -> List[Dict]:
    """按 OOT AUC 降序（None 排末尾），同值按 id。"""
    return sorted(rows, key=lambda r: (r["oot_auc"] is None, -(r["oot_auc"] or -1), r["id"]))


def top_k(rows: List[Dict], k: int = 10) -> List[Dict]:
    return [r for r in rows if r["status"] == "done"][:k]


def write_leaderboard(exp_root: str, specs: List[Dict],
                      out_prefix: Optional[str] = None) -> str:
    """写 leaderboard.md + leaderboard.xlsx（缺 openpyxl 仅 md），返回 md 路径。"""
    rows = collect_results(exp_root, specs)
    rows = sort_rows(rows)
    md_path = os.path.join(exp_root, out_prefix or "leaderboard.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    lines = ["# 实验 Leaderboard（OOT AUC 降序）", ""]
    lines.append("| rank | id | algo | sample | feat | OOT AUC | val AUC | n_feat | n_train | 标注 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        note = "⚠ 乐观偏差" if r["optimistic_bias"] else "-"
        status_note = "FAILED" if r["status"] == "failed" else ""
        lines.append("| {i} | {id} | {algo} | {sample} | {feat} | {oot} | {val} | {nf} | {ns} | {note}{status} |".format(
            i=i, id=r["id"], algo=r["algo"], sample=r["sample_scheme"], feat=r["feat_scheme"],
            oot="-" if r["oot_auc"] is None else "%.4f" % r["oot_auc"],
            val="-" if r["val_auc"] is None else "%.4f" % r["val_auc"],
            nf=r["n_features"], ns=r["n_samples"], note=note, status=status_note))
    # 失败清单
    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        lines.append("")
        lines.append("## 失败清单")
        lines.append("")
        for r in failed:
            lines.append("- `{id}`: {reason}".format(id=r["id"], reason=r["fail_reason"]))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # xlsx（可选）
    try:
        import pandas as pd

        pdf = pd.DataFrame([{k: v for k, v in r.items() if k != "optimistic_bias"} for r in rows])
        xlsx_path = os.path.join(exp_root, (out_prefix or "leaderboard").rsplit(".", 1)[0] + ".xlsx")
        if not out_prefix:
            xlsx_path = os.path.join(exp_root, "leaderboard.xlsx")
        pdf.to_excel(xlsx_path, index=False)
    except Exception:
        pass
    return md_path