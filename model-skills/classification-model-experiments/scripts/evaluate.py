# -*- coding: utf-8 -*-
"""精简四档评估（复用 _modelevo-shared/metrics.py：AUC/KS/IV/PSI/分桶）。

不 import training 的 eval_single.py —— 本模块自实现精简版，输出结构：
  evaluation/eval.json + evaluation/eval.md
  {
    "splits": {
      "train": {"n", "positive", "label_rate", "auc", "ks", "gini",
                "iv_psi": {"iv": {...} | None, "psi_oot": ...},    # 可选
                "buckets": [...]},
      "val": {...}, "oot": {...}, "all": {...}
    },
    "algo": ..., "features": N, "params": {...}, "optimistic_bias": bool,
  }
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  注入 _modelevo-shared
from metrics import calc_auc, calc_gini, calc_iv, calc_ks, decile_buckets, psi_from_series


def _label_stats(label: pd.Series) -> Dict:
    label = pd.to_numeric(label, errors="coerce")
    valid = label.dropna()
    n = len(valid)
    pos = int((valid > 0).sum())
    return {
        "n": n,
        "positive": pos,
        "label_rate": round(float(pos / n), 6) if n else None,
    }


def _split_eval(score: pd.Series, label: pd.Series,
                psi_base: Optional[pd.Series] = None,
                psi_score_name: str = "psi_oot") -> Dict:
    """单档评估；psi_base 非 None 时算分位 PSI（用于 OOT 参与 PSI 统计的红线例外格）。"""
    out: Dict = {}
    for key, fn in (("auc", calc_auc), ("ks", calc_ks), ("gini", calc_gini)):
        v = fn(score, label)
        out[key] = round(float(v), 6) if v is not None else None
    out.update(_label_stats(label))
    # 十分桶（以 score 降序，decile=N 最高分）
    sub = pd.DataFrame({"score": score, "label": label}).dropna()
    try:
        buckets = decile_buckets(sub, score_col="score")
        out["buckets"] = buckets
    except Exception:
        out["buckets"] = []
    if psi_base is not None:
        b_s = pd.to_numeric(psi_base, errors="coerce")
        s_s = pd.to_numeric(score, errors="coerce")
        v = psi_from_series(b_s, s_s, n_bins=10)
        out[psi_score_name] = round(float(v), 6) if v is not None else None
    return out


def evaluate(
    scores: Dict[str, pd.Series],
    labels: Dict[str, pd.Series],
    *,
    algo: str,
    features: List[str],
    params: Optional[Dict] = None,
    optimistic_bias: bool = False,
    oot_psi_base: Optional[pd.Series] = None,
    iv_features: Optional[List[str]] = None,
    iv_train_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """四档评估（train / val / oot / all）。

    Args:
        scores: {split: score_series}
        labels: {split: label_series}
        optimistic_bias: 该格 OOT 指标存在乐观偏差（对抗/IV-PSI 例外格）
        oot_psi_base: OOT 参与 PSI 统计时传 base（train 段分数）
        iv_features / iv_train_df: IV-PSI 格时传，计算全特征 IV 直算表
    """
    result: Dict[str, Dict] = {}
    for split in ("train", "val", "oot", "all"):
        if split not in scores:
            continue
        psi_base = None
        if split == "oot" and optimistic_bias:
            psi_base = oot_psi_base  # OOT 参与 PSI 统计（红线例外②，仅例外格传）
        result[split] = _split_eval(scores[split], labels[split], psi_base=psi_base)

    payload = {
        "splits": result,
        "algo": algo,
        "features": len(features),
        "params": params or {},
        "optimistic_bias": optimistic_bias,
    }
    if iv_features is not None and iv_train_df is not None:
        iv_table = {}
        label = pd.to_numeric(iv_train_df.get("label", pd.Series(dtype=float)), errors="coerce")
        for f in iv_features:
            if f in iv_train_df.columns:
                v = calc_iv(iv_train_df[f], label, n_bins=10)
                iv_table[f] = round(float(v), 6) if v is not None else None
            else:
                iv_table[f] = None
        payload["iv_table"] = iv_table
    return payload


def write_eval(payload: Dict, out_dir: str, exp_id: str) -> None:
    """落 evaluation/eval.json + eval.md。"""
    import os

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows = []
    for split, m in payload["splits"].items():
        rows.append({
            "split": split,
            "n": m.get("n"),
            "label_rate": m.get("label_rate"),
            "auc": m.get("auc"),
            "ks": m.get("ks"),
            "gini": m.get("gini"),
        })
    md = [f"# {exp_id} 评估", "",
          "| split | n | label_rate | AUC | KS | Gini |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {split} | {n} | {lr} | {auc} | {ks} | {gini} |".format(
            split=r["split"], n=r["n"],
            lr="-" if r["label_rate"] is None else f"{r['label_rate']:.4f}",
            auc="-" if r["auc"] is None else f"{r['auc']:.4f}",
            ks="-" if r["ks"] is None else f"{r['ks']:.4f}",
            gini="-" if r["gini"] is None else f"{r['gini']:.4f}"))
    if payload.get("optimistic_bias"):
        md.append("")
        md.append("> ⚠ 本格 OOT 参与对抗/PSI 统计，OOT 指标存在**乐观偏差**（红线例外，仅本模块授权）。")
    with open(os.path.join(out_dir, "eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")