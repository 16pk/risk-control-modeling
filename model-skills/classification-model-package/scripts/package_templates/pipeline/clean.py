#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据清理: 哨兵值 → NaN（仅特征列, 非交互 + WARN）。自包含, 零专家包依赖。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_INVALID_VALUES = [-1, -2, -9, -99, -999, -9999, -99999]


def load_invalid_values(assets_dir) -> List[float]:
    """读 assets/cleaning-scheme.json 的 invalid_values; 缺文件/缺字段回退默认集合并 WARN。"""
    scheme_path = assets_dir / "cleaning-scheme.json"
    if not scheme_path.exists():
        print(f"[WARN] 缺少 {scheme_path.name}, 使用默认哨兵集 {DEFAULT_INVALID_VALUES}")
        return list(DEFAULT_INVALID_VALUES)
    try:
        import json
        with scheme_path.open("r", encoding="utf-8") as f:
            scheme = json.load(f)
        vals = scheme.get("invalid_values") or scheme.get("invalidValues")
        if vals:
            return [float(v) for v in vals]
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 解析 {scheme_path.name} 失败({e}), 使用默认哨兵集 {DEFAULT_INVALID_VALUES}")
    return list(DEFAULT_INVALID_VALUES)


def clean_sentinel(df: pd.DataFrame, features: List[str], invalid_values: List[float]
                   ) -> tuple[pd.DataFrame, dict]:
    """仅对特征列做哨兵值替换为 NaN; 返回 (清洗后 df, 清洗报告 dict)。

    - 非交互: 命中不暂停, 只打 WARN（交付包必须无人值守可跑）。
    - 不做样本去重; 不校验 id/dt/label 列是否存在（允许缺 label）。
    - 仅替换数值列, 非数值列跳过。
    """
    if not invalid_values:
        return df, {"invalid_values": invalid_values, "features": [], "message": "空哨兵集"}
    df_clean = df.copy()
    report_rows = []
    for fc in features:
        if fc not in df_clean.columns:
            continue
        s = df_clean[fc]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        hit = [v for v in invalid_values if (s == v).any()]
        if not hit:
            continue
        mask = s.isin(hit)
        n_hit = int(mask.sum())
        hit_str = ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in hit)
        report_rows.append({
            "feature": fc,
            "hit_values": hit_str,
            "n_hit": n_hit,
            "hit_ratio": round(n_hit / len(s), 6),
        })
        df_clean.loc[mask, fc] = np.nan
    report = {"invalid_values": [float(v) for v in invalid_values], "features": report_rows}
    if report_rows:
        print(f"[WARN] 哨兵值命中 {len(report_rows)} 个特征（已静默替换为 NaN）:")
        for r in report_rows:
            print(f"[WARN]   - {r['feature']}: hit={r['hit_values']} n={r['n_hit']} "
                  f"ratio={r['hit_ratio']}")
    return df_clean, report