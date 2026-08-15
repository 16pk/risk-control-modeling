# -*- coding: utf-8 -*-
"""哨兵值/无效值替换为 NaN(自 feature-analysis 迁移, data-cleaning 统一收口)。

仅作用于入模特征列; label / id / dt 列不参与替换, 避免误伤标签与主键。
原 feature-analysis 中的替换动作已降级为「校验 + 提醒」, 本模块是哨兵值集合的
唯一权威管理点。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_INVALID_VALUES = [-1, -2, -9, -99, -999, -9999, -99999]


def parse_invalid_values(cfg_val, cli_val: Optional[str]) -> list:
    """解析哨兵值集合: CLI(--invalid-values) > yaml(model.invalid_values) > 默认。

    哨兵值 = 数据中代表"无数据/拒贷/异常"的占位取值(如 -1/-2/-999/-9999)。
    """
    raw = None
    if cli_val is not None and str(cli_val).strip():
        raw = str(cli_val)
    elif cfg_val is not None:
        # yaml 里可以是 list 或逗号分隔字符串
        if isinstance(cfg_val, (list, tuple)):
            return [float(v) for v in cfg_val]
        raw = str(cfg_val)
    if raw is None or not raw.strip():
        return []
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            print(f"[invalid-values] ⚠ 忽略非数值哨兵项: {part!r}")
    return vals


def replace_invalid_values(
    df: pd.DataFrame, features: List[str], invalid_values: list, label_col: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把特征列中的哨兵值(如 -1/-2/-999/-9999)替换为 NaN。

    仅作用于入模特征列(features); label / id / dt 列不参与。
    返回 (替换后的 df, 替换统计 DataFrame)。

    Args:
        df: 全量样本
        features: 入模特征清单(仅这些列会被检查替换)
        invalid_values: 哨兵值集合; 空列表则跳过
        label_col: 标签列名, 用于统计各特征替换前后的坏率变化(可选)

    Returns:
        (df_cleaned, report_df): report_df 列 = feature / hit_values / n_hit / hit_ratio
    """
    if not invalid_values:
        return df, pd.DataFrame(columns=["feature", "hit_values", "n_hit", "hit_ratio"])

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
        report_rows.append({
            "feature": fc,
            "hit_values": ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in hit),
            "n_hit": n_hit,
            "hit_ratio": round(n_hit / len(s), 6),
        })
        df_clean.loc[mask, fc] = np.nan

    report_df = pd.DataFrame(report_rows)
    return df_clean, report_df
