# -*- coding: utf-8 -*-
"""按「用户 + 日期」维度对样本去重。

冲突处理规则: 同一 (id_col, dt_col) 出现多行时, 优先保留 label 非空/非 NaN 的那一行;
组内 label 全空(或未提供 label_col)时保留首行兜底(保证结果确定性, 用稳定排序保留原始相对顺序)。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def dedup_by_user_date(
    df: pd.DataFrame,
    id_col: str,
    dt_col: str,
    label_col: Optional[str] = None,
) -> tuple:
    """按 (id_col, dt_col) 去重, 冲突保留 label 非空行。

    Args:
        df: 全量样本
        id_col: 用户粒度 ID 列
        dt_col: 日期分区列
        label_col: 标签列名(可选); 提供时组内优先保留 label 非空行

    Returns:
        (df_deduped, dedup_report): dedup_report = {n_before, n_after, n_removed, n_dup_groups}
    """
    n_before = int(len(df))
    keys = [id_col, dt_col]

    if label_col and label_col in df.columns:
        # 组内优先保留 label 非空行: 加临时 rank 列(非空=1 排前), 稳定排序后按 keys 去重取首行。
        # 稳定排序保证「组内 label 全空 / 多个非空」时仍按原始行序取首行, 结果确定。
        rank_col = "_dedup_label_rank"
        df = pd.concat(
            [df, df[label_col].notna().astype(int).to_frame(rank_col)],
            axis=1,
        )
        df = (
            df.sort_values(keys + [rank_col], ascending=[True, True, False], kind="stable")
            .drop_duplicates(subset=keys, keep="first")
            .drop(columns=[rank_col])
        )
    else:
        df = df.drop_duplicates(subset=keys, keep="first")

    df = df.reset_index(drop=True)
    n_after = int(len(df))
    report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_before - n_after,
    }
    return df, report
