# -*- coding: utf-8 -*-
"""把基础统计/IV/PSI 3 个结果渲染成单一 markdown 报告。

只产报告, 不写 selected_features.txt; 是否剔除特征由人工决定。
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def _df_to_md(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """DataFrame -> markdown 表; 空表给占位。"""
    if df is None or df.empty:
        return "_(无数据)_"
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([head, sep] + body)


def _render_split_stats(split_meta: dict) -> List[str]:
    """渲染切分统计段(三档样本量/正样本率/正负比/dropped_rows/跨档差异)。

    合并 task-spec 后置的切分统计口径, 与 task-spec 原「Train/Test/OOT 切分」段对齐。
    """
    sc = split_meta.get("sample_counts") or {}
    pr = split_meta.get("pos_rates") or {}
    pnr = split_meta.get("positive_negative_ratio") or {}
    lines = ["", "### 切分统计 (Train/Test/OOT)", ""]
    lines.append("| 集合 | 样本量 | 正样本率 | 正负比 |")
    lines.append("|------|--------|----------|--------|")
    for nm in ("train", "test", "oot"):
        n = sc.get(nm)
        rate = pr.get(nm)
        ratio = pnr.get(nm)
        rate_s = f"{rate:.2%}" if isinstance(rate, (int, float)) else "—"
        lines.append(f"| {nm} | {n if n is not None else '—'} | {rate_s} | {ratio or '—'} |")
    lines.append("")
    dropped = split_meta.get("dropped_rows")
    cross = split_meta.get("cross_split_pos_rate_diff_pp")
    if dropped is not None or cross is not None:
        lines.append(
            f"> 区间外/label 非法剔除行数(dropped_rows): {dropped if dropped is not None else '—'}; "
            f"三档正样本率跨档差异: {cross if cross is not None else '—'}pp"
        )
        lines.append("")
    return lines


def render_report(
    cfg: dict,
    features: List[str],
    n_total: int,
    n_train: int,
    n_oot: int,
    stats_df: pd.DataFrame,
    iv_df: pd.DataFrame,
    psi_df: pd.DataFrame,
    woe_df: pd.DataFrame = None,
    n_test: int = 0,
    woe_top_n: int = 20,
    invalid_report: pd.DataFrame = None,
    split_meta: dict = None,
) -> str:
    """渲染特征报告 markdown 文本。

    Args:
        cfg: 完整配置(读 model.name / analysis 阈值)
        features: 本次分析的特征清单
        n_total/n_train/n_oot: 样本规模
        stats_df / iv_df / psi_df: 各分析结果
        woe_df: WOE 分桶明细 long-format 表; 非空时渲染 三-bis 段, 取 IV Top N 特征展开
        n_test: 测试段样本量; 0 时不显示该段
        woe_top_n: WOE 段展开的特征数(按 iv_df 顺序取前 N); 默认 20
        invalid_report: 哨兵值替换明细(feature/hit_values/n_hit/hit_ratio); 非空时渲染提醒段
        split_meta: 切分统计(_split_sample_to_three 返回的 report), 非空时在概述段渲染三档样本量/正样本率/正负比/dropped_rows/跨档差异

    Returns:
        markdown 字符串
    """
    model = cfg.get("model") or {}
    analysis = cfg.get("analysis") or {}
    psi_warn = (analysis.get("psi") or {}).get("warn_threshold", 0.10)

    name = model.get("name", "(未命名)")
    label_col = model.get("label_col", "label")

    if n_test > 0:
        seg_line = f"- 全样本规模: **{n_total}**; 训练段: {n_train}; 测试段: {n_test}; OOT 段: {n_oot}"
    else:
        seg_line = f"- 全样本规模: **{n_total}**; 训练段: {n_train}; OOT 段: {n_oot}"

    lines: List[str] = []
    lines.append(f"# 特征分析报告 - {name}")
    lines.append("")
    lines.append("## 一、概述")
    lines.append("")
    lines.append("- 样本来源: 由 `data-cleaning` 产出 (sample.parquet)")
    lines.append(f"- 标签列: `{label_col}`")
    lines.append(f"- 特征数: **{len(features)}**")
    lines.append(seg_line)
    lines.append(f"- 配置: PSI 阈值={psi_warn}")
    if split_meta:
        lines.extend(_render_split_stats(split_meta))
    lines.append("")
    lines.append("> 本报告仅描述特征质量, **不**自动剔除特征。")
    lines.append("")

    lines.append("## 二、基础统计")
    lines.append("")
    lines.append(_df_to_md(stats_df))
    lines.append("")

    lines.append("## 三、单变量预测力 (IV / AUC), 按 IV 降序")
    lines.append("")
    lines.append("> **AUC 口径说明**: 本表的 AUC 是把特征做 WoE 编码后算 `roc_auc_score(y, woe(x))`,")
    lines.append("> 等价于按 bin 的正样本率排序的 AUC; 支持分类/缺失列, 但 WoE 用了样本内标签信息,")
    lines.append("> 数值轻微偏乐观, **不是 raw-feature ROC-AUC**。")
    lines.append("")
    lines.append(_df_to_md(iv_df))
    lines.append("")

    # 三-bis WOE 分桶明细: 取 IV Top N 特征展开各桶
    if woe_df is not None and not woe_df.empty and iv_df is not None and not iv_df.empty:
        top_features = list(iv_df.dropna(subset=["iv"]).head(woe_top_n)["feature"])
        if top_features:
            view = woe_df[woe_df["feature"].isin(top_features)].copy()
            # 保持 iv_df 的 Top N 顺序
            order = {f: i for i, f in enumerate(top_features)}
            view["_ord"] = view["feature"].map(order)
            view = view.sort_values(["_ord", "bin"]).drop(columns=["_ord"])
            lines.append(f"## 三-bis WOE 分桶明细 (IV Top {len(top_features)} 特征)")
            lines.append("")
            lines.append(_df_to_md(view))
            lines.append("")
            lines.append(f"> 全量 WOE 明细见 `woe_table.csv`。")
            lines.append("")

    lines.append(f"## 四、训练 vs OOT PSI (warn 阈值={psi_warn})")
    lines.append("")
    if psi_df is not None and not psi_df.empty:
        view = psi_df.copy()
        view["flag"] = view["warn"].map(lambda x: "[PSI_WARN]" if x else "")
        lines.append(_df_to_md(view[["feature", "psi", "flag"]]))
    else:
        lines.append("_(无 PSI 结果, 可能未切出 OOT)_")
    lines.append("")

    # 五、哨兵值校验提醒(替换已上移到 data-cleaning, 本阶段只检查残留)
    if invalid_report is not None and not invalid_report.empty:
        lines.append("## 五、无效值哨兵校验提醒")
        lines.append("")
        lines.append("> 哨兵值集合: `-1,-2,-9,-99,-999,-9999,-99999`(可配 `model.invalid_values`)。"
                     "哨兵值替换已由上游 `data-cleaning` 完成, 本阶段**仅校验是否仍有残留**, 不修改数据。"
                     "检测到以下特征仍残留哨兵值, 请确认是否已运行 data-cleaning 清洗。")
        lines.append("")
        lines.append(_df_to_md(invalid_report))
        lines.append("")
        lines.append("> 明细见 `invalid-values-report.csv`; 若为清洗前残留, 建议回退到 data-cleaning 处理。")
        lines.append("")

    return "\n".join(lines)
