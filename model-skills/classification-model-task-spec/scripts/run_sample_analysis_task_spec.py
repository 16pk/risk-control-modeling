# -*- coding: utf-8 -*-
"""纯样本分析脚本(切分已后置到 feature-analysis)。

输入: task-spec 自拉/用户提供的全量样本 parquet (含 id_col / label_col / time_col 三列及可选补充字段)。
      id_col 默认 fuid, 可由 --id-cols override (分析侧取单列作主 ID, 与 fetch 侧 --id-cols 多列命名对齐)。
      日期兼容 YYYY-MM-DD 与 8 位 YYYYMMDD 两种格式(内部统一归一化比较)。
输出: report.md / report.xlsx / _manifest.json, 全部落在 --output-dir。

本脚本只做「需求确认阶段」的标签质量分析:
  - compute_overall        全量样本概览(样本量/正样本率/正负比/pday 范围等)
  - segment_by_time        分时间段统计(默认按月 YYYYMM 聚合)
  - judge_stability        标签时间稳定性判定
  - judge_sufficiency      样本充足度判定
**不再**做三档切分(train/test/oot)与切分统计: 该职责已后置到 feature-analysis
(基于 data-cleaning 清洗后样本, 单一真相为 feature_config.yaml 的 model.split)。

日期归一化复用公共 _modelevo-shared/scripts/date_utils(通过 _bootstrap 注入)。

用法:
    python run_sample_analysis_task_spec.py \
        --sample .../sample.parquet \
        --model-name call_complaint \
        --timestamp 20260629-161231 \
        --output-dir .../data-profile/ \
        [--dt-col f_p_date] [--label-col label] [--id-cols fuid] [--sample-table your_db.xxx]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts 供 date_utils 使用
import pandas as pd


# ---------- 输入校验 ----------

def validate_input(df, args) -> None:
    """校验列存在 / label 取值 / 日期列格式(YYYY-MM-DD 或 YYYYMMDD 双兼容)。"""
    import date_utils

    for col in (args.dt_col, args.label_col, args._id_col_primary):
        if col not in df.columns:
            raise SystemExit("样本缺必要列: %s" % col)

    bad_labels = sorted(set(df[args.label_col].dropna().unique()) - {0, 1})
    if bad_labels:
        raise SystemExit("label 列存在非法取值 %s, 仅允许 0/1" % bad_labels)

    date_str = df[args.dt_col].astype(str)
    non_nan = set(date_str[date_str != "nan"])
    bad_dates = sorted(d for d in non_nan if not date_utils.is_date(d))
    if bad_dates:
        raise SystemExit("日期列存在非法值(须为 YYYY-MM-DD 或 8 位 YYYYMMDD): %s" % bad_dates[:5])


# ---------- 统计计算 ----------

def _fmt_ratio(pos: int, neg: int) -> str:
    if neg == 0:
        return "1 : 0" if pos > 0 else "NA"
    g = _gcd(pos, neg)
    return "1 : %.2f" % (neg / max(pos, 1))


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def compute_overall(df, args) -> dict:
    """全量样本统计。"""
    total = int(len(df))
    label_col = args.label_col
    time_col = args.dt_col
    id_col = args._id_col_primary

    valid = df[label_col].isin([0, 1])
    nan_label = int((~valid).sum())
    if nan_label:
        print("[run_sample_analysis_task_spec] [警告] %d 行 label 为 NaN 或非法, 已从统计中剔除" % nan_label, file=sys.stderr)
    pos = int((df.loc[valid, label_col] == 1).sum())
    neg = int((df.loc[valid, label_col] == 0).sum())
    rate = round(pos / total, 6) if total else 0.0

    pday_vals = sorted(df[time_col].astype(str).unique().tolist())
    pday_min = pday_vals[0] if pday_vals else None
    pday_max = pday_vals[-1] if pday_vals else None

    overall = {
        "total_samples": total,
        "positive_samples": pos,
        "negative_samples": neg,
        "positive_rate": rate,
        "positive_negative_ratio": _fmt_ratio(pos, neg),
        "pday_range": [pday_min, pday_max],
        "pday_unique_count": len(pday_vals),
        "pday_values": pday_vals,
        "null_counts": {c: int(df[c].isna().sum()) for c in [id_col, label_col, time_col]},
        "user_unique": int(df[id_col].nunique()),
        "dup_user_pday": int(df.duplicated(subset=[id_col, time_col]).sum()),
    }

    return overall


def segment_by_time(df, args) -> List[dict]:
    """按时间分段统计, 默认按月(YYYYMM)聚合。

    数据可能 YYYY-MM-DD 或 YYYYMMDD, 统一归一化后取月份前缀 YYYYMM。
    """
    import date_utils

    time_col = args.dt_col
    label_col = args.label_col

    df_dt = df.copy()
    df_dt["_month"] = df_dt[time_col].astype(str).map(
        lambda v: date_utils.month_prefix(v) if date_utils.is_date(v) else v
    )
    segments = []
    for month in sorted(df_dt["_month"].unique().tolist()):
        sub = df_dt[df_dt["_month"] == month]
        n = int(len(sub))
        pos = int((sub[label_col] == 1).sum())
        neg = int((sub[label_col] == 0).sum())
        segments.append({
            "pday": month,
            "samples": n,
            "positive": pos,
            "positive_rate": round(pos / n, 6) if n else 0.0,
            "pos_neg_ratio": _fmt_ratio(pos, neg),
        })
    return segments


def judge_stability(segments: List[dict]) -> dict:
    """稳定性判定: <1pp 稳定 / 1-3pp 轻微 / >=3pp 显著。"""
    if not segments:
        return {"positive_rate_max": 0, "positive_rate_min": 0, "volatility_pp": 0,
                "std_pp": 0, "judgment": "稳定", "note": "无时间段数据"}

    rates = [s["positive_rate"] for s in segments]
    rmax = max(rates)
    rmin = min(rates)
    vol_pp = round((rmax - rmin) * 100, 2)
    mean = sum(rates) / len(rates)
    std_pp = round(((sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5) * 100, 2)

    if vol_pp < 1:
        judgment = "稳定"
    elif vol_pp < 3:
        judgment = "轻微波动"
    else:
        judgment = "显著波动"

    min_seg = next(s for s in segments if s["positive_rate"] == rmin)
    max_seg = next(s for s in segments if s["positive_rate"] == rmax)
    min_label = min_seg.get("pday", min_seg.get("pday_segment", "NA"))
    max_label = max_seg.get("pday", max_seg.get("pday_segment", "NA"))
    note = "正样本率最低: %s (%.2f%%), 最高: %s (%.2f%%)" % (
        min_label, rmin * 100, max_label, rmax * 100)

    return {
        "positive_rate_max": rmax,
        "positive_rate_min": rmin,
        "volatility_pp": vol_pp,
        "std_pp": std_pp,
        "judgment": judgment,
        "note": note,
    }


def judge_sufficiency(overall: dict) -> dict:
    """样本充足度判定。"""
    pos = overall["positive_samples"]
    total = overall["total_samples"]
    if pos >= 10000 and total >= 100000:
        judgment = "充足"
    elif pos >= 500 and total >= 50000:
        judgment = "基本可用"
    else:
        judgment = "不足，建议补充样本"
    return {
        "judgment": judgment,
        "positive_meets_10k": pos >= 10000,
        "positive_count": pos,
        "total_meets_50k": total >= 50000,
        "total_count": total,
    }


# ---------- 输出 ----------

def write_report_md(overall, segments, stability, sufficiency, args, output_dir: str) -> None:
    n_seg = len(segments)

    def _pos_pct(x):
        return "%.2f%%" % (x * 100)

    seg_rows = "\n".join(
        "| %s | %s | %s | %s | %s |" % (
            s["pday"], f"{s['samples']:,}", f"{s['positive']:,}",
            _pos_pct(s["positive_rate"]), s["pos_neg_ratio"]
        ) for s in segments
    )

    id_col = args._id_col_primary
    time_col = args.dt_col

    source_label = args.local_parquet_path or args.sample or "未提供"
    source_line = "> 数据位置: %s" % source_label

    content = f"""# 样本分析报告

> 模型名称: {args.model_name}
> Session: {args.timestamp}
{source_line}

## 一、样本概览

| 指标 | 值 |
|------|-----|
| 总样本量 | {overall['total_samples']:,} |
| 正样本数 | {overall['positive_samples']:,} |
| 负样本数 | {overall['negative_samples']:,} |
| 正样本率 | {overall['positive_rate']:.2%} |
| 正负比 | {overall['positive_negative_ratio']} |
| pday 范围 | {overall['pday_range'][0]} ~ {overall['pday_range'][1]} |
| 时间段数 | {n_seg} |
| 用户数（{id_col} 去重） | {overall.get('user_unique', 'NA')} |
| {id_col} + {time_col} 重复 | {overall.get('dup_user_pday', 'NA')} |

## 二、分时间段标签分布（按月）

| 时间段 | 样本量 | 正样本数 | 正样本率 | 正负比 |
|--------|--------|----------|----------|--------|
{seg_rows}

## 三、标签稳定性

- 正样本率波动幅度（max - min） = {_pos_pct(stability['positive_rate_max'])} - {_pos_pct(stability['positive_rate_min'])} = **{stability['volatility_pp']:.2f}pp**
- 正样本率标准差 = {stability['std_pp']:.2f}pp

**判定：{stability['judgment']}**

{stability['note']}

## 四、综合判定与建模建议

### 4.1 样本可行性判定

| 维度 | 结果 |
|------|------|
| 标签分布 | {'✓' if 0.01 <= overall['positive_rate'] <= 0.99 else '⚠'} 正样本率 {overall['positive_rate']:.2%} |
| 正负比 | {'✓' if overall['positive_rate'] >= 0.01 else '⚠'} {overall['positive_negative_ratio']} |
| 时间稳定性 | {'✓' if stability['judgment'] == '稳定' else '⚠'} {stability['judgment']}（{stability['volatility_pp']:.2f}pp） |
| 样本量充足度 | {'✓' if sufficiency['judgment'] == '充足' else ('⚠' if sufficiency['judgment'] == '基本可用' else '✗')} {sufficiency['judgment']} |

### 4.2 建模建议

- {'无需重采样' if overall['positive_rate'] >= 0.05 else '建议重采样或调 scale_pos_weight'}：正样本率 {overall['positive_rate']:.2%}
- {'关注 Train→OOT 漂移' if stability['judgment'] != '稳定' else '标签稳定'}：波动幅度 {stability['volatility_pp']:.2f}pp，建模时关注 Train→OOT PSI
- 样本量：{sufficiency['judgment']}

## 五、下一步

1. ⏳ **等用户确认样本分析结果**
2. 之后 → `data-cleaning` 数据清洗（哨兵值替换 + 去重 + 派生特征清单）
3. 之后 → `feature-analysis` 特征分析（切分三档 + IV/PSI/基础统计）
4. 最后 → 建模决策（询问是否进入 model-development）
"""
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(content)


def write_report_xlsx(overall, segments, args, output_dir: str) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[run_sample_analysis_task_spec] [警告] 缺 openpyxl, 跳过 report.xlsx 产出", file=sys.stderr)
        return

    xlsx_path = os.path.join(output_dir, "report.xlsx")
    overview_df = pd.DataFrame([{
        "总样本量": overall["total_samples"],
        "正样本数": overall["positive_samples"],
        "负样本数": overall["negative_samples"],
        "正样本率": overall["positive_rate"],
        "正负比": overall["positive_negative_ratio"],
        "pday范围": "%s ~ %s" % (overall["pday_range"][0], overall["pday_range"][1]),
        "pday数": overall["pday_unique_count"],
    }])
    seg_df = pd.DataFrame(segments).rename(columns={
        "pday": "时间段", "samples": "样本量", "positive": "正样本数",
        "positive_rate": "正样本率", "pos_neg_ratio": "正负比",
    })

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        overview_df.to_excel(w, sheet_name="overview", index=False)
        seg_df.to_excel(w, sheet_name="time_segments", index=False)


def write_manifest(overall, segments, stability, sufficiency, args, output_dir: str) -> None:
    manifest = {
        "produced_by": "classification-skills/classification-model-task-spec",
        "schema_version": 1,
        "model_name": args.model_name,
        "timestamp": args.timestamp,
        "sample_table": args.sample_table,
        "sample_file": os.path.abspath(args.sample),
        "sample_summary": overall,
        "time_segments": segments,
        "stability": stability,
        "sample_sufficiency": sufficiency,
        "user_confirmed": False,
    }
    with open(os.path.join(output_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def print_summary(overall, stability, sufficiency) -> None:
    print()
    print("样本分析完成")
    print()
    print("样本概览:")
    print("- 总样本: %s，正样本率 %.2f%%" % (f"{overall['total_samples']:,}", overall["positive_rate"] * 100))
    print("- pday 范围: %s ~ %s" % (overall["pday_range"][0], overall["pday_range"][1]))
    print("- 标签稳定性: %s（%.2fpp）" % (stability["judgment"], stability["volatility_pp"]))
    print("- 样本充足度: %s" % sufficiency["judgment"])
    print()
    print("切分三档将在 feature-analysis 阶段完成(基于 data-cleaning 清洗后样本)")


# ---------- 主入口 ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="classification-model-task-spec 纯样本分析(切分已后置到 feature-analysis)")
    parser.add_argument("--sample", required=True, help="task-spec 拉取/用户提供的全量样本 parquet 路径")
    parser.add_argument("--model-name", required=True, help="模型英文简称")
    parser.add_argument("--timestamp", required=True, help="session 时间戳 YYYYMMDD-HHMMSS")
    parser.add_argument("--output-dir", required=True, help="输出目录 (通常是 data-profile/)")
    parser.add_argument("--dt-col", default="f_p_date", help="时间列名, 默认 f_p_date")
    parser.add_argument("--time-col", default=argparse.SUPPRESS, dest="time_col_deprecated",
                        help="[已弃用 alias] 用 --dt-col; 传入时覆盖 --dt-col")
    parser.add_argument("--label-col", default="label", help="标签列名, 默认 label")
    parser.add_argument("--id-cols", default="fuid", help="用户唯一标识列名(逗号分隔多列时取首列作主 ID), 默认 fuid")
    parser.add_argument("--id-col", default=argparse.SUPPRESS, dest="id_col_deprecated",
                        help="[已弃用 alias] 用 --id-cols; 传入时覆盖 --id-cols (单列)")
    parser.add_argument("--sample-table", default=None, help="样本源表名 (写入 manifest 用, 可选, 仅记录)")
    parser.add_argument("--local-parquet-path", default=None,
                        help="本地数据路径 (parquet/csv/feather), 用于报告头展示")
    args = parser.parse_args()

    # 兼容老参数: --time-col / --id-col 仍可传入 (已弃用, 推荐用 --dt-col / --id-cols)
    if hasattr(args, "time_col_deprecated") and args.time_col_deprecated:
        args.dt_col = args.time_col_deprecated
    if hasattr(args, "id_col_deprecated") and args.id_col_deprecated:
        args.id_cols = args.id_col_deprecated
    # 分析侧只取单列作主 ID
    args._id_col_primary = args.id_cols.split(",")[0].strip() if args.id_cols else "fuid"

    if not os.path.exists(args.sample):
        raise SystemExit("样本文件不存在: %s" % args.sample)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[run_sample_analysis_task_spec] 读取样本: %s" % args.sample)
    df = pd.read_parquet(args.sample)
    print("[run_sample_analysis_task_spec] 样本: %d 行 x %d 列" % (len(df), df.shape[1]))

    validate_input(df, args)

    overall = compute_overall(df, args)
    segments = segment_by_time(df, args)
    stability = judge_stability(segments)
    sufficiency = judge_sufficiency(overall)

    write_report_md(overall, segments, stability, sufficiency, args, args.output_dir)
    write_report_xlsx(overall, segments, args, args.output_dir)
    write_manifest(overall, segments, stability, sufficiency, args, args.output_dir)
    print_summary(overall, stability, sufficiency)


if __name__ == "__main__":
    main()
