#!/usr/bin/env python3
"""信贷特征分析脚本（ModelEvo credit-data-analysis skill 版）。

读取数据文件，生成包含以下分析结果的 Excel 文件：
  1. 样本分布    2. 特征分布    3. 覆盖率    4. 均值    5. 最小值
  6. 最大值      7. 标准差      8. Nunique   9. PSI     10. IV

用法：
  python3 feature_analysis.py --data-file ka_df.feather \\
      --feature-start tx_model_2_score --feature-end mob4_v5_score \\
      --feature-extra ascore_fpd7_v3 \\
      --base-month 2025-04 --iv-label fpd7 \\
      --output-dir <产物目录>

支持 .feather / .csv / .parquet（按扩展名自动选读取方式）。
产物：Excel + _manifest.json（参数溯源），落 --output-dir（默认当前目录）。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

PRODUCED_BY = "skills/credit-data-analysis"
MANIFEST_SCHEMA_VERSION = 1

# ============================================================
# 0. 默认配置（CLI 参数可覆盖）
# ============================================================
DEFAULTS = dict(
    data_file="ka_df.feather",
    output_file="特征分析结果.xlsx",
    output_dir=".",
    time_col="fsx_time",
    feature_start="tx_model_2_score",
    feature_end="mob4_v5_score",
    feature_extra="",               # 逗号分隔的额外特征列名
    base_month="2025-04",           # PSI 基准月份
    iv_label="fpd7",                # IV 计算所用的风险标签，留空则自动选择
    risk_prefixes="fpd,dpd",        # 风险标签前缀，逗号分隔；iv_label 为空时用于自动识别
    risk_labels="",                 # 手动指定风险标签列（逗号分隔），留空则按前缀自动识别
    psi_bins=10,
    iv_bins=10,
    invalid_values="-1,-2,-9,-99,-999,-9999,-99999",  # 无效值哨兵集合，逗号分隔
)


def parse_args():
    ap = argparse.ArgumentParser(description="信贷特征分析脚本")
    ap.add_argument("--data-file", default=DEFAULTS["data_file"])
    ap.add_argument("--output-file", default=DEFAULTS["output_file"])
    ap.add_argument("--output-dir", default=DEFAULTS["output_dir"],
                    help="产物（Excel + _manifest.json）输出目录，默认当前目录")
    ap.add_argument("--time-col", default=DEFAULTS["time_col"])
    ap.add_argument("--feature-start", default=DEFAULTS["feature_start"],
                    help="特征列范围的起始列名")
    ap.add_argument("--feature-end", default=DEFAULTS["feature_end"],
                    help="特征列范围的结束列名")
    ap.add_argument("--feature-extra", default=DEFAULTS["feature_extra"],
                    help="额外特征列名，逗号分隔")
    ap.add_argument("--base-month", default=DEFAULTS["base_month"],
                    help="PSI 基准月份，格式 YYYY-MM")
    ap.add_argument("--iv-label", default=DEFAULTS["iv_label"],
                    help="IV 计算所用的风险标签，留空则自动选择第一个可用的")
    ap.add_argument("--risk-prefixes", default=DEFAULTS["risk_prefixes"],
                    help="风险标签前缀，逗号分隔")
    ap.add_argument("--risk-labels", default=DEFAULTS["risk_labels"],
                    help="手动指定风险标签列，逗号分隔；覆盖自动识别")
    ap.add_argument("--psi-bins", type=int, default=DEFAULTS["psi_bins"])
    ap.add_argument("--iv-bins", type=int, default=DEFAULTS["iv_bins"])
    ap.add_argument("--invalid-values", default=DEFAULTS["invalid_values"],
                    help="无效值哨兵集合，逗号分隔；命中这些值的特征将在报告中标记提醒，建议建模时替换为空值")
    return ap.parse_args()


# ============================================================
# 1. 加载数据 & 准备列
# ============================================================
def _read_data(path: str) -> pd.DataFrame:
    """按扩展名选择读取方式：.feather / .csv / .parquet。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    suffix = Path(path).suffix.lower()
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"不支持的文件格式: {suffix}。支持 .feather / .csv / .parquet（pandas 可读格式）"
    )


def load_and_prepare(args):
    print("[1/4] 加载数据...")
    df = _read_data(args.data_file)

    # 时间列 & 月份切片
    if args.time_col not in df.columns:
        raise KeyError(
            f"时间列 {args.time_col} 不存在。可选: {list(df.columns[:10])}..."
        )
    if not pd.api.types.is_datetime64_any_dtype(df[args.time_col]):
        # csv 读入后常为字符串/object，自动尝试解析为 datetime
        try:
            df[args.time_col] = pd.to_datetime(df[args.time_col])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"时间列 {args.time_col} 无法解析为日期，请确认其内容为日期格式: {e}"
            )
    df["month"] = df[args.time_col].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())

    # 特征列
    cols = df.columns.tolist()
    if args.feature_start not in cols or args.feature_end not in cols:
        raise KeyError(
            f"特征区间列不存在: start={args.feature_start!r}, end={args.feature_end!r}。"
            f"请从数据列中重新指定（前10列: {cols[:10]}...）"
        )
    start_idx = cols.index(args.feature_start)
    end_idx = cols.index(args.feature_end)
    if start_idx > end_idx:
        raise ValueError(f"起始列在结束列之后: {args.feature_start} 出现在 {args.feature_end} 之后")
    feature_cols = cols[start_idx : end_idx + 1]
    if args.feature_extra:
        for extra in args.feature_extra.split(","):
            extra = extra.strip()
            if extra and extra not in feature_cols:
                feature_cols.append(extra)

    # 风险标签列
    if args.risk_labels:
        risk_labels = [x.strip() for x in args.risk_labels.split(",")]
    else:
        prefixes = [p.strip() for p in args.risk_prefixes.split(",")]
        risk_labels = [c for c in df.columns
                       if any(c.startswith(p) for p in prefixes)]

    # IV 标签：若未指定则从风险标签中自动选择
    iv_label = args.iv_label
    if not iv_label and risk_labels:
        iv_label = risk_labels[0]
        print(f"  → IV 标签自动选择: {iv_label}")

    # 无效值哨兵集合
    invalid_values = []
    for v in args.invalid_values.split(","):
        v = v.strip()
        if v:
            try:
                invalid_values.append(float(v))
            except ValueError:
                print(f"  ⚠ 无效值集合含非数值项，已忽略: {v!r}")
    if invalid_values:
        print(f"  无效值哨兵集合: {invalid_values}")

    print(f"  特征数: {len(feature_cols)}")
    print(f"  风险标签: {risk_labels}")
    print(f"  月份: {months}")
    print(f"  样本量: {len(df):,}")
    return df, months, feature_cols, risk_labels, iv_label, invalid_values


# ============================================================
# 2. 辅助函数
# ============================================================
def _get_bins(series, n_bins):
    s = series.dropna()
    if len(s) < n_bins * 2 or s.nunique() < 2:
        return None
    try:
        _, edges = pd.qcut(s, n_bins, duplicates="drop", retbins=True)
        if len(edges) < 2:
            return None
        return edges
    except (ValueError, IndexError):
        return None


def calc_psi(base_series, actual_series, bins=10):
    base = base_series.dropna()
    actual = actual_series.dropna()
    if len(base) < bins * 2 or len(actual) < bins * 2:
        return np.nan
    edges = _get_bins(base, bins)
    if edges is None:
        return np.nan
    base_dist = pd.cut(base, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    actual_dist = pd.cut(actual, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    all_idx = base_dist.index.union(actual_dist.index)
    base_dist = base_dist.reindex(all_idx, fill_value=0.0001)
    actual_dist = actual_dist.reindex(all_idx, fill_value=0.0001)
    return float(np.sum((actual_dist.values - base_dist.values)
                        * np.log(actual_dist.values / base_dist.values)))


def calc_iv(series, label, bins=10):
    mask = series.notna() & label.notna()
    x = series[mask]
    y = label[mask]
    if len(x) < bins * 2 or y.nunique() < 2:
        return np.nan
    edges = _get_bins(x, bins)
    if edges is None:
        return np.nan
    binned = pd.cut(x, edges, include_lowest=True)
    grouped_good = (y == 0).groupby(binned).sum()
    grouped_bad = (y == 1).groupby(binned).sum()
    total_good = grouped_good.sum()
    total_bad = grouped_bad.sum()
    if total_good == 0 or total_bad == 0:
        return np.nan
    good_pct = (grouped_good / total_good).clip(lower=0.0001)
    bad_pct = (grouped_bad / total_bad).clip(lower=0.0001)
    woe = np.log(good_pct / bad_pct)
    return float(np.sum((good_pct - bad_pct) * woe))


def check_invalid_values(df, feature_cols, invalid_values):
    """检测特征列中是否含无效值哨兵（如 -1/-2/-999/-9999 等）。

    命中规则：特征取值落在无效值集合内的样本数 > 0。
    输出：每特征的命中哨兵值集合、命中样本数、命中占比、是否建议替换为空值。
    这些值往往是"无数据/拒贷/异常"的占位符，若不替换为空值，建模时模型会
    学到虚假的取值边界（尤其树模型/分箱），故标记提醒，由人决策是否清洗。
    """
    if not invalid_values:
        return None
    rows = []
    for fc in feature_cols:
        s = df[fc]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        hit = set()
        for v in invalid_values:
            try:
                if (s == v).any():
                    hit.add(v)
            except (TypeError, ValueError):
                continue
        if hit:
            hit_list = sorted(hit)
            mask = s.isin(hit_list)
            n_hit = int(mask.sum())
            rows.append({
                "特征": fc,
                "命中无效值": ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in hit_list),
                "命中样本数": n_hit,
                "命中占比": round(n_hit / len(s), 6),
                "建议": "建议建模时替换为空值(NaN)",
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


# ============================================================
# 3. 计算所有分析结果
# ============================================================
def compute_all(df, months, feature_cols, risk_labels, iv_label, invalid_values, args):
    print("[2/4] 计算分析结果...")

    # --- 3.1 样本分布 ---
    print("  样本分布...")
    sample_total, sample_overdue, sample_rate = {}, {}, {}
    for month in months:
        mdf = df[df["month"] == month]
        sample_total[month] = {}
        sample_overdue[month] = {}
        sample_rate[month] = {}
        for rl in risk_labels:
            valid = mdf[rl].notna()
            total = valid.sum()
            overdue = (mdf.loc[valid, rl] == 1).sum()
            sample_total[month][rl] = int(total)
            sample_overdue[month][rl] = int(overdue)
            sample_rate[month][rl] = round(overdue / total, 6) if total > 0 else 0.0

    sample_total_df = pd.DataFrame.from_dict(sample_total, orient="index", columns=risk_labels)
    sample_total_df.index.name = "月份"
    sample_overdue_df = pd.DataFrame.from_dict(sample_overdue, orient="index", columns=risk_labels)
    sample_overdue_df.index.name = "月份"
    sample_rate_df = pd.DataFrame.from_dict(sample_rate, orient="index", columns=risk_labels)
    sample_rate_df.index.name = "月份"

    # --- 3.2 特征分布 ---
    print("  特征分布...")
    feat_dist_data = {}
    for fc in feature_cols:
        s = df[fc]
        feat_dist_data[fc] = {
            "覆盖率": s.notna().mean(),
            "平均值": s.mean(),
            "最小值": s.min(),
            "1%": s.quantile(0.01),
            "5%": s.quantile(0.05),
            "25%": s.quantile(0.25),
            "50%": s.quantile(0.50),
            "75%": s.quantile(0.75),
            "95%": s.quantile(0.95),
            "99%": s.quantile(0.99),
            "最大值": s.max(),
        }
    feat_dist_df = pd.DataFrame.from_dict(feat_dist_data, orient="index")
    feat_dist_df.index.name = "特征"

    # --- 3.3-3.8 分月统计表 ---
    monthly_funcs = {
        "覆盖率": lambda s: s.notna().mean(),
        "均值": lambda s: s.mean(),
        "最小值": lambda s: s.min(),
        "最大值": lambda s: s.max(),
        "标准差": lambda s: s.std(),
        "Nunique": lambda s: s.nunique(),
    }
    monthly_results = {}
    for sheet_name, func in monthly_funcs.items():
        print(f"  {sheet_name}...")
        data = {fc: df.groupby("month")[fc].apply(func) for fc in feature_cols}
        monthly_results[sheet_name] = pd.DataFrame(data).T
        monthly_results[sheet_name].index.name = "特征"

    # --- 3.9 PSI ---
    print(f"  PSI (以 {args.base_month} 为基准)...")
    base_mask = df["month"] == args.base_month
    psi_data = {}
    for fc in feature_cols:
        base_s = df.loc[base_mask, fc]
        row = {}
        for month in months:
            if month == args.base_month:
                row[month] = 0.0
            else:
                row[month] = calc_psi(base_s, df.loc[df["month"] == month, fc], args.psi_bins)
        psi_data[fc] = row
    psi_df = pd.DataFrame.from_dict(psi_data, orient="index")
    psi_df.index.name = "特征"

    # --- 3.10 IV ---
    print(f"  IV (标签={iv_label})...")
    label_series = df[iv_label]
    iv_data = {}
    for fc in feature_cols:
        row = {}
        for month in months:
            mask = df["month"] == month
            row[month] = calc_iv(df.loc[mask, fc], label_series[mask], args.iv_bins)
        iv_data[fc] = row
    iv_df = pd.DataFrame.from_dict(iv_data, orient="index")
    iv_df.index.name = "特征"

    # --- 3.11 无效值检查 ---
    print("  无效值检查...")
    invalid_df = check_invalid_values(df, feature_cols, invalid_values)
    if invalid_df is not None and len(invalid_df):
        print(f"  ⚠ 发现 {len(invalid_df)} 个特征含无效值哨兵（如 -1/-2/-999/-9999 等），"
              f"详情见「无效值检查」sheet，建议建模时替换为空值(NaN)")
        top_inv = invalid_df.sort_values("命中占比", ascending=False).head(5)
        for _, r in top_inv.iterrows():
            print(f"    - {r['特征']}: 命中 {r['命中无效值']} 占比 {r['命中占比']:.2%}")
    else:
        print("  ✓ 未发现无效值哨兵特征")

    return (sample_total_df, sample_overdue_df, sample_rate_df,
            feat_dist_df, monthly_results, psi_df, iv_df, invalid_df)


# ============================================================
# 4. 写入 Excel
# ============================================================
def write_excel(output_file, risk_labels, months,
                sample_total_df, sample_overdue_df, sample_rate_df,
                feat_dist_df, monthly_results, psi_df, iv_df, invalid_df):
    print("[3/4] 写入 Excel...")
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    sheet_order = ["特征分布", "覆盖率", "均值", "最小值", "最大值", "标准差", "Nunique", "PSI", "IV"]
    sheet_dfs_map = {
        "特征分布": feat_dist_df,
        "覆盖率": monthly_results["覆盖率"], "均值": monthly_results["均值"],
        "最小值": monthly_results["最小值"], "最大值": monthly_results["最大值"],
        "标准差": monthly_results["标准差"], "Nunique": monthly_results["Nunique"],
        "PSI": psi_df, "IV": iv_df,
    }

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for name in sheet_order:
            sheet_dfs_map[name].to_excel(writer, sheet_name=name, index=True)
        if invalid_df is not None and len(invalid_df):
            invalid_df.to_excel(writer, sheet_name="无效值检查", index=False)

    wb = load_workbook(output_file)
    ws = wb.create_sheet("样本分布", 0)

    def write_matrix(ws, start_row, title, df):
        ws.cell(row=start_row, column=1, value=title)
        ws.cell(row=start_row + 1, column=1, value="月份")
        for j, col in enumerate(df.columns):
            ws.cell(row=start_row + 1, column=j + 2, value=col)
        for i, (idx, row) in enumerate(df.iterrows()):
            ws.cell(row=start_row + 2 + i, column=1, value=str(idx))
            for j, val in enumerate(row):
                ws.cell(row=start_row + 2 + i, column=j + 2, value=val)

    write_matrix(ws, 1, "样本数", sample_total_df)
    write_matrix(ws, len(months) + 4, "逾期数", sample_overdue_df)
    write_matrix(ws, 2 * len(months) + 7, "逾期率", sample_rate_df)

    ws.column_dimensions["A"].width = 12
    for col_idx in range(2, len(risk_labels) + 2):
        ws.column_dimensions[get_column_letter(col_idx)].width = 14

    for name in sheet_order:
        wss = wb[name]
        wss.column_dimensions["A"].width = 18
        for col_idx in range(2, wss.max_column + 1):
            wss.column_dimensions[get_column_letter(col_idx)].width = 14

    wb.save(output_file)


# ============================================================
# 5. 落盘 manifest（参数溯源，供复现与追溯）
# ============================================================
def write_manifest(output_dir: Path, files: list, args) -> None:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "data_file": args.data_file,
            "feature_start": args.feature_start,
            "feature_end": args.feature_end,
            "feature_extra": args.feature_extra,
            "base_month": args.base_month,
            "iv_label": args.iv_label,
            "time_col": args.time_col,
            "risk_prefixes": args.risk_prefixes,
            "risk_labels": args.risk_labels,
            "psi_bins": args.psi_bins,
            "iv_bins": args.iv_bins,
            "invalid_values": args.invalid_values,
        },
        "files": sorted(files),
    }
    (output_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ============================================================
# 6. 主入口
# ============================================================
def main():
    args = parse_args()
    df, months, feature_cols, risk_labels, iv_label, invalid_values = load_and_prepare(args)
    results = compute_all(df, months, feature_cols, risk_labels, iv_label, invalid_values, args)
    (sample_total_df, sample_overdue_df, sample_rate_df,
     feat_dist_df, monthly_results, psi_df, iv_df, invalid_df) = results

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(out_dir / args.output_file)

    write_excel(output_file, risk_labels, months,
                sample_total_df, sample_overdue_df, sample_rate_df,
                feat_dist_df, monthly_results, psi_df, iv_df, invalid_df)
    write_manifest(out_dir, [args.output_file], args)

    print("[4/4] 完成!")
    print(f"  输出文件: {output_file}")
    print(f"  样本分布: {len(months)} 月 × {len(risk_labels)} 标签")
    print(f"  特征分布: {feat_dist_df.shape[0]} 特征 × {feat_dist_df.shape[1]} 统计量")
    for name in ["覆盖率", "均值", "最小值", "最大值", "标准差", "Nunique", "PSI", "IV"]:
        df_out = (monthly_results[name] if name in monthly_results
                  else psi_df if name == "PSI" else iv_df)
        print(f"  {name}: {df_out.shape[0]} 特征 × {df_out.shape[1]} 月份")
    if invalid_df is not None and len(invalid_df):
        print(f"  无效值检查: {len(invalid_df)} 个特征命中哨兵值（见「无效值检查」sheet）")


if __name__ == "__main__":
    sys.exit(main() or 0)
