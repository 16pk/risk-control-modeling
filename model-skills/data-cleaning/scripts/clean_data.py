# -*- coding: utf-8 -*-
"""data-cleaning 编排入口: 读数据 → 列名校验 → 哨兵值替换 → 强门禁 → 去重 → 派生特征清单 → 产清洗方案。

用法:
    python data-cleaning/scripts/clean_data.py \
        --input <用户本地/hive 落地数据文件> \
        --session-dir <session_dir> \
        --id-col fuid \
        --dt-col f_p_date \
        [--label-col label] \
        [--invalid-values -1,-2,-999,-9999] \
        [--feature-list-source model-knowledge/.../xxx.csv] \
        [--auto-confirm]

产物落 <session_dir>/sample-features/data-cleaning/:
  - sample.parquet        清洗后样本(下游 feature-analysis 的数据源)
  - feature-list.csv      派生特征清单
  - cleaning-scheme.json  机器可读清洗方案
  - cleaning-report.md    人工可读清洗报告
  - _manifest.json        产物清单

强门禁: 检测到哨兵值命中时, 默认暂停并等待用户确认(输入 y 继续); --auto-confirm 跳过交互,
用于编排层已确认的非交互续跑场景。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts

import pandas as pd

from gen_feature_list import write_feature_list_csv
from replace_invalid import DEFAULT_INVALID_VALUES, parse_invalid_values, replace_invalid_values
from dedup_sample import dedup_by_user_date
from derive_feature_list import derive_features, filter_by_list, _load_allow_list
from write_scheme import build_scheme, write_scheme_json, write_scheme_report


PRODUCED_BY = "skills/data-cleaning"
MANIFEST_SCHEMA_VERSION = 1
DEDUP_KEEP_RULE = "label_non_null"

# 输出目录: <session_dir>/sample-features/data-cleaning/
_OUT_SUBDIR = os.path.join("sample-features", "data-cleaning")


def read_sample(input_path: str) -> pd.DataFrame:
    """按扩展名读本地样本文件, 支持 pandas 可读的主流格式。"""
    p = str(input_path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if p.endswith(".csv"):
        return pd.read_csv(input_path)
    if p.endswith(".feather"):
        return pd.read_feather(input_path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(input_path)
    if p.endswith(".json"):
        return pd.read_json(input_path)
    raise ValueError(
        f"不支持的数据文件格式: {input_path} (支持 .parquet/.csv/.feather/.xlsx/.xls/.json)"
    )


def _validate_columns(df: pd.DataFrame, id_col: str, dt_col: str, label_col: Optional[str]) -> None:
    """校验 id/dt(label 可选)列均存在: label_col 为 None 或空时跳过。"""
    missing = [c for c in (id_col, dt_col, label_col) if c and c not in df.columns]
    if missing:
        raise ValueError(f"以下列不在样本中: {missing}")


def _print_anomaly_prompt(invalid_report: pd.DataFrame) -> None:
    """打印哨兵值命中提示(强门禁暂停点)。"""
    print("")
    print("=" * 64)
    print("[data-cleaning] ⚠ 检测到哨兵值/无效值命中, 任务暂停")
    print("=" * 64)
    print(f"共 {len(invalid_report)} 个特征命中哨兵值, 将替换为 NaN(视为缺失):")
    for _, row in invalid_report.iterrows():
        print(
            f"  - {row['feature']}: 命中 [{row['hit_values']}]"
            f" ({row['n_hit']} 行, 占比 {row['hit_ratio']:.4%})"
        )
    print("替换后这些样本在训练中按缺失处理, 不会影响标签与主键列。")
    print("=" * 64)


def _default_confirm() -> bool:
    """默认交互确认: 读 stdin, 非 y/yes 视为放弃。"""
    try:
        ans = input("是否继续清洗? (y/N): ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _write_manifest(out_dir: Path, files: List[str], overview: dict) -> None:
    """落 _manifest.json。"""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files),
        "overview": overview,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clean_data(
    input_path: str,
    session_dir: str,
    id_col: str,
    dt_col: str,
    label_col: Optional[str] = None,
    invalid_values: Optional[list] = None,
    feature_list_source: Optional[str] = None,
    auto_confirm: bool = False,
    confirm_fn=None,
) -> dict:
    """端到端执行数据清洗, 返回清洗结果 summary dict。

    Args:
        input_path: 用户本地/hive 落地数据文件路径
        session_dir: session 目录(产物落 <session_dir>/sample-features/data-cleaning/)
        id_col: 用户粒度 ID 列(数据探查自主识别 + 用户确认后传入)
        dt_col: 日期分区列
        label_col: 标签列(可选; 无标签场景传 None)
        invalid_values: 哨兵值集合; None 时用 DEFAULT_INVALID_VALUES
        feature_list_source: 可选, 特征清单过滤文件(派生 feature-list.csv 时取交集)
        auto_confirm: True 跳过异常值交互确认
        confirm_fn: 可注入的确认函数(测试用), 默认 input()

    Returns:
        summary dict: 含 output_dir / sample_parquet / feature_list_csv /
        cleaning_scheme_json / cleaning_report_md / invalid_report / dedup_report / aborted
    """
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"数据文件不存在: {input_path}")
    if not session_dir:
        raise ValueError("必须传 --session-dir")

    out_dir = Path(session_dir) / _OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读数据 + 列名校验 ----
    df = read_sample(input_path)
    _validate_columns(df, id_col, dt_col, label_col)
    n_raw = int(len(df))

    # ---- 2. 派生特征列(所有非 id/dt/label 列; 清洗作用于全部特征列) ----
    clean_cols = derive_features(list(df.columns), [id_col, dt_col, label_col])
    if not clean_cols:
        raise ValueError("样本除 id/dt/label 外无特征列, 无法清洗")

    # ---- 3. 哨兵值替换 ----
    inv_values = invalid_values if invalid_values is not None else list(DEFAULT_INVALID_VALUES)
    if inv_values:
        print(f"[data-cleaning] 哨兵值集合: {inv_values} (作用于 {len(clean_cols)} 个特征列)")
    df, invalid_report = replace_invalid_values(df, clean_cols, inv_values, label_col)

    # ---- 4. 强门禁: 发现异常值 → 暂停 → 用户确认 ----
    aborted = False
    if len(invalid_report) > 0 and not auto_confirm:
        _print_anomaly_prompt(invalid_report)
        confirm = confirm_fn() if confirm_fn is not None else _default_confirm()
        if not confirm:
            aborted = True
            print("[data-cleaning] 用户未确认, 任务中止, 未产出清洗后数据文件。")
            return {
                "output_dir": str(out_dir),
                "aborted": True,
                "invalid_report": invalid_report,
                "n_raw": n_raw,
            }

    # ---- 5. 去重(用户 + 日期) ----
    df, dedup_report = dedup_by_user_date(df, id_col, dt_col, label_col)
    print(
        f"[data-cleaning] 去重: {dedup_report['n_before']} → {dedup_report['n_after']}"
        f" (移除 {dedup_report['n_removed']})"
    )

    # ---- 6. 派生 feature-list.csv(可选按清单过滤) ----
    if feature_list_source:
        allow_list = _load_allow_list(feature_list_source)
        features, missing = filter_by_list(clean_cols, allow_list)
        if missing:
            preview = ", ".join(missing[:20])
            print(f"[data-cleaning] [WARN] {len(missing)}/{len(allow_list)} 个清单特征不在样本中, 已丢弃: {preview}")
            if len(missing) > 20:
                print(f"  ... 共 {len(missing)} 个, 仅显示前 20 个")
    else:
        features = clean_cols

    # ---- 7. 落产物 ----
    sample_path = out_dir / "sample.parquet"
    df.to_parquet(sample_path, index=False)

    feature_list_path = out_dir / "feature-list.csv"
    write_feature_list_csv(features, str(feature_list_path))

    scheme = build_scheme(
        invalid_values=inv_values,
        dedup_keys=[id_col, dt_col],
        dedup_keep_rule=DEDUP_KEEP_RULE,
        invalid_report=invalid_report,
        dedup_report=dedup_report,
    )
    scheme_json_path = out_dir / "cleaning-scheme.json"
    scheme_md_path = out_dir / "cleaning-report.md"
    write_scheme_json(scheme, str(scheme_json_path))
    write_scheme_report(scheme, str(scheme_md_path))

    overview = {
        "n_raw": n_raw,
        "n_after": int(len(df)),
        "n_features": len(features),
        "invalid_values": inv_values,
        "n_invalid_value_features": int(len(invalid_report)) if len(invalid_report) > 0 else 0,
        "dedup_report": dedup_report,
    }
    files = ["sample.parquet", "feature-list.csv", "cleaning-scheme.json", "cleaning-report.md"]
    _write_manifest(out_dir, files, overview)

    print(f"[data-cleaning] 完成: 清洗后样本 {len(df)} 行 / {len(features)} 特征 -> {sample_path}")

    return {
        "output_dir": str(out_dir),
        "sample_parquet": str(sample_path),
        "feature_list_csv": str(feature_list_path),
        "cleaning_scheme_json": str(scheme_json_path),
        "cleaning_report_md": str(scheme_md_path),
        "invalid_report": invalid_report,
        "dedup_report": dedup_report,
        "features": features,
        "n_raw": n_raw,
        "aborted": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="data-cleaning: 哨兵值替换 + 用户日期去重 + 派生特征清单 + 清洗方案")
    p.add_argument("--input", required=True, help="本地数据文件路径(支持 pandas 可读格式)")
    p.add_argument("--session-dir", required=True, help="session 目录 (runs/<timestamp>-<model_name>)")
    p.add_argument("--id-col", required=True, help="用户粒度 ID 列名")
    p.add_argument("--dt-col", required=True, help="日期分区列名")
    p.add_argument("--label-col", default=None, help="标签列名(可选; 无标签场景可不传)")
    p.add_argument(
        "--invalid-values",
        default=None,
        help="哨兵值集合(逗号分隔, 如 -1,-2,-999,-9999); 不传用默认 [-1,-2,-9,-99,-999,-9999,-99999]",
    )
    p.add_argument(
        "--feature-list-source",
        default=None,
        help="可选, 特征清单过滤文件(.csv 取 feature_name 列 / .txt 按行); 派生 feature-list.csv 时取交集",
    )
    p.add_argument(
        "--auto-confirm",
        action="store_true",
        help="跳过异常值交互确认(编排层已确认的非交互续跑场景)",
    )
    args = p.parse_args()

    inv_values = parse_invalid_values(None, args.invalid_values) or list(DEFAULT_INVALID_VALUES)

    summary = clean_data(
        input_path=args.input,
        session_dir=args.session_dir,
        id_col=args.id_col,
        dt_col=args.dt_col,
        label_col=args.label_col,
        invalid_values=inv_values,
        feature_list_source=args.feature_list_source,
        auto_confirm=args.auto_confirm,
    )

    if summary.get("aborted"):
        print("[data-cleaning] 已中止(未产出清洗后数据)。")
    else:
        print(f"[data-cleaning] 清洗后样本: {summary['sample_parquet']}")
        print(f"[data-cleaning] 特征清单:   {summary['feature_list_csv']}")
        print(f"[data-cleaning] 清洗方案:   {summary['cleaning_scheme_json']}")


if __name__ == "__main__":
    main()
