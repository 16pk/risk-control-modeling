# -*- coding: utf-8 -*-
"""样本前置轻量编排入口（development 轻量剧本）：串起「只清洗 / 只分析」独立任务。

背景: feature-classification / data-cleaning / credit-data-analysis 三个 skill 的 SKILL.md
均标注「仅由 classification-model-development 编排自动调起，不设独立触发词」。当用户
绕过完整建模主链路、只针对给定数据文件说「清洗」或「分析」时，若直接调 clean_data.py /
feature_analysis.py 会跳过特征列识别环节。本入口把这条独立链路补齐：

    clean   特征列识别(classify_features.py 探查) → 编排层交互确认
            → finalize_feature_list.py 固化权威 feature-list.csv
            → clean_data.py --feature-list-source <权威清单> 清洗
    analyze 在 clean 基础上追加 feature_analysis.py --feature-list <权威清单> 特征分析

设计原则:
- 本脚本只做「编排链」：以 subprocess 逐个调用各 sub-skill 的既有 CLI（绝对路径 python），
  不在 development 内重复实现探查/清洗/分析逻辑（反模式豁免: 这是复用 sub-skill CLI 的编排剧本）。
- 交互（id/dt/label 确认、三分类 exclude/keep 批量确认、PSI 基准月）由编排层大模型按
  development SKILL.md §5 决策点话术完成；脚本对非交互续跑显式传参
  --auto-confirm / --exclude / --keep / --base-month，不引入新的交互逻辑。
- 产物沿用标准 session 结构: <session-dir>/sample-features/
  (feature-classification.json + feature-list.csv + data-cleaning/ + credit-data-analysis/)，
  与 development 主链路产物完全互通，断点续跑靠 _manifest.json 推断。
- 权威清单唯一真相: finalize 固化的 sample-features/feature-list.csv 同时供 data-cleaning
  (--feature-list-source) 与 credit-data-analysis (--feature-list) 消费，杜绝各自派生。

用法:
    python prep_sample.py clean --input <数据文件> --session-dir <session_dir> \
        --id-col fuid --dt-col ftrans_date --label-col fpd7_sx30 \
        --exclude fser_date,sx_order_id,ftrans_time,... [--keep flag_ok,...]
    python prep_sample.py analyze --input <数据文件> --session-dir <session_dir> \
        --id-col fuid --dt-col ftrans_date --label-col fpd7_sx30 \
        --exclude ... [--keep ...] [--base-month 2025-04]

子命令:
    clean    特征列识别 + 固化权威清单 + 数据清洗（产 sample-features/data-cleaning/）
    analyze  在 clean 基础上追加特征分析（产 sample-features/credit-data-analysis/）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 本 skill 目录（scripts/ 的上一级）: model-skills/classification-model-development/
_SKILL_DIR = Path(__file__).resolve().parent.parent
# 各 sub-skill 脚本绝对路径（沿既有目录约定）
_FC_SCRIPTS = _SKILL_DIR.parent / "feature-classification" / "scripts"
_DC_SCRIPTS = _SKILL_DIR.parent / "data-cleaning" / "scripts"
_CDA_SCRIPTS = _SKILL_DIR.parent / "credit-data-analysis" / "scripts"

PRODUCED_BY = "skills/classification-model-development"
_SESSION_PREFIX = "prep"  # 自动建目录默认命名前缀

# 特征分类档案/权威清单在 session 内的落点（与 development Stage 1 一致）
_FEATURE_CLASSIFICATION_JSON = os.path.join("sample-features", "feature-classification.json")
_FEATURE_LIST_CSV = os.path.join("sample-features", "feature-list.csv")
# data-cleaning 产物子目录（clean_data.py 内部写 <session_dir>/sample-features/data-cleaning/）
_CLEAN_SAMPLE = os.path.join("sample-features", "data-cleaning", "sample.parquet")
# credit-data-analysis 产物子目录
_CDA_OUT_DIR = os.path.join("sample-features", "credit-data-analysis")


def run_skill(script: Path, *args: str) -> None:
    """subprocess 调 sub-skill CLI；非零返回码抛错并透传 stderr（不吞错）。"""
    if not script.is_file():
        raise FileNotFoundError(f"sub-skill 脚本不存在: {script}")
    cmd = [sys.executable, str(script), *args]
    print(f"[prep_sample] 调用: {cmd[0]} {cmd[1]}")
    for a in args:
        print(f"    {a}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[prep_sample] sub-skill 调用失败 (exit={proc.returncode}): {script.name} "
            f"args={args!r}"
        )


def ensure_session_dir(session_dir: Optional[str]) -> str:
    """解析 --session-dir；缺省按 runs/{ts}-prep-{task}/ 自动建目录。"""
    if session_dir:
        d = Path(session_dir)
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path("runs") / f"{ts}-{_SESSION_PREFIX}"
    d.mkdir(parents=True, exist_ok=True)
    print(f"[prep_sample] 未指定 --session-dir, 自动创建: {d}")
    return str(d)


def _resolve_absolute(script: Path) -> Path:
    return script


def clean_pipeline(
    session_dir: str,
    input_path: str,
    id_col: str,
    dt_col: str,
    label_col: str,
    exclude: List[str],
    keep: Optional[List[str]] = None,
) -> str:
    """1) classify_features.py 探查 → 2) finalize_feature_list.py 固化权威 feature-list.csv
    → 3) clean_data.py --feature-list-source <权威清单> --auto-confirm 清洗。

    交互（id/dt/label 确认 + 三分类 exclude/keep 批量确认）由编排层在调用前完成，
    本函数仅接收最终确认参数并逐个调用 sub-skill CLI。

    Returns: 权威 feature-list.csv 绝对路径。
    """
    cls_script = _resolve_absolute(_FC_SCRIPTS / "classify_features.py")
    fin_script = _resolve_absolute(_FC_SCRIPTS / "finalize_feature_list.py")
    clean_script = _resolve_absolute(_DC_SCRIPTS / "clean_data.py")

    # ---- 1. 特征列识别探查（扫描 → 语义三分类 → 通配符分组报告）----
    run_skill(
        cls_script,
        "--input", input_path,
        "--out-dir", os.path.join(session_dir, "sample-features"),
        "--id-col", id_col,
        "--dt-col", dt_col,
        "--label-col", label_col,
    )

    # ---- 2. 固化权威 feature-list.csv（应用用户批量确认的剔除/保留名单）----
    fin_args = [
        "--classification", os.path.join(session_dir, _FEATURE_CLASSIFICATION_JSON),
        "--out-dir", os.path.join(session_dir, "sample-features"),
        "--exclude", ",".join(exclude),
    ]
    if keep:
        fin_args += ["--keep", ",".join(keep)]
    run_skill(fin_script, *fin_args)

    # ---- 3. 数据清洗（经 --feature-list-source 消费权威清单取交集）----
    run_skill(
        clean_script,
        "--input", input_path,
        "--session-dir", session_dir,
        "--id-col", id_col,
        "--dt-col", dt_col,
        "--label-col", label_col,
        "--feature-list-source", os.path.join(session_dir, _FEATURE_LIST_CSV),
        "--auto-confirm",
    )
    return os.path.join(session_dir, _FEATURE_LIST_CSV)


def analyze_pipeline(
    session_dir: str,
    input_path: str,
    id_col: str,
    dt_col: str,
    label_col: str,
    exclude: List[str],
    keep: Optional[List[str]] = None,
    base_month: Optional[str] = None,
) -> None:
    """clean_pipeline 基础上追加 feature_analysis.py --feature-list <权威清单>
    --time-col <dt> --iv-label <label> [--base-month <YYYY-MM>]（独立体检模式）。

    分析输入为清洗后 sample.parquet（哨兵→NaN 已替换），PSI/IV 落在真实缺失语义上。
    """
    feature_list_csv = clean_pipeline(
        session_dir, input_path, id_col, dt_col, label_col, exclude, keep
    )
    analysis_script = _resolve_absolute(_CDA_SCRIPTS / "feature_analysis.py")
    ana_args = [
        "--data-file", os.path.join(session_dir, _CLEAN_SAMPLE),
        "--feature-list", feature_list_csv,
        "--time-col", dt_col,
        "--iv-label", label_col,
        "--output-dir", os.path.join(session_dir, _CDA_OUT_DIR),
    ]
    if base_month:
        ana_args += ["--base-month", base_month]
    run_skill(analysis_script, *ana_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prep_sample.py",
        description="样本前置轻量编排入口: 串起 feature-classification → data-cleaning "
                    "(→ credit-data-analysis), 供「只清洗 / 只分析」独立任务直接调用",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def _add_common(sp) -> None:
        sp.add_argument("--input", required=True, help="本地数据文件路径(parquet/csv/feather/xlsx/json)")
        sp.add_argument("--session-dir", default=None,
                        help="session 目录(产物落 <session_dir>/sample-features/; 缺省按 runs/{ts}-prep-*/ 自动建)")
        sp.add_argument("--id-col", required=True, help="用户粒度 ID 列(探查确认)")
        sp.add_argument("--dt-col", required=True, help="日期分区列(探查确认)")
        sp.add_argument("--label-col", required=True, help="标签列(探查确认)")
        sp.add_argument("--exclude", default="",
                        help="用户批量确认剔除的 non_feature 列(逗号分隔)")
        sp.add_argument("--keep", default="",
                        help="用户确认保留的列(逗号分隔, 可选; 如恢复规则误判)")

    sp_clean = sub.add_parser("clean", help="特征列识别 + 固化权威清单 + 数据清洗")
    _add_common(sp_clean)

    sp_analyze = sub.add_parser("analyze", help="clean 基础上追加特征分析(credit-data-analysis)")
    _add_common(sp_analyze)
    sp_analyze.add_argument("--base-month", default=None,
                            help="PSI 基准月 YYYY-MM(编排层确认后传入; 缺省用默认基准月)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    session_dir = ensure_session_dir(args.session_dir)
    exclude = [c.strip() for c in args.exclude.split(",") if c.strip()]
    keep = [c.strip() for c in args.keep.split(",") if c.strip()] or None

    if args.command == "clean":
        fl = clean_pipeline(
            session_dir, args.input, args.id_col, args.dt_col, args.label_col,
            exclude=exclude, keep=keep,
        )
        print(f"[prep_sample] clean 完成: 权威清单 {fl}")
        print(f"[prep_sample] 清洗产物: {os.path.join(session_dir, _CLEAN_SAMPLE)}")
        print(f"[prep_sample] 断点续跑/主链路衔接: 读 {os.path.join(session_dir, 'sample-features', 'data-cleaning', '_manifest.json')}")
    elif args.command == "analyze":
        analyze_pipeline(
            session_dir, args.input, args.id_col, args.dt_col, args.label_col,
            exclude=exclude, keep=keep, base_month=args.base_month,
        )
        print(f"[prep_sample] analyze 完成: 分析产物 {os.path.join(session_dir, _CDA_OUT_DIR)}")
    else:  # argparse required 已兜底
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
