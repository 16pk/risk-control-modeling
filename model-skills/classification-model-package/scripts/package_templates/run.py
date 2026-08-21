#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立交付包主入口：给定数据文件，完成「数据清理 → 定版模型打分 →（可选）FICO 转分」。

本脚本是自包含交付包的一部分，**不依赖任何建模专家包代码**（零引用专家包目录与共享代码），
只依赖 pip 包（pandas / numpy / pyarrow / lightgbm / xgboost）。

数据链路：
  clean: 哨兵值 → NaN（仅特征列，非交互 + WARN）          [pipeline.clean]
  score: 特征严格校验 + 按 feature_names 重排 + 推理 → score  [pipeline.score]
  fico:  若 assets/coef.json 存在 → 纯应用转分 → bscore     [pipeline.fico]

用法:
  python run.py --input <数据文件 parquet/csv> --output-dir <out> \
      [--score-col score] [--batch-size 500000]

输出到 --output-dir:
  score.parquet           透传非特征列 + score（含 FICO 时追加 bscore）
  cleaning-report.json    清洗统计（哨兵命中 feature / n_hit / hit_ratio）
  run-manifest.json       运行元信息（时间 / 资产快照 / 含 FICO / 统计）
  fico-summary.json       （含 FICO 模块时）coef/intc + bscore 分布
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 包内自包含：显式把本文件所在目录加入 sys.path，使 `import pipeline.*` 生效
# （无论从何处以何种方式调用 run.py 都能定位 pipeline 包）。
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd  # noqa: E402

from pipeline import clean, score, fico  # noqa: E402


def load_data(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise SystemExit(f"[ERROR] 仅支持 parquet/csv 输入: {path}")


def _save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="独立交付包：清理 → 打分 →（可选）FICO 转分")
    parser.add_argument("--input", required=True, help="输入数据文件（parquet/csv）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--score-col", default="score", help="输出违约概率分列名（默认 score）")
    parser.add_argument("--batch-size", type=int, default=500_000,
                        help="整批处理参考上限，超出仅提示按内存规划分批（当前为整批推理）")
    args = parser.parse_args()

    assets = Path(__file__).resolve().parent / "assets"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("[PACKAGE] 独立交付包运行开始")
    print(f"[PACKAGE] input={args.input} | output-dir={out_dir} | score-col={args.score_col}")

    # ---------- 加载资产 ----------
    model_meta = score.load_meta(assets)
    feature_names = list(model_meta.get("feature_names") or [])
    if not feature_names:
        raise SystemExit("[ERROR] assets/model_meta.json 缺 feature_names，请确认打包完整性")
    algo = (model_meta.get("algo") or "").lower()
    invalid_values = clean.load_invalid_values(assets)
    coef = fico.load_coef(assets)  # 无 coef.json 时返回 None（不含 FICO 模块）

    # ---------- 1. 数据清理 ----------
    df = load_data(args.input)
    print(f"[CLEAN] 输入 shape={df.shape} | 特征数={len(feature_names)}")
    df, report = clean.clean_sentinel(df, feature_names, invalid_values)
    _save_json(report, out_dir / "cleaning-report.json")
    if report.get("features"):
        print(f"[CLEAN] 哨兵值命中 {len(report['features'])} 个特征，已替换为 NaN（详见 cleaning-report.json）")
    else:
        print("[CLEAN] 未命中哨兵值")

    # ---------- 2. 打分 ----------
    df, score_col = score.apply_score(df, feature_names, algo, assets, args.score_col)
    print(f"[SCORE] score 列: {score_col} | 范围=[{float(df[score_col].min()):.6f}, "
          f"{float(df[score_col].max()):.6f}]")

    # ---------- 3. FICO 转分（条件包含） ----------
    has_fico = coef is not None
    fico_summary = None
    if has_fico:
        df, fico_summary = fico.apply_fico(df, score_col, coef)
        _save_json(fico_summary, out_dir / "fico-summary.json")
        print(f"[FICO] 转分完成 | bscore 范围=[{float(df['bscore'].min()):.1f}, "
              f"{float(df['bscore'].max()):.1f}]")
    else:
        print("[FICO] 本交付包未含 FICO 模块（打包时无 fico/coef.json），仅输出模型分")

    # ---------- 输出 ----------
    out_path = out_dir / "score.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[SAVE] score.parquet: {out_path} | shape={df.shape}")

    manifest = {
        "schema_version": 1,
        "produced_by": "delivery/run.py",
        "package_name": model_meta.get("package_name", "unknown"),
        "run_name": model_meta.get("run_name"),
        "algo": algo,
        "score_col": score_col,
        "has_fico": has_fico,
        "n_rows": int(len(df)),
        "sentinel_replaced_features": len(report.get("features", [])),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    _save_json(manifest, out_dir / "run-manifest.json")

    print("[DONE] 链路完成：清理 → 打分 →" + (" FICO 转分" if has_fico else "（无 FICO）"))
    print(f"[DONE] 产物: {out_dir}/score.parquet（含 bscore）" if has_fico
          else f"[DONE] 产物: {out_dir}/score.parquet（仅模型分）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
