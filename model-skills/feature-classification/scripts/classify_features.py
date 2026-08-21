# -*- coding: utf-8 -*-
"""feature-classification(特征列识别与复用) · 探查阶段: 扫描 → 语义三分类 → 通配符分组统计。

用法:
    python feature-classification/scripts/classify_features.py \
        --input <本地样本文件(parquet/csv/feather/xlsx/json)> \
        --out-dir <session_dir>/sample-features \
        [--id-col fuid] [--dt-col f_p_date] [--label-col label] \
        [--label-prefixes fpd,dpd] [--extra-patterns "pat1,pat2"] [--min-group 2]

产物落 <out-dir>/:
    feature-classification.json  逐列三分类档案(探查版, 判定人由 finalize 固化)
    _manifest.json               产出清单

后续: 编排层展示分类报告(render_report) → 用户批量确认剔除/保留名单 →
      finalize_feature_list.py 应用名单生成权威 feature-list.csv(全 pipeline 唯一真相)。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts

import pandas as pd

from rules import classify_column, DEFAULT_LABEL_PREFIXES


SCHEMA_VERSION = 1
PRODUCED_BY = "skills/feature-classification"
RULEBOOK = "v0"


def read_sample(path: str) -> pd.DataFrame:
    """按扩展名读样本(与 data-cleaning 同款轻量读取; 架构纪律禁跨 skill import)。"""
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".feather"):
        return pd.read_feather(path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if p.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(
        f"不支持的数据文件格式: {path} (支持 .parquet/.csv/.feather/.xlsx/.xls/.json)"
    )


def scan_features(
    df: pd.DataFrame,
    id_col: str,
    dt_col: str,
    label_col: str,
    label_prefixes: Optional[tuple] = None,
    ident_prefixes: Optional[tuple] = None,
    extra_patterns: Optional[list] = None,
    min_group: int = 2,
) -> dict:
    """探查扫描: 语义三分类 + 通配符分组统计。

    Returns dict:
        id_col/dt_col/label_col, counts(Counter 3 类), columns(逐列),
        groups([{group,n,cols,mixed,categories,null_median}])
    """
    exclude = {c for c in (id_col, dt_col, label_col) if c}
    feat_cols = [c for c in df.columns if c not in exclude]
    if not feat_cols:
        raise ValueError("样本除 id/dt/label 外无特征列, 无法识别")

    cls: Dict[str, dict] = {}
    for c in feat_cols:
        cat, reason = classify_column(
            c, df[c], label_prefixes=label_prefixes,
            ident_prefixes=ident_prefixes, extra_patterns=extra_patterns,
        )
        cls[c] = {
            "category": cat,
            "reason": reason,
            "dtype": str(df[c].dtype),
            "null_ratio": round(float(df[c].isna().mean()), 4),
        }
    null_by_col = {c: d["null_ratio"] for c, d in cls.items()}

    groups = _group_columns(feat_cols, cls, null_by_col, min_group)
    return {
        "id_col": id_col,
        "dt_col": dt_col,
        "label_col": label_col,
        "counts": dict(Counter(d["category"] for d in cls.values())),
        "columns": cls,
        "groups": groups,
    }


def _group_columns(
    feat_cols: List[str], cls: dict, null_by_col: dict, min_group: int
) -> List[dict]:
    """按列名首 token 前缀聚类; 组内列数 >= min_group 折叠为 `pfx_*`, 单列展示全名。"""
    def group_key(c: str) -> str:
        m = re.match(r"^([a-zA-Z]+)_", c)
        if m and sum(1 for x in feat_cols if x.startswith(m.group(1) + "_")) >= min_group:
            return m.group(1) + "_*"
        return c

    groups: "OrderedDict[str, list]" = OrderedDict()
    for c in feat_cols:
        groups.setdefault(group_key(c), []).append(c)

    gstat = []
    for g, cols in groups.items():
        cats = {cls[c]["category"] for c in cols}
        gstat.append({
            "group": g, "n": len(cols), "cols": cols,
            "mixed": len(cats) > 1, "categories": sorted(cats),
            "null_median": round(sorted(null_by_col[c] for c in cols)[len(cols) // 2], 4),
        })
    return gstat


def render_report(scan: dict) -> str:
    """生成人可读分类报告(组级折叠 + 混合/小组展开), 供编排层展示与用户批量确认。"""
    groups = scan["groups"]
    nonf = [g for g in groups if set(g["categories"]) == {"non_feature"}]
    amb = [g for g in groups if set(g["categories"]) == {"ambiguous"}]
    fea = [g for g in groups if set(g["categories"]) == {"feature"}]
    mixed = [g for g in groups if len(g["categories"]) > 1]

    cnt = scan["counts"]
    lines = []
    lines.append("=" * 70)
    lines.append(f"特征列识别 (feature-classification, 规则库 {RULEBOOK})")
    lines.append(f"id={scan['id_col']} dt={scan['dt_col']} label={scan['label_col']}")
    lines.append(
        f"类别计数: feature={cnt.get('feature', 0)}  ambiguous={cnt.get('ambiguous', 0)}"
        f"  non_feature={cnt.get('non_feature', 0)}"
    )
    lines.append("规则命中仅为候选标记: non_feature 需用户批量确认后才剔除; ambiguous 默认保留。")
    lines.append("=" * 70)

    def show(title: str, groups_sub) -> None:
        lines.append("")
        lines.append(f"[{title}]")
        cols = scan["columns"]
        for g in sorted(groups_sub, key=lambda x: -x["n"]):
            mark = " ⚠混合组(需展开确认)" if g["mixed"] else ""
            lines.append(
                f"  {g['group']:<24} {g['n']:>3} 列 缺中位 {g['null_median']:.1%}{mark}"
            )
            if g["n"] <= 4 or g["mixed"]:
                for c in g["cols"]:
                    lines.append(
                        f"      - {c:<44} {cols[c]['reason']} (缺 {cols[c]['null_ratio']:.1%})"
                    )

    show("non_feature 候选 (建议剔除, 待批量确认)", nonf)
    show("ambiguous (默认保留, 仅报数量)", amb)
    show("feature (默认保留)", fea)
    if mixed:
        show("警告 混合组 (需展开确认)", mixed)
    return "\n".join(lines)


def _write_manifest(out_dir: Path, files: list, overview: dict) -> None:
    """落 _manifest.json(断点续跑 / 可追溯)。"""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "rulebook": RULEBOOK,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files),
        "overview": overview,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_scan_json(scan: dict, out_path: str) -> None:
    """落 feature-classification.json(探查版; decided_by 由 finalize 固化)。

    schema(方案 §六):
        schema_version / generated_as / rulebook / id_col / dt_col / label_col /
        counts / groups / columns{列名 → {category, reason, dtype, null_ratio}}
    """
    rec = {
        "schema_version": SCHEMA_VERSION,
        "generated_as": "scan",
        "rulebook": RULEBOOK,
        "id_col": scan["id_col"],
        "dt_col": scan["dt_col"],
        "label_col": scan["label_col"],
        "counts": scan["counts"],
        "groups": [
            {"group": g["group"], "n": g["n"], "mixed": g["mixed"],
             "categories": g["categories"]}
            for g in scan["groups"]
        ],
        "columns": scan["columns"],
    }
    Path(out_path).write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="feature-classification: 特征列语义三分类 + 通配符分组")
    ap.add_argument("--input", required=True, help="本地样本文件(parquet/csv/feather/xlsx/json)")
    ap.add_argument("--out-dir", required=True, help="产物目录(建议 <session_dir>/sample-features)")
    ap.add_argument("--id-col", default="fuid", help="用户粒度 ID 列(不参与分类)")
    ap.add_argument("--dt-col", default="f_p_date", help="日期分区列(不参与分类)")
    ap.add_argument("--label-col", default="label", help="标签列(不参与分类)")
    ap.add_argument(
        "--label-prefixes",
        default=",".join(DEFAULT_LABEL_PREFIXES),
        help=f"标签列前缀红线(逗号分隔, 默认 {','.join(DEFAULT_LABEL_PREFIXES)}), 置顶规则",
    )
    ap.add_argument(
        "--extra-patterns",
        default=None,
        help="追加的 non_feature 正则(逗号分隔, 用户自定义规则, 优先级高于默认规则)",
    )
    ap.add_argument("--min-group", type=int, default=2,
                    help="组内列数 >= N 折叠为 pfx_*(默认 2)")
    args = ap.parse_args()

    label_prefixes = tuple(p.strip() for p in args.label_prefixes.split(",") if p.strip())
    extra_patterns = [p.strip() for p in (args.extra_patterns or "").split(",") if p.strip()]

    df = read_sample(args.input)
    scan = scan_features(
        df, args.id_col, args.dt_col, args.label_col,
        label_prefixes=label_prefixes, extra_patterns=extra_patterns,
        min_group=args.min_group,
    )

    print(render_report(scan))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "feature-classification.json"
    write_scan_json(scan, str(json_path))
    _write_manifest(
        out_dir,
        ["feature-classification.json", "_manifest.json"],
        {
            "n_cols": int(len(df.columns)),
            "n_feature_cols": len(scan["columns"]),
            "counts": scan["counts"],
        },
    )
    print(f"\n已落盘: {json_path} (探查档案, 判定人待 finalize 固化)")
    print("下一步: 向用户展示上述报告 → 用户批量确认剔除/保留名单 → "
          "运行 finalize_feature_list.py 生成权威 feature-list.csv")


if __name__ == "__main__":
    main()