# -*- coding: utf-8 -*-
"""写清洗方案产物: cleaning-scheme.json(机器可读) + cleaning-report.md(人工可读)。

记录「对哪些特征、做了怎样的处理」及命中统计, 供后续 session 复用/复现。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd


SCHEMA_VERSION = 1
PRODUCED_BY = "skills/data-cleaning"


def _normalize_hit_values(hit_str: str) -> list:
    """把 report 里的 'hit_values' 字符串(逗号分隔)还原为数值列表(整型化)。"""
    out = []
    for v in str(hit_str).split(","):
        v = v.strip()
        if not v:
            continue
        try:
            fv = float(v)
            out.append(int(fv) if fv.is_integer() else fv)
        except ValueError:
            out.append(v)
    return out


def build_scheme(
    invalid_values: list,
    dedup_keys: list,
    dedup_keep_rule: str,
    invalid_report: pd.DataFrame,
    dedup_report: dict,
) -> dict:
    """组装清洗方案 dict。

    Args:
        invalid_values: 本次使用的哨兵值集合
        dedup_keys: 去重键列
        dedup_keep_rule: 去重保留规则(如 label_non_null)
        invalid_report: replace_invalid_values 返回的明细表
        dedup_report: dedup_by_user_date 返回的统计

    Returns:
        cleaning-scheme dict(见 SKILL.md 产物说明)
    """
    features = []
    if invalid_report is not None and not invalid_report.empty:
        for _, row in invalid_report.iterrows():
            features.append({
                "feature": row["feature"],
                "action": "replace_invalid_to_nan",
                "hit_values": _normalize_hit_values(row["hit_values"]),
                "n_hit": int(row["n_hit"]),
                "hit_ratio": float(row["hit_ratio"]),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "invalid_values": [
            int(v) if float(v).is_integer() else float(v) for v in invalid_values
        ],
        "dedup_keys": list(dedup_keys),
        "dedup_keep_rule": dedup_keep_rule,
        "features": features,
        "dedup_report": dedup_report,
    }


def write_scheme_json(scheme: dict, out_path: str) -> str:
    """落 cleaning-scheme.json。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scheme, f, ensure_ascii=False, indent=2)
    return out_path


def write_scheme_report(scheme: dict, out_path: str) -> str:
    """落 cleaning-report.md(人工阅读)。"""
    lines: list = []
    lines.append("# 数据清洗报告 - data-cleaning")
    lines.append("")
    lines.append(f"- 生成时间: {scheme.get('generated_at', '—')}")
    lines.append(f"- 哨兵值集合: `{', '.join(str(v) for v in scheme.get('invalid_values', []))}`")
    lines.append(f"- 去重维度: `{' + '.join(scheme.get('dedup_keys', []))}` (保留规则: `{scheme.get('dedup_keep_rule', '—')}`)")
    lines.append("")

    dedup = scheme.get("dedup_report") or {}
    lines.append("## 一、去重")
    lines.append("")
    lines.append(
        f"- 去重前: **{dedup.get('n_before', '—')}** 行; 去重后: **{dedup.get('n_after', '—')}** 行;"
        f" 移除: **{dedup.get('n_removed', '—')}** 行"
    )
    lines.append("")

    feats = scheme.get("features") or []
    lines.append("## 二、哨兵值替换")
    lines.append("")
    if not feats:
        lines.append("未发现哨兵值命中, 无需替换。")
        lines.append("")
    else:
        lines.append("| 特征 | 命中哨兵值 | 命中行数 | 命中占比 |")
        lines.append("|------|-----------|---------|---------|")
        for f in feats:
            hit_vals = ", ".join(str(v) for v in f["hit_values"])
            lines.append(
                f"| `{f['feature']}` | {hit_vals} | {f['n_hit']} | {f['hit_ratio']:.4%} |"
            )
        lines.append("")
        lines.append("> 命中值已替换为 NaN, 后续训练按缺失处理。")
        lines.append("")

    lines.append("## 三、复用")
    lines.append("")
    lines.append("本方案的机器可读版本见 `cleaning-scheme.json`; 复用时可读该 json 的")
    lines.append("`invalid_values` 作为下次默认哨兵值集合, 保证可复现。")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
