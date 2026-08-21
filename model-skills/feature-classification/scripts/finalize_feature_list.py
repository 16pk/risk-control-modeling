# -*- coding: utf-8 -*-
"""feature-classification · 固化阶段: 应用用户批量确认的名单, 生成权威 feature-list.csv。

用法:
    python feature-classification/scripts/finalize_feature_list.py \
        --classification <session_dir>/sample-features/feature-classification.json \
        --out-dir <session_dir>/sample-features \
        --exclude if_tf,if_ka,fser_date,sx_order_id,jy_order_id,ftrans_time,fst_rn,last_rn \
        [--keep flag_ok]

行为(方案 §5.4 / §六):
    - exclude: 用户确认剔除 → category=non_feature + decided_by=user
    - keep:    用户明确保留(如恢复规则误判的 non_feature → feature, 或固化 ambiguous) → decided_by=user
    - 未确认的规则判定标 decided_by=rule
    - 权威 feature-list.csv = 全部非 non_feature 列(展示层通配符, 存储层逐列, 可复现可审计)
    - counts 保留初次分类快照; current_counts 记录固化后计数

复用约定(方案 §六): 下次 session 读档案直接复用, 不重复询问;
仅列集合变化时触发增量重分类。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts

import pandas as pd


def validate_names(rec: dict, exclude: List[str], keep: List[str]) -> List[str]:
    """校验剔除/保留名单都在分类档案中, 返回缺失列。"""
    cols = rec.get("columns", {})
    return [c for c in [*exclude, *keep] if c not in cols]


def finalize(
    classification_path: str,
    out_dir: str,
    exclude: Iterable[str] = (),
    keep: Iterable[str] = (),
    confirmed_at: Optional[str] = None,
) -> dict:
    """应用用户断定名单, 固化 feature-classification.json 并产出权威 feature-list.csv。

    Args:
        classification_path: feature-classification.json 路径(原地固化)
        out_dir: feature-list.csv 输出目录
        exclude: 用户确认剔除的列(非特征)
        keep: 用户确认保留的列(可选)
        confirmed_at: 确认时间(测试注入); 默认当前本地时间 ISO

    Returns:
        summary dict: n_excluded / n_kept / n_features / feature_list_csv /
        classification_json / counts / user_confirmed_at

    Raises:
        ValueError: 名单含档案外的未知列。
    """
    cls_path = Path(classification_path)
    rec = json.loads(cls_path.read_text(encoding="utf-8"))
    exclude = list(dict.fromkeys(c for c in exclude if c))  # 保序去重
    keep = list(dict.fromkeys(c for c in keep if c))

    missing = validate_names(rec, exclude, keep)
    if missing:
        raise ValueError(f"名单含未知列(不在分类档案中): {missing}")

    columns = rec["columns"]

    # 用户确认剔除: 一律置 non_feature + 判定人 user
    for c in exclude:
        columns[c]["category"] = "non_feature"
        columns[c]["decided_by"] = "user"
        columns[c]["reason"] = "用户确认非特征"

    # 用户确认保留: 恢复规则误判的 non_feature → feature; 其余固化判定人
    for c in keep:
        col = columns[c]
        if col.get("category") == "non_feature":
            col["category"] = "feature"
        col["decided_by"] = "user"
        col["reason"] = "用户确认保留"

    # 未确认的规则判定
    for d in columns.values():
        d.setdefault("decided_by", "rule")

    rec["generated_as"] = "final"
    rec["user_confirmed_exclude"] = exclude
    if keep:
        rec["user_confirmed_keep"] = keep
    rec["user_confirmed_at"] = (
        confirmed_at or datetime.now().astimezone().isoformat(timespec="seconds")
    )
    # counts 保留初次分类快照(可审计初判), current_counts 为固化后计数
    rec["current_counts"] = dict(Counter(d["category"] for d in columns.values()))

    # 权威特征清单: 非 non_feature 的全部列(逐列精确, 无通配符)
    feats = [c for c, d in columns.items() if d["category"] != "non_feature"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fl_path = out / "feature-list.csv"
    pd.DataFrame({"feature_name": feats}).to_csv(fl_path, index=False)
    cls_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    return {
        "n_excluded": len(exclude),
        "n_kept": len(keep),
        "n_features": len(feats),
        "feature_list_csv": str(fl_path),
        "classification_json": str(cls_path),
        "counts": rec["current_counts"],
        "user_confirmed_at": rec["user_confirmed_at"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="feature-classification: 固化权威特征清单")
    ap.add_argument("--classification", required=True,
                    help="feature-classification.json 路径(原地固化)")
    ap.add_argument("--out-dir", required=True, help="feature-list.csv 输出目录")
    ap.add_argument("--exclude", default="", help="用户确认剔除的列(逗号分隔)")
    ap.add_argument("--keep", default="", help="用户确认保留的列(逗号分隔, 可选)")
    args = ap.parse_args()

    exclude = [c.strip() for c in args.exclude.split(",") if c.strip()]
    keep = [c.strip() for c in args.keep.split(",") if c.strip()]
    summary = finalize(args.classification, args.out_dir, exclude=exclude, keep=keep)

    print(f"剔除 {summary['n_excluded']} 列 / 保留固化 {summary['n_kept']} 列"
          f" -> 权威特征清单 {summary['n_features']} 列")
    print(f"feature-list.csv:            {summary['feature_list_csv']}")
    print(f"feature-classification.json: {summary['classification_json']} (decided_by 已固化)")
    print(f"分类计数: {summary['counts']} / 确认时间: {summary['user_confirmed_at']}")


if __name__ == "__main__":
    main()