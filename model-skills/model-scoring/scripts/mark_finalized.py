#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""model-scoring 定版标记工具: 落 finalized_model.json。

由 classification-model-development Stage 4 收口、用户确认上线候选后调用, 把
「定版 run 的模型文件 + 元信息」落 session 根 finalized_model.json, 供
Stage 5 model-scoring 定位定版模型(机器可读, 取代此前仅停留在 report.md 附录
与 model_catalog.csv 的人工文字标记)。

finalized_model.json 结构:
{
  "schema_version": 1,
  "produced_by": "skills/model-scoring",
  "run_name": "xgb-v1",
  "algo": "xgb",
  "model_path": "new-models/xgb-v1/model/model.json",   # 相对 session_dir
  "model_dir": "new-models/xgb-v1/model",
  "feature_names": [...],
  "oot_auc": 0.82,                                       # 可空
  "finalized_at": "2026-08-13T12:00:00"
}

用法:
  python mark_finalized.py --session-dir <session_dir> --run-name xgb-v1 [--oot-auc 0.82]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from score_data import infer_algo, read_model_meta  # 复用同目录打分脚本的元信息读取


def main() -> int:
    parser = argparse.ArgumentParser(description="落定版标记 finalized_model.json")
    parser.add_argument("--session-dir", required=True, help="session 根目录")
    parser.add_argument("--run-name", required=True, help="定版 run 名(如 xgb-v1)")
    parser.add_argument("--oot-auc", type=float, default=None,
                        help="定版模型 OOT AUC(可选, 供报告/台账追溯)")
    parser.add_argument("--out", default=None,
                        help="标记文件输出路径(默认 <session_dir>/finalized_model.json)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    run_dir = session_dir / "new-models" / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f"[ERROR] 定版 run 目录不存在: {run_dir}")

    model_dir = run_dir / "model"
    if not model_dir.is_dir():
        raise SystemExit(f"[ERROR] 模型目录不存在: {model_dir}(请确认已跑完 model 阶段)")

    meta = read_model_meta(model_dir)
    algo = infer_algo(model_dir, meta, None)
    model_file = "model.json" if algo == "xgb" else "model.pkl"
    if not (model_dir / model_file).exists():
        raise SystemExit(f"[ERROR] 模型文件缺失: {model_dir / model_file}")

    feature_names = list(meta.get("feature_names") or [])
    if not feature_names:
        raise SystemExit(f"[ERROR] model_meta.json 缺 feature_names: {model_dir / 'model_meta.json'}")

    payload = {
        "schema_version": 1,
        "produced_by": "skills/model-scoring",
        "run_name": args.run_name,
        "algo": algo,
        "model_path": f"new-models/{args.run_name}/model/{model_file}",
        "model_dir": f"new-models/{args.run_name}/model",
        "feature_names": feature_names,
        "oot_auc": args.oot_auc,
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
    }

    out = Path(args.out) if args.out else session_dir / "finalized_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[FINALIZED] 已落定版标记: {out}")
    print(f"[FINALIZED] run={args.run_name} algo={algo} 特征数={len(feature_names)} oot_auc={args.oot_auc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
