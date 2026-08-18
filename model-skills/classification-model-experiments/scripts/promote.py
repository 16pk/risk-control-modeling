# -*- coding: utf-8 -*-
"""转正闭环：top10 展示 + 用户确认/改选 → 复制 new-models/{algo}-v{N} + finalized_model.json（plan §2.1 A1 / D3）。

转正产物保持 model-scoring 消费契约：
  <session_dir>/new-models/{run_name}/model/model.pkl + model_meta.json + config.json
  <session_dir>/finalized_model.json   # 结构对齐 mark_finalized.py 的 schema_version=1

注意：model-scoring 的 score_data.py 目前原生支持 xgb/dnn/lr；本模块产 lgb/xgb 实验，
若 lgb 需下游打分，请后续在 model-scoring 扩展 lgb 加载（本模块不跨 skill import）。
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional


def build_candidates(exp_root: str, specs: List[Dict], k: int = 10) -> List[Dict]:
    """汇总 done 且 OOT AUC 非空的前 k 名（含 -opt 格）作为转正候选。"""
    rows = []
    for s in specs:
        if s["status"] != "done":
            continue
        exp_dir = os.path.join(exp_root, s["id"])
        eval_json = os.path.join(exp_dir, "evaluation", "eval.json")
        if not os.path.exists(eval_json):
            continue
        with open(eval_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        oot_auc = payload.get("splits", {}).get("oot", {}).get("auc")
        val_auc = payload.get("splits", {}).get("val", {}).get("auc")
        if oot_auc is None:
            continue
        rows.append({
            "id": s["id"], "algo": s["algo"], "sample_scheme": s["sample_scheme"],
            "feat_scheme": s["feat_scheme"], "oot_auc": oot_auc, "val_auc": val_auc,
            "optimistic_bias": bool(payload.get("optimistic_bias", False)),
            "is_tuned": bool(s.get("is_tuned", False)),
            "base_exp": s.get("plan", {}).get("base_exp"),
        })
    rows.sort(key=lambda r: (-r["oot_auc"], r["id"]))
    return rows[:k]


def render_candidates(cands: List[Dict]) -> str:
    lines = ["| # | id | algo | sample | feat | OOT AUC | val AUC | tuned | 标注 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cands, 1):
        note = "⚠ 乐观偏差" if c["optimistic_bias"] else "-"
        tuned = "opt" if c.get("is_tuned") else "-"
        lines.append("| {i} | {id} | {algo} | {sample} | {feat} | {oot} | {val} | {tuned} | {note} |".format(
            i=i, id=c["id"], algo=c["algo"], sample=c["sample_scheme"], feat=c["feat_scheme"],
            oot="%.4f" % c["oot_auc"], val="-" if c["val_auc"] is None else "%.4f" % c["val_auc"],
            tuned=tuned, note=note))
    return "\n".join(lines)


def select_promote(cands: List[Dict], auto: bool = False,
                   promote_id: Optional[str] = None) -> Optional[Dict]:
    """用户确认/改选转正实验。

    - promote_id 指定 → 直接选
    - auto → 选第 1 名（OOT AUC 最优）
    - 否则交互：默认回车选推荐（第 1 名），或输入序号 / id 改选；输入 n 取消。
    """
    if promote_id:
        for c in cands:
            if c["id"] == promote_id:
                return c
        raise ValueError(f"指定转正实验 {promote_id} 不在 top{len(cands)} 候选内")
    if auto:
        return cands[0] if cands else None
    # 交互
    print("\n===== 转正候选 top%d（默认推荐 OOT AUC 最优）=====" % len(cands))
    print(render_candidates(cands))
    try:
        choice = input("\n回车选推荐 #1；输入序号/id 改选；输入 n 取消转正：").strip()
    except EOFError:
        choice = ""
    if choice.lower() in ("n", "no"):
        print("[promote] 用户取消转正")
        return None
    if choice == "":
        return cands[0] if cands else None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(cands):
            return cands[idx - 1]
        print(f"[promote] 序号越界，使用推荐 #1")
        return cands[0] if cands else None
    for c in cands:
        if c["id"] == choice:
            return c
    print(f"[promote] 未匹配 id={choice!r}，使用推荐 #1")
    return cands[0] if cands else None


def _next_run_version(session_dir: str, algo: str) -> str:
    """扫描 new-models/ 下 {algo}-v{N} 取 max+1；无则 v1。"""
    base = os.path.join(session_dir, "new-models")
    max_n = 0
    if os.path.isdir(base):
        for name in os.listdir(base):
            prefix = f"{algo}-v"
            if name.startswith(prefix):
                rest = name[len(prefix):]
                if rest.isdigit():
                    max_n = max(max_n, int(rest))
    return f"{algo}-v{max_n + 1}"


def promote(cands: List[Dict], exp_root: str, session_dir: str,
            auto: bool = False, promote_id: Optional[str] = None,
            oot_auc_override: Optional[float] = None) -> Optional[Dict]:
    """执行转正：复制实验模型到 new-models/{algo}-v{N} + 写 finalized_model.json。

    Returns:
        转正信息 dict（含 run_name / candidate / finalized_path），None 表示取消。
    """
    sel = select_promote(cands, auto=auto, promote_id=promote_id)
    if sel is None:
        return None

    algo = sel["algo"]
    run_name = _next_run_version(session_dir, algo)
    run_dir = os.path.join(session_dir, "new-models", run_name)
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    src_exp = os.path.join(exp_root, sel["id"])
    src_model = os.path.join(src_exp, "model")
    shutil.copyfile(os.path.join(src_model, "model.pkl"), os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(src_model, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    shutil.copyfile(os.path.join(src_model, "model_meta.json"),
                    os.path.join(model_dir, "model_meta.json"))

    # config.json（含 runtime：来源实验 / 方案 / 指标）
    with open(os.path.join(src_exp, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    config = {
        "produced_by": "skills/model-experiments",
        "run_name": run_name,
        "algo": algo,
        "source_exp": sel["id"],
        "sample_scheme": sel["sample_scheme"],
        "feat_scheme": sel["feat_scheme"],
        "params": manifest.get("params", {}),
        "features": meta.get("feature_names", []),
        "metrics": {"oot_auc": sel["oot_auc"], "val_auc": sel["val_auc"]},
        "optimistic_bias": sel["optimistic_bias"],
        "is_tuned": sel.get("is_tuned", False),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # finalized_model.json（对齐 mark_finalized schema_version=1）
    oot_auc = oot_auc_override if oot_auc_override is not None else sel["oot_auc"]
    payload = {
        "schema_version": 1,
        "produced_by": "skills/model-experiments",
        "run_name": run_name,
        "algo": algo,
        "model_path": f"new-models/{run_name}/model/model.pkl",
        "model_dir": f"new-models/{run_name}/model",
        "feature_names": meta.get("feature_names", []),
        "oot_auc": oot_auc,
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
        "source_exp": sel["id"],
    }
    finalized_path = os.path.join(session_dir, "finalized_model.json")
    with open(finalized_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[PROMOTE] 已转正: {sel['id']} → new-models/{run_name} (OOT AUC={oot_auc:.4f})")
    print(f"[PROMOTE] 定版标记: {finalized_path}")
    return {"run_name": run_name, "candidate": sel, "finalized_path": finalized_path}