#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""classification-model-experiments 主入口 CLI：范围确认→规划→矩阵串行→评选→Optuna→转正确认。

用法：
  python run_experiments.py \
      --session-dir <session_dir> --sample <sample.parquet> --feature-list <feature-list.csv> \
      --config <feature_config.yaml> \
      [--split-train ...] [--split-test ...] [--split-oot ...] \
      [--label-col ...] [--id-col fuid] [--dt-col f_p_date] \
      [--algos lgb xgb] [--no-sample-select] [--no-feat-select] \
      [--no-adversarial] [--no-tune] \
      [--max-experiments-per-algo 12] [--n-trials 25] \
      [--auto-apply] [--resume] [--until matrix|tune|promote] [--promote-id <id>]

主流程（plan §5.1）：
  实验范围确认(v2.5：算法三选一 + 样本/特征选择/对抗验证/Optuna 4 开关, 对抗与 Optuna 附耗时提醒)
  → 矩阵规划 → 波1 all 格（lgb-full-all-v1 兼 baseline）→ 波2 importance/iv-psi →
  对抗格（幅度确认）→ leaderboard → 每算法 winner Optuna（-opt，关闭时规则诊断一并跳过）
  → top10 展示 → 转正。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import sample_schemes as ss
from config_io import check_sensitive
from date_utils import parse_date_pair
from feature_schemes import adversarial_features, importance_features, iv_psi_features
from leaderboard import collect_results, sort_rows, write_leaderboard
from plan_matrix import build_matrix, load_state, save_state, update_spec, get_spec
from plan_scope import resolve_scope, scope_summary
from promote import build_candidates, promote as promote_run
from run_single_experiment import run_experiment, split_dev
from safety_filter import filter_boundary_features_from_df


# ---------------------------------------------------------------------------
# 输入加载
# ---------------------------------------------------------------------------
def load_split_from_df(df: pd.DataFrame, dt_col: str, rng: list) -> pd.DataFrame:
    """按 model.split 区间从全量 df 过滤出单档。rng=[start,end] 归一化 8 位。"""
    raw = df[dt_col].astype(str).str.strip()
    out_mask = np.zeros(len(df), dtype=bool)
    for i, v in enumerate(raw):
        try:
            from date_utils import normalize_date

            d = normalize_date(v)
            out_mask[i] = rng[0] <= d <= rng[1]
        except Exception:
            continue
    return df[out_mask].copy()


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="建模实验台：矩阵实验+对抗验证+Optuna调优+转正")
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--sample", required=True, help="样本 parquet")
    ap.add_argument("--feature-list", required=True, help="特征清单 csv")
    ap.add_argument("--config", default=None, help="含 model.split 的 yaml（与 --split-* 二选一）")
    ap.add_argument("--split-train", default=None)
    ap.add_argument("--split-test", default=None)
    ap.add_argument("--split-oot", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--id-col", default="fuid")
    ap.add_argument("--dt-col", default="f_p_date")
    ap.add_argument("--algos", nargs="+", default=None,
                    help="实验算法（lgb/xgb/两者；缺省时规划前交互询问）")
    ap.add_argument("--no-sample-select", action="store_true",
                    help="不做样本选择（仅全量 full 方案，跳过 recent-N/时间加权）")
    ap.add_argument("--no-feat-select", action="store_true",
                    help="不做特征选择（仅 all 方案，跳过 importance/iv-psi 格）")
    ap.add_argument("--no-adversarial", action="store_true",
                    help="不做对抗验证（跳过波3 对抗格）")
    ap.add_argument("--no-tune", action="store_true",
                    help="不做 Optuna 调优（winner 规则诊断一并跳过，仅出 leaderboard）")
    ap.add_argument("--max-experiments-per-algo", type=int, default=12)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--auto-apply", action="store_true", help="跳过所有交互确认")
    ap.add_argument("--resume", action="store_true", help="断点续跑：跳过 done 实验")
    ap.add_argument("--until", default="promote", choices=["matrix", "tune", "promote"])
    ap.add_argument("--promote-id", default=None)
    ap.add_argument("--force-tune", action="store_true",
                    help="v2.3: winner 规则诊断为 well_fit 时仍强制 Optuna 调优（默认跳过）")
    return ap.parse_args()


def resolve_split(args: argparse.Namespace, sample_dir: str):
    """从 --config 或 --split-* 解析 (train_range, test_range, oot_range)（8 位归一化）。"""
    if args.config:
        from config_io import load_config, validate_split_ranges

        cfg = load_config(args.config)
        model = cfg.get("model") or {}
        if not model.get("split"):
            raise SystemExit(f"[ERROR] {args.config} 缺 model.split（切分唯一真相）")
        validate_split_ranges(model)
        split = model["split"]
        ranges = {name: parse_date_pair(split[f"{name}_range"]) for name in ("train", "test", "oot")}
        return ranges
    if args.split_train and args.split_test and args.split_oot:
        ranges = {
            "train": parse_date_pair(args.split_train, what="--split-train"),
            "test": parse_date_pair(args.split_test, what="--split-test"),
            "oot": parse_date_pair(args.split_oot, what="--split-oot"),
        }
        return ranges
    raise SystemExit("[ERROR] 请提供 --config（含 model.split）或 --split-train/--split-test/--split-oot 三档")


def infer_label(df: pd.DataFrame, cfg_label: str) -> str:
    if cfg_label:
        return cfg_label
    for c in df.columns:
        if "label" in c.lower():
            return c
    raise SystemExit("[ERROR] 无法推断标签列；请 --label-col 显式指定")


def load_inputs(args: argparse.Namespace):
    """加载 sample/feature-list/split，返回 (df, label_col, dt_col, id_col, base_features, ranges)。"""
    sample_path = Path(args.sample)
    fl_path = Path(args.feature_list)
    if not sample_path.exists():
        raise SystemExit(f"[ERROR] 样本不存在: {sample_path}")
    if not fl_path.exists():
        raise SystemExit(f"[ERROR] 特征清单不存在: {fl_path}")
    df = pd.read_parquet(sample_path)
    # 数据安全红线
    check_sensitive(",".join(map(str, df.columns)))

    from gen_feature_list import load_feature_list

    features = load_feature_list(str(fl_path))
    feat_in = [f for f in features if f in df.columns]
    if not feat_in:
        raise SystemExit(f"[ERROR] 特征清单中无任何列存在于样本: {fl_path}")

    cfg_label = None
    if args.config:
        from config_io import load_config

        cfg = load_config(args.config)
        cfg_label = (cfg.get("model") or {}).get("label_col")
    label_col = infer_label(df, args.label_col or cfg_label)
    if label_col not in df.columns:
        raise SystemExit(f"[ERROR] 标签列不存在: {label_col}")

    ranges = resolve_split(args, str(sample_path.parent))
    return df, label_col, args.dt_col, args.id_col, feat_in, ranges


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_cli()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("exp")

    session_dir = Path(args.session_dir).resolve()
    exp_root = os.path.join(str(session_dir), "experiments")
    os.makedirs(exp_root, exist_ok=True)
    plan_json = os.path.join(exp_root, "matrix-plan.json")
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "templates", "train_template.py")

    df, label_col, dt_col, id_col, raw_features, ranges = load_inputs(args)

    # 安全过滤 → 基础特征（数据直算，开发池基准）
    dev_all = load_split_from_df(df, dt_col, [ranges["train"][0], ranges["test"][1]])
    oot_all = load_split_from_df(df, dt_col, ranges["oot"])
    dev_filter_input = dev_all.drop(columns=[c for c in (id_col, dt_col) if c in dev_all.columns],
                                    errors="ignore")
    base_features, dropped = filter_boundary_features_from_df(
        dev_filter_input, id_col=id_col, label_col=label_col)
    log.info("[plan] 安全过滤: %d 特征保留, 剔除 %d 个: %s",
             len(base_features), len(dropped), [d[0] for d in dropped])

    if len(base_features) == 0:
        raise SystemExit("[ERROR] 安全过滤后无可用特征")

    # 1) 矩阵规划（规划前交互确认算法+4 开关 → 组数自决 + 理由）
    if args.resume and os.path.exists(plan_json):
        specs = load_state(plan_json)
        scope = {"algos": args.algos or ["lgb", "xgb"],
                 "optuna": not args.no_tune}
        log.info("[plan] 恢复断点：%d 个实验", len(specs))
    else:
        n_months = len(set(ss._month_series(dev_all, dt_col).dropna().astype(str)))
        oot_available = len(oot_all) > 0
        scope = resolve_scope(
            algos_cli=args.algos,
            no_sample_select=args.no_sample_select,
            no_feat_select=args.no_feat_select,
            no_adversarial=args.no_adversarial,
            no_tune=args.no_tune,
            n_months=n_months,
            oot_available=oot_available,
            auto=args.auto_apply)
        log.info("[plan] 实验范围：%s", scope_summary(scope))
        sample_plans = ss.decide_sample_schemes(dev_all, dt_col, label_col)
        if not scope["sample_select"]:
            sample_plans = [p for p in sample_plans if p["name"] == "full"]
        reasons = ["开发池 %d 样本 / %d 特征 / %d 月" % (
            len(dev_all), len(base_features), n_months)]
        reasons.extend(scope["reasons"])
        for p in sample_plans:
            reasons.append(f"  - {p['name']}: {p['reason']}")
        specs = build_matrix(scope["algos"], sample_plans,
                             oot_available=oot_available,
                             max_experiments=args.max_experiments_per_algo,
                             feat_select=scope["feat_select"],
                             adversarial=scope["adversarial"])
        save_state(plan_json, specs, reasons)
        log.info("[plan] 矩阵规划完成：%d 个实验（%s）", len(specs), ", ".join(scope["algos"]))

    # 2) 波1 + 波2：串行执行非对抗格
    def _flush_reasons(specs):
        try:
            with open(plan_json, "r", encoding="utf-8") as f:
                return json.load(f).get("planning_reasons", [])
        except Exception:
            return []

    def _run_spec(spec, sample_scheme=None, feat_scheme=None, feat_override=None,
                  importance_source=None, optimistic=False):
        run_experiment(
            spec, dev=dev_all, oot=oot_all, label_col=label_col, dt_col=dt_col,
            id_col=id_col, base_features=base_features, exp_root=exp_root,
            template_path=template_path, sample_scheme=sample_scheme,
            feat_scheme=feat_scheme or spec.get("feat_scheme", "all"),
            feat_override=feat_override, importance_source=importance_source,
            optimistic_bias=optimistic, resume=args.resume)
        save_state(plan_json, specs, _flush_reasons(specs))

    # 波1：所有样本方案 all 格（lgb-full-all-v1 兼 baseline）
    wave1 = [s for s in specs if s["wave"] == 1]
    for spec in wave1:
        sname = spec["sample_scheme"]
        if sname == "full":
            scheme = ss.full_scheme(dev_all)
        elif sname.startswith("recent"):
            n = int(sname.replace("recent", ""))
            scheme = ss.recent_n_scheme(dev_all, dt_col, n)
        else:  # timeweight
            scheme = ss.linear_time_weight_scheme(dev_all, dt_col)
        _run_spec(spec, sample_scheme=scheme)

    # 波2：importance（依赖同样本 all 格）+ iv-psi（单格直算）
    wave2 = [s for s in specs if s["wave"] == 2]
    for spec in wave2:
        if spec["feat_scheme"] == "importance":
            dep = get_spec(specs, spec["depends_on"])
            if dep is None or dep["status"] != "done":
                spec["status"] = "failed"
                spec["fail_reason"] = f"依赖 {spec['depends_on']} 未完成"
                save_state(plan_json, specs, _flush_reasons(specs))
                log.warning("[exp] %s 依赖缺失，跳过", spec["id"])
                continue
            dep_dir = os.path.join(exp_root, dep["id"])
            imp_df = pd.read_csv(os.path.join(dep_dir, "feature_importance.csv"))
            imp_scheme = _scheme_for(specs, spec, dev_all, dt_col)
            _run_spec(spec, sample_scheme=imp_scheme,
                      importance_source={"exp_dir": dep_dir, "importance_df": imp_df})
        else:  # iv-psi
            iv_scheme = _scheme_for(specs, spec, dev_all, dt_col)
            _run_spec(spec, sample_scheme=iv_scheme, optimistic=True)

    # 波3：对抗格（lgb 主跑；幅度确认后双产出合并应用）
    adv_specs = [s for s in specs if s["wave"] == 3]
    for spec in adv_specs:
        if spec["status"] == "done":
            continue
        try:
            import adversarial

            model, imp_adv, oot_auc = adversarial.train_adversarial(
                dev_all, oot_all, base_features, seed=42, log=log)
            proba_dev = model.predict_proba(
                dev_all[base_features].apply(pd.to_numeric, errors="coerce"))[:, 1]
            rec = adversarial.recommend_drop(proba_dev, oot_auc)
            top_k = max(3, int(len(base_features) * 0.15))
            ans = rec["recommended_sample_drop_pct"]
            if not args.auto_apply:
                try:
                    inp = input(
                        f"[对抗] {spec['id']}: {rec['desc']}（剔除 proba 最低、最不像 OOT 的样本）；"
                        f"特征剔除 top{top_k}。回车接受/输入新剔除比例(0~1)/n 跳过：").strip()
                except EOFError:
                    inp = ""
                if inp.lower() in ("n", "no"):
                    spec["status"] = "failed"
                    spec["fail_reason"] = "user_cancelled"
                    save_state(plan_json, specs, _flush_reasons(specs))
                    continue
                if inp:
                    try:
                        ans = min(max(float(inp), 0.0), 1.0)
                    except ValueError:
                        ans = rec["recommended_sample_drop_pct"]
            masks = adversarial.compute_drop_masks(proba_dev, ans, imp_adv, top_k, base_features)
            adv_scheme = ss.adversarial_filter_scheme(dev_all, masks["sample_drop_mask"],
                                                      meta={**rec, "sample_drop_n": masks["sample_drop_n"]})
            adv_feats = adversarial_features(base_features, top_k, imp_adv)
            # 落盘对抗验证元数据（oot_auc / 推荐与实际剔除量 / 剔除特征 top-K / desc）
            adv_dir = os.path.join(exp_root, spec["id"])
            adv_meta = {
                "oot_auc": rec["oot_auc"],
                "recommended_sample_drop_pct": ans,
                "sample_drop_n": masks["sample_drop_n"],
                "sample_drop_pct_actual": round(masks["sample_drop_n"] / max(len(dev_all), 1), 4),
                "feature_drop_topk": masks["feature_drop_list"],
                "feature_drop_n": masks["feature_drop_n"],
                "desc": rec["desc"],
            }
            adversarial.save_adversarial_meta(adv_dir, adv_meta)
            log.info("[对抗] %s 落盘对抗验证元数据: %s", spec["id"], os.path.join(adv_dir, "adversarial_meta.json"))
            _run_spec(spec, sample_scheme=adv_scheme, feat_scheme="adversarial",
                      feat_override=adv_feats, optimistic=True)
        except Exception as e:
            spec["status"] = "failed"
            spec["fail_reason"] = f"对抗训练失败: {e}"
            save_state(plan_json, specs, _flush_reasons(specs))
            log.warning("[exp] 对抗格失败: %s", spec["id"])

    save_state(plan_json, specs, _flush_reasons(specs))

    if args.until == "matrix":
        write_leaderboard(exp_root, specs)
        log.info("[done] 矩阵执行完成（--until matrix 停止）")
        return 0

    # 3) leaderboard（OOT AUC 排序 + 乐观偏差标注）
    rows = sort_rows(collect_results(exp_root, specs))
    lb_md = write_leaderboard(exp_root, specs)
    log.info("[lb] leaderboard: %s", lb_md)

    # 4) 每算法 winner → Optuna 邻域调优（产 -opt run；-opt 复用 winner 数据快照）
    #    关闭 Optuna（--no-tune / 交互回答不做）→ 规则诊断一并跳过，直接进入 leaderboard 与转正
    #    v2.5 幂等：候选 winner 排除已调优格（is_tuned）+ 已有 done 的 -opt 格则跳过（防重放重复调优）
    winners = {}
    if scope["optuna"]:
        for algo in scope["algos"]:
            algo_rows = [r for r in rows if r["algo"] == algo and r["status"] == "done"]
            if not algo_rows:
                continue
            winner = algo_rows[0]
            winners[algo] = winner
            win_spec = get_spec(specs, winner["id"])
            if win_spec is None:
                continue
            # ① 跳过已调优格：winner 本身已是 -opt 格（is_tuned）→ 不再重复调优
            if win_spec.get("is_tuned"):
                log.info("[tune] %s 已是调优格（is_tuned），跳过", winner["id"])
                continue
            # ② 幂等：已有 {winner}-opt 且 done → 跳过
            opt_id = f"{win_spec['id']}-opt"
            existing = get_spec(specs, opt_id)
            if existing and existing.get("status") == "done":
                log.info("[tune] %s 已存在且 done，跳过（幂等）", opt_id)
                continue
            from tune_winner import tune_winner

            # v2.6.1（方案 A）：winner 格 data 快照不再落盘，这里按 winner 的样本方案
            # 运行时重切 dev/oot（seed=42 纯确定性，与 winner 完全同基线）后透传给调优。
            w_split = _resplit_for_optuna(win_spec, exp_root, dev_all, oot_all,
                                          dt_col, label_col, seed=42)
            if w_split is None:
                # 无法重切（如对抗格缺 data/features.json）→ 跳过调优并记录
                log.warning("[tune] %s 无法重切基线，跳过调优", win_spec["id"])
                continue
            w_train_df, w_val_df, w_oot_df = w_split
            tune_spec = tune_winner(
                win_spec, exp_root=exp_root, template_path=template_path,
                n_trials=args.n_trials, seed=42, resume=args.resume,
                force_tune=args.force_tune,
                train_df=w_train_df, val_df=w_val_df, oot_df=w_oot_df)
            if tune_spec is not None:
                specs = load_state(plan_json) or specs
                # 追加 -opt spec 到矩阵状态（well_fit 跳过的 skipped 记录同样纳入，便于展示）
                if not get_spec(specs, tune_spec["id"]):
                    specs.append(tune_spec)
                    save_state(plan_json, specs, _flush_reasons(specs))
    else:
        log.info("[tune] Optuna 已关闭：跳过 winner 规则诊断与调优（仅出 leaderboard）")

    if args.until == "tune":
        write_leaderboard(exp_root, specs)
        if not scope["optuna"]:
            log.info("[done] Optuna 已关闭，--until tune 等价执行至矩阵完成")
        else:
            log.info("[done] 调优完成（--until tune 停止）")
        return 0

    # 5) 汇总 top10 + 转正确认
    cands = build_candidates(exp_root, specs, k=10)
    if not cands:
        log.warning("[promote] 无 done 实验可转正")
        return 1
    res = promote_run(cands, exp_root, str(session_dir), auto=args.auto_apply,
                      promote_id=args.promote_id)
    write_leaderboard(exp_root, specs)
    if res is None:
        log.info("[promote] 取消转正")
        return 0
    log.info("[done] 转正完成: new-models/%s", res["run_name"])
    return 0


def _scheme_for(specs, spec, dev, dt_col):
    """按 spec.sample_scheme 构造样本方案（对抗格不走本函数）。"""
    sname = spec["sample_scheme"]
    if sname == "full":
        return ss.full_scheme(dev)
    if sname.startswith("recent"):
        n = int(sname.replace("recent", ""))
        return ss.recent_n_scheme(dev, dt_col, n)
    if sname == "timeweight":
        return ss.linear_time_weight_scheme(dev, dt_col)
    return ss.full_scheme(dev)


def _resplit_for_optuna(win_spec, exp_root, dev, oot, dt_col, label_col, seed=42):
    """v2.6.1（方案 A）：按 winner 格的样本方案运行时重切 dev/oot，返回 (train_df, val_df, oot_df)。

    - full / recentN / timeweight：`_scheme_for` 直接构造，纯确定性（seed=42）。
    - adversarial：对抗样本剔除依赖对抗分类器（seed=42 固定）。drop_mask 不落盘，但从
      `adversarial_meta.json` 可拿回用户确认的剔除比例（sample_drop_pct_actual），
      据此重训对抗分类器 + `compute_drop_masks` 精确重建，与 winner 完全同基线。
    - 返回的 DataFrame 统一将 label 列重命名为 `label`（tune_winner 硬编码列名）。
    """
    sname = win_spec.get("sample_scheme", "full")
    if sname == "adversarial":
        # 重建对抗样本方案（纯确定性重放）：对抗分类器用基础特征训练，
        # winner 格 data/features.json 保留的是对抗剔除后的最终特征
        import adversarial as adv

        feats_path = os.path.join(exp_root, win_spec["id"], "data", "features.json")
        try:
            with open(feats_path, "r", encoding="utf-8") as f:
                win_feats = json.load(f)
        except Exception:
            win_feats = []
        if not win_feats:
            print(f"[tune] 对抗格 winner {win_spec['id']} 缺 data/features.json，无法重切")
            return None
        base_feats = win_feats
        model, imp_adv, _ = adv.train_adversarial(dev, oot, base_feats, seed=seed)
        proba_dev = model.predict_proba(
            dev[base_feats].apply(pd.to_numeric, errors="coerce"))[:, 1]
        # 用户确认的剔除比例来自 adversarial_meta.json（无则用推荐值兜底）
        meta_path = os.path.join(exp_root, win_spec["id"], "adversarial_meta.json")
        drop_pct = 0.0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    drop_pct = json.load(f).get("sample_drop_pct_actual", 0.0)
            except Exception:
                pass
        if drop_pct <= 0:
            rec = adv.recommend_drop(proba_dev, 0.6)  # 无元数据时保守不剔除
            drop_pct = rec["recommended_sample_drop_pct"]
        top_k = max(3, int(len(base_feats) * 0.15))
        masks = adv.compute_drop_masks(proba_dev, drop_pct, imp_adv, top_k, base_feats)
        adv_scheme = ss.adversarial_filter_scheme(dev, masks["sample_drop_mask"],
                                                  meta={"reconstructed": True})
        filtered = ss.apply_sample_scheme(adv_scheme, dev).reset_index(drop=True)
    else:
        scheme = _scheme_for(None, win_spec, dev, dt_col)
        filtered = ss.apply_sample_scheme(scheme, dev).reset_index(drop=True)

    train_df, val_df = split_dev(filtered, label_col, seed=seed)
    if label_col != "label":
        train_df = train_df.rename(columns={label_col: "label"})
        val_df = val_df.rename(columns={label_col: "label"})
    oot_df = oot.rename(columns={label_col: "label"}) if label_col != "label" else oot
    return train_df, val_df, oot_df


if __name__ == "__main__":
    sys.exit(main())