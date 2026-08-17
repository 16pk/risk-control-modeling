# -*- coding: utf-8 -*-
"""model-training 编排器: 进程内调优训练 -> 多阶段产物落盘。

输出根目录由 --output_dir 决定(通常 `<session_dir>`),
本编排器在其下落 `new-models/{algo}-v{N}/`,
各阶段产物(config/features/model/evaluation/predictions/explainability/logs)
分别由对应 `write_*_stage` 模块负责。

数据来源 (v2.1 切分后置):
  - 新链路: <session_dir>/sample-features/data-cleaning/sample.parquet + train_config 的 model.split
    由本编排器在训练消费时即时切分为 train/test/oot 三档(写 run 内部 data/splits/ 临时目录,
    不作为 session 交付层产物)。
  - 旧 session 兼容: <session_dir>/sample-features/splits/{train,test,oot}.parquet 直接读取。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _resolve_version(cli_version: Optional[str], model_cfg: dict) -> Optional[str]:
    """决定本 run 的 version(形如 v1 / v2 / custom-tag)。

    优先级: CLI --label > yaml model.run_label > None(由 RunLayout.create 自动调 next_version 自增)。
    yaml.model.run_label 与 CLI --label 传"version 标识"语义; 留空则自动自增(v1→v2→...)。
    """
    if cli_version:
        return cli_version
    yaml_version = (model_cfg.get("run_label") or "").strip()
    if yaml_version:
        return yaml_version
    return None


def _clean_object_features_three(
    train_path: str, test_path: str, oot_path: str, features: list,
) -> Tuple[str, str, str]:
    """对三档 parquet 分别清洗 object 特征列 -> float64 落临时 parquet。

    薄包装到 data_clean.clean_object_features,逐档调用;三档独立清洗保证
    训练/测试/OOT 列类型一致(若某段无 object 列则原路径返回)。

    Returns:
        (cleaned_train_path, cleaned_test_path, cleaned_oot_path)
    """
    from data_clean import clean_object_features
    return (
        clean_object_features(train_path, features),
        clean_object_features(test_path, features),
        clean_object_features(oot_path, features),
    )


def _dispatch_train(algo: str, common: dict):
    """分发到 trainer_dispatch.dispatch_train。"""
    from trainer_dispatch import dispatch_train
    return dispatch_train(algo, common)


def _load_pre_split_data(
    splits_dir: Path, target: str, dt_col: str,
) -> Tuple[str, str, str, dict]:
    """读旧版三档 parquet + _manifest.json 摘要 split_report（兼容旧 session）。

    旧流程 feature-analysis 产出 <session_dir>/sample-features/splits/{train,test,oot}.parquet。
    v2.1 起切分后置到本 skill 消费时即时切分（见 _load_and_split_from_sample），
    但旧 session 产物仍可直接读取，本函数保留兼容。

    Args:
        splits_dir: <session_dir>/sample-features/splits 目录
        target: 标签列名(仅用于兜底 pos_rate 计算)
        dt_col: 时间列名(仅用于兜底)

    Returns:
        (train_path, test_path, oot_path, split_report_dict)

    Raises:
        SystemExit: 三档 parquet 任一缺失
    """
    train_path = splits_dir / "train.parquet"
    test_path = splits_dir / "test.parquet"
    oot_path = splits_dir / "oot.parquet"
    missing = [p for p in (train_path, test_path, oot_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"[run_build] 切分数据缺失: {missing}\n"
            "v2.1 起切分后置到训练消费时即时进行, 请确认上游 data-cleaning 已产出 sample.parquet "
            "且 train_config.yaml 含 model.split 区间。"
        )

    split_report: dict = {}
    import json
    # 尝试读取 feature-analysis 旧 manifest 提取 split_report（旧 session 兼容）
    manifest_path = splits_dir.parent / "feature-analysis" / "analysis" / "_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ov = manifest.get("overview") or {}
            sc = ov.get("sample_counts") or {}
            pr = ov.get("pos_rates") or {}
            split_report = {
                "strategy": ov.get("split_strategy", "explicit"),
                "oot_boundary": ov.get("oot_boundary", ""),
                "counts": {
                    "train": int(sc.get("train", 0)),
                    "val": int(sc.get("val", 0)),
                    "oot": int(sc.get("oot", 0)),
                },
                "pos_rates": {
                    "train": pr.get("train"),
                    "val": pr.get("val"),
                    "oot": pr.get("oot"),
                },
                "time_col_used": ov.get("time_col_used", dt_col),
                "warnings": tuple(),
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[run_build] warn: 解析 feature-analysis manifest 失败: {e}, split_report 留空")

    print(
        f"[run_build] 读旧版切分数据: train={split_report.get('counts', {}).get('train', '?')} "
        f"test={split_report.get('counts', {}).get('val', '?')} "
        f"oot={split_report.get('counts', {}).get('oot', '?')} <- {splits_dir}"
    )
    return str(train_path), str(test_path), str(oot_path), split_report


def _load_and_split_from_sample(
    sample_path: Path,
    split_cfg: dict,
    target: str,
    dt_col: str,
    out_splits_dir: Path,
) -> Tuple[str, str, str, dict]:
    """v2.1 即时切分: 从 data-cleaning 产出的 sample.parquet 按 model.split 区间切三档。

    切分发生在训练消费时(本 skill 内), 不依赖上游 feature-analysis 落盘 splits;
    切分结果写临时 parquet 到 run 内部 out_splits_dir(<run_dir>/data/splits/),
    供 trainer 按路径读取。切分唯一真相 = feature_config.yaml 的 model.split。

    Args:
        sample_path: data-cleaning 产出的 sample.parquet 全量样本
        split_cfg: model.split 配置(train_range/test_range/oot_range + dt_col)
        target: 标签列名
        dt_col: 时间列名
        out_splits_dir: run 内部临时 splits 目录

    Returns:
        (train_path, test_path, oot_path, split_report_dict)

    Raises:
        ValueError: split 区间缺失/不合法, 或 sample.parquet 不存在
    """
    if not sample_path.exists():
        raise ValueError(
            f"[run_build] sample.parquet 不存在: {sample_path}\n"
            "请先完成 data-cleaning 产出 sample.parquet, 再训练。"
        )
    if not split_cfg:
        raise ValueError(
            "[run_build] model.split 缺失: 需要 train_range/test_range/oot_range 三档区间。"
            "切分唯一真相 = feature_config.yaml 的 model.split。"
        )

    from date_utils import normalize_date

    def _parse_range(key: str) -> tuple:
        val = split_cfg.get(key)
        if not val or len(val) != 2:
            raise ValueError(f"[run_build] model.split.{key} 缺失或非 [起,止] 两元素: {val!r}")
        return normalize_date(val[0]), normalize_date(val[1])

    train_start, train_end = _parse_range("train_range")
    test_start, test_end = _parse_range("test_range")
    oot_start, oot_end = _parse_range("oot_range")

    print(f"[run_build] 即时切分: train [{train_start}, {train_end}] "
          f"test [{test_start}, {test_end}] oot [{oot_start}, {oot_end}] <- {sample_path}")
    df = pd.read_parquet(sample_path)
    if dt_col not in df.columns:
        raise ValueError(
            f"[run_build] 时间列 '{dt_col}' 不在 sample.parquet 中, "
            f"可用列: {list(df.columns)[:20]}"
        )
    # 时间列归一化为 8 位 YYYYMMDD(兼容 YYYY-MM-DD 与 YYYYMMDD)后按区间过滤
    norm = df[dt_col].astype(str).str.replace("-", "", regex=False)
    df["_dt_norm"] = norm

    def _in_range(s: str, e: str) -> pd.DataFrame:
        sub = df[(df["_dt_norm"] >= s) & (df["_dt_norm"] <= e)].copy()
        # 剔除 label 缺失/非法样本(训练/评估均不参与)
        if target in sub.columns:
            sub = sub[sub[target].isin([0, 1])].reset_index(drop=True)
        return sub

    train_df = _in_range(train_start, train_end)
    test_df = _in_range(test_start, test_end)
    oot_df = _in_range(oot_start, oot_end)
    df.drop(columns=["_dt_norm"], inplace=True)

    if len(train_df) == 0 or len(test_df) == 0 or len(oot_df) == 0:
        raise ValueError(
            f"[run_build] 切分后存在空档: "
            f"train={len(train_df)} test={len(test_df)} oot={len(oot_df)}。"
            "请检查 model.split 区间是否与数据时间范围匹配。"
        )

    out_splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_splits_dir / "train.parquet"
    test_path = out_splits_dir / "test.parquet"
    oot_path = out_splits_dir / "oot.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    oot_df.to_parquet(oot_path, index=False)

    pos_rates = {
        "train": float(train_df[target].mean()) if target in train_df.columns else None,
        "val": float(test_df[target].mean()) if target in test_df.columns else None,
        "oot": float(oot_df[target].mean()) if target in oot_df.columns else None,
    }
    split_report = {
        "strategy": "explicit",
        "oot_boundary": f"{dt_col} >= {oot_start}",
        "counts": {"train": len(train_df), "val": len(test_df), "oot": len(oot_df)},
        "pos_rates": pos_rates,
        "time_col_used": dt_col,
        "warnings": tuple(),
    }
    print(
        f"[run_build] 即时切分完成: train={len(train_df)} test={len(test_df)} "
        f"oot={len(oot_df)} -> {out_splits_dir}"
    )
    return str(train_path), str(test_path), str(oot_path), split_report


def run(
    cfg: dict,
    data_dir: str,
    output_dir: str,
    version: Optional[str] = None,
    source_yaml_path: Optional[str] = None,
) -> Dict[str, Any]:
    """端到端编排: 调优训练 + 评估/预测/可解释性/日志/配置快照 全部落盘。

    Args:
        cfg: load_config + validate 后的字典
        data_dir: 含 splits/{train,test,oot}.parquet 的目录(通常 <session_dir>/sample-features/)
        output_dir: session_dir(run_build 会在其下落 new-models/)
        version: 本次 run 的版本号(如 v1 / v2 / custom-tag);None 时自动自增
        source_yaml_path: 入参 yaml 路径;若提供则复制到 run_dir/config/train_config.yaml

    Returns:
        摘要 dict: run_dir / model_path / evaluation_report / predictions_parquet
    """
    from stages.layout import RunLayout, write_config_snapshot, write_train_config_yaml
    from stages.logs import tee_stdout, process_tee, finalize_logs_stage
    from stages.features import write_features_stage
    from stages.model import write_model_stage
    from stages.eval_data import assemble as assemble_eval_data
    from invoke.evaluation import invoke_evaluation_stage
    from stages.predictions import write_predictions_stage
    from stages.explainability import write_explainability_stage
    from stages.run_summary import write_run_summary_stage

    model = cfg["model"]
    features = list(model["features"])
    algo = str(model.get("algo", "xgb")).lower()
    id_cols = list(model.get("id_cols") or [])

    target = model.get("label_col", "label")
    if model.get("label_expr"):
        target = "label"

    # 数据决议 (v2.1 切分后置): 优先从 data-cleaning/sample.parquet + model.split 即时切分;
    # 兼容旧 session 直接读 sample-features/splits/{train,test,oot}.parquet。
    splits_dir = Path(data_dir) / "splits"
    sample_path = Path(data_dir) / "data-cleaning" / "sample.parquet"
    split_cfg = model.get("split") or {}
    dt_col = model.get("dt_col", "f_p_date")
    if sample_path.exists() and split_cfg:
        # 即时切分: 结果写 run 内部临时目录(由 run_dir 决议后补建)
        split_mode = "instant-split"
        _split_ctx = {"sample_path": sample_path, "split_cfg": split_cfg,
                      "target": target, "dt_col": dt_col}
    elif (splits_dir / "train.parquet").exists():
        train_path, test_path, oot_path, split_report_dict = _load_pre_split_data(
            splits_dir=splits_dir,
            target=target,
            dt_col=dt_col,
        )
        split_mode = "pre-split"
    else:
        raise SystemExit(
            f"[run_build] 数据源缺失:\n"
            f"  - 新链路(v2.1): {sample_path} + model.split {split_cfg or '(缺失)'}\n"
            "  - 旧链路: " + str(splits_dir / "train.parquet") + "\n"
            "请确认已完成 data-cleaning(产 sample.parquet)并配置 model.split, "
            "或旧 session 已有 splits/ 三档。"
        )

    # v2.1 精简: 全仓库已废除 spark 取数与分布式裁决(task-spec Gate P0 / feature-analysis 已删),
    # 计算路由恒为 local, 不做分布式信号探测。

    layout = RunLayout.create(output_dir=output_dir, algo=algo, suffix="", version=version)
    print(f"[run_build] run_dir = {layout.run_dir}")

    # v2.1 即时切分: 把 split_ctx 落实为三档 parquet(写 run 内部 data/splits/, 非 session 交付层)
    if split_mode == "instant-split":
        train_path, test_path, oot_path, split_report_dict = _load_and_split_from_sample(
            sample_path=_split_ctx["sample_path"],
            split_cfg=_split_ctx["split_cfg"],
            target=_split_ctx["target"],
            dt_col=_split_ctx["dt_col"],
            out_splits_dir=layout.run_dir / "data" / "splits",
        )

    # 进程级日志: 从 run_dir 创建到完成回执的全部 stdout/stderr 落 logs/run_build.log
    # (tee_stdout 只覆盖训练核心阶段, process_tee 额外捕获 run_dir print / import 警告 / 完成回执等)
    process_log = layout.logs_dir / "run_build.log"
    with process_tee(process_log):
        with tee_stdout(layout):
            # 边界特征过滤: 剔除常量/泄漏/ID-like/全缺失 4 类会让训练失败或泄漏的特征
            # v2.1: instant-split 走数据直算(从 train 段直接算 stats/IV, 不依赖 feature-analysis csv);
            #       pre-split 兼容旧 session 读 feature-analysis 落的 csv(缺失则跳过规则)。
            bf_cfg = model.get("boundary_filter") or {}
            n_before_filter = len(features)
            if split_mode == "instant-split":
                from boundary_filter import filter_boundary_features_from_df
                bf_result = filter_boundary_features_from_df(
                    features,
                    pd.read_parquet(train_path),
                    target=target,
                    enable_constant=bf_cfg.get("enable_constant", True),
                    enable_leakage=bf_cfg.get("enable_leakage", True),
                    enable_id_like=bf_cfg.get("enable_id_like", True),
                    enable_all_missing=bf_cfg.get("enable_all_missing", True),
                    iv_max=bf_cfg.get("iv_max", 1.0),
                    const_unique_max=bf_cfg.get("const_unique_max", 1),
                    id_like_ratio=bf_cfg.get("id_like_ratio", 0.9),
                    missing_max=bf_cfg.get("missing_max", 1.0),
                )
            else:
                from boundary_filter import filter_boundary_features
                analysis_manifest_path = Path(data_dir) / "feature-analysis" / "analysis" / "_manifest.json"
                sample_total = 0
                if analysis_manifest_path.exists():
                    import json as _json
                    try:
                        _ov = (_json.loads(analysis_manifest_path.read_text(encoding="utf-8")).get("overview") or {})
                        sample_total = int(_ov.get("n_total") or 0)
                    except (ValueError, KeyError) as _e:
                        print(f"[run_build] warn: 解析 feature-analysis manifest n_total 失败: {_e}")
                bf_result = filter_boundary_features(
                    features,
                    analysis_dir=str(Path(data_dir) / "feature-analysis" / "analysis"),
                    sample_total=sample_total,
                    enable_constant=bf_cfg.get("enable_constant", True),
                    enable_leakage=bf_cfg.get("enable_leakage", True),
                    enable_id_like=bf_cfg.get("enable_id_like", True),
                    enable_all_missing=bf_cfg.get("enable_all_missing", True),
                    iv_max=bf_cfg.get("iv_max", 1.0),
                    const_unique_max=bf_cfg.get("const_unique_max", 1),
                    id_like_ratio=bf_cfg.get("id_like_ratio", 0.9),
                    missing_max=bf_cfg.get("missing_max", 1.0),
                )
            print(
                f"[run_build] 边界过滤: {len(bf_result.dropped_features)} 个特征被剔除, "
                f"保留 {len(bf_result.kept_features)}/{n_before_filter}"
            )
            if bf_result.dropped_features:
                for _rule, _feats in bf_result.dropped_by_rule.items():
                    if _feats:
                        print(f"[run_build]   {_rule}: {len(_feats)} 个 -> {_feats}")
            features = bf_result.kept_features

            cleaned_train, cleaned_test, cleaned_oot = _clean_object_features_three(
                train_path, test_path, oot_path, features,
            )
            common = dict(
                train_path=cleaned_train, test_path=cleaned_test, oot_path=cleaned_oot,
                target=target, features=features,
            )

            predictor, metrics, used_params, train_info = _dispatch_train(algo, common)

            # config.json 快照(放训练后,以便记录训练细节 / 实际 features 数)
            # metrics 是 Dict[str, BinMetrics] (dataclass), 这里转 dict 以保证 JSON 可序列化
            # train_info 字段随 algo 不同(xgb: best_iteration; dnn: best_epoch/total/early_stopped;
            # lr: n_iter/converged), 整体 merge 进 runtime 透传
            metrics_payload = {k: v.to_dict() for k, v in metrics.items()} if metrics else {}
            runtime_extra = {
                "version": layout.version,
                "suffix": layout.suffix,
                "n_features": len(features),
                "metrics": metrics_payload,
                "split_mode": split_mode,
                "split_report": split_report_dict,
                "boundary_filter": {
                    "n_before": bf_result.n_before,
                    "n_after": len(bf_result.kept_features),
                    "n_dropped": len(bf_result.dropped_features),
                    "dropped_by_rule": {k: len(v) for k, v in bf_result.dropped_by_rule.items()},
                    "thresholds": bf_result.thresholds,
                    "rules_enabled": bf_result.rules_enabled,
                    "sample_total": bf_result.sample_total,
                },
            }
            runtime_extra.update(train_info or {})
            write_config_snapshot(
                layout, cfg, data_dir,
                extra=runtime_extra,
            )
            # 入参 yaml 副本落到 run_dir/config/train_config.yaml
            write_train_config_yaml(layout, source_yaml_path)

            write_features_stage(
                layout, features=features,
                upstream_source=cfg.get("model", {}).get("feature_list_source"),
                dropped=[],
                dropped_by_rule=bf_result.dropped_by_rule if bf_result.dropped_features else None,
            )

            model_path = write_model_stage(
                layout, predictor=predictor,
                used_params=used_params, train_info=train_info,
            )

            eval_data = assemble_eval_data(
                predictor=predictor,
                train_path=cleaned_train, test_path=cleaned_test, oot_path=cleaned_oot,
                target=target, features=features,
                metrics=metrics,
            )

            predictions_parquet = write_predictions_stage(
                layout, data=eval_data, id_cols=id_cols,
            )
            invoke_evaluation_stage(
                layout, model_cfg=model, used_params=used_params,
            )

            # baseline_eval_dir 决议: 仅读 yaml 显式配置(不再默认扫描历史模型推荐目录);
            # 未配置或为空则跳过 comparison 阶段
            baseline_eval_dir = model.get("baseline_eval_dir") or None
            if baseline_eval_dir:
                from invoke.comparison import invoke_comparison_stage
                invoke_comparison_stage(layout, baseline_eval_dir=baseline_eval_dir)
            else:
                print("[run_build] 未配置 model.baseline_eval_dir, 跳过 comparison 阶段")

            # 会话级横向对比聚合: 扫 new-models/ 下所有 eval JSON,
            # 在 model-comparison/ 下产 N-way 汇总三件套 (跨 run 对比, 与单 run 的 comparison/ 互补)
            from invoke.session_aggregate import invoke_session_aggregate
            invoke_session_aggregate(output_dir)

            write_explainability_stage(layout, predictor=predictor, data=eval_data)

            finalize_logs_stage(layout, extra_logs=[process_log])
            write_run_summary_stage(layout)

        print(f"[run_build] 完成: run_dir={layout.run_dir}")

    return {
        "run_dir": str(layout.run_dir),
        "model_path": str(model_path),
        "evaluation_report": str(layout.evaluation_dir / f"{layout.run_name}_oot_eval.md"),
        "predictions_parquet": str(predictions_parquet),
        "version": layout.version,
        "label": layout.version,
    }


def _resolve_data_dir(cli_data_dir: Optional[str], output_dir: str) -> str:
    """决定训练数据目录(需含 data-cleaning/sample.parquet 或 splits/{train,test,oot}.parquet)。

    v2.1 优先级: CLI --data_dir > 同 session 下 sample-features/ > 报错。
    sample-features/ 下数据源二选一:
      - 新链路: data-cleaning/sample.parquet (+ train_config 的 model.split 即时切分)
      - 旧链路: splits/{train,test,oot}.parquet (旧 session 兼容)
    """
    if cli_data_dir:
        return cli_data_dir

    inferred = str(Path(output_dir) / "sample-features")
    if (Path(inferred) / "data-cleaning" / "sample.parquet").exists() or \
       (Path(inferred) / "splits" / "train.parquet").exists():
        print(f"[run_build] 未传 --data_dir, 自动推断 = {inferred}")
        return inferred

    raise SystemExit(
        f"[run_build] 未传 --data_dir 且默认路径不存在: {inferred}/data-cleaning/sample.parquet\n"
        "请先跑 data-cleaning 产出 sample.parquet 并配置 model.split, "
        "或显式传 --data_dir 指向含数据子目录的目录。"
    )


def main() -> None:
    """命令行入口。"""
    from validate_config import load_config, validate_config

    parser = argparse.ArgumentParser(description="model-training 编排")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default=None,
                        help="数据目录(通常 <session_dir>/sample-features/); 需含 data-cleaning/sample.parquet "
                             "(v2.1 即时切分)或 splits/{train,test,oot}.parquet(旧 session 兼容); "
                             "留空则从 <output_dir>/sample-features/ 推断")
    parser.add_argument("--output_dir", required=True,
                        help="session_dir (本编排器在其下落 new-models/)")
    parser.add_argument("--label", default=None,
                        help="本次 run 的版本标识(别名, 形如 v1 / v2 / custom-tag);"
                             "留空则按优先级回退到 yaml.model.run_label, 再空则自动自增 v1/v2/...")
    parser.add_argument("--version", default=None,
                        help="同 --label, 显式版本号; 二者都传时 --version 优先")
    args = parser.parse_args()

    _warn_if_config_in_output_dir(args.config, args.output_dir)

    cfg = load_config(args.config)
    validate_config(cfg)
    # CLI --version/--label 校验 (yaml.model.run_label 已在 validate_config 内校验);
    # 拦截 `xgb-v1` / `tuned-v1` / `feat` 这类会导致 xgb-xgb-v1 / xgb-tuned-tuned-v1
    # 重复前缀的输入, 在 run_dir 创建之前。
    from stages.layout import validate_version_label
    validate_version_label(args.version or args.label)
    version = _resolve_version(args.version or args.label, cfg.get("model") or {})
    data_dir = _resolve_data_dir(args.data_dir, args.output_dir)

    res = run(cfg, data_dir, args.output_dir, version, source_yaml_path=args.config)
    print(f"[run_build] 完成: {res}")


def _warn_if_config_in_output_dir(config_path: str, output_dir: str) -> None:
    """检查 --config 路径是否符合 SKILL.md §6 输入 yaml 落盘约束。

    规范: 输入 yaml 应放 <session_dir>/new-models/{algo}-v{N}/config/train_config.yaml
    (model 内部 config 目录)。本函数在 run_dir 创建前做近似检查 — 因为此时还不知道
    {algo}-v{N}, 只能判断 --config 是否落在 --output_dir 之下:
      - 落在 output_dir 之下: 视为合规(model 内部目录), 不提醒
      - 落在 output_dir 之外: 疑似放错位置(如 <skill_dir>/config/ 或 session 根目录), 打 warning

    只提醒不阻塞, 兜底历史 run 重跑。
    """
    try:
        cfg_abs = Path(config_path).resolve()
        out_abs = Path(output_dir).resolve()
    except OSError:
        return
    try:
        cfg_abs.relative_to(out_abs)
    except ValueError:
        # config 不在 output_dir 之下, 疑似放错位置(如 <skill_dir>/config/ 或 session 根目录)
        print(
            f"[run_build] warn: --config {config_path} 不在 --output_dir {output_dir} 之下。"
            "输入 yaml 应放 <session_dir>/new-models/{algo}-v{N}/config/train_config.yaml "
            "(model 内部 config 目录), 不放 <skill_dir>/config/ 或 session 根目录。"
            "详见 SKILL.md §6 输入 yaml 落盘约束。"
        )

if __name__ == "__main__":
    main()
