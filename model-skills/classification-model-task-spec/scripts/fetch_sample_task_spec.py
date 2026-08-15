# -*- coding: utf-8 -*-
"""classification-model-task-spec 样本拉取脚本。

职责: 需求确认后, 从本地样本文件(parquet/csv/feather)提取 fuid/label/f_p_date
+ 补充字段(不拉特征列), 落成 sample.parquet 供 run_sample_analysis_task_spec.py 分析。

切分边界: 本脚本仅「记录/透传」Train/Test/OOT 三档区间到 sample_config.*.yaml 的
model.split(供用户确认需求规格), 不再驱动本地切分。切分与切分统计已后置到
feature-analysis, 切分区间单一真相 = feature_config.yaml 的 model.split。

本脚本独立完成 task-spec 阶段样本拉取, 与下游 data-cleaning 的清洗职责分离:
  - 不做特征清单加载 / feature-list.csv 派生(task-spec 阶段不需要特征)
  - 不支持 feature_list_source / 全列模式
  - features 字段在 task-spec 语义里 = "补充字段清单"(供后续报告补充分析)
  - 全仓库已废除 spark 取数, 本脚本仅支持 local_file 模式

CLI 模式: --session-dir + 关键参数直传, 脚本内部自动生成 yaml 落到
<session_dir>/task-spec/sample_config.<model_name>.yaml, 不再接收 --config,
机制上强制配置文件落 session 目录, 避免误落 skill config/ 目录。

公共模块: model-evo/_modelevo-shared/scripts, 通过 _bootstrap.py 注入。
安全: 仅取所需列, 不输出用户级明细到日志。
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from typing import List

import yaml


def _parse_range(text: str) -> List[str]:
    """解析 '起,止' 为 [起, 止] 列表 (YYYY-MM-DD / YYYYMMDD 双兼容, 归一化为 8 位)。"""
    import date_utils

    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("区间须为两元素 起,止, 如 2026-03-12,2026-04-30(或 YYYYMMDD), 当前 %r" % text)
    norm = [date_utils.parse_date(d, what="range") for d in parts]
    if norm[0] > norm[1]:
        raise ValueError("区间起始 %s 不应大于结束 %s" % (norm[0], norm[1]))
    return norm


def _build_cfg(args: argparse.Namespace) -> dict:
    """从 CLI 参数构造配置 dict (仅 local_file 模式)。

    sample_table/fetch_dt 仅作占位/记录(本地样本无取数窗口概念)。
    """
    features: List[str] = []
    if args.features:
        features = [c.strip() for c in args.features.split(",") if c.strip()]

    id_cols: List[str] = [c.strip() for c in (args.id_cols or "fuid").split(",") if c.strip()]

    cfg = {
        "model": {
            "name": args.model_name,
            "version": args.version,
            "mode": "local_file",
            "sample_table": args.sample_table or "local_file",
            "local_parquet_path": args.local_parquet_path,
            "dt_col": args.dt_col,
            "id_cols": id_cols,
            "fetch_dt": [args.fetch_start or "", args.fetch_end or ""],
            "where": args.where,
            # split 三档区间仅「记录/透传」供用户确认需求规格, 不再驱动本地切分;
            # 切分与切分统计已后置到 feature-analysis, 切分真相单一 = feature_config.yaml 的 model.split。
            "split": {
                "train_range": _parse_range(args.train_range),
                "test_range": _parse_range(args.test_range),
                "oot_range": _parse_range(args.oot_range),
            },
            "features": features,
            "feature_list_source": None,
            "label_col": args.label_col,
            "label_expr": args.label_expr,
        },
    }
    return cfg


def _dump_yaml(cfg: dict, session_dir: str, model_name: str) -> str:
    """把 cfg 落成 yaml 到 <session_dir>/task-spec/sample_config.<model_name>.yaml。"""
    out_dir = os.path.join(session_dir, "task-spec")
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "sample_config.%s.yaml" % model_name)
    # 去掉 _config_dir 等运行时字段
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# task-spec 样本拉取配置 (由 fetch_sample_task_spec.py 自动落盘)\n")
        f.write("# model: %s\n\n" % model_name)
        yaml.safe_dump(clean, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return yaml_path


def _link_or_copy_local(src: str, dst: str) -> None:
    """把本地样本文件落到 dst: 优先硬链接(同盘零拷贝), 失败则软链接, 再失败才 copyfile。

    local_file 模式下 sample.parquet 由后续脚本只读消费(run_sample_analysis_task_spec /
    feature-analysis / model-training 都不改源文件), 无需整文件复制。
    .csv 输入仍走 pandas 读后写 parquet(格式转换必须复制)。

    Args:
        src: 源文件路径(.parquet 或 .csv)
        dst: 目标 parquet 路径
    """
    src_lower = src.lower()
    if src_lower.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(src)
        df.to_parquet(dst, index=False)
        return
    if src_lower.endswith(".feather"):
        # feather 非 parquet, 必须读后转写(不能链接/复制)
        import pandas as pd
        df = pd.read_feather(src)
        df.to_parquet(dst, index=False)
        return
    # .parquet: 优先硬链接(同盘 inode 共享, 零拷贝), 跨盘则软链接, 都不行才 copyfile
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(os.path.abspath(src), dst)
        return
    except OSError:
        pass
    shutil.copyfile(src, dst)


def main() -> None:
    """命令行入口: 读参数 → 构造 cfg → 校验 → 落 yaml → 本地样本转写落 sample.parquet。"""
    import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts
    from config_io import validate_split_ranges, check_sensitive

    parser = argparse.ArgumentParser(description="task-spec 样本拉取(--session-dir 自动落盘模式, 仅 local_file)")
    parser.add_argument("--session-dir", required=True, help="session 目录 (runs/<timestamp>-<model_name>)")
    parser.add_argument("--model-name", required=True, help="模型简称, 如 draw_willingness-t7")
    parser.add_argument("--mode", choices=["local_file"], default="local_file",
                        help="取数模式: 全仓库已废除 spark, 仅支持 local_file(本地文件转写)")
    parser.add_argument("--local-parquet-path", required=True,
                        help="[必填] 本地样本文件路径(.parquet/.csv/.feather), 含 id_cols+label_col+dt_col+(可选)features")
    parser.add_argument("--sample-table", default=None,
                        help="样本表 库.表, 如 your_db.xxx (仅记录用, 本地文件模式不取数)")
    parser.add_argument("--fetch-start", default=None, help="取数起始日期 YYYY-MM-DD(兼容 YYYYMMDD) (仅记录用)")
    parser.add_argument("--fetch-end", default=None, help="取数结束日期 YYYY-MM-DD(兼容 YYYYMMDD) (仅记录用)")
    parser.add_argument("--train-range", required=True, help="Train 日期闭区间 起,止 (YYYY-MM-DD, 兼容 YYYYMMDD)")
    parser.add_argument("--test-range", required=True, help="Test 日期闭区间 起,止 (YYYY-MM-DD, 兼容 YYYYMMDD)")
    parser.add_argument("--oot-range", required=True, help="OOT 日期闭区间 起,止 (YYYY-MM-DD, 兼容 YYYYMMDD)")
    parser.add_argument("--label-col", default="label", help="标签列名 (默认 label)")
    parser.add_argument("--label-expr", default=None, help="标签表达式, 非空时替代 --label-col (仅记录, 本地文件需已含标签列)")
    parser.add_argument("--features", default=None, help="补充字段清单(逗号分隔), 不直接入模; 留空=仅样本三列")
    parser.add_argument("--id-cols", default="fuid", help="ID 列(逗号分隔), 默认 fuid")
    parser.add_argument("--dt-col", default="f_p_date", help="日期分区字段, 默认 f_p_date")
    parser.add_argument("--where", default=None, help="可选客群筛选条件 (仅记录)")
    parser.add_argument("--version", default="v1", help="模型版本, 默认 v1")
    parser.add_argument("--out", default=None, help="输出 parquet 路径, 默认 <session_dir>/data-profile/<model_name>_sample_<YYYYMMDD>.parquet")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        raise SystemExit("session 目录不存在: %s" % session_dir)

    # local_file 模式必填校验
    if not os.path.isfile(args.local_parquet_path):
        raise SystemExit("本地样本文件不存在: %s" % args.local_parquet_path)
    _ext = os.path.splitext(args.local_parquet_path)[1].lower()
    if _ext not in (".parquet", ".csv", ".feather"):
        raise SystemExit(
            "本地样本文件扩展名不支持: %s, 仅支持 .parquet / .csv / .feather" % _ext
        )

    cfg = _build_cfg(args)
    model = cfg["model"]

    # 校验: 安全线 + split 区间合法性(仅校验「记录」区间的时序/格式合法性,
    # 不驱动切分; 切分真相在 feature-analysis 的 feature_config.yaml)
    check_sensitive(model.get("where") or "")
    check_sensitive(model.get("sample_table") or "")
    validate_split_ranges(model)

    # 落 yaml 到 session 目录 (机制上强制, 不可能落 skill config/)
    yaml_path = _dump_yaml(cfg, session_dir, model["name"])
    print("[fetch_sample_task_spec] 配置已落盘: %s" % yaml_path)


    # 输出 parquet 路径: 默认 <session_dir>/data-profile/<model_name>_sample_<YYYYMMDD>.parquet
    today = datetime.now().strftime("%Y%m%d")
    out_path = args.out or os.path.join(
        session_dir, "data-profile", "%s_sample_%s.parquet" % (model["name"], today)
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 用硬链接(同盘)或软链接(跨盘)替代 shutil.copyfile, 避免把同一份本地 parquet
    # 整文件复制一份到 data-profile/(后续 run_sample_analysis_task_spec.py 只读不改, 无需副本)。
    # 源是 .csv / .feather 时仍 copyfile 转 parquet。
    _link_or_copy_local(args.local_parquet_path, out_path)
    print("[fetch_sample_task_spec] mode=local_file, 已链接本地 parquet:")
    print("  源: %s" % args.local_parquet_path)
    print("  目: %s" % out_path)
    extra_cols = model.get("features") or []
    if extra_cols:
        print("补充字段(不直接入模): %s" % ", ".join(extra_cols))
    else:
        print("仅样本三列(%s/%s/%s), 无补充字段" % (
            "/".join(model.get("id_cols") or []),
            model.get("label_col"),
            model.get("dt_col"),
        ))


if __name__ == "__main__":
    main()
