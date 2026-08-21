# -*- coding: utf-8 -*-
"""report.md 自动回填器。

按 classification-model-development/SKILL.md §6 契约,从 session_dir 下各 sub-skill
产出的 manifest/JSON/CSV 中提取信息,幂等更新 report.md 的三~六段。

段落归属(v2.1 收敛为 4 节, 全部由 `classification-model-development` 触发回填):
- 二(样本与特征) — Stage 2 credit-data-analysis 完成后
- 三(模型迭代) — 每个 run 完成后
- 四(结论与交付) — Stage 5/6 收口后

幂等: 同一段落多次调用结果一致 — 用 `## 三、` 等 H2 锚点切分,替换段落内容,
保留首部 (一~二段) 与其他段落不变。

用法:
    python fill_report.py --session-dir <path> --section {IV|V|VI|VII|all}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SECTION_ANCHORS = {
    "IV": ("## 三、特征宽表", "## III. 特征宽表"),
    "V": ("## 四、特征分析", "## IV. 特征分析"),
    "VI": ("## 五、模型迭代", "## V. 模型迭代"),
    "VII": ("## 六、横向对比", "## VI. 横向对比"),
}
# 元组 (中文锚点, 英文锚点): 中文为写出形式, 英文为读取形式(识别已有 report.md)。
# report.md 章节编号(汉字 一~六)与本脚本 section key 对应(历史模型推荐一节已移除,
# 原四~七 顺延为 三~六):
#   - 一/二 (需求 / 样本) 由 orchestration 自填
#   - 三~六 (特征宽表 / 特征分析 / 模型迭代 / 横向对比) 由本脚本接管
#   - 附录「待处理项与下一步建议」由 orchestration 自填(不带编号)
# 如改 report.md 章节顺序, 务必同步本表锚点。


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[fill_report] WARN: {path} 解析失败: {e}", file=sys.stderr)
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(x)


def _fmt_num(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_auc(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


def _latest_run_config(session_dir: Path) -> Optional[dict]:
    """v2.3: 最新 run 的 config.json(按目录名倒序)。无则 None。"""
    new_models = session_dir / "new-models"
    if not new_models.is_dir():
        return None
    for run_dir in sorted(new_models.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        cfg = _read_json(run_dir / "config.json")
        if cfg is not None:
            return cfg
    return None


def _is_experiments_run(cfg: dict) -> bool:
    """v2.3: 识别 experiments 矩阵转正 run(config.json.produced_by == skills/model-experiments)。"""
    return cfg.get("produced_by") == "skills/model-experiments"


def _experiments_source_dir(session_dir: Path, cfg: dict) -> Optional[Path]:
    """定位 experiments 转正 run 的矩阵源格目录: experiments/{source_exp}/。

    experiments 型 run 的 config.json.source_exp 记录 winner/opt 格 id(如 lgb-full-all-v1-opt)。
    返回该格目录(含 data/、feature_importance.csv、evaluation/);缺失返回 None。
    """
    if not _is_experiments_run(cfg):
        return None
    source_exp = cfg.get("source_exp") or ""
    if not source_exp:
        return None
    exp_dir = session_dir / "experiments" / source_exp
    return exp_dir if exp_dir.is_dir() else None


def _experiments_split_manifest(session_dir: Path, cfg: dict) -> Optional[dict]:
    """experiments 型 run 的切分信息: 从矩阵源格 data/{train,val,oot}.parquet 重建。

    experiments 无独立 test 档(开发池=train+test 合并后随机 70/30 切 train/val),
    以 val 档作为 test 档展示并注明语义, 避免 §IV 回填空档。
    """
    exp_dir = _experiments_source_dir(session_dir, cfg)
    if exp_dir is None:
        return None
    data_dir = exp_dir / "data"
    try:
        import pandas as pd
    except ImportError:
        return None
    split_names = ("train", "val", "oot")
    splits: Dict[str, dict] = {}
    total = 0
    for name in split_names:
        p = data_dir / f"{name}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        label_col = "label" if "label" in df.columns else None
        pos = int(df[label_col].sum()) if label_col else 0
        rows_n = int(len(df))
        time_col = None
        for c in ("f_p_date", "dt", "pday"):
            if c in df.columns:
                time_col = c
                break
        prange = None
        if time_col:
            vals = df[time_col].dropna().astype(str)
            if len(vals):
                prange = [str(vals.min()), str(vals.max())]
        splits[name] = {
            "rows": rows_n, "pos": pos,
            "pos_rate": (pos / rows_n) if rows_n else 0.0,
            "pday_range": prange,
        }
        total += rows_n
    ranges = {name: {"start": splits[name]["pday_range"][0] if splits[name]["pday_range"] else None,
                     "end": splits[name]["pday_range"][1] if splits[name]["pday_range"] else None}
              for name in split_names}
    actual_ratios = {name: (splits[name]["rows"] / total) if total else 0.0 for name in split_names}
    return {
        "strategy": "experiments 矩阵(开发池=train+test 合并, seed=42 随机 70/30 切 train/val; test=实验台 val)",
        "splits": splits, "ranges": ranges, "actual_ratios": actual_ratios,
        "dropped_rows": 0, "time_col": "f_p_date", "label_col": "label",
    }


def _build_split_manifest_from_parquets(splits_dir: Path) -> Optional[dict]:
    """从 sample-features/splits/{train,test,oot}.parquet 重建 split manifest 供 section IV 回填。

    读三档 parquet 的行数 + 正样本数 + dt 范围, 拼成与 task-spec
    同构的 dict (splits/ranges/actual_ratios/dropped_rows/strategy)。
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    split_names = ("train", "test", "oot")
    splits: Dict[str, dict] = {}
    total = 0
    for name in split_names:
        p = splits_dir / f"{name}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        # 标签列: 优先常见名, 兜底取唯一的 0/1 整数列
        label_col = None
        for c in ("TARGET", "label", "y"):
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            num_cols = [c for c in df.columns if df[c].dtype.kind in "iu"]
            for c in num_cols:
                if set(df[c].dropna().unique()).issubset({0, 1}):
                    label_col = c
                    break
        pos = int(df[label_col].sum()) if label_col else 0
        rows_n = int(len(df))
        # 时间列: 优先 dt / pday
        time_col = None
        for c in ("dt", "pday"):
            if c in df.columns:
                time_col = c
                break
        prange = None
        if time_col:
            prange = [str(int(df[time_col].min())), str(int(df[time_col].max()))]
        splits[name] = {
            "rows": rows_n,
            "pos": pos,
            "pos_rate": (pos / rows_n) if rows_n else 0.0,
            "pday_range": prange,
        }
        total += rows_n

    ranges = {
        name: {"start": splits[name]["pday_range"][0] if splits[name]["pday_range"] else None,
               "end": splits[name]["pday_range"][1] if splits[name]["pday_range"] else None}
        for name in split_names
    }
    actual_ratios = {
        name: (splits[name]["rows"] / total) if total else 0.0
        for name in split_names
    }
    return {
        "strategy": "time_explicit (local_file, rebuilt from splits parquet)",
        "splits": splits,
        "ranges": ranges,
        "actual_ratios": actual_ratios,
        "dropped_rows": 0,
    }


# ===== Section IV: 特征宽表 =====

def build_section_iv(session_dir: Path) -> str:
    """特征宽表段 — 来源: feature-list.csv + 三档切分。

    v2.3 主链路: experiments 转正 run 从 experiments/{source_exp}/ 矩阵源格的
    data/{train,val,oot}.parquet 重建切分信息(test 档用实验台 val 表示, 注明语义)。
    """
    dc_dir = session_dir / "sample-features" / "data-cleaning"
    manifest = None
    source_tag = "—"

    # v2.3: 若最新 run 是 experiments 转正 → 从其矩阵源格 data/ 重建切分信息
    latest_cfg = _latest_run_config(session_dir)
    if latest_cfg is not None and _is_experiments_run(latest_cfg):
        exp_manifest = _experiments_split_manifest(session_dir, latest_cfg)
        if exp_manifest is not None:
            manifest = exp_manifest
            source_tag = "experiments 矩阵源格 data/(test=实验台 val)"

    # 旧 session 兼容: sample-features/splits/
    if not manifest:
        splits_dir = session_dir / "sample-features" / "splits"
        if splits_dir.exists() and (splits_dir / "train.parquet").exists():
            manifest = _build_split_manifest_from_parquets(splits_dir)
            source_tag = "feature-analysis(splits, 旧 session)"

    if not manifest:
        return (
            f"{SECTION_ANCHORS['IV'][0]}\n\n"
            "（特征宽表/切分尚未执行: 无 experiments 矩阵源格 data/）\n"
        )

    splits = manifest.get("splits", {})
    ranges = manifest.get("ranges", {})
    time_col = manifest.get("time_col", "f_p_date")
    label_col = manifest.get("label_col", "label")

    feature_list_csv = dc_dir / "feature-list.csv"
    n_features_listed = 0
    if feature_list_csv.exists():
        with feature_list_csv.open(encoding="utf-8") as f:
            n_features_listed = sum(1 for _ in f) - 1

    rows = [
        f"{SECTION_ANCHORS['IV'][0]}\n",
        f"- 特征清单文件: `feature-list.csv` (共 {n_features_listed} 列)",
        f"- 时间列: `{time_col}` / 标签列: `{label_col}`",
        f"- 切分策略: `{manifest.get('strategy', '—')}`  (数据来源: `{source_tag}`)",
        "",
        "| 集合 | 样本量 | 正样本 | 正样本率 | {col} 范围 |".format(col=time_col),
        "|------|--------|--------|----------|-----------|",
    ]
    # experiments 型 manifest 的 splits 键为 train/val/oot(无独立 test 档),
    # 展示时统一映射到 train/test/oot(test = 实验台 val)
    split_key_map = {"train": "train", "test": "test", "oot": "oot"}
    if manifest.get("strategy", "").startswith("experiments"):
        split_key_map = {"train": "train", "test": "val", "oot": "oot"}
    for split_name in ("train", "test", "oot"):
        key = split_key_map[split_name]
        s = splits.get(key, {})
        r = ranges.get(key, {})
        # 兼容两种区间字段名: data-cleaning/data-profile 用 'start'/'end',
        # data-profile split 内也可能用 'pday_range' (列表)
        start = r.get("start") if isinstance(r, dict) else None
        end = r.get("end") if isinstance(r, dict) else None
        if not start or not end:
            pr = s.get("pday_range") if isinstance(s, dict) else None
            if isinstance(pr, list) and len(pr) == 2:
                start, end = pr[0], pr[1]
        rows.append(
            f"| {split_name} | {_fmt_num(s.get('rows'))} | "
            f"{_fmt_num(s.get('pos', s.get('positive')))} | "
            f"{_fmt_pct(s.get('pos_rate', s.get('positive_rate')))} | "
            f"{start or '—'} ~ {end or '—'} |"
        )

    actual_ratios = manifest.get("actual_ratios", {})
    if actual_ratios:
        ratio_str = " / ".join(
            f"{k}={_fmt_pct(v)}" for k, v in actual_ratios.items()
        )
        rows.append("")
        rows.append(f"> 实际切分比例: {ratio_str}")
        rows.append(f"> dropped_rows: {manifest.get('dropped_rows', 0)}")

    rows.append("")
    return "\n".join(rows)


# ===== Section V: 特征分析 =====

def build_section_v(session_dir: Path) -> str:
    """特征分析段 — 来源: credit-data-analysis 产物(sample-features/credit-data-analysis/) + 新模型 run 内部 IV 表。

    v2.1: 原 feature-analysis 已并入 credit-data-analysis(pipeline 特征分析),
    产物为 特征分析结果.md / 特征分析结果.xlsx / _manifest.json(分月视角);
    若不存在则回退读取最新 run 的 explainability/feature-importance.csv 兜底展示。
    """
    cda_dir = session_dir / "sample-features" / "credit-data-analysis"
    manifest = _read_json(cda_dir / "_manifest.json")

    # 主路径: credit-data-analysis 已执行 → 引用其 md 报告 + 最新 run 特征重要性
    if manifest:
        md_report = cda_dir / "特征分析结果.md"
        params = manifest.get("params", {})
        lines = [
            f"{SECTION_ANCHORS['V'][0]}\n",
            f"- 特征分析(credit-data-analysis)已执行, 报告见 `sample-features/credit-data-analysis/特征分析结果.md`",
            f"- 数据文件: `{params.get('data_file', '—')}`",
            f"- PSI 基准月: `{params.get('base_month', '—')}`"
            f"{' ｜ split_config: `' + str(params.get('split_config')) + '`' if params.get('split_config') else ''}",
            "",
        ]
        fi_path = _latest_run_feature_importance(session_dir)
        lines.append("### 最新 run 特征重要性 (Top 20)")
        lines.append("")
        lines.extend(_feature_importance_lines(fi_path))
        lines.append(f"> 分月 PSI/IV/无效值明细见 `sample-features/credit-data-analysis/特征分析结果.md`")
        lines.append("")
        return "\n".join(lines)

    # 兜底: credit-data-analysis 未执行, 从最新 run 读取特征重要性
    fi_path = _latest_run_feature_importance(session_dir)
    if fi_path is None:
        return (
            f"{SECTION_ANCHORS['V'][0]}\n\n"
            "（特征分析尚未执行: credit-data-analysis 产物缺失）\n"
        )
    return _build_section_v_fallback(fi_path)


def _feature_importance_lines(fi_path: Optional[Path]) -> list:
    """从 feature-importance.csv 生成 Top 20 展示行; 无文件返回说明行。"""
    if fi_path is None:
        return ["（无可用特征重要性数据）"]
    try:
        import pandas as pd
        fi = pd.read_csv(fi_path)
    except Exception:
        return ["（特征重要性文件读取失败）"]
    if "feature" not in fi.columns:
        return ["（特征重要性文件缺 feature 列）"]
    gain_col = next((c for c in ("gain", "total_gain", "importance") if c in fi.columns), None)
    lines = ["| # | feature | importance |", "|---|---------|-----------|"]
    for i, row in fi.head(20).iterrows():
        imp = row.get(gain_col, "—") if gain_col else "—"
        lines.append(f"| {i + 1} | {row.get('feature', '—')} | {imp} |")
    return lines


def _latest_run_feature_importance(session_dir: Path) -> Optional[Path]:
    """v2.3 兜底: 找最新 run 的特征重要性 csv。

    - experiments 型 run: 从其 config.json.source_exp 定位 experiments/{id}/feature_importance.csv
    - 其他 run: new-models/*/explainability/feature-importance.csv(历史兼容)
    """
    new_models = session_dir / "new-models"
    if not new_models.is_dir():
        return None
    for run_dir in sorted(new_models.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        p = run_dir / "explainability" / "feature-importance.csv"
        if p.exists():
            return p
        cfg = _read_json(run_dir / "config.json")
        if cfg is not None and _is_experiments_run(cfg):
            exp_dir = _experiments_source_dir(session_dir, cfg)
            if exp_dir is not None:
                ep = exp_dir / "feature_importance.csv"
                if ep.exists():
                    return ep
    return None


def _build_section_v_fallback(fi_path: Path) -> str:
    """v2.1 兜底: credit-data-analysis 未执行时, 用最新 run 的特征重要性简要展示。"""
    lines = [
        f"{SECTION_ANCHORS['V'][0]}\n",
        "- 特征分析(credit-data-analysis)尚未执行, 以下为最新 run 的特征重要性兜底展示:",
        "",
    ]
    lines.extend(_feature_importance_lines(fi_path))
    lines.append("")
    return "\n".join(lines)


# ===== Section VI: 模型迭代 =====

def _classify_run(config: dict) -> str:
    """从 config.json 判断 run 类型,返回一句话描述。"""
    if _is_experiments_run(config):
        # v2.3: experiments 矩阵转正 run
        feat_scheme = config.get("feat_scheme") or ""
        sample_scheme = config.get("sample_scheme") or ""
        tag = "experiments 矩阵转正"
        if config.get("is_tuned"):
            tag += " + Optuna"
        if config.get("optimistic_bias"):
            tag += " [乐观偏差候选]"
        if sample_scheme or feat_scheme:
            tag += f" ({sample_scheme} × {feat_scheme})"
        return tag

    return "baseline"


def build_section_vi(session_dir: Path) -> str:
    """模型迭代段 — 来源: new-models/*/config.json。"""
    new_models_dir = session_dir / "new-models"
    if not new_models_dir.exists():
        return (
            f"{SECTION_ANCHORS['VI'][0]}\n\n"
            "（尚未训练任何模型, new-models/ 不存在）\n"
        )

    run_dirs = sorted(
        [d for d in new_models_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not run_dirs:
        return (
            f"{SECTION_ANCHORS['VI'][0]}\n\n"
            "（new-models/ 为空, Stage 2 baseline 待跑）\n"
        )

    lines = [
        f"{SECTION_ANCHORS['VI'][0]}\n",
        f"共 {len(run_dirs)} 个 run (按目录名排序):",
        "",
        "| # | run_name | algo | n_feat | train AUC | val AUC | oot AUC | 关键变更 |",
        "|---|----------|------|--------|-----------|---------|---------|----------|",
    ]

    for i, run_dir in enumerate(run_dirs, 1):
        cfg = _read_json(run_dir / "config.json")
        if not cfg:
            lines.append(f"| {i} | {run_dir.name} | ? | ? | ? | ? | ? | config.json 缺失 |")
            continue

        runtime = cfg.get("runtime", {})
        metrics = runtime.get("metrics", {})
        train_auc = _fmt_auc(metrics.get("train", {}).get("auc"))
        val_auc = _fmt_auc(metrics.get("val", {}).get("auc"))
        oot_auc = _fmt_auc(metrics.get("oot", {}).get("auc"))
        n_feat = runtime.get("n_features", "—")
        algo = cfg.get("algo", "?")

        # v2.3: experiments 型 run 的 config.json 用顶层 metrics{oot_auc,val_auc} + features 列表
        if _is_experiments_run(cfg):
            top_metrics = cfg.get("metrics", {})
            val_auc = _fmt_auc(top_metrics.get("val_auc"))
            oot_auc = _fmt_auc(top_metrics.get("oot_auc"))
            features = cfg.get("features") or []
            n_feat = len(features) if features else "—"

        change = _classify_run(cfg)

        lines.append(
            f"| {i} | {run_dir.name} | {algo} | {n_feat} | "
            f"{train_auc} | {val_auc} | {oot_auc} | {change} |"
        )

    lines.append("")
    return "\n".join(lines)


# ===== Section VII: 横向对比 =====

def build_section_vii(session_dir: Path) -> str:
    """横向对比段 — 纯占位（v2.7 起 comparison 模块已移除，主链路评选用 experiments leaderboard）。"""
    return (
        f"{SECTION_ANCHORS['VII'][0]}\n\n"
        "（experiments 主链路评选以 leaderboard 为准：OOT AUC 排序 + 乐观偏差标注；本段为占位）\n"
    )


# ===== 报告替换逻辑 =====

SECTION_BUILDERS = {
    "IV": build_section_iv,
    "V": build_section_v,
    "VI": build_section_vi,
    "VII": build_section_vii,
}


def _split_report(content: str) -> List[Tuple[str, str]]:
    """把 report.md 切成 [(anchor_or_header, body), ...] 列表。

    每个顶层 `## ` H2 标题起一段。保留 H1 与首部 metadata。
    """
    parts: List[Tuple[str, str]] = []
    current_anchor = ""
    current_lines: List[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_anchor or current_lines:
                parts.append((current_anchor, "\n".join(current_lines)))
            current_anchor = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_anchor or current_lines:
        parts.append((current_anchor, "\n".join(current_lines)))

    return parts


def _is_anchor_match(anchor: str, targets: Tuple[str, str]) -> bool:
    """检查 anchor 是否匹配 targets 中的任意一个。

    targets = ("## 三、特征宽表", "## III. 特征宽表") 等。
    匹配规则: 比较"序号 token"(三、 / III.)与"标题关键词"(特征宽表),都一致才算匹配。
    """
    if not anchor:
        return False

    def _parse(line: str) -> Tuple[str, str]:
        """返回 (序号token, 标题剩余)。例: '## 三、特征宽表' → ('三、', '特征宽表')。"""
        if not line.startswith("## "):
            return ("", line)
        rest = line[3:].strip()
        # 序号 token = 第一个空白前或符号后
        parts = rest.split(maxsplit=1)
        if not parts:
            return ("", "")
        num = parts[0]
        title = parts[1] if len(parts) > 1 else ""
        return (num, title)

    a_num, a_title = _parse(anchor)
    for tgt in targets:
        t_num, t_title = _parse(tgt)
        if a_num == t_num and a_title == t_title:
            return True
    return False


def _replace_section(content: str, section_key: str, new_body: str) -> str:
    """替换 report.md 中指定 H2 段落的内容,保留其他段落。

    段落由 H2 锚点开头,到下一个 `## ` 或文件末尾结束。
    锚点为 `## 四、...` 形式(中文序号)。若不存在则追加到末尾。
    """
    primary, fallback = SECTION_ANCHORS[section_key]
    parts = _split_report(content)

    target_anchors = (primary, fallback)

    replaced_idx = None
    for i, (anchor, _body) in enumerate(parts):
        if _is_anchor_match(anchor, target_anchors):
            replaced_idx = i
            break

    body_only = new_body
    for tgt in target_anchors:
        if body_only.startswith(tgt):
            body_only = body_only[len(tgt):].lstrip("\n")
            break

    if replaced_idx is not None:
        parts[replaced_idx] = (primary, body_only)
    else:
        parts.append((primary, body_only))

    out_lines: List[str] = []
    for i, (anchor, body) in enumerate(parts):
        chunk_lines: List[str] = []
        if anchor:
            chunk_lines.append(anchor)
        body_str = body.strip()
        if body_str:
            chunk_lines.append(body_str)
        if not chunk_lines:
            continue
        if out_lines:
            out_lines.append("")
        out_lines.extend(chunk_lines)

    return "\n".join(out_lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="report.md 自动回填器")
    parser.add_argument("--session-dir", required=True, type=Path,
                        dest="session_dir",
                        help="session 目录 (runs/<timestamp>-<model_name>)")
    parser.add_argument(
        "--section",
        required=True,
        choices=list(SECTION_ANCHORS.keys()) + ["all"],
    )
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        sys.exit(f"[fill_report] session_dir 不存在: {session_dir}")

    report_path = session_dir / "report.md"
    if not report_path.exists():
        sys.exit(f"[fill_report] report.md 不存在: {report_path}")

    content = report_path.read_text(encoding="utf-8")

    sections = (
        list(SECTION_ANCHORS.keys()) if args.section == "all" else [args.section]
    )

    builders = {
        "IV": lambda: build_section_iv(session_dir),
        "V": lambda: build_section_v(session_dir),
        "VI": lambda: build_section_vi(session_dir),
        "VII": lambda: build_section_vii(session_dir),
    }

    for key in sections:
        new_body = builders[key]()
        content = _replace_section(content, key, new_body)
        print(f"[fill_report] 已回填段落 {key}")

    report_path.write_text(content, encoding="utf-8")
    print(f"[fill_report] report.md 已更新: {report_path}")


if __name__ == "__main__":
    main()
