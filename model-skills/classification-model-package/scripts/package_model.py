#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""classification-model-package 打包器: 把已定版训练任务组装为可独立运行的交付代码包。

输入 session 定版产物:
  - <session_dir>/finalized_model.json      (run_name / algo / model_path)
  - <session_dir>/new-models/{run}/model/   (model.pkl 或 model.json + model_meta.json)
  - <session_dir>/sample-features/data-cleaning/cleaning-scheme.json  (哨兵集, 缺则默认+WARN)
  - <session_dir>/sample-features/feature-list.csv                    (权威特征清单, 一致性 WARN)
  - <session_dir>/fico/coef.json            (存在 → 交付包含 FICO 转分模块)

输出 <session_dir>/delivery/ 交付包:
  run.py                    主入口(清理→打分→可选 FICO)
  pipeline/                 自包含实现(clean/score/fico)
  assets/                   模型 + 清洗方案 + 权威特征清单 + (可选)coef.json + model_meta.json
  requirements.txt          最小依赖(按 algo 裁剪 lgb/xgb)
  README.md                 用法/输入 schema/输出说明(渲染会话信息)
  package-manifest.json     打包元信息(来源/算法/是否含FICO/特征一致性)

设计要点:
  - 资产驱动: 包内脚本行为全部从 assets/ 既有资产文件读取, 零占位符、可审计。
  - 纯拷贝模板: package_templates/ 下脚本即交付包内最终脚本, 打包器只做 shutil.copy
    (仅 requirements.txt 与 README.md 做字符串渲染); 模板与单测共享同一份源码。
  - 独立运行: 交付包零引用专家包目录与 _modelevo-shared。
  - 算法边界: 仅支持主链路产物 lgb/xgb(含 xgb 历史 model.json); dnn/lr 无法在自包含
    包内反序列化(DnnPredictor/LrPredictor 需 training 脚本依赖), 报错拒绝。

用法:
  python package_model.py --session-dir <session_dir> [--out-dir <delivery 目录>]

依赖: pandas / numpy / pyarrow / joblib (仅读资产与组装); 共享代码经 _bootstrap.py 注入。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts
from gen_feature_list import load_feature_list

# 支持算法: 主链路产物 lgb/xgb(含 xgb 历史 model.json)
SUPPORTED_ALGOS = ("lgb", "xgb")
# 默认哨兵集(data-cleaning 默认, 打包时 cleaning-scheme.json 缺失则用之)
DEFAULT_INVALID_VALUES = [-1, -2, -9, -99, -999, -9999, -99999]


def _err(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def read_json(path: Path) -> dict:
    if not path.exists():
        _err(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 校验链
# ---------------------------------------------------------------------------
def resolve_finalized(session_dir: Path) -> dict:
    fin = session_dir / "finalized_model.json"
    if not fin.exists():
        _err(
            f"{session_dir} 未定版: 缺少 {fin.name}。请先走主链路收口(development Stage 6)落定版标记。"
        )
    return read_json(fin)


def resolve_model_dir(session_dir: Path, fin: dict) -> Path:
    run_name = fin.get("run_name") or fin.get("run")
    if not run_name:
        _err("finalized_model.json 缺 run_name, 无法定位定版模型")
    model_dir = session_dir / "new-models" / str(run_name) / "model"
    if not model_dir.is_dir():
        # 兜底: finalized 里若直接给了 model_path
        mp = fin.get("model_path")
        if mp:
            cand = Path(mp)
            model_dir = cand if cand.is_dir() else (session_dir / mp)
    if not model_dir.is_dir():
        _err(f"定版模型目录不存在: {model_dir}")
    return model_dir


def validate_model_assets(model_dir: Path) -> tuple[str, list]:
    """校验 model.pkl 或 model.json + model_meta.json 齐全, 返回 (algo, feature_names)。"""
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        _err(f"定版模型缺 model_meta.json: {meta_path}")
    meta = read_json(meta_path)
    algo = (meta.get("algo") or "").lower()
    feature_names = list(meta.get("feature_names") or [])
    if not feature_names:
        _err(f"model_meta.json 缺 feature_names: {meta_path}")
    if algo not in SUPPORTED_ALGOS:
        _err(
            f"定版模型 algo={algo!r} 不在交付包支持范围({SUPPORTED_ALGOS})。"
            "自包含交付包无法携带 dnn/lr 反序列化所需的 training 脚本依赖, 请走主链路(lgb/xgb)。"
        )
    has_pkl = (model_dir / "model.pkl").exists()
    has_json = (model_dir / "model.json").exists()
    if not (has_pkl or has_json):
        _err(f"定版模型目录缺 model.pkl / model.json: {model_dir}")
    return algo, feature_names


def resolve_cleaning_scheme(session_dir: Path) -> list:
    """读 cleaning-scheme.json 的 invalid_values; 缺失/异常回退默认集合并 WARN。"""
    path = session_dir / "sample-features" / "data-cleaning" / "cleaning-scheme.json"
    if not path.exists():
        print(f"[WARN] 缺少 {path.name}, 使用默认哨兵集 {DEFAULT_INVALID_VALUES}")
        return list(DEFAULT_INVALID_VALUES)
    try:
        scheme = read_json(path)
        vals = scheme.get("invalid_values") or scheme.get("invalidValues")
        if vals:
            return [float(v) for v in vals]
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 解析 {path.name} 失败({e}), 使用默认哨兵集 {DEFAULT_INVALID_VALUES}")
    return list(DEFAULT_INVALID_VALUES)


def resolve_feature_list(session_dir: Path, feature_names: list) -> list:
    """读权威 feature-list.csv; 与 model feature_names 做一致性 WARN。"""
    path = session_dir / "sample-features" / "feature-list.csv"
    if not path.exists():
        print(f"[WARN] 缺少权威特征清单 {path}, 以 model_meta.feature_names 作为打包特征集")
        return list(feature_names)
    try:
        features = load_feature_list(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 读取权威特征清单失败({e}), 以 model_meta.feature_names 作为打包特征集")
        return list(feature_names)
    if not features:
        print("[WARN] 权威特征清单为空, 以 model_meta.feature_names 作为打包特征集")
        return list(feature_names)
    # 一致性 WARN: 特征清单应覆盖定版模型特征
    missing = [f for f in feature_names if f not in features]
    extra = [f for f in features if f not in feature_names]
    if missing:
        print(f"[WARN] 权威特征清单缺少定版模型特征 {len(missing)} 个: {missing}")
    if extra:
        print(f"[WARN] 权威特征清单含定版模型之外特征 {len(extra)} 个: {extra}")
    return features


def resolve_fico(session_dir: Path) -> Optional[Path]:
    """存在 fico/coef.json → 交付包含 FICO 转分模块; 否则不含。"""
    p = session_dir / "fico" / "coef.json"
    if p.exists():
        # 校验 coef/intc 可读
        try:
            read_json(p)
            return p
        except SystemExit:
            print(f"[WARN] {p} 解析失败, 交付包不含 FICO 模块")
            return None
    return None


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def _copy_templates(templates_dir: Path, delivery: Path) -> None:
    """纯拷贝模板: pipeline/ + run.py + requirements.txt(占位待渲染)。"""
    for name in ("run.py", "requirements.txt"):
        src = templates_dir / name
        if not src.exists():
            _err(f"模板缺失: {src}")
        shutil.copy2(src, delivery / name)
    shutil.copytree(templates_dir / "pipeline", delivery / "pipeline", dirs_exist_ok=True)


def _render_requirements(delivery: Path, algo: str) -> None:
    path = delivery / "requirements.txt"
    req_algo = "lightgbm>=3.3" if algo == "lgb" else "xgboost>=1.7"
    text = path.read_text(encoding="utf-8").replace("{{REQ_ALGO}}", req_algo)
    path.write_text(text, encoding="utf-8")


def _render_readme(readme_tpl: Path, delivery: Path, ctx: dict) -> None:
    if not readme_tpl.exists():
        _err(f"README 模板缺失: {readme_tpl}")
    text = readme_tpl.read_text(encoding="utf-8")
    for key, val in ctx.items():
        text = text.replace("{{" + key + "}}", str(val))
    (delivery / "README.md").write_text(text, encoding="utf-8")


def build_package(session_dir: Path, out_dir: Path) -> Path:
    """核心组装逻辑(可被单测直接 import)。"""
    templates_dir = Path(__file__).resolve().parent / "package_templates"
    session_dir = session_dir.resolve()
    out_dir = out_dir.resolve()

    fin = resolve_finalized(session_dir)
    model_dir = resolve_model_dir(session_dir, fin)
    algo, feature_names = validate_model_assets(model_dir)
    invalid_values = resolve_cleaning_scheme(session_dir)
    feature_list = resolve_feature_list(session_dir, feature_names)
    coef_path = resolve_fico(session_dir)

    # 组装
    delivery = out_dir / "delivery"
    if delivery.exists():
        shutil.rmtree(delivery)
    delivery.mkdir(parents=True, exist_ok=True)
    (delivery / "assets").mkdir(parents=True, exist_ok=True)
    (delivery / "pipeline").mkdir(parents=True, exist_ok=True)

    # 资产拷贝
    for name in ("model.pkl", "model.json"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, delivery / "assets" / name)
    shutil.copy2(model_dir / "model_meta.json", delivery / "assets" / "model_meta.json")
    # 清洗方案(哨兵集) + 权威特征清单
    (delivery / "assets" / "cleaning-scheme.json").write_text(
        json.dumps({"schema_version": 1, "invalid_values": invalid_values},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (delivery / "assets" / "feature-list.csv").write_text(
        "feature_name\n" + "\n".join(feature_list) + "\n", encoding="utf-8")
    has_fico = coef_path is not None
    if has_fico:
        shutil.copy2(coef_path, delivery / "assets" / "coef.json")

    # 模板拷贝 + 渲染
    _copy_templates(templates_dir, delivery)
    _render_requirements(delivery, algo)
    run_name = str(fin.get("run_name") or fin.get("run") or "unknown")
    readme_ctx = {
        "RUN_NAME": run_name,
        "ALGO": algo,
        "N_FEATURES": len(feature_names),
        "FEATURES": "、".join(feature_names),
        "HAS_FICO": "是" if has_fico else "否",
        "PACKAGED_AT": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _render_readme(templates_dir / "README.md", delivery, readme_ctx)

    # manifest
    manifest = {
        "schema_version": 1,
        "produced_by": "skills/classification-model-package",
        "source_session": str(session_dir),
        "package_dir": str(delivery),
        "run_name": run_name,
        "algo": algo,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "invalid_values": invalid_values,
        "feature_list_source": str(session_dir / "sample-features" / "feature-list.csv"),
        "feature_list_matches_model": set(feature_names) <= set(feature_list),
        "has_fico": has_fico,
        "coef_source": str(coef_path) if has_fico else None,
        "packaged_at": readme_ctx["PACKAGED_AT"],
    }
    (delivery / "package-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[PACKAGE] 交付包组装完成: {delivery}")
    print(f"[PACKAGE] run={run_name} | algo={algo} | features={len(feature_names)} | "
          f"FICO={'是' if has_fico else '否'}")
    return delivery


def main() -> int:
    parser = argparse.ArgumentParser(description="把已定版训练任务组装为独立交付代码包")
    parser.add_argument("--session-dir", required=True, help="已定版 session 目录")
    parser.add_argument("--out-dir", default=None,
                        help="交付包输出目录(默认 <session_dir>/delivery)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    if not session_dir.is_dir():
        _err(f"session 目录不存在: {session_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else session_dir
    build_package(session_dir, out_dir)
    print("[DONE] 打包完成。交付包位于 <out>/delivery/, 运行: "
          "python run.py --input <数据文件> --output-dir <out>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
