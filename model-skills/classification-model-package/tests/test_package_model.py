#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""classification-model-package 打包器单测 + 交付包模板共享冒烟。

覆盖:
- 打包器: 校验链(finalized/model_meta/清洗方案/特征清单/FICO 探测) + 组装(delivery 结构) +
  README/requirements 渲染 + dnn/lr 拒绝
- 模板共享: 直接 import package_templates 下的 run/pipeline 源码(与打包产物同一份代码),
  验证清理/打分/转分/缺特征报错 行为

运行: python -m pytest tests/ -q
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "package_templates"))

import package_model  # noqa: E402
from pipeline import clean as pclean  # noqa: E402
from pipeline import fico as pfico  # noqa: E402
from pipeline import score as pscore  # noqa: E402


def _py():
    return sys.executable


# ---------------------------------------------------------------------------
# 造一个最小定版 session
# ---------------------------------------------------------------------------
def make_session(tmp_path) -> Path:
    session = tmp_path / "session"
    (session / "new-models" / "lgb-v1" / "model").mkdir(parents=True)
    (session / "sample-features" / "data-cleaning").mkdir(parents=True)

    # model_meta + model.pkl(假模型: 单列概率)
    feat = ["f0", "f1", "f2"]
    meta = {"algo": "lgb", "feature_names": feat, "run_name": "lgb-v1"}
    (session / "new-models" / "lgb-v1" / "model" / "model_meta.json").write_text(
        json.dumps(meta), encoding="utf-8")
    (session / "new-models" / "lgb-v1" / "model" / "model.pkl").write_bytes(b"dummy")

    # finalized_model.json
    (session / "finalized_model.json").write_text(
        json.dumps({"schema_version": 1, "run_name": "lgb-v1", "algo": "lgb",
                    "model_path": "new-models/lgb-v1/model"}),
        encoding="utf-8")

    # 清洗方案(哨兵集) + 权威特征清单
    (session / "sample-features" / "data-cleaning" / "cleaning-scheme.json").write_text(
        json.dumps({"invalid_values": [-1, -2, -999]}), encoding="utf-8")
    (session / "sample-features" / "feature-list.csv").write_text(
        "feature_name\nf0\nf1\nf2\n", encoding="utf-8")
    return session


# ---------------------------------------------------------------------------
# 打包器: 校验链
# ---------------------------------------------------------------------------
def test_resolve_finalized_missing(tmp_path):
    with pytest.raises(SystemExit) as ei:
        package_model.resolve_finalized(tmp_path)
    assert "未定版" in str(ei.value)


def test_validate_model_assets_dnn_rejected(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_meta.json").write_text(
        json.dumps({"algo": "dnn", "feature_names": ["a"]}), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        package_model.validate_model_assets(model_dir)
    assert "dnn" in str(ei.value) and "lgb/xgb" in str(ei.value)


def test_resolve_cleaning_scheme_fallback(tmp_path):
    # 缺失 → 默认集 + WARN(不抛)
    vals = package_model.resolve_cleaning_scheme(tmp_path)
    assert vals == package_model.DEFAULT_INVALID_VALUES


def test_resolve_feature_list_consistency(tmp_path):
    session = tmp_path / "s"
    (session / "sample-features").mkdir(parents=True)
    (session / "sample-features" / "feature-list.csv").write_text(
        "feature_name\nf0\nf1\n", encoding="utf-8")
    features = package_model.resolve_feature_list(session, ["f0", "f1", "f2"])
    assert features == ["f0", "f1"]  # 返回清单内容
    # 缺失 feature-list → 回退 feature_names
    assert package_model.resolve_feature_list(tmp_path / "nosuch", ["a", "b"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# 打包器: 组装
# ---------------------------------------------------------------------------
def test_build_package_structure(tmp_path):
    session = make_session(tmp_path)
    out = tmp_path / "out"
    delivery = package_model.build_package(session, out)

    assert (delivery / "run.py").exists()
    assert (delivery / "pipeline" / "clean.py").exists()
    assert (delivery / "pipeline" / "score.py").exists()
    assert (delivery / "pipeline" / "fico.py").exists()
    assert (delivery / "assets" / "model.pkl").exists()
    assert (delivery / "assets" / "model_meta.json").exists()
    assert (delivery / "assets" / "cleaning-scheme.json").exists()
    assert (delivery / "assets" / "feature-list.csv").exists()
    assert not (delivery / "assets" / "coef.json").exists()  # 无 FICO
    assert (delivery / "requirements.txt").exists()
    assert (delivery / "README.md").exists()
    assert (delivery / "package-manifest.json").exists()

    # 模板零占位符泄漏(除 requirements)
    for f in ("run.py", "README.md"):
        assert "{{" not in (delivery / f).read_text(encoding="utf-8"), f

    manifest = json.loads((delivery / "package-manifest.json").read_text(encoding="utf-8"))
    assert manifest["algo"] == "lgb"
    assert manifest["run_name"] == "lgb-v1"
    assert manifest["has_fico"] is False
    assert "lightgbm" in (delivery / "requirements.txt").read_text(encoding="utf-8")
    assert "{{" not in (delivery / "requirements.txt").read_text(encoding="utf-8")


def test_build_package_with_fico(tmp_path):
    session = make_session(tmp_path)
    (session / "fico").mkdir(parents=True)
    (session / "fico" / "coef.json").write_text(json.dumps({"coef": 1.5, "intc": -2.0}),
                                                encoding="utf-8")
    out = tmp_path / "out"
    delivery = package_model.build_package(session, out)
    assert (delivery / "assets" / "coef.json").exists()
    manifest = json.loads((delivery / "package-manifest.json").read_text(encoding="utf-8"))
    assert manifest["has_fico"] is True
    assert "FICO" in (delivery / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 模板共享: pipeline 行为(直接 import 模板源码, 与打包产物同一份)
# ---------------------------------------------------------------------------
def test_clean_sentinel_only_features(tmp_path):
    df = pd.DataFrame({
        "f0": [1.0, -1.0, 3.0, -999.0],
        "f1": [0.1, 0.2, 0.3, 0.4],
        "fuid": ["U1", "U2", "U3", "U4"],      # 非特征: 不参与
        "label": [-1, 1, 0, 1],                # 非特征: -1 不替换(允许缺/保留)
    })
    cleaned, report = pclean.clean_sentinel(df, ["f0", "f1"], [-1, -999])
    assert cleaned.loc[1, "f0"] != cleaned.loc[1, "f0"]  # NaN
    assert np.isnan(cleaned.loc[1, "f0"])
    assert cleaned["label"].tolist() == [-1, 1, 0, 1]     # 非特征不动
    assert cleaned["fuid"].tolist() == ["U1", "U2", "U3", "U4"]
    assert len(report["features"]) == 1
    assert report["features"][0]["feature"] == "f0"


def test_clean_default_invalid(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    assert pclean.load_invalid_values(assets) == pclean.DEFAULT_INVALID_VALUES


def test_score_missing_feature_error(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "model_meta.json").write_text(
        json.dumps({"algo": "lgb", "feature_names": ["a", "b"]}), encoding="utf-8")
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(SystemExit) as ei:
        pscore.apply_score(df, ["a", "b"], "lgb", assets, "score")
    assert "缺失" in str(ei.value) and "b" in str(ei.value)


def test_fico_apply(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    assert pfico.load_coef(assets) is None  # 无 coef → 不含 FICO
    (assets / "coef.json").write_text(json.dumps({"coef": 2.0, "intc": -3.0}), encoding="utf-8")
    # 恒等校准（coef=1,intc=0, p'=p）+ 真实违约概率分布 → 标准分区间内（p<=0.5 时 bscore>=400）
    coef = {"coef": 1.0, "intc": 0.0}
    df = pd.DataFrame({"score": [0.02, 0.05, 0.1, 0.3, 0.5]})
    out, summary = pfico.apply_fico(df, "score", coef)
    assert "bscore" in out.columns
    assert out["bscore"].between(400, 780).all()
    assert summary["n"] == 5
    # 与专家包公式一致: bscore = 400 - 35/ln2 * ln(p/(1-p))
    p = pfico.logistic_prob(pfico.calc_odds(df["score"].values), 1.0, 0.0)
    expect = 400.0 - 35.0 / np.log(2.0) * np.log(p / (1 - p))
    np.testing.assert_allclose(out["bscore"].values, expect)


def test_fico_apply_coef_nonpositive_warns(tmp_path, capsys):
    df = pd.DataFrame({"score": [0.5, 0.6]})
    out, _ = pfico.apply_fico(df, "score", {"coef": -1.0, "intc": 0.5})
    assert "WARN" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 端到端: 打包 → 交付包运行(需 lightgbm; 缺失则 skip)
# ---------------------------------------------------------------------------
def _train_lgb(assets_dir, features):
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({f: rng.normal(size=n) for f in features})
    y = (X["f0"] + rng.normal(0, 0.3, n) > 0).astype(int)
    m = lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=42)
    m.fit(X, y)
    import joblib
    joblib.dump(m, str(assets_dir / "model.pkl"))


@pytest.mark.skipif(not pytest.importorskip("lightgbm"), reason="lightgbm 缺失")
def test_e2e_package_run(tmp_path):
    session = make_session(tmp_path)
    # 用真实 lgb 模型替换 dummy
    model_dir = session / "new-models" / "lgb-v1" / "model"
    _train_lgb(model_dir, ["f0", "f1", "f2"])
    (session / "fico").mkdir(parents=True)
    (session / "fico" / "coef.json").write_text(json.dumps({"coef": 2.0, "intc": -3.0}),
                                                encoding="utf-8")

    delivery = package_model.build_package(session, tmp_path / "out")

    # 造生产数据(含哨兵值 + 非特征列, 无 label)
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame({
        "fuid": [f"U{i:05d}" for i in range(n)],
        "f_p_date": 20260101 + (rng.integers(0, 20, n)),
        "f0": np.where(rng.random(n) < 0.2, -1.0, rng.normal(size=n)),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    in_path = tmp_path / "batch.parquet"
    df.to_parquet(in_path, index=False)

    out_dir = tmp_path / "batch_out"
    res = subprocess.run(
        [_py(), str(delivery / "run.py"), "--input", str(in_path),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(delivery), timeout=120)
    assert res.returncode == 0, res.stderr + res.stdout

    scored = pd.read_parquet(out_dir / "score.parquet")
    assert "score" in scored.columns and "bscore" in scored.columns
    assert "f0" not in scored.columns  # 特征列不输出
    assert "fuid" in scored.columns and "f_p_date" in scored.columns
    assert len(scored) == n
    assert (out_dir / "cleaning-report.json").exists()
    assert (out_dir / "fico-summary.json").exists()
    assert (out_dir / "run-manifest.json").exists()
    # 哨兵替换: 清洗报告中应有 f0 命中
    cr = json.loads((out_dir / "cleaning-report.json").read_text(encoding="utf-8"))
    assert any(f["feature"] == "f0" for f in cr["features"])


def test_e2e_missing_feature_fails(tmp_path):
    session = make_session(tmp_path)
    model_dir = session / "new-models" / "lgb-v1" / "model"
    _train_lgb(model_dir, ["f0", "f1", "f2"])
    delivery = package_model.build_package(session, tmp_path / "out")

    df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [1.0, 2.0]})  # 缺 f2
    in_path = tmp_path / "bad.parquet"
    df.to_parquet(in_path, index=False)
    res = subprocess.run(
        [_py(), str(delivery / "run.py"), "--input", str(in_path),
         "--output-dir", str(tmp_path / "bad_out")],
        capture_output=True, text=True, cwd=str(delivery), timeout=120)
    assert res.returncode != 0
    assert "f2" in res.stderr + res.stdout
