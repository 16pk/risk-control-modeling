"""credit-data-analysis pipeline 模式测试（v2.1）。

验证:
1. --split-config 推导 PSI 基准月 = 第一个 OOT 月（model.split.oot_range 首月）
2. 显式 --base-month 覆盖推导
3. md + xlsx + _manifest.json 三产物
4. manifest 记录 split_config
5. 不做切分、不产 splits / IV / PSI 筛选 csv
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "feature_analysis.py"
PYTHON = sys.executable


@pytest.fixture(scope="module")
def sample_data(tmp_path_factory):
    """构造 3 个月样本数据（含 label + 日期 + 3 特征）。"""
    tmp = tmp_path_factory.mktemp("cda_pipeline")
    rng = np.random.default_rng(42)
    n = 4000
    dates = pd.date_range("2026-03-01", "2026-05-24", freq="D")
    pdays = np.random.choice([d.strftime("%Y%m%d") for d in dates], size=n)
    df = pd.DataFrame({
        "fuid": [f"u{i:06d}" for i in range(n)],
        "pday": pdays,
        "f0": rng.normal(size=n) + 0.5,
        "f1": rng.normal(size=n),
        "f2": rng.integers(0, 10, size=n).astype(float),
        "label": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
    })
    df.to_parquet(tmp / "sample.parquet", index=False)
    return tmp


@pytest.fixture(scope="module")
def split_config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cda_cfg")
    cfg = tmp / "feature_config.yaml"
    cfg.write_text(
        "model:\n"
        "  split:\n"
        "    train_range: ['20260301', '20260430']\n"
        "    test_range:  ['20260501', '20260516']\n"
        "    oot_range:   ['20260517', '20260524']\n"
        "analysis:\n"
        "  psi:\n"
        "    n_bins: 10\n"
        "    warn_threshold: 0.10\n",
        encoding="utf-8",
    )
    return cfg


def _run(args: list) -> dict:
    out = subprocess.run([PYTHON, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"脚本失败: {out.stderr[-2000:]}"
    return out


def _latest_month(data_file: Path) -> str:
    df = pd.read_parquet(data_file)
    months = sorted(df["pday"].astype(str).str[:6].unique())
    return f"{months[-1][:4]}-{months[-1][4:6]}"


def test_pipeline_mode_derives_base_month_from_oot(sample_data, split_config, tmp_path):
    """pipeline 模式未显式 base-month 时, PSI 基准月 = 第一个 OOT 月。"""
    out = tmp_path / "out"
    _run([
        "--data-file", str(sample_data / "sample.parquet"),
        "--feature-start", "f0", "--feature-end", "f2",
        "--iv-label", "label", "--time-col", "pday",
        "--split-config", str(split_config),
        "--output-dir", str(out), "--output-file", "特征分析结果.xlsx",
    ])
    # 产物三件套
    for f in ["特征分析结果.xlsx", "特征分析结果.md", "_manifest.json"]:
        assert (out / f).exists(), f"缺产物: {f}"
    # md 含 PSI 基准月 = 2026-05（oot_range 首月 20260517）
    md = (out / "特征分析结果.md").read_text(encoding="utf-8")
    assert "三、分月 PSI（基准月 `2026-05`）" in md, md[:800]
    # manifest 记录 split_config 与 base_month
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["params"]["split_config"] == str(split_config)
    assert manifest["params"]["base_month"] == "2026-05"
    # 不产 splits / 筛选 csv
    assert not (out / "splits").exists()
    for csv_name in ["stats.csv", "iv_table.csv", "psi_table.csv"]:
        assert not (out / csv_name).exists()


def test_explicit_base_month_overrides(sample_data, split_config, tmp_path):
    """显式 --base-month 覆盖 split_config 推导。"""
    out = tmp_path / "out2"
    _run([
        "--data-file", str(sample_data / "sample.parquet"),
        "--feature-start", "f0", "--feature-end", "f2",
        "--iv-label", "label", "--time-col", "pday",
        "--base-month", "2026-04",
        "--split-config", str(split_config),
        "--output-dir", str(out), "--output-file", "特征分析结果.xlsx",
    ])
    md = (out / "特征分析结果.md").read_text(encoding="utf-8")
    assert "三、分月 PSI（基准月 `2026-04`）" in md, md[:800]
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["params"]["base_month"] == "2026-04"


def test_missing_split_range_errors(sample_data, tmp_path):
    """split_config 缺 oot_range 时明确报错, 不静默兜底。"""
    bad = tmp_path / "bad_config.yaml"
    bad.write_text("model:\n  split:\n    train_range: ['20260301', '20260430']\n", encoding="utf-8")
    out = tmp_path / "out3"
    proc = subprocess.run(
        [PYTHON, str(SCRIPT),
         "--data-file", str(sample_data / "sample.parquet"),
         "--feature-start", "f0", "--feature-end", "f2",
         "--iv-label", "label", "--time-col", "pday",
         "--split-config", str(bad),
         "--output-dir", str(out), "--output-file", "特征分析结果.xlsx"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "oot_range" in proc.stderr


# ---------------- v2.5 修复：--feature-list 透传特征清单 ----------------

def test_feature_list_mode_selects_exact_columns(sample_data, tmp_path):
    """--feature-list 精确选列（主入口）：报告只含清单列，manifest 记录清单。"""
    fl = sample_data / "feature-list.csv"
    pd.DataFrame({"feature_name": ["f1", "f2"]}).to_csv(fl, index=False)
    out = tmp_path / "out_fl"
    proc = subprocess.run(
        [PYTHON, str(SCRIPT),
         "--data-file", str(sample_data / "sample.parquet"),
         "--feature-list", str(fl),
         "--iv-label", "label", "--time-col", "pday",
         "--base-month", "2026-05",
         "--output-dir", str(out), "--output-file", "特征分析结果.xlsx"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"脚本失败: {proc.stderr[-2000:]}"
    # 报告只含清单列 f1/f2（不含 f0）
    md = (out / "特征分析结果.md").read_text(encoding="utf-8")
    assert "| f1 |" in md
    assert "| f2 |" in md
    assert "| f0 |" not in md
    assert "特征来源: 特征清单" in md
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["params"]["feature_list"] == str(fl)
    assert manifest["params"]["feature_source"] == "feature-list"
    assert "feature_start" not in manifest["params"]  # 清单模式不记录区间


def test_feature_list_mode_extra_columns(sample_data, tmp_path):
    """--feature-list + --feature-extra 追加额外列。"""
    fl = sample_data / "feature-list.csv"
    pd.DataFrame({"feature_name": ["f1"]}).to_csv(fl, index=False)
    out = tmp_path / "out_fl_extra"
    proc = subprocess.run(
        [PYTHON, str(SCRIPT),
         "--data-file", str(sample_data / "sample.parquet"),
         "--feature-list", str(fl), "--feature-extra", "f2",
         "--iv-label", "label", "--time-col", "pday",
         "--base-month", "2026-05",
         "--output-dir", str(out), "--output-file", "特征分析结果.xlsx"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"脚本失败: {proc.stderr[-2000:]}"
    md = (out / "特征分析结果.md").read_text(encoding="utf-8")
    assert "| f1 |" in md and "| f2 |" in md
    assert "| f0 |" not in md


def test_feature_list_missing_cols_warn_not_fail(sample_data, tmp_path):
    """清单含不在样本中的列：仅 WARN 不报错（容忍列漂移）。"""
    fl = sample_data / "feature-list.csv"
    pd.DataFrame({"feature_name": ["f1", "ghost_col"]}).to_csv(fl, index=False)
    out = tmp_path / "out_fl_warn"
    proc = subprocess.run(
        [PYTHON, str(SCRIPT),
         "--data-file", str(sample_data / "sample.parquet"),
         "--feature-list", str(fl),
         "--iv-label", "label", "--time-col", "pday",
         "--base-month", "2026-05",
         "--output-dir", str(out), "--output-file", "特征分析结果.xlsx"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"脚本失败: {proc.stderr[-2000:]}"
    assert "ghost_col" in proc.stdout  # WARN 打印
    md = (out / "特征分析结果.md").read_text(encoding="utf-8")
    assert "缺失 1 列" in md
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["params"]["feature_list_missing"] == ["ghost_col"]


def test_no_feature_source_errors(sample_data, tmp_path):
    """--feature-list 与区间参数均缺 → 报错提示迁移（不再默认区间）。"""
    out = tmp_path / "out_none"
    proc = subprocess.run(
        [PYTHON, str(SCRIPT),
         "--data-file", str(sample_data / "sample.parquet"),
         "--iv-label", "label", "--time-col", "pday",
         "--base-month", "2026-05",
         "--output-dir", str(out), "--output-file", "特征分析结果.xlsx"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "--feature-list" in proc.stderr
