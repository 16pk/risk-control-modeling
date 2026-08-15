# -*- coding: utf-8 -*-
"""run_sample_analysis_task_spec.py 测试: 标准流程 / 参数校验 / 按月分段。"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_sample_analysis_task_spec as rsa


def _make_sample(tmp_path, n_per_pday=10, pday_list=None, extra_rows=None, date_format="%Y%m%d"):
    """造一个小 parquet 样本(默认列名 fuid/label/f_p_date)。"""
    import datetime as _dt

    if pday_list is None:
        pday_list = ["20260413", "20260430", "20260508", "20260516", "20260524"]
    rows = []
    for pday in pday_list:
        if date_format == "%Y-%m-%d" and "-" not in pday:
            pday = "%s-%s-%s" % (pday[:4], pday[4:6], pday[6:8])
        for i in range(n_per_pday):
            rows.append({
                "fuid": "u_%s_%d" % (pday, i),
                "label": 1 if i < 2 else 0,   # 2/10 正样本
                "f_p_date": pday,
            })
    if extra_rows:
        rows.extend(extra_rows)
    df = pd.DataFrame(rows)
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    return p, df


def _run_script(sample_path, tmp_path, model_name="test_model", timestamp="20260629-161231"):
    out_dir = tmp_path / "data-profile"
    out_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(_SCRIPTS / "run_sample_analysis_task_spec.py"),
        "--sample", str(sample_path),
        "--model-name", model_name,
        "--timestamp", timestamp,
        "--output-dir", str(out_dir),
    ]
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, out_dir


def test_standard_flow(tmp_path):
    """用例 1: 标准流程, 纯样本分析(无切分), 产出 report/manifest。"""
    p, df = _make_sample(tmp_path)
    proc, out_dir = _run_script(p, tmp_path)
    assert proc.returncode == 0, "脚本失败: %s\n%s" % (proc.stdout, proc.stderr)

    for fname in ["report.md", "report.xlsx", "_manifest.json"]:
        assert (out_dir / fname).exists(), "缺文件: %s" % fname

    # 切分后置: 不再产出三档 parquet 与 _split_manifest.json
    for fname in ["_split_manifest.json", "train.parquet", "test.parquet", "oot.parquet"]:
        assert not (out_dir / fname).exists(), "不应产出(已后置到 feature-analysis): %s" % fname

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_summary"]["total_samples"] == 50
    assert manifest["model_name"] == "test_model"
    assert manifest["timestamp"] == "20260629-161231"
    assert manifest["user_confirmed"] is False
    # 无切分统计字段
    assert "split" not in manifest
    assert "split_files" not in manifest

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "样本分析报告" in report_md
    assert "test_model" in report_md
    # 报告不再含切分段
    assert "Train/Test/OOT 切分" not in report_md

    assert "样本分析完成" in proc.stdout


def test_monthly_segments(tmp_path):
    """用例 2: segment_by_time 默认按月聚合(YYYYMM)。"""
    p, df = _make_sample(tmp_path)  # pday 跨 202604 / 202605 两个月
    args = type("Args", (), {"dt_col": "f_p_date", "label_col": "label"})()
    segs = rsa.segment_by_time(df, args)
    months = [s["pday"] for s in segs]
    assert months == ["202604", "202605"]
    # 202604: 4/13 + 4/30 = 20 样本(4 正); 202605: 5/08 + 5/16 + 5/24 = 30 样本(6 正)
    assert segs[0]["samples"] == 20
    assert segs[1]["samples"] == 30
    assert segs[0]["positive"] == 4
    assert segs[1]["positive"] == 6


def test_missing_columns(tmp_path):
    """用例 3a: 缺 fuid 列。"""
    df = pd.DataFrame([{"label": 0, "f_p_date": "20260413"} for _ in range(10)])
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    proc, _ = _run_script(p, tmp_path)
    assert proc.returncode != 0
    assert "fuid" in proc.stdout or "fuid" in proc.stderr


def test_bad_label(tmp_path):
    """用例 3b: label 含 2。"""
    rows = [{"fuid": "u_%d" % i, "label": 2, "f_p_date": "20260413"} for i in range(10)]
    df = pd.DataFrame(rows)
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    proc, _ = _run_script(p, tmp_path)
    assert proc.returncode != 0
    assert "非法" in proc.stdout or "非法" in proc.stderr


def test_bad_pday(tmp_path):
    """用例 3c: f_p_date 含非法日期。"""
    rows = [{"fuid": "u_%d" % i, "label": 0, "f_p_date": "2026"} for i in range(10)]
    df = pd.DataFrame(rows)
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    proc, _ = _run_script(p, tmp_path)
    assert proc.returncode != 0


def test_sample_not_exist(tmp_path):
    """用例 3d: --sample 文件不存在。"""
    proc, _ = _run_script(tmp_path / "nonexistent.parquet", tmp_path)
    assert proc.returncode != 0
    assert "不存在" in proc.stdout or "不存在" in proc.stderr


def test_judge_stability():
    """稳定性判定阈值。"""
    segs = [{"pday": "202604", "positive_rate": 0.11},
            {"pday": "202605", "positive_rate": 0.13},
            {"pday": "202606", "positive_rate": 0.15}]
    s = rsa.judge_stability(segs)
    assert s["volatility_pp"] == 4.0
    assert s["judgment"] == "显著波动"

    segs2 = [{"pday": "202604", "positive_rate": 0.13},
             {"pday": "202605", "positive_rate": 0.135}]
    s2 = rsa.judge_stability(segs2)
    assert s2["judgment"] == "稳定"


def test_judge_sufficiency():
    """充足度判定。"""
    assert rsa.judge_sufficiency({"positive_samples": 15000, "total_samples": 150000})["judgment"] == "充足"
    assert rsa.judge_sufficiency({"positive_samples": 5000, "total_samples": 50000})["judgment"] == "基本可用"
    assert rsa.judge_sufficiency({"positive_samples": 100, "total_samples": 1000})["judgment"] == "不足，建议补充样本"


def test_standard_flow_dual_format(tmp_path):
    """用例 4: 数据列用 YYYY-MM-DD, 按月分段归一化正确。"""
    p, df = _make_sample(tmp_path, date_format="%Y-%m-%d")
    proc, out_dir = _run_script(p, tmp_path)
    assert proc.returncode == 0, "脚本失败: %s\n%s" % (proc.stdout, proc.stderr)
    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_summary"]["total_samples"] == 50
    # 按月聚合产出 YYYYMM 前缀
    segs = manifest["time_segments"]
    assert all(isinstance(s["pday"], str) and len(s["pday"]) == 6 for s in segs)
    assert [s["pday"] for s in segs] == ["202604", "202605"]
