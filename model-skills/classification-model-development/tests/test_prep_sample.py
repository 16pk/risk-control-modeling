# -*- coding: utf-8 -*-
"""prep_sample.py 轻量编排入口单测: 编排链参数传递顺序 / 非零返回码透传 / 自动建目录。

不实际跑大文件/子进程: mock subprocess.run 与文件存在性, 仅验证编排链逻辑。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import prep_sample


@pytest.fixture(autouse=True)
def _mock_subprocess():
    """mock subprocess.run: 返回 returncode=0, 记录调用序列。"""
    proc = mock.Mock()
    proc.returncode = 0
    with mock.patch.object(prep_sample.subprocess, "run", return_value=proc) as m:
        yield m


@pytest.fixture
def session_dir(tmp_path) -> str:
    return str(tmp_path / "runs" / "20260820-150000-prep-test")


def _real_scripts_ok():
    """前置断言: 编排链引用的 sub-skill 脚本真实存在(防路径漂移)。

    不存在时抛 AssertionError 而非静默失败, 使测试保真。
    """
    for p in (
        prep_sample._FC_SCRIPTS / "classify_features.py",
        prep_sample._FC_SCRIPTS / "finalize_feature_list.py",
        prep_sample._DC_SCRIPTS / "clean_data.py",
        prep_sample._CDA_SCRIPTS / "feature_analysis.py",
    ):
        assert p.is_file(), f"sub-skill 脚本缺失: {p}"


def _call_args(run_mock, index: int) -> list:
    """取第 index 次 subprocess.run 调用的位置参数列表(去掉 python 与脚本路径后的 args)。"""
    call = run_mock.call_args_list[index]
    return list(call.args[0][2:])  # [python, script, ...args]


def test_clean_pipeline_chain_order(session_dir):
    """clean: classify_features → finalize_feature_list → clean_data, 参数契约正确。"""
    _real_scripts_ok()
    exclude = ["fser_date", "sx_order_id", "ftrans_time"]
    keep = ["flag_ok"]

    prep_sample.clean_pipeline(
        session_dir=session_dir, input_path="data.csv",
        id_col="fuid", dt_col="ftrans_date", label_col="fpd7_sx30",
        exclude=exclude, keep=keep,
    )

    assert len(prep_sample.subprocess.run.call_args_list) == 3

    # 1) classify_features 探查: --out-dir sample-features, id/dt/label 透传
    c1 = _call_args(prep_sample.subprocess.run, 0)
    assert "classify_features.py" in str(prep_sample.subprocess.run.call_args_list[0].args[0][1])
    assert "--out-dir" in c1
    assert f"{session_dir}/sample-features" in c1
    assert "--id-col" in c1 and "fuid" in c1
    assert "--dt-col" in c1 and "ftrans_date" in c1
    assert "--label-col" in c1 and "fpd7_sx30" in c1

    # 2) finalize: --classification/--out-dir/--exclude/--keep
    c2 = _call_args(prep_sample.subprocess.run, 1)
    assert "finalize_feature_list.py" in str(prep_sample.subprocess.run.call_args_list[1].args[0][1])
    assert f"{session_dir}/sample-features/feature-classification.json" in c2
    assert ",".join(exclude) in c2
    assert ",".join(keep) in c2

    # 3) clean_data: --feature-list-source 权威清单 + --auto-confirm
    c3 = _call_args(prep_sample.subprocess.run, 2)
    assert "clean_data.py" in str(prep_sample.subprocess.run.call_args_list[2].args[0][1])
    assert f"{session_dir}/sample-features/feature-list.csv" in c3
    assert "--auto-confirm" in c3
    assert "--id-col" in c3 and "fuid" in c3


def test_clean_pipeline_without_keep(session_dir):
    """未传 keep 时 finalize 不追加 --keep。"""
    _real_scripts_ok()
    prep_sample.clean_pipeline(
        session_dir=session_dir, input_path="data.csv",
        id_col="fuid", dt_col="ftrans_date", label_col="fpd7_sx30",
        exclude=["fser_date"],
    )
    c2 = _call_args(prep_sample.subprocess.run, 1)
    assert "--keep" not in c2


def test_analyze_pipeline_appends_analysis(session_dir):
    """analyze: clean 三步 + feature_analysis 第四步, 分析消费清洗后 sample + 权威清单。"""
    _real_scripts_ok()
    prep_sample.analyze_pipeline(
        session_dir=session_dir, input_path="data.csv",
        id_col="fuid", dt_col="ftrans_date", label_col="fpd7_sx30",
        exclude=["fser_date"], base_month="2025-04",
    )
    calls = prep_sample.subprocess.run.call_args_list
    assert len(calls) == 4

    c4 = _call_args(prep_sample.subprocess.run, 3)
    assert "feature_analysis.py" in str(calls[3].args[0][1])
    assert f"{session_dir}/sample-features/data-cleaning/sample.parquet" in c4  # 清洗产物
    assert f"{session_dir}/sample-features/feature-list.csv" in c4              # 权威清单
    assert "--time-col" in c4 and "ftrans_date" in c4
    assert "--iv-label" in c4 and "fpd7_sx30" in c4
    assert "--base-month" in c4 and "2025-04" in c4
    assert f"{session_dir}/sample-features/credit-data-analysis" in c4


def test_analyze_without_base_month(session_dir):
    """analyze 未传 base_month 时不追加 --base-month(脚本用默认基准月)。"""
    _real_scripts_ok()
    prep_sample.analyze_pipeline(
        session_dir=session_dir, input_path="data.csv",
        id_col="fuid", dt_col="ftrans_date", label_col="fpd7_sx30",
        exclude=["fser_date"],
    )
    c4 = _call_args(prep_sample.subprocess.run, 3)
    assert "--base-month" not in c4


def test_nonzero_returncode_raises(session_dir):
    """sub-skill 非零返回码 → RuntimeError, 不吞错。"""
    _real_scripts_ok()
    proc = mock.Mock()
    proc.returncode = 2
    with mock.patch.object(prep_sample.subprocess, "run", return_value=proc):
        with pytest.raises(RuntimeError, match="exit=2"):
            prep_sample.clean_pipeline(
                session_dir=session_dir, input_path="data.csv",
                id_col="fuid", dt_col="ftrans_date", label_col="fpd7_sx30",
                exclude=["fser_date"],
            )


def test_ensure_session_dir_auto_create(tmp_path, monkeypatch):
    """--session-dir 缺省 → 自动建 runs/{ts}-prep-*/ 并返回绝对路径。"""
    monkeypatch.chdir(tmp_path)
    d = prep_sample.ensure_session_dir(None)
    p = Path(d)
    assert p.is_dir()
    assert p.parent.name == "runs"
    assert p.name.endswith("prep")
    assert p.name.split("-")[0].isdigit()  # 时间戳前缀


def test_ensure_session_dir_explicit():
    """显示 --session-dir → 原样返回并建目录。"""
    with mock.patch("pathlib.Path.mkdir") as mk:
        d = prep_sample.ensure_session_dir("my/session")
    assert d == "my/session"
    mk.assert_called_once()