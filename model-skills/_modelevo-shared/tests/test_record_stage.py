# -*- coding: utf-8 -*-
"""record_stage.py 单测: 阶段脚本快照记录工具。"""
import json
import os
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from record_stage import record_stage  # noqa: E402


@pytest.fixture()
def fake_script(tmp_path):
    """生成一个假的入口脚本, 返回其绝对路径。"""
    p = tmp_path / "run_experiments.py"
    p.write_text("# fake entry script\nprint('hi')\n", encoding="utf-8")
    return str(p)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- 正常落盘 ----

def test_creates_dir_and_files(tmp_path, fake_script):
    """记录后应生成 scripts/<stage>/<脚本名>.py、command.json 与 _manifest.json。"""
    session = tmp_path / "session"
    entry = record_stage(str(session), "experiments", fake_script, "python run_experiments.py --version v1")
    assert entry is not None
    assert (session / "scripts" / "experiments" / "run_experiments.py").exists()
    assert (session / "scripts" / "experiments" / "command.json").exists()
    assert (session / "scripts" / "_manifest.json").exists()
    # 快照内容与源一致
    assert (session / "scripts" / "experiments" / "run_experiments.py").read_text(encoding="utf-8") \
        == "# fake entry script\nprint('hi')\n"


def test_command_json_fields(tmp_path, fake_script):
    """command.json 应包含核心字段: stage/cmd/timestamp/script_path/snapshot/sha256/python。"""
    session = tmp_path / "session"
    record_stage(str(session), "experiments", fake_script, "python run_experiments.py --version v1",
                 label="Stage4 实验矩阵")
    cmd_json = _read_json(session / "scripts" / "experiments" / "command.json")
    assert cmd_json["stage"] == "experiments"
    assert cmd_json["cmd"] == "python run_experiments.py --version v1"
    assert cmd_json["label"] == "Stage4 实验矩阵"
    assert cmd_json["script_path"] == os.path.abspath(fake_script)
    assert cmd_json["snapshot"] == "experiments/run_experiments.py"
    assert cmd_json["timestamp"]
    assert len(cmd_json["sha256"]) == 64  # sha256 hex
    assert cmd_json["python"]


def test_manifest_merges_multiple_stages(tmp_path, fake_script):
    """多阶段记录后, _manifest.json 应含各 stage 条目(集中可查)。"""
    session = tmp_path / "session"
    record_stage(str(session), "experiments", fake_script, "python run_experiments.py")
    other = tmp_path / "tune_winner.py"
    other.write_text("print('tune')\n", encoding="utf-8")
    record_stage(str(session), "tune", str(other), "python tune_winner.py")
    manifest = _read_json(session / "scripts" / "_manifest.json")
    assert set(manifest["stages"].keys()) == {"experiments", "tune"}
    assert manifest["stages"]["experiments"][0]["snapshot"] == "experiments/run_experiments.py"
    assert manifest["stages"]["tune"][0]["snapshot"] == "tune/tune_winner.py"


def test_same_script_idempotent_override(tmp_path, fake_script):
    """同名脚本重复记录: 快照被覆盖为最新内容, manifest 条目更新而非无限追加。"""
    session = tmp_path / "session"
    record_stage(str(session), "experiments", fake_script, "cmd v1")
    # 修改源脚本内容后再次记录
    with open(fake_script, "w", encoding="utf-8") as f:
        f.write("# v2\nprint('new')\n")
    record_stage(str(session), "experiments", fake_script, "cmd v2")
    snap = session / "scripts" / "experiments" / "run_experiments.py"
    assert snap.read_text(encoding="utf-8") == "# v2\nprint('new')\n"
    manifest = _read_json(session / "scripts" / "_manifest.json")
    assert len(manifest["stages"]["experiments"]) == 1  # 覆盖旧条目, 不追加
    assert manifest["stages"]["experiments"][0]["cmd"] == "cmd v2"


def test_same_stage_different_scripts_appends(tmp_path, fake_script):
    """同一 stage 不同脚本 → manifest 中该 stage 追加为新条目。"""
    session = tmp_path / "session"
    record_stage(str(session), "tune", fake_script, "cmd a")
    another = tmp_path / "diagnose_winner.py"
    another.write_text("print('select')\n", encoding="utf-8")
    record_stage(str(session), "tune", str(another), "cmd b")
    manifest = _read_json(session / "scripts" / "_manifest.json")
    assert len(manifest["stages"]["tune"]) == 2


def test_session_dir_auto_created(tmp_path, fake_script):
    """session_dir 不存在时自动创建。"""
    session = tmp_path / "nested" / "deep" / "session"
    entry = record_stage(str(session), "fico", fake_script, "python score_to_fico.py")
    assert entry is not None
    assert (session / "scripts" / "fico" / "command.json").exists()


# ---- 错误分支 ----

def test_invalid_stage_name(tmp_path, fake_script):
    """stage 含路径分隔符 / .. 等必须拒绝(防路径穿越)。"""
    for bad in ("../x", "a/b", "..", ""):
        with pytest.raises(ValueError, match="stage"):
            record_stage(str(tmp_path / "s"), bad, fake_script, "cmd")


def test_script_missing_raises(tmp_path):
    """入口脚本不存在 → ValueError。"""
    with pytest.raises(ValueError, match="不存在"):
        record_stage(str(tmp_path / "s"), "experiments", str(tmp_path / "nope.py"), "cmd")


def test_non_py_script_raises(tmp_path):
    """入口脚本非 .py → ValueError。"""
    p = tmp_path / "run.sh"
    p.write_text("#!/bin/bash\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".py"):
        record_stage(str(tmp_path / "s"), "experiments", str(p), "cmd")


def test_unwritable_session_warns_not_blocking(tmp_path, fake_script):
    """session_dir 不可写 → 打印 warning 返回 None, 不抛错(不阻断建模)。"""
    session = tmp_path / "ro_session"
    session.mkdir()
    os.chmod(session, 0o444)
    try:
        entry = record_stage(str(session), "experiments", fake_script, "cmd")
        assert entry is None
    finally:
        os.chmod(session, 0o755)
