#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline 阶段脚本快照记录工具(共享)。

在分类建模 pipeline 中, 每个阶段的 python CLI 执行完成后, 由编排 agent 调用本工具,
把「实际执行命令 + 入口脚本源码快照」落盘到任务目录 <session_dir>/scripts/<stage>/,
便于结果可复现、可追溯 —— 即使 skill 代码后续演进, 也能还原当时跑的代码与调用方式。

落盘结构:
    <session_dir>/scripts/
    ├── _manifest.json          # 按 stage 汇总所有已记录命令(集中可查)
    └── <stage>/
        ├── <入口脚本名>.py     # 源码快照(原样复制, 可读可 diff)
        └── command.json        # 本阶段执行命令详情(cmd/timestamp/sha256/python)

用法:
    python record_stage.py --session-dir runs/20260812-model_a \
        --stage experiments \
        --script model-skills/classification-model-experiments/scripts/run_experiments.py \
        --cmd "python run_experiments.py --session-dir runs/20260812-model_a --until promote" \
        [--label "Stage4 实验矩阵"]

安全与容错:
- 仅记录命令字符串与脚本源码, 不触碰样本数据;
- stage 名仅允许小写字母/数字/连字符(防路径穿越);
- 落盘失败(如 session_dir 不可写)仅打印 warning 并返回 None, 不阻断建模主流程;
- 参数/脚本路径错误属使用错误, 直接抛 ValueError(退出码 1), 提示 agent 修正。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime

MANIFEST_VERSION = 1
MANIFEST_NAME = "_manifest.json"

# stage 名白名单: 仅允许小写字母/数字/连字符(防路径穿越)
_STAGE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _sha256(path: str) -> str:
    """计算文件 sha256(分块读, 兼容大文件)。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, obj: dict) -> None:
    """写 JSON(先写临时文件再 rename, 防半截文件)。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def record_stage(session_dir: str, stage: str, script: str, cmd: str,
                 label: str = None) -> dict | None:
    """把某阶段执行命令与入口脚本源码快照落盘到 <session_dir>/scripts/。

    Args:
        session_dir: 任务目录(如 runs/20260812-model_a)
        stage: 阶段名(task-spec/data-cleaning/experiments/
               fico/fill_report 等, 仅小写字母/数字/连字符; v2.1 起不再被编排调用)
        script: 被执行的入口脚本路径(.py)
        cmd: 实际执行的完整命令行(含全部参数)
        label: 阶段说明(可选)

    Returns:
        dict: 落盘记录(含 snapshot/command_path/manifest_path 等), 落盘 IO 失败时 None

    Raises:
        ValueError: stage 非法 / script 不存在 / 非 .py 文件(使用错误, 调用方需修正)
    """
    if not _STAGE_RE.match(stage or ""):
        raise ValueError(
            "stage 名不合法: %r, 仅允许小写字母/数字/连字符(如 training / task-spec)" % stage
        )
    script = os.path.abspath(script)
    if not os.path.isfile(script):
        raise ValueError("入口脚本不存在: %s" % script)
    if not script.endswith(".py"):
        raise ValueError("入口脚本须为 .py 文件: %s" % script)

    try:
        return _do_record(session_dir, stage, script, cmd, label)
    except OSError as e:
        # 落盘失败(如 session_dir 不可写)仅 warn 不阻断建模主流程
        print("[record_stage][warn] 落盘失败, 不阻断建模: %s" % e, file=sys.stderr)
        return None


def _do_record(session_dir: str, stage: str, script: str, cmd: str,
               label: str) -> dict:
    """record_stage 的落盘实现(IO 异常向上抛给调用方统一 warn)。"""
    scripts_root = os.path.join(session_dir, "scripts")
    stage_dir = os.path.join(scripts_root, stage)
    os.makedirs(stage_dir, exist_ok=True)

    snapshot_name = os.path.basename(script)
    snapshot_path = os.path.join(stage_dir, snapshot_name)

    # 复制源码快照(覆盖旧快照, 幂等)
    with open(script, "r", encoding="utf-8") as src, \
            open(snapshot_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())

    entry = {
        "stage": stage,
        "label": label,
        "cmd": cmd,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_path": script,
        "snapshot": "%s/%s" % (stage, snapshot_name),
        "sha256": _sha256(snapshot_path),
        "python": sys.version.split()[0],
    }

    _write_json(os.path.join(stage_dir, "command.json"), entry)

    # 合并写 _manifest.json: 按 stage 追加; 同名脚本(同 snapshot)重复执行时
    # 更新为最新一次, 反映当时实际跑的代码
    manifest_path = os.path.join(scripts_root, MANIFEST_NAME)
    manifest: dict = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest.setdefault("version", MANIFEST_VERSION)
    stages = manifest.setdefault("stages", {})
    stages.setdefault(stage, [])
    stages[stage] = [e for e in stages[stage] if e.get("snapshot") != entry["snapshot"]]
    stages[stage].append(entry)
    _write_json(manifest_path, manifest)

    entry["command_path"] = os.path.join(stage_dir, "command.json")
    entry["manifest_path"] = manifest_path
    return entry


def main() -> int:
    """CLI 入口: 解析参数 → record_stage → 打印结果。"""
    parser = argparse.ArgumentParser(description="pipeline 阶段脚本快照记录工具")
    parser.add_argument("--session-dir", required=True, help="任务目录(session_dir)")
    parser.add_argument(
        "--stage", required=True,
        help="阶段名: task-spec/data-cleaning/experiments/fico/fill_report 等",
    )
    parser.add_argument("--script", required=True, help="被执行的入口脚本绝对路径(.py)")
    parser.add_argument("--cmd", required=True, help="实际执行的完整命令行(含全部参数)")
    parser.add_argument("--label", default=None, help="阶段说明(可选)")
    args = parser.parse_args()

    try:
        entry = record_stage(args.session_dir, args.stage, args.script, args.cmd, args.label)
    except ValueError as e:
        print("[record_stage] 参数错误, 未记录: %s" % e, file=sys.stderr)
        return 1
    if entry is None:
        return 0  # 落盘失败已 warn, 不阻断
    print("[record_stage] 已记录 stage=%s -> %s" % (entry["stage"], entry["snapshot"]))
    print("[record_stage] 命令: %s" % entry["cmd"])
    print("[record_stage] 清单: %s" % entry["manifest_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
