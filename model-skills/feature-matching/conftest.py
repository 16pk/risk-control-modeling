# -*- coding: utf-8 -*-
"""pytest 收集辅助:_modelevo-shared/scripts 稳定注入 sys.path。

在安装形态(.claude/skills/…)与仓库源(model-evo/model-skills)两种布局下,
tests/*.py 里的 `from fetch_spark / config_io import …` 都能解析到共享脚本。
等价于 _bootstrap.py 的注入逻辑,但只做 sys.path(不在测试进程内切 CWD)。
"""
import sys
from pathlib import Path

_SHARED = (Path(__file__).resolve().parent.parent.parent / "_modelevo-shared" / "scripts").resolve()
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))