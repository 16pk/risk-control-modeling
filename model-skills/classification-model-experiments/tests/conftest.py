# -*- coding: utf-8 -*-
"""测试公共夹具：注入本 skill scripts 与 _modelevo-shared 路径。"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # tests/
_SCRIPTS = _HERE.parent / "scripts"              # scripts/
_SKILLS_ROOT = _HERE.parents[2]                  # model-skills/
_SHARED = _SKILLS_ROOT / "_modelevo-shared" / "scripts"

for p in (str(_SCRIPTS), str(_SHARED)):
    if p not in sys.path:
        sys.path.insert(0, p)
