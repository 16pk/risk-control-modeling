# -*- coding: utf-8 -*-
"""样本集 JOIN 红线单测 (ModelEvo-RED-0102)。

覆盖 _modelevo-shared/fetch_spark.validate_join_keys:
  - join keys 必须 = [ID类键 + 日期分区列], 缺一不可
  - 仅以单 ID(fuid/user_no)为 key → raise ValueError(硬拦截)
  - 空 keys / 全日期列无 ID → raise ValueError
  - 合法双键通过; dt_col 显式换名(f_p_date)时须纳入 keys 才放行
"""
import os
import sys
from pathlib import Path

import pytest


def _insert_shared():
    """把 model-skills/_modelevo-shared/scripts 稳定注入 sys.path。

    兼容安装形态(.claude/skills/)与仓库源(model-evo/model-skills)两种布局:
    从本测试文件逐级向上探测祖先目录,取第一个含 _modelevo-shared/scripts/
    (带 config_io.py + fetch_spark.py)者为共享脚本根, 不依赖 CWD / parents[N]。
    """
    here = Path(os.path.abspath(__file__)).resolve().parent  # .../<skill>/tests
    for ancestor in [here, *here.parents]:
        cand = ancestor / "_modelevo-shared" / "scripts"
        if (cand / "config_io.py").exists() and (cand / "fetch_spark.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return True
    raise RuntimeError("未定位到 model-skills/_modelevo-shared/scripts")


_insert_shared()

from fetch_spark import validate_join_keys  # noqa: E402


def test_valid_id_and_date_pass():
    """标准双键 [user_no(=fuid), pday] 应通过。"""
    assert validate_join_keys(["user_no", "pday"], "pday") is None


def test_single_id_key_rejected():
    """仅以单个用户 ID(user_no) 作唯一连接键 → 必抛 ValueError。"""
    with pytest.raises(ValueError, match="缺失日期分区列"):
        validate_join_keys(["user_no"], "pday")


def test_single_dt_only_rejected():
    """keys=[dt_col] 没有 ID 类键 → 必抛 ValueError。"""
    with pytest.raises(ValueError, match="缺少 ID 类键"):
        validate_join_keys(["pday"], "pday")


def test_empty_keys_rejected():
    """空 join keys → 必抛 ValueError。"""
    with pytest.raises(ValueError, match="不能为空"):
        validate_join_keys([], "pday")


def test_non_default_date_col_must_be_in_keys():
    """日期列实名为 f_p_date(非默认 pday):只给 user_no 仍违反红线;
    把 f_p_date 显式放进 join_keys 则放行(容错约定:不自动猜列名)。"""
    with pytest.raises(ValueError, match="缺失日期分区列.*f_p_date"):
        validate_join_keys(["user_no"], "f_p_date")
    # 显式含 f_p_date 双键 => 合规
    assert validate_join_keys(["user_no", "f_p_date"], "f_p_date") is None


def test_lag_mode_without_dt_key_still_rejected_via_builder():
    """回归护栏: build_sample_feature_sql 在 lag=0/1 两条路径都强制走红线段校验,
    即使调用方试图只传单 ID join_keys 也会被拦下(lag=1 还额外要求 dt_col ∈ keys)。"""
    from fetch_spark import build_sample_feature_sql

    kwargs = dict(
        sample_table="db.sample", feature_table="db.feat",
        label_expr="label", id_cols=["user_no"], features=["f0"],
        fetch_start="20260101", fetch_end="20260131", where=None,
    )
    # lag=0, 仅单 ID → 拒绝
    with pytest.raises(ValueError, match="缺失日期分区列"):
        build_sample_feature_sql(dt_col="pday", join_keys=["user_no"], feature_lag_day=0, **kwargs)
    # lag=1, 仅单 ID → 同样被拦(lag 分支的 dt∈keys 前置断言之外,红线段先触发)
    with pytest.raises(ValueError):
        build_sample_feature_sql(dt_col="pday", join_keys=["user_no"], feature_lag_day=1, **kwargs)