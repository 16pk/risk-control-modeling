# -*- coding: utf-8 -*-
"""build_sample_feature_sql t-1 滞后 JOIN 单测: SQL 字符串断言。

覆盖 lag=0/1 两种模式的 ON 子句 + 特征表窗口, 以及边界异常。
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

from fetch_spark import build_sample_feature_sql


def test_lag_zero_on_clause_same_day():
    """lag=0: ON 子句为 a.user_no=b.user_no AND a.pday=b.pday, 特征窗口 = 样本窗口。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0", "f1"],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=0,
    )
    assert "a.user_no=b.user_no" in sql
    assert "a.pday=b.pday" in sql
    # 特征表子查询窗口 = 样本窗口
    assert "FROM db.feat WHERE pday >= '20260101' AND pday <= '20260131'" in sql
    # 样本表子查询窗口
    assert "FROM db.sample WHERE pday >= '20260101' AND pday <= '20260131'" in sql
    # 不应出现日期算术
    assert "date_add" not in sql
    assert "date_format" not in sql


def test_lag_one_on_clause_date_arithmetic():
    """lag=1: ON 子句用日期算术对齐 a.pday = date_format(date_add(b.pday,1)),
    特征窗口 = 样本窗口 - 1 天。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0", "f1"],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=1,
    )
    # ON 子句: a.user_no=b.user_no + 日期算术 (无 a.pday=b.pday)
    assert "a.user_no=b.user_no" in sql
    assert "a.pday=b.pday" not in sql
    assert "a.pday = date_format(date_add(to_date(b.pday, 'yyyyMMdd'), 1), 'yyyyMMdd')" in sql
    # 特征表窗口平移 -1 天: 20251231 ~ 20260130
    assert "FROM db.feat WHERE pday >= '20251231' AND pday <= '20260130'" in sql
    # 样本表窗口不变
    assert "FROM db.sample WHERE pday >= '20260101' AND pday <= '20260131'" in sql


def test_lag_one_cross_year_boundary():
    """跨年边界: fetch_start=20260101 → feat_start=20251231 (datetime 跨年处理)。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0"],
        fetch_start="20260101", fetch_end="20260105", where=None,
        feature_lag_day=1,
    )
    assert "FROM db.feat WHERE pday >= '20251231' AND pday <= '20260104'" in sql


def test_lag_one_full_column_except_includes_dt_col():
    """lag=1 + 全列模式 (features=[]): b.* EXCEPT 子句含 dt_col, 避免重复列。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=[],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=1,
    )
    # lag=1: EXCEPT 子句需含 dt_col (因 b.pday 不再等值匹配 a.pday)
    assert "b.* EXCEPT (user_no, pday)" in sql


def test_lag_zero_full_column_except_excludes_dt_col():
    """lag=0 + 全列模式: EXCEPT 子句只含 join_keys (含 dt_col=pday, 仍出现在 EXCEPT 中)。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=[],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=0,
    )
    # lag=0: join_keys=user_no,pday, EXCEPT = (user_no, pday)
    assert "b.* EXCEPT (user_no, pday)" in sql


def test_lag_one_without_dt_col_in_join_keys_raises():
    """lag=1 + join_keys 只含单 ID(缺 dt_col): 样本集 JOIN 红线段先行拦下。

    此输入同时违反 lag=1 的「dt∈keys」前置与【ID+日期】双键红线,
    新版在 build_sample_feature_sql 入口由 validate_join_keys 统一硬拦(reason 为缺失日期分区列)。
    """
    with pytest.raises(ValueError, match="缺失日期分区列"):
        build_sample_feature_sql(
            sample_table="db.sample", feature_table="db.feat",
            join_keys=["user_no"], dt_col="pday",
            label_expr="label", id_cols=["user_no"], features=["f0"],
            fetch_start="20260101", fetch_end="20260131", where=None,
            feature_lag_day=1,
        )


def test_lag_two_raises():
    """lag=2 不支持: 仅 0/1, 抛 ValueError。"""
    with pytest.raises(ValueError, match="feature_lag_day 仅支持 0"):
        build_sample_feature_sql(
            sample_table="db.sample", feature_table="db.feat",
            join_keys=["user_no", "pday"], dt_col="pday",
            label_expr="label", id_cols=["user_no"], features=["f0"],
            fetch_start="20260101", fetch_end="20260131", where=None,
            feature_lag_day=2,
        )


def test_lag_negative_raises():
    """lag 负数: 仅 0/1, 抛 ValueError。"""
    with pytest.raises(ValueError, match="feature_lag_day 仅支持 0"):
        build_sample_feature_sql(
            sample_table="db.sample", feature_table="db.feat",
            join_keys=["user_no", "pday"], dt_col="pday",
            label_expr="label", id_cols=["user_no"], features=["f0"],
            fetch_start="20260101", fetch_end="20260131", where=None,
            feature_lag_day=-1,
        )


def test_lag_one_specified_features_keeps_dt_in_b_cols_and_outer():
    """lag=1 + 指定 features: b_cols 保留 pday(lag 日期对齐所需), 外层投影只挑 a.* + b.{f0,f1}。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0", "f1"],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=1,
    )
    # 特征表子查询 b_cols = user_no, pday, f0, f1:
    #   - pday 必须保留: lag=1 JOIN ON 需引用 b.pday 做 a.pday = date_add(b.pday,+1) 日期对齐;
    #     同时外部 window 过滤也基于 b.pday —— 这是既有正确行为(见 fetch_spark.py L128-139)。
    feat_sub_idx = sql.find("FROM db.feat")
    assert feat_sub_idx >= 0
    select_idx = sql.rfind("SELECT ", 0, feat_sub_idx)
    feat_select_cols = sql[select_idx:feat_sub_idx]
    for col in ("user_no", "pday", "f0", "f1"):
        assert col in feat_select_cols
    # JOIN ON 使用日期算术对齐而非等值(pday 不进 join_on 等值比较)
    assert "a.pday=b.pday" not in sql
    assert "date_format(date_add(to_date(b.pday, 'yyyyMMdd'), 1), 'yyyyMMdd')" in sql
    # 外层 SELECT 只出现一个 pday 列(a.pday), 未因 lag=1 产生 b.pday/b.* EXCEPT 的二义重复投影;
    # b.* EXCEPT 全列模式才需要显式排除 pday(该分支由既有 test_lag_one_all_columns_except_dt 覆盖)
    outer_select = sql[:sql.find(" FROM (SELECT")]
    assert outer_select.count("pday") == 1
