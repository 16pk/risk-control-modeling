# -*- coding: utf-8 -*-
"""config_io 通用校验单测。"""
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import pandas as pd

from config_io import (
    LOCAL_BYTES_LIMIT,
    BYTES_ROUTING_SCHEMA_VERSION,
    estimate_size_bytes,
    route_by_bytes,
    load_config,
    validate_common,
    validate_split_ranges,
)


def _base_cfg():
    """返回一份合法的最小通用配置字典(name/sample_table/dt_col/label/fetch_dt/features)。"""
    return {
        "spark": {"app_name": "t", "master": "local[*]"},
        "model": {
            "name": "m", "sample_table": "db.t",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "fetch_dt": ["20250101", "20250131"], "where": None,
            "features": ["f0", "f1"],
        },
    }


def test_validate_ok():
    """合法配置应通过校验。"""
    validate_common(_base_cfg())


def test_validate_empty_features_allowed_local():
    """local_file 语义: features 为空放行(视为用本地 parquet 全部列, 除 id/label/dt)。"""
    cfg = _base_cfg()
    cfg["model"]["features"] = []
    validate_common(cfg)


def test_validate_missing_label():
    """label_col 与 label_expr 都缺必须报错。"""
    cfg = _base_cfg()
    cfg["model"].pop("label_col")
    with pytest.raises(ValueError, match="label"):
        validate_common(cfg)


def test_validate_rejects_hardcoded_id_in_where():
    """where 里出现疑似身份证/手机号必须报错(数据安全红线)。"""
    cfg = _base_cfg()
    cfg["model"]["where"] = "id_card='110101199001011234'"
    with pytest.raises(ValueError, match="安全红线|敏感"):
        validate_common(cfg)


def test_load_config_reads_yaml(tmp_path):
    """load_config 读 yaml 并返回 dict。"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_base_cfg(), allow_unicode=True), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["model"]["name"] == "m"


def test_validate_rejects_hardcoded_phone_in_where():
    """where 里出现疑似手机号必须报错(数据安全红线)。"""
    cfg = _base_cfg()
    cfg["model"]["where"] = "mobile='13800138000'"
    with pytest.raises(ValueError, match="安全红线|敏感"):
        validate_common(cfg)


def test_validate_rejects_sensitive_in_sample_table():
    """sample_table 里出现疑似敏感信息也必须报错(校验覆盖 where 与 sample_table)。"""
    cfg = _base_cfg()
    cfg["model"]["sample_table"] = "db.13800138000"
    with pytest.raises(ValueError, match="安全红线|敏感"):
        validate_common(cfg)


def test_validate_bad_fetch_dt_shape():
    """fetch_dt 非两元素列表必须报错。"""
    cfg = _base_cfg()
    cfg["model"]["fetch_dt"] = ["20250101"]
    with pytest.raises(ValueError, match="fetch_dt"):
        validate_common(cfg)


def test_validate_features_file(tmp_path):
    """features_file 指向外部文件时,应加载为 features 列表。"""
    feats_path = tmp_path / "feats.txt"
    feats_path.write_text("a\nb\nc\n", encoding="utf-8")
    cfg = _base_cfg()
    cfg["model"].pop("features")
    cfg["model"]["features_file"] = str(feats_path)
    validate_common(cfg)
    assert cfg["model"]["features"] == ["a", "b", "c"]


# ---- validate_split_ranges: 可选 model.split 时间划分校验 ----

def _split_model():
    """返回含合法 model.split 的 model 段。"""
    return {
        "fetch_dt": ["20260312", "20260524"],
        "split": {
            "train_range": ["20260312", "20260430"],
            "test_range": ["20260501", "20260516"],
            "oot_range": ["20260517", "20260524"],
        },
    }


def test_split_ranges_none_noop():
    """无 model.split 时直接返回, 不报错。"""
    validate_split_ranges({"fetch_dt": ["20260312", "20260524"]})


def test_split_ranges_ok_list_and_str():
    """合法三档(列表)通过; 字符串 '起,止' 也支持。"""
    validate_split_ranges(_split_model())
    m = _split_model()
    m["split"]["train_range"] = "20260312,20260430"
    validate_split_ranges(m)


def test_split_ranges_missing_tier():
    """三档缺一报错。"""
    m = _split_model()
    m["split"].pop("oot_range")
    with pytest.raises(ValueError, match="oot_range"):
        validate_split_ranges(m)


def test_split_ranges_bad_date():
    """非法日期(非 YYYY-MM-DD 亦非 8 位 YYYYMMDD)报错。"""
    m = _split_model()
    m["split"]["train_range"] = ["2026031", "20260430"]
    with pytest.raises(ValueError, match="日期格式不合法|不合法"):
        validate_split_ranges(m)


def test_split_ranges_dual_format_dash():
    """YYYY-MM-DD 双格式输入通过, 内部归一化比较正确。"""
    m = _split_model()
    m["split"]["train_range"] = ["2026-03-12", "2026-04-30"]
    m["split"]["test_range"] = ["2026-05-01", "20260516"]
    m["split"]["oot_range"] = ["20260517", "2026-05-24"]
    validate_split_ranges(m)


def test_split_ranges_dual_format_mix_overlap():
    """混合双格式下重叠/逆序不再拦截(已删除时序递增强制校验)。"""
    m = _split_model()
    m["split"]["test_range"] = ["2026-04-30", "20260516"]  # 与 train 20260430 同日重叠
    validate_split_ranges(m)  # 不再报"重叠或逆序"


def test_split_ranges_exceed_fetch_dt_dual_format():
    """双格式下划分并集超出 fetch_dt 仍报错(归一化后比较)。"""
    m = _split_model()
    m["split"]["oot_range"] = ["2026-05-17", "20260601"]  # 超出 fetch_dt 末端
    with pytest.raises(ValueError, match="超出取数窗口"):
        validate_split_ranges(m)


def test_split_ranges_start_gt_end():
    """单档起 > 止报错。"""
    m = _split_model()
    m["split"]["test_range"] = ["20260516", "20260501"]
    with pytest.raises(ValueError, match="不应大于"):
        validate_split_ranges(m)


@pytest.mark.parametrize("test_range", [
    ["20260420", "20260516"],   # 与 train 重叠
    ["20260430", "20260516"],   # 与 train 同日(前档结束日=后档开始日)
])
def test_split_ranges_overlap_no_longer_rejected(test_range):
    """三档重叠/同日不再报错(已删除时序递增强制校验)。"""
    m = _split_model()
    m["split"]["test_range"] = test_range
    validate_split_ranges(m)  # 不再报"重叠或逆序"


def test_split_ranges_reverse_order_ok():
    """三档逆序(oot 早于 train)通过: 不强制时间递增。"""
    m = _split_model()
    m["split"]["train_range"] = ["20260517", "20260524"]
    m["split"]["test_range"] = ["20260501", "20260516"]
    m["split"]["oot_range"] = ["20260312", "20260430"]
    validate_split_ranges(m)


def test_split_ranges_adjacent_ok():
    """相邻间隔(前档结束日次日=后档开始日)允许通过。"""
    m = _split_model()
    # train 20260312~20260430, test 次日 20260501 起
    m["split"]["test_range"] = ["20260501", "20260516"]
    # test 20260501~20260516, oot 次日 20260517 起
    m["split"]["oot_range"] = ["20260517", "20260524"]
    validate_split_ranges(m)  # 不报错


def test_split_ranges_exceed_fetch_dt():
    """划分并集超出 fetch_dt 报错。"""
    m = _split_model()
    m["split"]["oot_range"] = ["20260517", "20260601"]  # 超出 fetch_dt 末端
    with pytest.raises(ValueError, match="超出取数窗口"):
        validate_split_ranges(m)


# ---- 计算资源路由: route_by_bytes / estimate_size_bytes (字节口径 EXP-G-004) ----

def test_route_none_is_local(capsys):
    """无法估计(None)→ 放行 local,不误杀。"""
    assert route_by_bytes(None, where="unit") == "local"
    out = capsys.readouterr().out
    assert "[compute-routing]" not in out


@pytest.mark.parametrize("size", [0, LOCAL_BYTES_LIMIT - 1])
def test_route_below_limit_local(size, capsys):
    """<1GB → local 且无告警。"""
    assert route_by_bytes(size, where="unit") == "local"
    assert "[compute-routing]" not in capsys.readouterr().out


@pytest.mark.parametrize("size", [LOCAL_BYTES_LIMIT, int(2e9)])
def test_route_at_or_over_limit_distributed(size, capsys):
    """>=1GB → distributed + 醒目 WARNING(含 bytes/limit/schema_v)。"""
    assert route_by_bytes(size, where="unit-test") == "distributed"
    out = capsys.readouterr().out
    assert "DISTRIBUTED" in out or "ray-distributed-train" in out
    assert str(LOCAL_BYTES_LIMIT) in out
    assert f"schema_v{BYTES_ROUTING_SCHEMA_VERSION}" in out


def test_route_limit_override():
    """自定义 limit 覆盖默认阈值。"""
    assert route_by_bytes(100, where="unit", limit=200) == "local"
    assert route_by_bytes(300, where="unit", limit=200) == "distributed"


def test_estimate_from_partition_total_size_preferred(tmp_path):
    """partition_total_size(经 MCP 实测的分区求和)优先于 df/path 探测。"""
    df = pd.DataFrame({"a": range(10), "b": list(range(10))})
    p = tmp_path / "tiny.parquet"
    df.to_parquet(p)
    # 即便存在本地小文件,也应以显式传入的 partition size 为准(Gate P0 首选来源)
    est = estimate_size_bytes(df=df, path=str(p), partition_total_size=int(5e9))
    assert est == int(5e9)


def test_estimate_df_deep_memory():
    """df= 时按 memory_usage(deep=True).sum() 估算。"""
    df = pd.DataFrame({"int_col": range(1000)})
    est = estimate_size_bytes(df=df)
    assert est is not None and est > 7000  # int64×1000 ≈ 8000B


def test_estimate_path_small_parquet_nonzero(tmp_path):
    """path= 指向真实 parquet → 返回有限正字节数(非 None)。"""
    df = pd.DataFrame({"a": range(25), "b": [float(i) for i in range(25)]})
    p = tmp_path / "s.parquet"
    df.to_parquet(p)
    est = estimate_size_bytes(path=str(p))
    assert est is not None and est > 40


def test_estimate_path_missing_is_none():
    """path 不存在/非法 → None(调用方记 reason=estimate_unavailable 后放行 local)。"""
    assert estimate_size_bytes(path="/nonexistent/nope.parquet") is None
    assert estimate_size_bytes(df=None, path=None, partition_total_size=None) is None


def test_legacy_routing_json_decode_stable(tmp_path):
    """存量 _routing.json {"route":"local"/"distributed"} 字段语义不变(downstream 兼容)。"""
    import json as _json
    for r in ("local", "distributed"):
        f = tmp_path / "_routing.json"
        f.write_text(_json.dumps({"route": r}), encoding="utf-8")
        loaded = _json.loads(f.read_text(encoding="utf-8"))
        assert loaded["route"] == r
        # downstream(experiments)对 route 的判断:
        distributed_flag = str((loaded or {}).get("route") or "local").lower() == "distributed"
        assert distributed_flag == (r == "distributed")


def test_constants_present():
    """关键常量暴露给 task-spec Gate P0 与三处消费引用。"""
    assert LOCAL_BYTES_LIMIT == int(1e9)
    assert BYTES_ROUTING_SCHEMA_VERSION >= 1



