# -*- coding: utf-8 -*-
"""model-evo/_modelevo-shared 公共日期工具: 双格式(YYYY-MM-DD / YYYYMMDD)兼容与归一化。

背景(2026-08 需求优化):
  - 默认日期格式统一为 YYYY-MM-DD(用户可读, 非分区硬编码场景);
  - 同时自动识别兼容旧 8 位 YYYYMMDD(向后兼容, Hive 分区/存量配置仍可能是 YYYYMMDD);
  - 内部比较(字符串/整数序)统一用归一化后的 8 位 YYYYMMDD, 保证 start<=p<=end 语义正确。

本模块提供单一真相, 供 config_io / task-spec /
feature-analysis / experiments 等多处消费。
"""
from __future__ import annotations

import datetime as _dt

# 两种受支持的输入格式
_FORMATS = ("%Y-%m-%d", "%Y%m%d")


def parse_date(d, *, what="date"):
    """校验并归一化单个日期字符串, 接受 YYYY-MM-DD 或 8 位 YYYYMMDD。

    先按规范格式字符串解析(YYYY-MM-DD 优先), 再回退 8 位; 二者皆失败即报错。
    返回归一化后的 8 位字符串(YYYYMMDD), 用于内部统一比较。

    Args:
        d: 日期字符串或可 str() 的对象(如 int 20260312)
        what: 报错文案里的语义标签(如 "split.train_range")

    Returns:
        归一化 8 位 YYYYMMDD 字符串

    Raises:
        ValueError: 非 YYYY-MM-DD 亦非 YYYYMMDD
    """
    s = str(d).strip()
    if not s:
        raise ValueError("%s 日期为空" % what)
    # strptime 的 %d/%m 允许 1 位数字, 需按期望长度严格回代, 避免 "2026-3-5"/"2026031" 被放行
    expected_len = {"%Y-%m-%d": 10, "%Y%m%d": 8}
    for fmt in _FORMATS:
        if len(s) != expected_len[fmt]:
            continue
        try:
            _dt.datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
        if fmt == "%Y-%m-%d":
            return s.replace("-", "")
        return s
    raise ValueError(
        "%s 日期格式不合法: %r, 须为 YYYY-MM-DD 或 8 位 YYYYMMDD" % (what, s)
    )


def parse_date_pair(value, *, what="range"):
    """把 [起, 止] 或 '起,止' 规整为归一化后的 (start, end) 两 8 位日期。

    Args:
        value: 形如 ["2026-03-12", "20260430"] 或 "2026-03-12,20260430"
        what: 报错文案语义标签

    Returns:
        (start, end) 归一化 8 位 YYYYMMDD

    Raises:
        ValueError: 非两元素 / 日期不合法 / 起 > 止
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        raise ValueError("%s 须为 [起, 止] 列表或 '起,止' 字符串" % what)
    if len(parts) != 2:
        raise ValueError("%s 须为两元素 [起, 止], 当前 %r" % (what, value))
    start = parse_date(parts[0], what=what + ".start")
    end = parse_date(parts[1], what=what + ".end")
    if start > end:
        raise ValueError("%s 起始 %s 不应大于结束 %s" % (what, start, end))
    return start, end


def month_prefix(d):
    """返回月份前缀(YYYYMM), 兼容 YYYY-MM-DD 与 YYYYMMDD 两种输入。

    归一化后统一取前 6 位, 避免 YYYY-MM-DD 取前 6 位得到 "2026-0" 的经典坑。

    Args:
        d: 日期字符串(YYYY-MM-DD 或 YYYYMMDD)

    Returns:
        6 位 YYYYMM
    """
    return normalize_date(d)[:6]


def normalize_date(d):
    """内部比较用归一化: 接受双格式, 返回 8 位 YYYYMMDD。

    Args:
        d: 日期字符串(YYYY-MM-DD 或 YYYYMMDD)

    Returns:
        归一化 8 位 YYYYMMDD
    """
    return parse_date(d, what="normalize")


def is_date(d):
    """宽松判断是否为受支持的日期格式(不抛异常)。"""
    if not d:
        return False
    try:
        parse_date(d, what="check")
        return True
    except ValueError:
        return False


def hive_date_str(d):
    """生成 Hive SQL 里 to_date/date_format 用的 'yyyy-MM-dd' 格式串。

    Hive 侧统一按 yyyy-MM-dd 规范化(对双格式输入都成立, to_date(col,'yyyy-MM-dd')
    对 '20260312' 这类 8 位输入需配合 date_format 先转), 返回标准 yyyy-MM-dd。

    Args:
        d: 日期字符串(YYYY-MM-DD 或 YYYYMMDD)

    Returns:
        yyyy-MM-dd 字符串
    """
    norm = normalize_date(d)
    return "%s-%s-%s" % (norm[:4], norm[4:6], norm[6:8])


def shift_days(d, delta):
    """日期加减 delta 天, 返回归一化 8 位 YYYYMMDD(兼容双格式输入)。

    Args:
        d: 日期字符串(YYYY-MM-DD 或 YYYYMMDD)
        delta: 天数增量(可负)

    Returns:
        归一化 8 位 YYYYMMDD
    """
    norm = normalize_date(d)
    dt = _dt.datetime.strptime(norm, "%Y%m%d") + _dt.timedelta(days=int(delta))
    return dt.strftime("%Y%m%d")


def today_canonical():
    """返回今天的归一化 8 位 YYYYMMDD(用于默认文件名等)。"""
    return _dt.datetime.now().strftime("%Y%m%d")
