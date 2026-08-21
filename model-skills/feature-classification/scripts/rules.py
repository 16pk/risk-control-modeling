# -*- coding: utf-8 -*-
"""feature-classification 规则库 v0(内置默认, 可配置扩展)。

语义三分类(特征列识别方案 §四):
    feature      默认类别(含匿名编码列中带业务词的列), 保留进特征清单
    non_feature  规则命中: 日期/时间戳/订单号/ID/序号/纯标识列/标签列(用户红线), 列候选待用户批量确认
    ambiguous    列名无语义信息(匿名编码列)或疑似标签列, 默认保留, 仅报数量

优先级: 用户红线(标签前缀) > 用户自定义规则 > 默认规则 > 启发式。
红线规则必须置顶, 否则会被更具体的模式抢先匹配而失效。

设计原则(方案 §二):
- 误剔真特征的代价 > 误留非特征: 规则只做"候选标记", 不自动剔除;
  非特征有特征分析(IV/PSI)环节兜底, 真特征被剔则信息永久丢失。
- ambiguous 默认保留, 仅报数量; non_feature 也必须经用户批量确认后才剔除。
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import pandas as pd

# ---- 内置默认规则库 v0(实证验证过, 直接复用) ----
DEFAULT_NON_FEATURE_PATTERNS = [
    (r'(?i)(^|_)date$',            "日期列"),
    (r'(?i)(^|_)time$|(_|^)time_', "时间戳列"),
    (r'(?i)order_?id$|order_?id_', "订单号列"),
    (r'(?i)(^|_)(id|uid)$',        "ID列"),
    (r'(?i)(^|_)(rn|seq|no)$',     "序号列"),
    (r'(?i)^f_p_',                 "分区日期列"),
]

# 标识前缀: 需结合 0/1 值域验证(纯 0/1 → 纯标识列 non_feature; 否则 ambiguous 待确认)
DEFAULT_IDENT_PREFIXES = ("if_", "is_", "has_", "flag_")

# 用户红线: 这些前缀的列一律视为标签列, 禁止入特征集(必须置顶, 先于一切规则)
DEFAULT_LABEL_PREFIXES = ("fpd", "dpd")


def classify_column(
    name: str,
    s: pd.Series,
    label_prefixes: Optional[Iterable[str]] = None,
    ident_prefixes: Optional[Iterable[str]] = None,
    extra_patterns: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """对单个列做语义三分类, 返回 (category, reason)。

    category ∈ feature / non_feature / ambiguous。
    规则命中仅标"候选", 最终由用户在 finalize 阶段批量确认后才剔除。
    """
    labels = tuple(label_prefixes) if label_prefixes is not None else DEFAULT_LABEL_PREFIXES
    idents = tuple(ident_prefixes) if ident_prefixes is not None else DEFAULT_IDENT_PREFIXES

    pats = list(DEFAULT_NON_FEATURE_PATTERNS)
    if extra_patterns:
        pats.extend((p, "用户自定义规则") for p in extra_patterns if p)

    # 1) 用户红线: 标签列前缀, 置顶(先于其他所有规则)
    if labels and re.match(rf"^({'|'.join(map(re.escape, labels))})", name):
        return "non_feature", f"标签列(用户红线: {'/'.join(labels)}*)"

    # 2) 默认规则 + 用户自定义规则
    for pat, reason in pats:
        if re.search(pat, name):
            return "non_feature", reason

    # 3) 标识前缀启发式: 数值且唯一值 ⊆ {0,1} → 纯标识列; 否则 ambiguous 值域待确认
    for pfx in idents:
        if name.startswith(pfx):
            if pd.api.types.is_numeric_dtype(s):
                uniq = set(pd.unique(s.dropna())[:10])
                if uniq <= {0, 1}:
                    return "non_feature", f"纯标识列({pfx}*)"
            return "ambiguous", f"标识前缀({pfx}*) 值域待确认"

    # 注: fst_rn/last_rn 会被上面"序号列"规则 (rn|seq|no)$ 先命中 → non_feature 候选
    #     (可能是排名特征, 必须经用户确认后才剔除; 本数据集用户已确认剔除)

    # 4) 匿名编码列: 前缀+纯数字后缀, 无语义 → ambiguous(可能误伤评分列如 score_80002,
    #    归 ambiguous 而非剔除, 是安全的)
    if re.match(r"^([a-zA-Z]+)_(\d+)$", name):
        return "ambiguous", "匿名编码列(无业务词)"

    # 5) 默认保留
    return "feature", "默认保留"