# -*- coding: utf-8 -*-
"""model-evo/_modelevo-shared 公共配置读写: yaml 加载 + 通用必填校验 + 数据安全红线。

各 skill(feature-analysis / classification-model-training / classification-model-tuning)在本模块之上叠加自己的专有校验。
"""
from __future__ import annotations

import csv
import os
import re
import sys

import yaml

# 定位 model-skills 根(用于解析 feature_list_source 相对路径)。
# 兼容三种部署形态:
#   ① 仓库源:     model-evo/_modelevo-shared              → 父目录(model-evo)下有 model-skills/
#   ② 仓库软链接: model-skills/_modelevo-shared           → 父目录 basename == "model-skills"
#   ③ 安装后:     SKILL_ROOT/_modelevo-shared             → 各 skill 直接平铺在父目录下(无 model-skills 层)
# model-knowledge 在 model-skills/ 下; 相对路径(如 feature_list_source: model-knowledge/assets/.../
# xxx.csv)按 model-skills 根解析, 与 gen_feature_list.load_feature_list 一致, 不依赖 yaml 文件位置。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELEVO_SHARED_DIR = os.path.dirname(_THIS_DIR)
_PARENT = os.path.dirname(_MODELEVO_SHARED_DIR)
if os.path.basename(_PARENT) == "model-skills":
    _MODEL_SKILLS_ROOT = _PARENT
elif os.path.isdir(os.path.join(_PARENT, "model-skills")):
    _MODEL_SKILLS_ROOT = os.path.join(_PARENT, "model-skills")
else:
    _MODEL_SKILLS_ROOT = _PARENT

# gen_feature_list.py 已迁移到本目录(_modelevo-shared/scripts), 与 config_io 同层;
# 各 skill 通过 _bootstrap 注入本目录后, `from gen_feature_list import load_feature_list`
# 可直接解析, 无需再注入 feature-matching/scripts。

# 疑似敏感信息正则: 18位身份证 / 11位手机号
_SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{17}[\dxX]\b"),   # 身份证
    re.compile(r"\b1[3-9]\d{9}\b"),    # 手机号
]

def _load_feature_list(fpath: str) -> list:
    """加载特征清单, 复用本目录 gen_feature_list 的解析逻辑。

    .csv 取 feature_name 列(跳过表头), .txt 按行(跳过 # 注释), 去重保序。
    复用而非重写, 保证「特征清单如何解析」只有一处真相。

    Args:
        fpath: 特征清单文件绝对路径

    Returns:
        去重保序的 feature 名列表
    """
    from gen_feature_list import load_feature_list

    return load_feature_list(fpath)



def load_config(path: str) -> dict:
    """读取 yaml 配置文件并返回字典。

    Args:
        path: 配置 yaml 路径

    Returns:
        配置字典(含 _config_dir 用于解析相对路径; feature_list_source 已按 repo 根解析)
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def check_sensitive(text: str) -> None:
    """检查字符串是否含疑似用户ID/手机号/身份证号,命中即抛错。

    Args:
        text: 待检查文本(如 where 条件)

    Raises:
        ValueError: 命中敏感信息红线
    """
    if not text:
        return
    for pat in _SENSITIVE_PATTERNS:
        if pat.search(text):
            raise ValueError(f"触碰数据安全红线: 配置中疑似硬编码敏感信息: {text!r}")


def validate_common(cfg: dict) -> None:
    """通用配置校验: 必填字段 + features 加载 + 敏感信息红线。

    全仓库已废除 spark 取数, 仅支持 local_file 语义: 数据来自本地 parquet,
    不再要求 sample_table / fetch_dt 必填。各 skill 在此基础上追加专有校验
    (如 model-training 校验 base_score_col)。

    Args:
        cfg: load_config 返回的字典

    Raises:
        ValueError: 缺必填项 / 命中敏感信息
    """
    model = cfg.get("model") or {}

    required = ["name", "dt_col"]

    has_label = bool(model.get("label_col")) or bool(model.get("label_expr"))

    for key in required:
        if not model.get(key):
            raise ValueError(f"配置 model.{key} 缺失或为空")
    if not has_label:
        raise ValueError("配置 model.label_col 与 model.label_expr 必须至少填一个")

    # features: 列表直接填,或通过外部文件加载(features_file 或 feature_list_source)
    # 相对路径解析顺序: ① yaml 所在目录(直觉行为, 支持 session 内相对引用)
    #                 ② model-skills 根(向后兼容 model-knowledge/... 风格)
    features_file = model.get("features_file") or model.get("feature_list_source")
    if features_file:
        if os.path.isabs(features_file):
            fpath = features_file
        else:
            cfg_dir = cfg.get("_config_dir")
            candidates = []
            if cfg_dir:
                candidates.append(os.path.join(cfg_dir, features_file))
            candidates.append(os.path.join(_MODEL_SKILLS_ROOT, features_file))
            fpath = next((c for c in candidates if os.path.exists(c)), candidates[-1])
        # 复用 gen_feature_list.load_feature_list 正确解析
        # (.csv 取 feature_name 列 / .txt 按行 / 跳过 # 注释 / 去重保序),
        # 避免朴素按行读取把 CSV 表头 feature_name 当成特征名。
        # 透传 _config_dir 让 load_feature_list 的相对路径基准与本函数一致。
        if cfg.get("_config_dir"):
            os.environ["_CONFIG_DIR"] = cfg["_config_dir"]
        model["features"] = _load_feature_list(fpath)
    # local_file 语义: 允许 features 为空, 视为"用本地 parquet 全部列(除 id/label/dt)"

    # fetch_dt 不强求; 若填仍按列表两元素校验
    fetch_dt = model.get("fetch_dt")
    if fetch_dt is not None and not (isinstance(fetch_dt, list) and len(fetch_dt) == 2):
        raise ValueError("model.fetch_dt 须为 [起始, 结束] 两元素列表")

    check_sensitive(model.get("where") or "")
    check_sensitive(model.get("sample_table") or "")


def _parse_range_pair(name: str, value) -> tuple:
    """把单档区间值规整成 (起, 止) 并校验日期格式(YYYY-MM-DD / YYYYMMDD 双兼容)、起 ≤ 止。

    Args:
        name: 档名(train/test/oot), 仅用于报错信息
        value: 形如 ["2026-03-12", "2026-04-30"] 或 "2026-03-12,20260430"

    Returns:
        (start, end) 两个归一化 8 位日期字符串

    Raises:
        ValueError: 非两元素 / 日期不合法 / 起 > 止
    """
    from date_utils import parse_date_pair

    return parse_date_pair(value, what="split.%s_range" % name)


def validate_split_ranges(model: dict) -> None:
    """校验 model.split(可选): train/test/oot 三档 pday 区间的合法性。

    切分唯一真相 = model.split(feature_config.yaml / train_config.yaml), v2.1 起由 training
    在训练消费时按区间即时切分; task-spec 的 sample_config.*.yaml 的 split 段仅「记录/透传」,
    调用本函数只校验记录区间的合法性, 不再驱动切分。

    仅当 model.split 存在时触发。约束:
      1. 三档齐全(train_range/test_range/oot_range 缺一报错)
      2. 每档 [起, 止] 为合法日期(YYYY-MM-DD / YYYYMMDD 双兼容)且起 ≤ 止
      3. 三档并集 ⊆ model.fetch_dt(划分范围不得超出取数窗口);
         local_file 模式不强制 fetch_dt, 若未填则跳过本条
      注: 不再强制 train/test/oot 三档时间递增(允许任意时序排布, 由业务侧保证切分合理性)。

    Args:
        model: 配置 model 段

    Raises:
        ValueError: 任一约束不满足
    """
    split = model.get("split")
    if not split:
        return
    ranges = {}
    for name in ("train", "test", "oot"):
        key = "%s_range" % name
        if not split.get(key):
            raise ValueError("model.split 须三档齐全, 缺 %s" % key)
        ranges[name] = _parse_range_pair(name, split[key])

    fetch_dt = model.get("fetch_dt")
    if isinstance(fetch_dt, list) and len(fetch_dt) == 2:
        # 并集取三档起的最小、止的最大(不假设三档时序递增)
        union_start = min(r[0] for r in ranges.values())
        union_end = max(r[1] for r in ranges.values())
        # local_file 模式不强制 fetch_dt(本地 parquet 无取数窗口概念);
        # fetch_dt 为空字符串或占位时跳过本条校验
        f_vals = [str(v).strip() for v in fetch_dt if str(v).strip()]
        if len(f_vals) == 2:
            from date_utils import parse_date

            f_start = parse_date(f_vals[0], what="fetch_dt.start")
            f_end = parse_date(f_vals[1], what="fetch_dt.end")
            if union_start < f_start or union_end > f_end:
                raise ValueError(
                    "train/test/oot 划分并集 [%s,%s] 超出取数窗口 fetch_dt [%s,%s]"
                    % (union_start, union_end, f_start, f_end)
                )


# ---------------------------------------------------------------------------
# 计算资源路由工具 (compute routing, 字节口径 EXP-G-004): 共享工具函数。
#
# 口径(2026-08 固化为字节): 预估『拉到本地后的体量』——<1GB → local; ≥1GB → distributed。
#
# v2.1 精简说明: task-spec Gate P0 裁决 / feature-analysis 已删除, run_build 计算路由恒为 local,
# 不再消费 engine.ruling; 本模块的 estimate_size_bytes / route_by_bytes 保留为纯工具函数
# (task-spec fetch_sample 仍可用于样本体量预估), 但不再作为分布式路由的唯一真源。
# ---------------------------------------------------------------------------

LOCAL_BYTES_LIMIT = int(1e9)          # 1GB:<1GB 本地 / ≥1GB 分布式
BYTES_ROUTING_SCHEMA_VERSION = 1      # manifest 存档版本, 防止存量 ruling 升级后被静默改判


def estimate_size_bytes(df=None, *, path=None, partition_total_size=None):
    """预估『拉到本地后的体量』(语义钉死: 口径 = 本地驻留矩阵近似字节, 非 Hive 原始落盘)。

    优先级链:
      ① partition_total_size(int): Hive 分区级 totalSize 求和(LLM 层经 MCP 测得传入)——首选,最准
      ② df                    : pandas DataFrame 的 memory_usage(deep=True).sum()
      ③ path                  : pyarrow ParquetFile.metadata.num_rows × Σ(schema dtype 位宽/8),
                                 逻辑尺寸≈本地驻留;元数据不足则退回 os.path.getsize
      None                 : 无法估计, 调用方放行 local 但在日志/manifest 记 reason=estimate_unavailable

    ⚠️ 不明示分区 size 时不自行猜全表 stats(Hive 无 ANALYZE 时常空/失真)——宁可 None→local+warn。

    Args:
        df: pandas.DataFrame, 非 None 时用其 deep memory usage
        path: 备选的本地 .parquet 路径, 当 df 为 None 时尝试读元数据
        partition_total_size: MCP 已实测的窗口内命中分区 totalSize 之和(bytes)

    Returns:
        预估字节数(int), 无法估计时 None
    """
    if partition_total_size is not None:
        try:
            return max(0, int(partition_total_size))
        except Exception:
            pass
    if df is not None:
        try:
            return int(df.memory_usage(deep=True).sum())
        except Exception:
            pass
    if path:
        p = str(path)
        try:
            from pyarrow import parquet as pq

            pf = pq.ParquetFile(p)
            row_bytes = 0
            for field in pf.schema:
                bw = getattr(field.type, "bit_width", None)
                # 定长数值型有 bit_width;字符串/嵌套等可变长按 8B/单元粗估(约等于行内指针+内容上界)
                row_bytes += (bw // 8) if isinstance(bw, int) else 24
            return int(pf.metadata.num_rows) * row_bytes
        except Exception:
            pass
        try:
            return int(os.path.getsize(p))  # 最后手段: 文件物理大小(偏低估压缩前, 方向一致即可)
        except Exception:
            pass
    return None


def route_by_bytes(size_or_none, *, where="", limit=LOCAL_BYTES_LIMIT,
                   schema_version=BYTES_ROUTING_SCHEMA_VERSION) -> str:
    """按字节判据裁决引擎路由(单一真相)。

    - size < limit        ⇒ 'local'(应拉取到本地走本地训练流程)
    - size ≥ limit        ⇒ 打印醒目 WARNING 并返回 'distributed'(应转 ray-distributed-train)
    - None(无法估计)      ⇒ 兼容旧行为放行 'local', 由调用方在日志/manifest 记 reason=estimate_unavailable

    Args:
        size_or_none: estimate_size_bytes 的返回值(可为 None=无法估计)
        where: 调用位置标签(如 "run_build Stage1"/"task-spec Gate P0"), 仅用于告警文案定位
        limit: 字节阈值, 默认 LOCAL_BYTES_LIMIT(=1GB)
        schema_version: 存档版本号, 供 manifest 溯源本次裁决依据

    Returns:
        "local" 或 "distributed"
    """
    if size_or_none is None or int(size_or_none) < limit:
        # None=无法估计, 宁放过不放杀:放行本地, 下游记录原因可追溯
        return "local"
    print(
        "[compute-routing] %s bytes=%d >= limit=%d (bytes, schema_v%d) "
        "=> MUST use distributed (ray-distributed-train), NOT local single-process."
        % (where, int(size_or_none), limit, schema_version)
    )
    return "distributed"
