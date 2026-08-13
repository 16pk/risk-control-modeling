#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""skills 公共取数 PySpark 脚本(提交到集群运行)。

从现成宽表取样本,落成单份 parquet 文件。
SQL 拼装逻辑内联,无本地依赖,可独立提交到集群。

提交方式:
    spark-submit fetch_spark.py --table db.wide --dt-col f_p_date \
        --features f0,f1,f2 --fetch-start 2026-03-01 --fetch-end 2026-05-30 \
        --label-expr "(CASE WHEN req>0 THEN 1 ELSE 0 END)" \
        [--where "seg='活跃户'"] [--id-cols fuid] --out sample.parquet

或样本表⋈特征表模式:
    spark-submit fetch_spark.py --sample-table db.sample --feature-table db.features \
        --join-keys fuid,f_p_date --fetch-start 2026-03-01 --fetch-end 2026-05-30 \
        --label-col label --features f0,f1 --id-cols fuid --out sample.parquet

数据安全: 仅取所需列,不输出用户级明细到日志。
"""

import argparse

# ---------------------------------------------------------------------------
# 样本集 JOIN 红线 (ModelEvo): join key 必须 = ID列 + 日期分区列, 二者缺一不可。
#
#   - fuid/id_cols:   用户粒度唯一实体
#   - dt_col(pday):   时间切片; 同一用户在多个 pday 各有快照, 仅按 ID join 会跨日错配,
#                     把「今天的特征」贴到「昨天的标签」上 → 泄漏/口径错误, 线上必崩。
#   - 日期列名容错约定: 脚本不做隐式猜测——实际叫 f_p_date / fsx_time 时必须显式传
#                     --dt-col <真实日期列>, 不得指望自动识别。
# ---------------------------------------------------------------------------


def validate_join_keys(keys, dt_col):
    """校验样本⋈特征 JOIN keys 满足红线: 至少一个 ID 类键 + 含日期分区列 dt_col。

    幂等、无 IO、纯参数断言, fetch_spark / gen_fetch_command /
    feature-matching / classification-model-recommend 多处共用单一实现。

    Args:
        keys: join 键列表(如 ["user_no", "pday"])
        dt_col: 日期分区字段名(如 pday)

    Raises:
        ValueError: keys 为空 / 不含任何 ID 类键 / 不含 dt_col(或 dt_col 未展开)
    """
    if not keys:
        raise ValueError(
            "样本集 JOIN 红线: join_keys 不能为空, 必须以 [ID列, 日期列] 为 key "
            "(例如 %r), 禁止仅以单列作为连接键。" % _default_join_hint(dt_col)
        )
    missing_id = all(str(k) == str(dt_col) for k in keys)
    if missing_id:
        raise ValueError(
            "样本集 JOIN 红线: join_keys=%r 缺少 ID 类键(如 fuid), "
            "须为 [ID列, 日期列(%s)] 双键联接。" % (list(keys), dt_col)
        )
    if str(dt_col) not in {str(k) for k in keys}:
        raise ValueError(
            "样本集 JOIN 红线: join_keys=%r 缺失日期分区列 %s, 仅以单ID联接会跨日错配,"
            " 必须同时包含日期列(示例: ['fuid', '%s'])。"
            " 若表中日期列实际名为其他值(如 pday), 请显式传 dt_col, 不依赖自动猜测。"
            % (list(keys), dt_col, dt_col)
        )


def _default_join_hint(dt_col):
    return ["fuid", dt_col]


def build_sample_feature_sql(
    sample_table, feature_table, join_keys,
    dt_col, label_expr, id_cols, features,
    fetch_start, fetch_end, where,
    feature_lag_day=0,
):
    """样本表⋈特征表模式: 两表 LEFT JOIN 全在 spark 完成。

    样本表(主表 a): 提供 join_keys + label, 决定样本范围。
    特征表(副表 b): 提供全部入模 features。
    时间窗过滤下推到各子查询,避免扫全表。

    feature_lag_day: 0=同日JOIN(默认, a.{dt_col}=b.{dt_col}); 1=t-1 滞后JOIN,
        样本表 day t LEFT JOIN 特征表 day t-1, ON 子句去 dt_col 等值比较改用
        a.{dt_col} = date_format(date_add(to_date(b.{dt_col},'yyyyMMdd'),1),'yyyyMMdd'),
        特征表时间窗自动平移为 [fetch_start-1, fetch_end-1]。要求 dt_col 必须在 join_keys 中。
    """
    from date_utils import parse_date

    # 内部统一用归一化 8 位 YYYYMMDD 做比较/算术, 保证 start<=p<=end 语义正确(双格式兼容)
    fetch_start = parse_date(fetch_start, what="fetch_start")
    fetch_end = parse_date(fetch_end, what="fetch_end")

    keys = list(join_keys) if join_keys else ["fuid", dt_col]
    validate_join_keys(keys, dt_col)

    if feature_lag_day not in (0, 1):
        raise ValueError("feature_lag_day 仅支持 0(同日) / 1(t-1 滞后), 当前: %r" % feature_lag_day)

    if feature_lag_day == 1:
        if dt_col not in keys:
            raise ValueError(
                "feature_lag_day=1 要求 dt_col=%s 必须在 join_keys 中(用于做 t-1 日期对齐), "
                "当前 join_keys=%r" % (dt_col, keys)
            )
        join_keys_eq = [k for k in keys if k != dt_col]
        from date_utils import shift_days
        feat_start = shift_days(fetch_start, -1)
        feat_end = shift_days(fetch_end, -1)
        join_on_ab = " AND ".join(f"a.{k}=b.{k}" for k in join_keys_eq) + \
            f" AND a.{dt_col} = date_format(date_add(to_date(b.{dt_col}, 'yyyyMMdd'), 1), 'yyyyMMdd')"
    else:
        join_keys_eq = keys
        feat_start, feat_end = fetch_start, fetch_end
        join_on_ab = " AND ".join(f"a.{k}=b.{k}" for k in keys)

    # 样本表子查询: 仅取 join_keys + id_cols + label, 限定窗口
    a_cols = []
    for c in list(keys) + list(id_cols):
        if c and c not in a_cols:
            a_cols.append(c)
    if label_expr:
        a_cols.append(f"{label_expr} AS label")
    sample_sub = (
        f"SELECT {', '.join(a_cols)} FROM {sample_table}"
        f" WHERE {dt_col} >= '{fetch_start}' AND {dt_col} <= '{fetch_end}'"
    )
    if where:
        sample_sub += f" AND ({where})"

    if features:
        # 指定特征: 特征表子查询取 join_keys(含 dt_col) + 指定 features
        # 注意 dt_col 必须保留在 b_cols 中: lag=1 时 JOIN ON 需引用 b.{dt_col} 做 t-1 日期对齐;
        # 外层投影只挑 a.{keys+id+label} + b.{features}, 不会引 b.{dt_col}, 无列名冲突
        b_cols = []
        for c in list(keys) + list(features):
            if c and c not in b_cols:
                b_cols.append(c)
        feature_sub = (
            f"SELECT {', '.join(b_cols)} FROM {feature_table}"
            f" WHERE {dt_col} >= '{feat_start}' AND {dt_col} <= '{feat_end}'"
        )
        # 外层: a 的 keys + id_cols + label, b 的指定 features
        outer = [f"a.{c}" for c in a_cols if not c.endswith(" AS label")]
        if label_expr:
            outer.append("a.label")
        seen = set(keys)
        for f in features:
            if f and f not in seen:
                outer.append(f"b.{f}")
                seen.add(f)
    else:
        # 全列模式: 特征表 SELECT *, 外层 b.* EXCEPT(...) 取除 join_keys 外全部列
        # lag=1 时 EXCEPT 需含 dt_col (b.pday 不再等值匹配 a.pday, 避免重复列)
        feature_sub = (
            f"SELECT * FROM {feature_table}"
            f" WHERE {dt_col} >= '{feat_start}' AND {dt_col} <= '{feat_end}'"
        )
        outer = [f"a.{c}" for c in a_cols if not c.endswith(" AS label")]
        if label_expr:
            outer.append("a.label")
        except_cols = join_keys_eq + [dt_col] if feature_lag_day == 1 else list(keys)
        outer.append(f"b.* EXCEPT ({', '.join(except_cols)})")

    sql = (
        f"SELECT {', '.join(outer)} FROM ({sample_sub}) a"
        f" LEFT JOIN ({feature_sub}) b ON {join_on_ab}"
    )

    return sql


def build_fetch_sql(
    table, dt_col, label_expr, id_cols, features,
    fetch_start, fetch_end, where,
):
    """拼装取数 SQL(内联,无本地依赖)。

    单表模式: 直接从宽表取特征+标签。
    """
    from date_utils import parse_date

    # 内部统一用归一化 8 位 YYYYMMDD 做比较(双格式兼容)
    fetch_start = parse_date(fetch_start, what="fetch_start")
    fetch_end = parse_date(fetch_end, what="fetch_end")

    cols = []
    for c in list(id_cols) + list(features) + [dt_col]:
        if c and c not in cols:
            cols.append(c)

    if label_expr:
        cols.append(f"{label_expr} AS label")

    sql = (
        f"SELECT {', '.join(cols)} FROM {table} "
        f"WHERE {dt_col} >= '{fetch_start}' AND {dt_col} <= '{fetch_end}'"
    )
    if where:
        sql += f" AND ({where})"

    return sql


def main():
    """脚本入口:建 SparkSession、执行 SQL、落 parquet。"""
    parser = argparse.ArgumentParser(description="skills 公共取数(PySpark)")
    parser.add_argument("--table", required=False, help="样本宽表 库.表(单表模式)")
    parser.add_argument("--sample-table", default=None, help="样本表 库.表(样本⋈特征模式, 提供 label)")
    parser.add_argument("--feature-table", default=None, help="特征表 库.表(样本⋈特征模式, 提供全部特征)")
    parser.add_argument("--join-keys", default="fuid,f_p_date", help="样本⋈特征 JOIN 键(逗号分隔), 默认 fuid,f_p_date")
    parser.add_argument("--dt-col", default="f_p_date", help="日期分区字段(默认 f_p_date)")
    parser.add_argument("--features", default="", help="逗号分隔特征列(拼接模式留空=取特征表全部列)")
    parser.add_argument("--fetch-start", required=True, help="取数起始日期 YYYY-MM-DD(兼容 YYYYMMDD)")
    parser.add_argument("--fetch-end", required=True, help="取数结束日期 YYYY-MM-DD(兼容 YYYYMMDD)")
    parser.add_argument("--label-expr", default=None, help="SQL 标签表达式,如 '(CASE WHEN x>0 THEN 1 ELSE 0 END)'")
    parser.add_argument("--label-col", default=None, help="标签列名(无 label-expr 时使用)")
    parser.add_argument("--id-cols", default="", help="逗号分隔 ID 列")
    parser.add_argument("--where", default=None, help="可选客群筛选")
    parser.add_argument("--out", required=True, help="输出 parquet 路径(HDFS 或本地)")
    parser.add_argument("--feature-lag-day", type=int, default=0,
                        help="特征表滞后天数: 0=同日JOIN(默认), 1=特征表 t-1 vs 样本表 t (样本表 day t LEFT JOIN 特征表 day t-1)")
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    label_expr = args.label_expr or None
    label_col = args.label_col or None
    if not label_expr and not label_col:
        raise SystemExit("必须指定 --label-expr 或 --label-col")
    # label_col 模式下直接当列名拼入
    if label_col and not label_expr:
        label_expr = label_col

    features = [c.strip() for c in args.features.split(",") if c.strip()]
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    spark = (
        SparkSession.builder
        .appName("skills_fetch")
        .enableHiveSupport()
        .getOrCreate()
    )
    try:
        if args.sample_table and args.feature_table:
            # 样本表⋈特征表模式
            sf_keys = [k.strip() for k in args.join_keys.split(",") if k.strip()]
            validate_join_keys(sf_keys, args.dt_col)
            if not features:
                # 全列模式: 运行时读特征表 schema, 排除 join_keys 后生成显式列清单
                # (Spark 3.3.2 不支持 SELECT * EXCEPT 语法, 故动态展开)
                all_cols = spark.table(args.feature_table).columns
                features = [c for c in all_cols if c not in sf_keys]
                print(f"[fetch_spark] 特征表全列模式: 解析 {len(features)} 列(已排除 join_keys {sf_keys})")
            sql = build_sample_feature_sql(
                sample_table=args.sample_table, feature_table=args.feature_table,
                join_keys=sf_keys, dt_col=args.dt_col,
                label_expr=label_expr, id_cols=id_cols, features=features,
                fetch_start=args.fetch_start, fetch_end=args.fetch_end,
                where=args.where,
                feature_lag_day=args.feature_lag_day,
            )
        else:
            if not args.table:
                raise SystemExit("必须指定 --table(单表模式)或 --sample-table+--feature-table(拼接模式)")
            if args.feature_lag_day != 0:
                raise SystemExit("--feature-lag-day 仅适用于样本⋈特征表 JOIN 模式, 单表模式不适用")
            # 单表模式允许 features 为空: task-spec 阶段仅拉样本三列(id+dt+label)做标签稳定性分析
            sql = build_fetch_sql(
                table=args.table, dt_col=args.dt_col,
                label_expr=label_expr, id_cols=id_cols, features=features,
                fetch_start=args.fetch_start, fetch_end=args.fetch_end,
                where=args.where,
            )

        print(f"[fetch_spark] 执行 SQL:\n{sql}")
        sdf = spark.sql(sql)
        n = sdf.count()
        print(f"[fetch_spark] 取数 {n} 行, 落盘 -> {args.out}")
        sdf.write.mode("overwrite").parquet(args.out)
        print(f"[fetch_spark] 完成")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
