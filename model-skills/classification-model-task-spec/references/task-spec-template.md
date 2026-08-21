# task-spec.md 4 段式需求规格文档模板

> 本文件从 `classification-model-task-spec/SKILL.md` 抽出，包含 task-spec.md 的完整 4 段式模板（建模目标 / 核心参数 / 样本数据 / 待处理项）。

```
# 建模需求规格

> 沟通时间: {日期}
> 模型简称: {model_name}

## 一、建模目标

{一句话说清楚：在 XX 人群上预测 XX 行为，好坏标签定义，用于 XX 决策}

## 二、核心参数

| 维度 | 内容 | 状态 |
|------|------|:---:|
| 预测目标 | {未来N天是否发生X行为；好坏标签定义} | |
| 目标变量 | `label`（样本文件固定列名） | |
| 表现窗口 | {N天} | |
| 预估正样本率 | ~{X}%（如有） | |
| 数据文件 | {本地 parquet/csv/feather 路径} | |
| ID / 标签 / 日期列 | `{fuid}` / `{label}` / `{f_p_date}` | |

## 三、样本数据与切分窗口

**样本文件**: `{local_sample_path}`

| 集合 | 起 | 止 |
|------|----|----|
| Train | | |
| Test | | |
| OOT | | |

> 切分方式：三档区间不强制时间递增（时序排布由业务侧保证）；train/test 开发集可随机切分（记录 seed，val 偏乐观以 OOT 为裁决）。
> 切分在 experiments 消费时即时进行，不落盘 splits。

## 四、待处理项

| 优先级 | 事项 | 类型 | 建议 |
|:---:|------|:---:|------|
| P0 | {阻塞建模的} | 待确认/待探查 | |
| P1 | {不阻塞的} | 待确认/待探查 | |

## 下一步

- [ ] 用户确认需求规格
- [ ] 用户确认是否建模 → classification-model-development
```

> 关联: `classification-model-task-spec/SKILL.md` 4.2 节 task-spec.md 模板
