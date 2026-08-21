---
name: data-cleaning
description: 建模 pipeline 内部的数据清洗环节，位于特征分析(credit-data-analysis)之前。承接用户提供的本地数据文件，完成哨兵值/无效值替换为 NaN、按用户+日期去重，产出清洗后 parquet + feature-list.csv + 可复用清洗方案(cleaning-scheme.json + cleaning-report.md)。发现异常值时任务暂停、弹出提示让用户确认是否继续。由编排层自动调起（development 主链路 / 轻量入口 prep_sample.py clean），不设独立触发词。
---

# data-cleaning

建模 pipeline 中样本进入流程后的**第一道工序**，位于 `credit-data-analysis` 之前。它承接用户提供的本地数据文件，统一收口原先散落在各 skill 中的「哨兵值/无效值替换为 NaN」逻辑，并按「用户 + 日期」维度去重，产出清洗后 parquet 供后续任务消费。

> ⚠️ **触发定位**：本 skill 由 `classification-model-development` 编排自动调起（主链路 Stage 2，或轻量入口 `prep_sample.py clean/analyze` 的独立任务场景），**不设独立触发词**。独立任务必须先经 `feature-classification` 产出权威 `feature-list.csv` 后再清洗（轻量入口已串联，禁止直接跳过特征识别单跑清洗）。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| 数据文件 | ✅ | 用户提供 / 从 hive 下载到本地 | 含 `id 列 + 特征列 + label 列(可选) + 日期列`，支持所有 pandas 可读格式（.parquet/.csv/.feather/.xlsx/.xls/.json）；本地文件唯一链路 |
| `id_col` / `dt_col` | ✅ | 数据探查由大模型自主识别 + 用户确认后传入 | 不硬编码默认列名；脚本校验这两列均存在 |
| `label_col` | 否 | 无标签场景可不传 | 提供时参与去重保留决策（组内优先保留 label 非空行）与哨兵替换统计；缺失时仅影响这两处，清洗主体逻辑不变 |
| 哨兵值集合 | 否 | 默认 `[-1,-2,-9,-99,-999,-9999,-99999]`，CLI `--invalid-values` 覆盖 | 命中这些值的特征列替换为 NaN |
| `feature_list_source` | 否 | 特征清单文件（.csv 取 `feature_name` 列 / .txt 按行） | 派生 feature-list.csv 时取交集；**v2.4 起由 `feature-classification` 的权威清单（`<session_dir>/sample-features/feature-list.csv`）承接**，不传则派生全量特征列 |

> 本 skill **不提供 hive 下载能力**（下载由用户在 skill 外完成），只消费已落地的本地文件。

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录，执行时替换为实际绝对路径，不要依赖当前工作目录。

```bash
python <skill_dir>/scripts/clean_data.py \
    --input <本地数据文件路径> \
    --session-dir <session_dir> \
    --id-col fuid \
    --dt-col f_p_date \
    [--label-col label] \
    [--invalid-values -1,-2,-999,-9999] \
    [--feature-list-source model-knowledge/assets/feature-knowledge/feature-list/xxx.csv] \
    [--auto-confirm]
```

**强门禁行为**：脚本检测到哨兵值命中时，默认打印提示并等待 `input()` 确认（输入 `y` 继续，否则中止、不产出清洗后数据）。编排层已完成交互确认时用 `--auto-confirm` 跳过交互续跑。

## 3. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--input` | ✅ | - | 本地数据文件路径 |
| `--session-dir` | ✅ | - | session 目录（产物落 `<session_dir>/sample-features/data-cleaning/`） |
| `--id-col` | ✅ | - | 用户粒度 ID 列名 |
| `--dt-col` | ✅ | - | 日期分区列名 |
| `--label-col` | 否 | `None` | 标签列名；无标签场景可不传（去重退化为保首行，哨兵替换不做坏率统计） |
| `--invalid-values` | 否 | `-1,-2,-9,-99,-999,-9999,-99999` | 哨兵值集合（逗号分隔） |
| `--feature-list-source` | 否 | `None` | 特征清单过滤文件，派生 feature-list.csv 时取交集 |
| `--auto-confirm` | 否 | `False` | 跳过异常值交互确认 |

## 4. 输出产物

产物统一落 `<session_dir>/sample-features/data-cleaning/`：

```text
<session_dir>/sample-features/data-cleaning/
├── sample.parquet         # 清洗后样本(哨兵值→NaN + 去重), pipeline 后续任务数据源
├── feature-list.csv       # 派生特征清单(1 列 feature_name)
├── cleaning-scheme.json   # 机器可读清洗方案(对哪些特征、做了怎样的处理 + 命中统计)
├── cleaning-report.md     # 人工可读清洗报告
└── _manifest.json         # 产物清单
```

### 4.1 清洗方案（cleaning-scheme.json）结构

```json
{
  "schema_version": 1,
  "produced_by": "skills/data-cleaning",
  "invalid_values": [-1, -2, -9, -99, -999, -9999, -99999],
  "dedup_keys": ["fuid", "f_p_date"],
  "dedup_keep_rule": "label_non_null",
  "features": [
    {"feature": "f0", "action": "replace_invalid_to_nan", "hit_values": [-1, -999], "n_hit": 123, "hit_ratio": 0.0123}
  ],
  "dedup_report": {"n_before": 100000, "n_after": 98000, "n_removed": 2000}
}
```

复用约定：下次 session 可读 `cleaning-scheme.json` 的 `invalid_values` 作为默认哨兵值集合，保证可复现。

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | 用户本地 parquet/csv/feather | 数据源（本 skill 是样本进入 pipeline 的第一道清洗工序，承接 task-spec 转写的本地样本） |
| 下游 | `credit-data-analysis` | 读 `data-cleaning/sample.parquet` + `feature-list.csv`，做分月 PSI/IV 体检（pipeline 特征分析） |
| 下游 | `classification-model-experiments` | 消费 `data-cleaning/sample.parquet` + `model.split` 即时切分训练 |
| 依赖 | `_modelevo-shared` | 复用 `gen_feature_list`（特征清单解析唯一真相）、`config_io`（配置/安全红线） |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| 哨兵值仅作用于特征列 | id / dt / label 列不参与替换，避免误伤标签与主键 |
| 去重保留规则 | 按 `(id_col, dt_col)` 去重，组内优先保留 label 非空行；全空或未提供 label_col 则保留首行兜底 |
| label 非法值剔除**暂不处理** | 该职责保留在下游训练模块（experiments）消费切分后的 OOT 防御逻辑中，本 skill 不动 |
| 强门禁 | 发现哨兵值命中 → 任务暂停 → 弹提示 → 用户确认是否继续 |

> 覆盖范围、异常处理、交互约定详情见 `references/constraints-and-exceptions.md`。

## 7. 异常处理

异常分类与处理方式详见 `references/constraints-and-exceptions.md` 第 7 节。

## 8. 测试

```bash
python -m pytest data-cleaning/tests/ -q
```
