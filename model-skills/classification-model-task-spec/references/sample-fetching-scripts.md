# 样本拉取与分析脚本

> 本文件从 `classification-model-task-spec/SKILL.md` 3.6 节抽出,包含本地文件模式样本拉取、样本分析脚本的 bash 调用、脚本行为、产出文件清单。SKILL.md 中保留 4 行摘要 + 指向本文件的指针。

## 1. 拉取样本数据（local_file 模式）

当上游选择"本地文件"模式透传 `mode=local_file` 标记时，或用户首轮即表达"已有本地文件"/"离线实验"时：

**首轮提问调整**：
- SAMPLE 维度问题为 **"本地 parquet/csv/feather 路径是什么？"**
- 同时确认 `--label-col`/`--dt-col`/`--id-cols`（本地文件列名可能不是 `fuid`/`label`/`f_p_date`，必须显式问清楚）
- 其他维度按标准流程询问；用户不提供时 WHO/WHAT/HOW GOOD/CONSTRAINTS 按默认值填写，**Train-Test-OOT 三档区间仍强制由用户提供**（仅记录/透传，切分由 feature-analysis 完成）

**调用方式**：

```bash
python scripts/fetch_sample_task_spec.py \
    --session-dir <session_dir> \
    --model-name {model_name} \
    --local-parquet-path /path/to/local_sample.parquet \
    --train-range {train_start},{train_end} \
    --test-range  {eval_start},{eval_end} \
    --oot-range   {oot_start},{oot_end} \
    --label-col {label_col} \
    --dt-col {dt_col} \
    --id-cols {id_cols} \
    [--where ""]
```

> `--local-parquet-path` 接受 `.parquet`（直接硬链接/软链接）或 `.csv` / `.feather`（读后写 parquet）。

**脚本行为**：
- 全仓库已废除 spark 取数,仅支持 local_file: `_link_or_copy_local` 转写,`.parquet` 直接链接,`.csv` / `.feather` 读后写 parquet
- 落 yaml 含 `model.mode="local_file"`、`model.local_parquet_path=<path>`
- `--sample-table / --fetch-start / --fetch-end` 仅作记录用
- 校验仅做 `check_sensitive` + `validate_split_ranges`（后者仅校验「记录」区间的时序/格式合法性，不驱动切分）

**`_manifest.json` 额外字段**：`mode: "local_file"` / `local_parquet_path: <path>` / `id_cols`/`label_col`/`dt_col` 沿用用户指定

> task-spec 完成后，进入 data-cleaning（调 `data-cleaning/scripts/clean_data.py` 完成哨兵值替换 + 去重）。

## 2. 运行样本分析脚本

调起 `scripts/run_sample_analysis_task_spec.py`，由脚本统一完成：分时间段（默认按月）标签分布计算、稳定性判定、样本充足度判定、报告/清单产出。**不再做 Train/Test/OOT 切分**（切分与切分统计已后置到 feature-analysis，单一真相 = `feature_config.yaml` 的 `model.split`）。

```bash
python scripts/run_sample_analysis_task_spec.py \
    --sample .../data-profile/{model_name}_sample_{YYYYMMDD}.parquet \
    --model-name {model_name} \
    --timestamp {timestamp} \
    --output-dir .../data-profile/ \
    --dt-col {dt_col} \
    --label-col {label_col} \
    --id-cols {id_cols} \
    [--sample-table {source_table}] [--local-parquet-path {path}]
```

> **列名约定**：`--dt-col` / `--label-col` / `--id-cols` 与 `fetch_sample_task_spec.py` 对齐。
> - 本地文件列名可能不是 `f_p_date`/`label`/`fuid`（如 Home Credit 的 `dt`/`TARGET`/`SK_ID_CURR`），必须显式传入，否则校验失败。
> - `--id-cols` 支持逗号分隔多列，分析侧取首列作主 ID；老参数 `--time-col` / `--id-col` 仍可传入（已弃用 alias）。

> **报告头**：展示 `--local-parquet-path`（未传则回退到 `--sample`）的数据位置。

**产出文件**：

| 产出 | 路径 |
|------|------|
| 样本分析报告 (md/xlsx) | `data-profile/report.md` / `report.xlsx` |
| 关键信息清单 | `data-profile/_manifest.json` |

> 三档切分与切分统计已后置到 feature-analysis，产 `sample-features/splits/{train,test,oot}.parquet`；`data-profile/` 不再产 `_split_manifest.json` 与三档 parquet。

> 关联: `classification-model-task-spec/SKILL.md` 3.6 节需求确认后的样本分析
