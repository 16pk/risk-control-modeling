# 样本链接与 split 区间记录脚本

> 本文件从 `classification-model-task-spec/SKILL.md` 抽出，包含本地文件模式样本链接脚本的 bash 调用、脚本行为、产出文件清单。

## 1. 样本链接 + split 区间记录（local_file 模式）

需求澄清 3 问确认后，调 `scripts/fetch_sample_task_spec.py`：
- 记录 Train/Test/OOT 三档区间到 `<session_dir>/task-spec/sample_config.{model_name}.yaml` 的 `model.split`（切分窗口唯一真相的**记录入口**，供 feature_config.yaml 使用）
- 把本地样本链接/转写到 `<session_dir>/data-profile/{model_name}_sample_{YYYYMMDD}.parquet`（供 data-cleaning 消费）

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
- 全仓库已废除 spark 取数，仅支持 local_file：`_link_or_copy_local` 转写，`.parquet` 直接链接，`.csv` / `.feather` 读后写 parquet
- 落 yaml 含 `model.mode="local_file"`、`model.local_parquet_path=<path>`、`model.split.{train,test,oot}_range`
- 校验仅做 `check_sensitive` + `validate_split_ranges`（校验「记录」区间的格式/范围合法性，不驱动切分；不强制三档时序递增）

**`_manifest.json` 额外字段**：`mode: "local_file"` / `local_parquet_path: <path>` / `id_cols`/`label_col`/`dt_col` 沿用用户指定

> task-spec 完成后，进入 data-cleaning（调 `data-cleaning/scripts/clean_data.py` 完成哨兵值替换 + 去重）。

## 2. （已废弃）独立样本分析报告

**不再产独立样本分析报告**（`data-profile/report.md` + `report.xlsx`）。样本充足度在需求澄清时口头确认，标签质量/分月分布由 `credit-data-analysis`（建模 pipeline 特征分析）在 Stage 0 统一产出。

`run_sample_analysis_task_spec.py` 保留脚本但**不再由编排强制调用**，仅在用户明确要求"先看样本标签分布"时手动运行。

## 3. 切分说明

三档切分与切分统计**不在本 skill 完成**：`model.split` 仅记录/透传，切分由 `classification-model-training`（`prepare_splits`）等消费方在训练/调优/评估时按区间即时切分，**不落盘 splits**。

> 关联: `classification-model-task-spec/SKILL.md`
