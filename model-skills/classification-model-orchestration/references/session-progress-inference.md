# 会话启动进度推断规则

> 本文件从 `classification-model-orchestration/SKILL.md` 3.1 节抽出,包含 7 阶段进度推断表（完成标志 / 缺失时推断）。SKILL.md 中保留 2 行摘要 + 指向本文件的指针。

扫描 `runs/` 下所有 `{timestamp}-{model_name}` 命名的任务文件夹，按时间戳倒序取最近 5 个，对每个文件夹读 `task-spec/_manifest.json` 推断进度。

**进度推断规则**：

| 阶段 | 完成标志 | 缺失时推断 |
|------|----------|-----------|
| task-spec | `task-spec/.done` 存在 | 待跑或半途中断 |
| 样本分析 | `data-profile/_manifest.json` 存在 | data-profile 待跑 |
| data-cleaning | `sample-features/data-cleaning/sample.parquet` + `feature-list.csv` 存在 | 待跑 |
| feature-analysis（Dev Stage 0） | `sample-features/feature-analysis/analysis/_manifest.json` 存在 | 缺失 → Stage 0 待跑 |
| Dev Stage 1 | `new-models/` 非空（分布式场景以远端 job 返回的模型产物回填 new-models/） | Stage 1 待跑 |
| Dev Stage 2 | `new-models/` 下多 run，但 `model-comparison/_manifest.json` 不存在 | Stage 2 loop 中 |
| Dev Stage 3 | `model-comparison/_manifest.json` 存在 | Stage 3 已完成，待收口 |

> 关联: `classification-model-orchestration/SKILL.md` 3.1 节会话启动检查
