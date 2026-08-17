---
name: model-scoring
description: 建模 pipeline 内部的定版模型打分环节，位于模型定版(收口)之后、score-to-fico 之前。用定版模型对 data-cleaning 清洗后的数据文件跑推理，输出违约概率分 score（不做校准、不转 FICO），透传所有非特征列减少存储成本。由 classification-model-development 收口后**默认执行**（用户可叫停），不设独立触发词。
---

# model-scoring

建模 pipeline 中「定版模型 → 打分」的**唯一推理环节**，位于 `classification-model-development` 的 **Stage 6**（v2.1 起默认执行，不再"总是询问"），承接收口确认的定版模型（`finalized_model.json`），对清洗后数据跑推理产出违约概率分 `score`，供下游 `score-to-fico`（用户主动触发时）拟合校准转 FICO。

> ⚠️ **触发定位**：v2.1 起收口后**默认执行**（用户可叫停），由 `classification-model-development` Stage 6 编排调起；不设独立触发词。

## 1. 职责边界

| 做 | 不做 |
|---|---|
| 用定版模型对清洗后数据跑推理，产出违约概率分 `score` | 不做 LR 校准、不转 FICO（交给 `score-to-fico`） |
| 透传所有非特征列（id/date/label 等），减少存储成本 | 不做「是否已定版」判定（定版是编排层职责，本 skill 只收 `--model-path`） |
| 严格校验特征对齐（缺特征报错、按 feature_names 重排） | 不做特征衍生 / 缺失填充（数据已由 data-cleaning 清洗） |

## 2. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| 定版模型 | ✅ | Stage 4 收口落 `finalized_model.json` → 定位 `new-models/{run}/model/` | `model.json`(xgb) / `model.pkl`(dnn/lr) + `model_meta.json`(含 `feature_names`) |
| 清洗后数据 | ✅ | `sample-features/data-cleaning/sample.parquet` | 含特征列 + 非特征列（id/date/label），schema 与定版模型特征对齐 |

## 3. 执行命令

`<skill_dir>` 指本 skill 所在目录，执行时替换为实际绝对路径。

```bash
python <skill_dir>/scripts/score_data.py \
    --model-path <session_dir>/new-models/{run}/model \
    --data <session_dir>/sample-features/data-cleaning/sample.parquet \
    --out <session_dir>/scoring/score_sample.parquet \
    [--score-col score] \
    [--algo xgb|dnn|lr]
```

编排层（development Stage 6）先从 `finalized_model.json` 读 `run_name` / `algo` / `model_path`，再拼出上面对应参数；`--algo` 一般可省（脚本从 `model_meta.json` + 文件扩展名自动判定）。

## 4. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--model-path` | ✅ | - | 定版模型文件（`model.json`/`model.pkl`）或 `model/` 目录 |
| `--data` | ✅ | - | 清洗后数据文件（parquet/csv） |
| `--out` | ✅ | - | 打分输出 parquet（按扩展名自动 CSV/parquet） |
| `--score-col` | 否 | `score` | 输出概率分列名 |
| `--algo` | 否 | 自动判定 | 算法覆盖：`xgb`/`dnn`/`lr` |

## 5. 输出产物

产物落 session 根独立 `scoring/` 目录：

```text
<session_dir>/scoring/
└── score_sample.parquet    # 透传所有非特征列(id/date/label 等) + score 列; 不含原特征列
```

`score` 列即**违约概率分**，为下游 `score-to-fico`（用户主动触发时）的输入。

## 6. 定版标记（finalized_model.json）

收口确认上线候选后，由编排层调用本 skill 的定版标记工具落 session 根 `finalized_model.json`：

```bash
python <skill_dir>/scripts/mark_finalized.py \
    --session-dir <session_dir> \
    --run-name {run} \
    [--oot-auc 0.82]
```

结构：`{schema_version, produced_by, run_name, algo, model_path, model_dir, feature_names, oot_auc, finalized_at}`。该文件是「定版」唯一的机器可读落盘标记，供 Stage 5 定位定版模型。

## 7. 与其他 skill 的关联

| skill / 模块 | 关系 | 说明 |
|---|---|---|
| `classification-model-development` | **编排调起（Stage 6，默认执行）** | 收口确认上线候选后调起，落 `finalized_model.json` + 打分 |
| `data-cleaning` | 上游 | 产 `sample-features/data-cleaning/sample.parquet`（本 skill 输入） |
| `classification-model-training` | 上游 | 产 `new-models/{run}/model/`（定版模型 + `model_meta.json`） |
| `score-to-fico` | 下游（可选） | 用户主动触发时消费本 skill 打分结果（含 label + score），拟合校准转 FICO |
| `_modelevo-shared` | 依赖 | 共享 metrics（打分评估用）；record_stage 链已删除 |

## 8. 执行约束

| 约束 | 说明 |
|---|---|
| 特征对齐红线 | 用 `model_meta.json.feature_names` 严格校验，缺失特征**报错列出**，不静默填 NaN；按 `feature_names` 顺序重排后喂模型，避免列序错位打错分 |
| 只读上游 | 只读 `finalized_model.json` + `new-models/{run}/model/` + 清洗后数据；唯一写目录是 `scoring/` |
| 不校准不转分 | 输出仅 `score`（违约概率），LR 校准与 FICO 转换严格交由下游 `score-to-fico` |
| 数据安全红线 | 透传非特征列时仅重发输入已有的列（数据已由 data-cleaning 保证无身份证/手机号明文），不新增任何敏感信息 |
| 依赖 | `pandas / numpy / pyarrow`；按算法额外 `xgboost`(xgb) / `torch`(dnn) / `scikit-learn`(lr) |

## 9. 异常处理

| 异常 | 处理方式 |
|---|---|
| `model_meta.json` 缺失 / 无 `feature_names` | 报错退出，提示确认 model 阶段已落盘 |
| 输入数据缺特征 | 报错并列出全部缺失特征，提示检查数据清洗/特征清单是否与定版模型一致 |
| `model.pkl` 存在但 meta 缺 `algo` | 报错，提示用 `--algo dnn|--algo lr` 显式指定 |
| 推理输出长度与输入不一致 | 报错退出，提示检查模型/数据对齐 |
| 模型文件缺失 | 报错退出，提示确认定版 run 的 model 阶段产物 |

## 10. 测试

```bash
python -m pytest model-scoring/tests/ -q
```
