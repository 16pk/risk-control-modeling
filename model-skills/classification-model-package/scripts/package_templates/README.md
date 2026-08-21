# 独立交付包 — 定版模型打分

本包由建模专家（classification-model-package）从已定版训练任务组装，**可独立运行**：
给定一份数据文件，一条命令完成「数据清理 → 定版模型打分 →（若含 FICO 模块）FICO 转分」。

## 包信息

| 项 | 值 |
|---|---|
| 定版模型 run | `{{RUN_NAME}}` |
| 算法 | `{{ALGO}}` |
| 特征数 | `{{N_FEATURES}}` |
| 含 FICO 转分模块 | `{{HAS_FICO}}` |
| 打包时间 | `{{PACKAGED_AT}}` |

> 包文件与 `package-manifest.json` 一一对应；如需追溯来源，见 `package-manifest.json` 的
> `source_session` / `run_name` / `feature_names`。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 输入数据要求

- 支持 `parquet` / `csv`。
- 必须包含定版模型的**全部特征列**（见本README「特征列表」，缺少任一特征将报错退出）。
- 可含任意非特征列（id / 日期 / label 等），全部透传到输出。
- **label 列非必需**；id/dt/label 列不参与校验与替换。

### 特征列表（{{N_FEATURES}} 个）

`{{FEATURES}}`

## 运行

```bash
python run.py --input <数据文件> --output-dir <输出目录>
```

可选参数：`--score-col score`（输出概率分列名，默认 `score`）、`--batch-size 500000`（整批
内存处理参考上限；若数据量超出内存，请按行数分片多次执行并自行合并）。

## 输出（`--output-dir` 下）

| 文件 | 说明 |
|---|---|
| `score.parquet` | 透传所有非特征列 + `score`（违约概率）列；**含 FICO 模块时**追加 `odds` / `logistic_prob` / `bscore`（FICO 标准分，约 [400,780]，分高险低） |
| `cleaning-report.json` | 清洗统计：哨兵值命中特征 / 命中值 / 命中数 / 命中比例（命中即替换为 NaN，仅 WARN 不中断） |
| `fico-summary.json` | （含 FICO 模块时）校准参数 coef/intc + bscore 分布 |
| `run-manifest.json` | 本次运行的元信息（时间 / 资产 / 统计） |

## 行为说明

- **数据清理**：仅对特征列将哨兵值（`cleaning-scheme.json` 的 `invalid_values`）替换为 NaN，
  非交互；**不做样本去重**。
- **打分**：严格校验特征齐全后按 `feature_names` 重排推理，产出违约概率 `score`，不含特征列。
- **FICO**：纯应用模式，直接使用打包时固化的校准参数 `coef/intc` 转分，不做拟合；
  `bscore` 越界仅 WARN 不中断。
- 失败即打印 `[ERROR]` 并退出，退出的同时不产出半成品。