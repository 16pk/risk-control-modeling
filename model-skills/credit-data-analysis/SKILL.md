---
name: credit-data-analysis
description: 信贷特征分析 skill，双模式：①独立数据体检（用户主动发起"样本分析/特征分析/特征IV/特征PSI/数据体检/分月监控/逾期率走势"时优先使用，分月视角 11-sheet Excel）；②建模 pipeline 特征分析（development Stage 0 编排调起，从 feature_config.yaml 推导 PSI 基准月=第一个 OOT 月，产出 md+xlsx 报告）。不做三档切分、不产 IV/PSI 筛选 csv（切分后置到 training/tuning/evaluation 消费时即时进行，训练过程不通过 IV/PSI 指标筛选特征）。
---

# credit-data-analysis

信贷特征分析 skill：对一张数据文件做**分月视角**的多维度描述性统计，产出 Excel + Markdown 报告（样本分布 / 特征分布 / 覆盖率 / 均值 / 最小 / 最大 / 标准差 / Nunique / PSI / IV / 无效值检查）+ 可复现参数清单。**只产报告、不自动筛特征**，筛与不筛由人决定。

## 定位与双模式

| 模式 | 触发场景 | 时间视角 | PSI 基准月 | 输出 |
|---|---|---|---|---|
| 独立体检 | 用户主动发起样本/特征分析、数据体检、分月监控 | 分月矩阵（PSI=基准月对比各月，IV=分月） | 用户指定 `--base-month` | 11-sheet Excel + md + `_manifest.json` |
| pipeline（development Stage 0） | 建模流程内部特征分析，由 `classification-model-development` 编排调起 | 分月矩阵（PSI=基准月对比各月，IV=分月） | **默认取第一个 OOT 月**（读 `--split-config` 的 `model.split.oot_range` 首月，**须用户确认**，`--base-month` 可覆盖） | Excel + md + `_manifest.json` |

> **职责边界**：
> - **不做三档切分**：train/test/oot 切分后置到 training/tuning/evaluation 消费时即时进行（复用 training 的 `prepare_splits`），本 skill **不产 `splits/{train,test,oot}.parquet`**。
> - **不产 IV/PSI 筛选 csv**：训练过程不通过 IV/PSI 指标筛选特征，本 skill 只做体检报告。
> - 本 skill 承担建模 pipeline 特征分析角色（Stage 0）。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|---|---:|---|---|
| 数据文件 | ✅ | 用户指定路径（`.feather` / `.csv` / `.parquet`，pandas 可读格式） | 须含时间列（默认 `fsx_time`，可覆盖）+ 数值型特征列 + 风险标签列 |
| 特征列范围 | ✅ | **交互确认**（门禁） | 起始列名 → 结束列名（区间连续含两端）+ 区间外额外特征列；起始/结束列名不在数据中直接报错重输 |
| PSI 基准月份 | ✅ | **交互确认**（门禁） | pipeline 模式默认第一个 OOT 月；该月 PSI 记为 0，其余各月与之对比 |
| IV 风险标签 | 否 | 默认自动选择第一个 `fpd`/`dpd` 前缀列 | 可用 `--iv-label` 指定，或 `--risk-labels` 手动指定多个标签 |
| 输出位置 | 否 | 默认当前工作目录 | 推荐落 `<session_dir>/sample-features/credit-data-analysis/`（在建模 session 内时） |

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录（即本文件所在目录），执行时替换为实际绝对路径，**不要依赖当前工作目录，不要复制脚本到 cwd**。

### 2.1 独立体检模式

```bash
python <skill_dir>/scripts/feature_analysis.py \
  --data-file <数据文件绝对路径> \
  --feature-start tx_model_2_score \
  --feature-end mob4_v5_score \
  --feature-extra ascore_fpd7_v3 \
  --base-month 2025-04 \
  --iv-label fpd7 \
  [--time-col fsx_time] \
  [--output-dir <产物目录>]
```

### 2.2 pipeline 模式（development Stage 0 编排调起）

```bash
python <skill_dir>/scripts/feature_analysis.py \
  --data-file <session_dir>/sample-features/data-cleaning/sample.parquet \
  --feature-start <首特征> --feature-end <末特征> \
  --iv-label <label_col> \
  --time-col <dt_col> \
  --split-config <session_dir>/sample-features/feature_config.yaml \
  --base-month <YYYY-MM>   # 可选；缺省时从 split_config 推导第一个 OOT 月
  [--output-dir <session_dir>/sample-features/credit-data-analysis/]
```

**PSI 基准月确认流程（门禁）**：
1. 编排层读取 `--split-config` 的 `model.split.oot_range`，取起始日所在月为默认基准月
2. **向用户确认**："PSI 基准月默认取第一个 OOT 月 `{YYYY-MM}`，是否确认？或指定其他月份"
3. 用户确认后以 `--base-month` 显式传入；脚本内未传 `--base-month` 时自动推导并打印提示（不静默拍板）

**交互顺序**（遵循关键决策确认门禁）：
1. **探查数据**：加载并输出结构摘要（行/列数、时间列月份范围、潜在风险标签列 `fpd`/`dpd` 前缀列、首尾列名），供用户确认特征列范围。
2. **一次性收集参数**：特征列范围（起始→结束 + 额外列）、PSI 基准月份（pipeline 模式给默认+确认）、IV 风险标签、输出文件名，一并列出等用户回答；缺哪项补问哪项。
3. **运行并报告**：跑完后报告 Excel + Markdown 路径 + 各 sheet 的规模。
4. **产物落盘**：Excel + md + `_manifest.json` 落 `--output-dir`（默认 cwd），复现时直接重跑同一命令即可。

## 3. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--data-file` | ✅ | - | 数据文件（`.feather` / `.csv` / `.parquet`，按扩展名自动选读取方式） |
| `--feature-start` / `--feature-end` | ✅ | - | 特征列区间（含两端）；不在数据中报错 |
| `--feature-extra` | 否 | 空 | 逗号分隔的区间外额外特征列 |
| `--base-month` | 否 | pipeline 模式推导 | PSI 基准月份 `YYYY-MM`；pipeline 模式缺省时从 `--split-config` 的 `model.split.oot_range` 首月推导（第一个 OOT 月） |
| `--split-config` | pipeline 必填 | 空 | pipeline 模式：`feature_config.yaml` 路径，用于推导 PSI 基准月 |
| `--time-col` | 否 | `fsx_time` | 时间列名，须为 datetime 类型（脚本转月度） |
| `--iv-label` | 否 | 空（自动选第一个 fpd/dpd 列） | IV 计算所用的风险标签列 |
| `--risk-prefixes` | 否 | `fpd,dpd` | 风险标签前缀（`--risk-labels` 为空时按此前缀自动识别） |
| `--risk-labels` | 否 | 空 | 手动指定风险标签列（逗号分隔），覆盖自动识别 |
| `--psi-bins` / `--iv-bins` | 否 | `10` / `10` | PSI / IV 等频分箱数 |
| `--invalid-values` | 否 | `-1,-2,-9,-99,-999,-9999,-99999` | 无效值哨兵集合（逗号分隔）。命中这些值的特征在「无效值检查」sheet 标记提醒，建议建模时替换为空值 |
| `--output-file` | 否 | `特征分析结果.xlsx` | Excel 文件名（Markdown 同名自动生成 `.md`） |
| `--output-dir` | 否 | 当前目录 | 产物（Excel + md + `_manifest.json`）输出目录 |

## 4. 输出产物

主交付为 Excel（11-sheet：样本分布 + 特征分布 + 6 张分月表 + PSI + IV + 无效值检查）+ **Markdown 报告**（与 Excel 同源），同目录另落 `_manifest.json`（参数溯源）：

```text
<output-dir>/
├── 特征分析结果.xlsx        # 11 sheet（见下表）
├── 特征分析结果.md          # Markdown 报告（一、样本分布 二、特征分布 三、分月PSI 四、分月IV 五、无效值检查）
└── _manifest.json           # schema_version / produced_by / params（含 split_config）/ files
```

| Sheet | 说明 |
|---|---|
| 样本分布 | 样本数 / 逾期数 / 逾期率三个矩阵上下堆叠，行=月份，列=风险标签（支持多标签） |
| 特征分布 | 全时段：覆盖率、均值、最小/最大值、1%/5%/25%/50%/75%/95%/99% 分位数 |
| 覆盖率 / 均值 / 最小值 / 最大值 / 标准差 / Nunique | 分月：行=特征，列=月份 |
| PSI | 分月 PSI，指定基准月为 0，剔除缺失值计算；<20 条有效记录标 NaN |
| IV | 分月 IV，指定风险标签，剔除缺失值计算；<20 条有效记录标 NaN |
| 无效值检查 | **仅当命中时生成**：列出含无效值哨兵（默认 `-1/-2/-999/-9999` 等）的特征、命中值、命中样本数/占比与替换建议。这些值往往是"无数据/拒贷/异常"占位符，建模时必须先替换为空值(NaN)，否则树模型/分箱会学到虚假取值边界 |

> 风险标签列中 `1` = 逾期（bad），`0` = 未逾期（good）；`_manifest.json` 记录全部 CLI 参数与文件清单，供复现与追溯。

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `classification-model-development` | pipeline 模式由其在 Stage 0 编排调起，传入 `--split-config`（feature_config.yaml）与 `--base-month` 确认结论 |
| 上游 | `data-cleaning` | 建模 session 内直接分析其产出的 `sample.parquet` |
| 下游 | `classification-model-training` | 消费 `sample.parquet` + `feature_config.yaml` 的 `model.split` 即时切分训练（本 skill 不切分） |
| 平行 | `classification-model-tuning` | 训练过程不通过 IV/PSI 筛选特征；本 skill 报告仅作人工参考 |
| 依赖 | `_modelevo-shared` | 可选复用 `config_io.load_config`（yaml 解析，脚本内已有独立 yaml 读取） |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 关键决策先确认 | 特征列范围、PSI 基准月（pipeline 默认第一个 OOT 月）、IV 标签三项属「关键决策确认门禁」范畴，**先列方案等用户确认再跑**；用户说"按默认"才用默认值 |
| ⚠️ 特征列必须数值型 | 非数值特征列报错（风险标签列除外） |
| ⚠️ 数据安全红线 | 分析报告不得透出身份证 / 手机号等明文个人数据；如数据含敏感列，先排除再分析 |
| ⚠️ 小样本标记 | 某月有效记录 <20 条时，该月 PSI/IV 单元格标 NaN，不硬算 |
| ⚠️ 无效值提醒（不自动清洗） | 检测到 `-1/-2/-999/-9999` 等哨兵值时在报告提醒，**仅标记不替换**；是否清洗、如何清洗（替换为空值 / 剔除）由人在建模阶段决策，本 skill 只产报告不动数据 |
| ⚠️ 不切分不筛选 | 本 skill 不产三档 splits、不产 IV/PSI 筛选 csv；切分后置到消费方，特征筛选不做（由 data-cleaning 的 feature-list + training 的边界安全过滤承担） |

## 7. 异常处理

| 异常 | 处理 |
|---|---|
| 数据文件不存在 / 格式不支持 | 报错并列出支持的格式（feather/csv/parquet） |
| 时间列不存在 | 报错并列出前 10 个可用列名 |
| 起始/结束特征列不在数据中 | 报错并让用户重输（区间含两端） |
| 特征列非数值 | 报错指出列名，提示排除或转换 |
| 分箱边界不足（`<2` 个唯一值） | 该特征 PSI/IV 返回 NaN，不中断 |
| `--split-config` 缺失 `model.split.oot_range` | 报错指明缺哪个字段，不静默兜底 |
| `--invalid-values` 含非数值项 | 忽略该项并打印警告，不影响主流程 |

## 8. 测试

```bash
python -m pytest model-skills/credit-data-analysis/tests/ -q
```
