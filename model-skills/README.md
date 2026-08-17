# Model Skills

业务建模全流程 Skill 集合。v2.0 精简后，编排层由三层合并为一层：`classification-model-development` 是唯一调度者，从需求澄清到收口打分的整个主链路都由其编排。所有 Skill 遵循统一的命名约定与产物规范。

> 每个 Skill 位于独立目录下的 `SKILL.md`，包含 frontmatter（`name` + `description`）与正文（输入依赖 / 执行命令 / 参数说明 / 输出产物 / 关联 skill / 执行约束 / 异常处理）。

## 整体能力

- **需求澄清（3 问）**：预测目标 Y 定义 / 数据路径+列名 / Train-Test-OOT 切分窗口（`classification-model-task-spec`，产单文件 `task-spec.md`；非二分类诉求在 task-spec 第零步一句话确认后终止）
- **分类建模（5 步主链路）**：需求澄清 → 数据清洗 → 特征分析（credit-data-analysis，分月 PSI/IV 报告）→ baseline 训练（training 内嵌评估，切分后置即时切分）→ 收口默认打分（model-scoring）→ 可选 FICO / 业务报告
  - 迭代方向（仅用户主动要求）：超参调优（Optuna）、换算法（xgb / dnn / lr）、特征筛选（select_features，训练不通过 IV/PSI 自动筛特征）
- **共享能力**：数据清洗、特征分析（credit-data-analysis 双模式：pipeline 特征分析 + 独立数据体检）、模型知识库
- **会话连续性**：基于 session 根 `_manifest.json` 自动推断进度，支持断点续跑

## 目录结构

```
model-skills/
├── model-knowledge/                     # 模型知识库（业务领域 / 特征 / 历史模型 / 建模经验，LLM 内部知识）
│
├── classification-model-task-spec/      # 需求澄清 3 问，产单文件 task-spec.md
├── classification-model-development/    # 模型开发总控（唯一调度者，吸收原 orchestration）
├── classification-model-training/       # 模型训练（xgb / dnn / lr）+ 内嵌评估（eval_single.py）
├── classification-model-tuning/         # 模型调参 / 特征筛选（可选）
├── classification-model-comparison/     # 多模型 N-way 对比（可选）
├── credit-model-report/                 # 业务评估报告（回溯表/Lift/SWAP/打分分布，模板化 Excel，可选）
├── score-to-fico/                       # 概率分 → FICO 标准分转换（可选，仅用户主动触发）
├── model-scoring/                       # 定版模型打分（development 收口后默认执行）
│
├── data-cleaning/                       # 数据清洗（哨兵值替换 + 用户日期去重 + 派生特征清单）
└── credit-data-analysis/                # 特征分析（双模式：pipeline 特征分析 + 独立数据体检，分月 xlsx + md）
```

> **公共代码说明**：`model-skills/_modelevo-shared/scripts/`（含统一指标 `metrics.py`）经各 skill 的 `_bootstrap.py` 自动注入。

## Skill 清单

### 共享

| Skill | 说明 | 触发词示例 |
|---|---|---|
| `data-cleaning` | 数据清洗：哨兵值/无效值替换为 NaN + 按用户+日期去重 + 派生 `feature-list.csv`，产出清洗后 `sample.parquet` 与可复用清洗方案。**仅由编排层调起，不设独立触发词** | （无独立触发词；`classification-model-development` 调起） |
| `credit-data-analysis` | 双模式：①独立数据体检（分月 xlsx + md，PSI 基准月用户指定）；②pipeline 特征分析（development Stage 2 调起，PSI 基准月默认第一个 OOT 月须用户确认）。**不切分、不产筛选 csv** | 样本分析、特征分析、特征IV、特征PSI、数据体检、分月监控、逾期率走势 |
| `model-scoring` | 定版模型打分：用收口确认的定版模型（`finalized_model.json`）对清洗后 `sample.parquet` 跑推理，产出违约概率分 `score`，透传所有非特征列。**收口后默认执行（用户可叫停）** | （无独立触发词；`classification-model-development` Stage 6 调起） |
| `model-knowledge` | 沉淀建模方法论、业务领域知识、特征资产、历史模型档案与建模经验教训，供检索复用（LLM 内部知识，不作强制前置） | 查历史建模经验、归档建模知识、查业务字段定义 |

### Classification 建模

| Skill | 说明 |
|---|---|
| `classification-model-task-spec` | 需求澄清 3 问（Y 定义 / 数据路径+列名 / 切分窗口），输出单文件 `task-spec.md` + `_manifest.json`（split_ranges 记录入口）；非二分类诉求在第零步一句话确认后终止 |
| `classification-model-development` | 开发总控（唯一调度者）：串联 task-spec → data-cleaning → credit-data-analysis → training → 收口打分，管理路径接力、决策点询问（2 必问 + 1 确认）、report.md 回填（4 节）、断点续跑 |
| `classification-model-training` | 训练 xgb / dnn / lr 模型，读 `sample.parquet` + `model.split` 即时切分（写 run 内部 `data/splits/` 临时目录），内嵌 `eval_single.py` 评估产标准化三件套，并与历史 baseline 做 AUC/KS/分档多维对比 |
| `classification-model-tuning` | 基于 baseline run 做超参调优（Optuna）或特征筛选（PSI/IV/缺失率，数据直算），产 `-tuned` / `-feat` 新 run。**可选：仅用户主动要求时调度** |
| `classification-model-comparison` | 多模型 N-way 横向对比，消费 training 产出的 eval JSON 做 delta 分析与缺口清单，输出含条件格式的 Excel。**可选：仅用户主动要求或配 `baseline_eval_dir` 时** |
| `credit-model-report` | 从打分 CSV 生成**业务评估报告**（Excel：回溯表/建模信息/KS/特征重要性/Lift+SWAP/打分分布 PSI+分桶+分段逾期率），支持新 vs 基线模型 SWAP 迁移与客群过滤。**可选：仅用户主动要求** |
| `score-to-fico` | **概率分 → FICO 标准分转换**（LR 校准 + 标准分映射，范围约 [400,780]）。**可选：仅用户主动要求（收口后不再默认询问）**，消费 model-scoring 打分结果，产 session 根 `fico/` |

> 模型上线（`model-publication`）、指标匹配（`metric-matching`）、演化方案（`classification-model-evolution-plan`）、分群建模（`classification-segment-model`）等为规划中的能力，**当前尚未实现**。

## 完整流程

```
用户建模诉求
   │
   ▼
classification-model-development（唯一调度者）
   │
   ├─ Stage 0: task-spec 需求澄清 3 问（Y定义 / 数据路径+列名 / 切分窗口）
   │            → 产 task-spec/task-spec.md + _manifest.json
   ├─ Stage 1: data-cleaning → sample.parquet + feature-list.csv
   ├─ Stage 2: credit-data-analysis（pipeline 模式，分月 PSI/IV 报告，PSI 基准月=首个OOT月须确认）
   ├─ Stage 3: training（读 sample.parquet + model.split 即时切分，内嵌评估）
   │            → new-models/{algo}-v{N}/
   ├─ Stage 4: 迭代（可选，loop）：run_tuning / 换算法 / select_features（仅用户主动要求）
   ├─ Stage 5: 收口 → report.md（4 节）+ finalized_model.json
   ├─ Stage 6: model-scoring（默认执行，用户可叫停）→ scoring/score_sample.parquet
   └─ Stage 7: 可选（仅用户主动触发）：score-to-fico / credit-model-report
```

## 前置依赖

### 基础环境

参考仓库主 [`README.md`](../README.md) 的前置依赖部分。

### 数据前置

全仓库已废除 spark 取数，建模 pipeline 仅支持本地文件模式（parquet/csv/feather），无需 Spark 集群。

### 上下游数据前置

| 场景 | 前置数据 |
|---|---|
| 分类建模 | 一份含 `id + 特征列 + label`（可含日期列）的 parquet/csv/feather |

## 使用说明

参考仓库 [`README.md`](../README.md) 的**使用说明**部分。

## Session 产物结构

每个建模任务以 `runs/{YYYYMMDD-HHMMSS}-{model_name}/` 组织：

```
runs/20260624-114630-draw_willingness/
├── report.md                              # 项目总报告（4 节：需求 / 样本与特征 / 模型迭代 / 结论与交付）
├── task-spec/
│   ├── task-spec.md                       # 单文件需求规格（3 问结论）
│   └── _manifest.json                     # 结构化核心信息（含 split_ranges）
├── sample-features/
│   ├── data-cleaning/
│   │   ├── sample.parquet                 # 清洗后样本（id + features + label）
│   │   └── feature-list.csv
│   └── credit-data-analysis/
│       ├── 特征分析结果.xlsx / .md         # 分月 PSI/IV 体检报告
│       └── _manifest.json
├── new-models/                            # 各次 run 的训练产物
│   └── {algo}-v{N}/                       # xgb-v1 / xgb-v1-feat / xgb-v1-tuned ...
│       ├── config/train_config.yaml       # 含 model.split（切分唯一真相）
│       ├── data/splits/                   # 即时切分的 run 内部临时三档（非 session 交付层）
│       ├── features/ · model/ · evaluation/（内嵌 eval_single 产三件套）
│       ├── predictions/ · explainability/
│       ├── comparison/ · logs/
│       └── _manifest.json
├── finalized_model.json                   # 定版标记（收口确认上线候选后落）
├── scoring/                               # 定版模型打分产物（默认执行）
│   └── score_sample.parquet               # 透传非特征列 + score 概率列
└── fico/                                  # FICO 转换产物（可选，仅用户主动触发）
    ├── coef.json                          # LR 校准参数
    ├── fico_predictions.parquet           # 转分结果（含 bscore）
    └── fitting-summary.{json,md}          # 拟合方案
```

> v2.0 已删除：`splits/{train,test,oot}.parquet`（session 级）、`feature-analysis/`、`scripts/` 快照层、`deliverables.md`。

## 命名约定

| 类型 | 格式 | 示例 |
|---|---|---|
| Session 目录 | `YYYYMMDD-HHMMSS-{model_name}` | `20260624-114630-draw_willingness` |
| 模型简称 | 全小写英文 + 下划线，`{业务动作}_{预测目标}` | `draw_willingness`、`coupon_response` |
| 需求文档 | `task-spec.md` | 固定名称 |
| 元信息 | `_manifest.json` | 固定名称 |
| 项目报告 | `report.md` | 固定名称 |

命名前缀规则：仅 classification 专属 skill 加 `classification-` 前缀，跨流程共享 skill（`data-cleaning`、`credit-data-analysis`、`model-knowledge`、`model-scoring`、`score-to-fico`）不加前缀。每个 `SKILL.md` 的 `name` 字段必须等于其所在目录名。

## 公共代码

| 位置 | 作用 |
|---|---|
| `_modelevo-shared/scripts/config_io.py` | yaml 配置读写 + 必填校验 + 数据安全红线（`load_config` / `validate_common` / `check_sensitive`，命中身份证/手机号即抛错） |
| `_modelevo-shared/scripts/date_utils.py` | 日期归一化工具（YYYY-MM-DD / YYYYMMDD 双兼容，`parse_date` / `parse_date_pair` / `month_prefix` / `shift_days` 等） |
| `_modelevo-shared/scripts/gen_feature_list.py` | 特征清单加载/识别（`.csv` 取 `feature_name` 列 / `.txt` 按行 / 跳过注释 / 去重保序） |
| `_modelevo-shared/scripts/metrics.py` | **统一指标库（v2.0 新增）**：AUC / KS / Gini / PSI / IV / 分类指标 / 分桶排序性，各 skill 经 `_bootstrap.py` 复用 |
| `_modelevo-shared/scripts/feature_knowledge.py` | 特征清单索引解析（按 feature_table / business_domain 从 feature-knowledge.md 匹配） |
| `_modelevo-shared/scripts/record_stage.py` | （保留文件，v2.0 起不再被编排调用；可追溯性收敛为 `_manifest.json` + `report.md`） |
| `_modelevo-shared/tests/` | 公共代码单元测试 |

## 扩展指南

本专家包设计为可插拔扩展，新增能力按以下约定接入，保证契约一致、可追溯。

### 新增 Skill 的标准流程

1. **创建目录**：`model-skills/{name}/SKILL.md`（frontmatter `name` 必须等于目录名），按需附 `scripts/`（可执行脚本 + `_bootstrap.py`）、`references/`（方法论文档）、`tests/`、`config/`
2. **登记点（缺一不可）**：
   - `.codebuddy-plugin/plugin.json` 的 `skills` 数组（新增后须重新校验 + 注册）
   - 本 README 的「Skill 清单」表格（含触发词示例）
   - `agents/risk-control-modeling.md` 的「可调用的建模技能」章节
   - 涉及关键决策的 skill 须声明走「关键决策确认门禁」对应节点
3. **命名约定**：classification 专属加 `classification-` 前缀、跨流程共享不加前缀
4. **公共代码**：优先复用 `_modelevo-shared`（config_io 配置读写 + 数据安全红线 + metrics 统一指标），不重复造轮子
5. **校验 + 注册**：用平台 **expert-manager** 插件脚本 `validate_expert.py` → `register_expert.py`（位于 `~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/`），禁止直接改 `marketplace.json`

### 接入 MCP 数据源 / 外部服务的契约

- **数据源类 MCP**（数仓、特征平台等）：拉取结果须符合 `data-cleaning` 的样本契约——含 `id + 特征列 + label`（可含日期列），落盘 `sample.parquet` + `feature-list.csv`，或直接透传已有格式；同时在本 README「上下游数据前置」表格登记新数据来源
- **外部服务类 MCP**（模型服务、指标平台、监控告警等）：在对应 skill 的 SKILL.md 中声明调用方式与参数
- **数据安全红线**：任何 MCP 取数不得透出身份证 / 手机号等明文个人数据（`config_io.check_sensitive` 拦截）

### 关键决策确认门禁

所有 skill 执行过程中，凡影响建模结论的决策（预测目标、切分窗口、超参数）必须**先给方案、等用户确认、再执行**；默认值及门禁节点定义见 `agents/risk-control-modeling.md` 的「关键决策确认门禁」章节。

## 关键约束

- **单一调度者**：任何建模任务由 `classification-model-development` 编排，各 skill 不绕过总控自写脚本
- **切分唯一真相**：`model.split`（feature_config.yaml / train_config.yaml）三档区间；切分在训练消费时即时进行，不落 session 级 `splits/`
- **训练不筛特征**：训练过程不通过 IV/PSI 指标筛选特征（boundary_filter 只做常量/泄漏/ID/全缺失安全过滤）
- **文件落盘**：需求和分析结果必须保存为文件，不能只留在对话中
- **演化闭环**：training 内嵌评估 → 可选 comparison → 回流 development/training 形成多轮迭代；完成后归档至 `model-knowledge`
- **数据安全红线**：`config_io.check_sensitive` 拦截配置中硬编码的身份证号 / 手机号；`where` / `sample_table` 等字段严禁写入明细个人数据
