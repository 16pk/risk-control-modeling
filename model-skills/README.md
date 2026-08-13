# Model Skills

业务建模全流程 Skill 集合，覆盖从需求采集、样本准备、特征工程、模型开发、评估对比到归档沉淀的完整链路。所有 Skill 遵循统一的命名约定与产物规范；用户的建模诉求先经 `model-task-routing` 评估，确认属于 **classification（分类）** 建模需求后由下游承接（本版本仅支持 classification）。

> 每个 Skill 位于独立目录下的 `SKILL.md`，包含 frontmatter（`name` + `description`）与正文（输入依赖 / 执行命令 / 参数说明 / 输出产物 / 关联 skill / 执行约束 / 异常处理）。

## 整体能力

- **需求评估（分类守门）**：一次性提问评估需求是否为 classification / 拒绝（**本版本仅支持 classification**）
- **分类建模**：需求规格化 → 历史模型推荐 → 特征准备 → baseline 模型开发（特征分析 / 训练 / 评估）→ LOOP 迭代式开发（确定优化方案 → 模型开发 → 横向对比，多轮循环）→ 收口归档 → **FICO 转换**（Stage 5，收口后询问，对 top1 上线候选概率分转 FICO 标准分）
  - 当前已支持的优化方向：超参调优（Optuna）、特征筛选（PSI/IV/缺失率）、换算法（xgb / dnn / lr）；特征衍生、样本调整、loss 优化、网络结构优化等为规划中的演化方向
- **共享能力**：特征匹配、特征分析（建模 pipeline 用 `feature-analysis`；用户主动发起样本/特征分析用 `credit-data-analysis` 分月体检）、模型知识库（业务领域知识 / 特征资产 / 历史模型档案 / 建模经验）
- **会话连续性**：基于 `runs/` 下的 `_manifest.json` 自动推断进度，支持断点续跑


## 目录结构

```
model-skills/
├── model-task-routing/                  # 需求评估守门：判定 classification / 拒绝
├── model-knowledge/                     # 模型知识库（业务领域 / 特征 / 历史模型 / 建模经验）
│
├── classification-model-orchestration/  # 分类流程编排器（接收 routing_input）
├── classification-model-task-spec/      # 分类任务规格化 + 样本分析
├── classification-model-recommend/      # 历史模型检索推荐
├── classification-model-development/    # 模型开发总控（迭代式编排）
├── classification-model-training/       # 模型训练（xgb / dnn / lr）
├── classification-model-tuning/         # 模型调参 / 特征筛选
├── classification-model-evaluation/     # 单模型标准化评估
├── classification-model-comparison/     # 多模型 N-way 对比
├── classification-model-report/         # session 聚合报告（6-sheet Excel）
├── credit-model-report/                 # 业务评估报告（打分 CSV → 回溯表/Lift/SWAP/打分分布，模板化 Excel）
├── score-to-fico/                       # 概率分 → FICO 标准分转换（development Stage 5 收口后调起；亦可独立调用）
│
├── feature-matching/                    # 特征匹配（拉样本+特征，跨流程共用）
├── feature-analysis/                    # 特征分析（建模 pipeline Stage 0：IV+AUC / 训练-OOT PSI / 切分，仅编排调起）
├── credit-data-analysis/                # 样本与特征分析（独立数据体检：分月 10-sheet Excel，用户主动发起时优先）
```

> **公共代码说明**：[`model-evo/_modelevo-shared/`](../_modelevo-shared/)也会自动安装到SKILL_ROOT目录下供各skill公共使用。

## Skill 清单

### 总入口 / 共享

| Skill | 说明 | 触发词示例 |
|---|---|---|
| `model-task-routing` | 建模需求评估守门，一次性提问评估需求是否为 classification / 拒绝，构造 routing_input JSON 透传下游 | 建模、新模型、模型需求、帮我建模、模型立项 |
| `feature-matching` | 从 Spark 宽表拉取样本+特征+标签，生成 spark-submit 提交脚本，确认后自动提交集群落 `sample.parquet`；或 `local_file` 模式下从本地 parquet/csv 直接转写+派生 `feature-list.csv` | 取数、拉样本、拉宽表、准备建模样本 |
| `feature-analysis` | 对候选特征做基础统计 / 单变量预测力（IV+AUC）/ 训练-OOT 稳定性（PSI）报告，仅产报告不自动剔特征，并按配置切分 Train/Test/OOT。**仅建模 pipeline Stage 0 使用，由 development 编排调起，不响应独立关键词触发** | （无独立触发词；`classification-model-development` Stage 0 调起） |
| `credit-data-analysis` | 独立数据体检：分月视角 10-sheet Excel（样本分布 / 特征分布 / 覆盖率 / 均值 / min/max/std/Nunique / PSI / IV）。用户主动发起样本及特征分析任务时**优先调用** | 样本分析、特征分析、特征IV、特征PSI、数据体检、分月监控、逾期率走势 |
| `model-knowledge` | 沉淀建模方法论、业务领域知识、特征资产、历史模型档案与建模经验教训，供检索复用 | 查历史建模经验、归档建模知识、查业务字段定义 |

### Classification 建模

| Skill | 说明 |
|---|---|
| `classification-model-orchestration` | 分类流程总调度，承接 routing_input，串联 task-spec → recommend → feature-matching → development，管理 session 目录与断点续跑 |
| `classification-model-task-spec` | 需求挖掘 + 样本分析，输出 4 段式 task-spec.md 与 `_manifest.json`，拉取样本（仅 fuid/label/f_p_date + 补充字段）并切分 Train/Test/OOT（OOT 按时间顺序且晚于训练窗；train/val 开发集可随机切分保证同分布） |
| `classification-model-recommend` | 从历史模型台账检索可复用模型，语义筛选排序 + 适配度评估，可选委托 evaluation 产三档评估 |
| `classification-model-development` | 开发总控，按 Stage 0~4 迭代式编排 feature-analysis / training / tuning / comparison，管理路径接力、决策点询问、report.md 回填 |
| `classification-model-training` | 训练 xgb / dnn / lr 模型，读上游 feature-analysis 切分数据，产八阶段产物，并与历史 baseline 做 AUC/KS/分档多维对比 |
| `classification-model-tuning` | 基于 baseline run 做超参调优（Optuna）或特征筛选（PSI/IV/缺失率），产 `-tuned` / `-feat` 新 run |
| `classification-model-evaluation` | 单模型标准化评估：AUC/KS/准确率/F1/十分桶排序性/业务指标 + 客群拆分，输出 JSON+MD+XLSX 三件套 |
| `classification-model-comparison` | 多模型 N-way 横向对比，消费 evaluation 的 JSON 做 delta 分析与缺口清单，输出含条件格式的 Excel |
| `classification-model-report` | 聚合 session 内建模信息产出 6-sheet Excel 报告，用户主动调起 |
| `credit-model-report` | 从打分 CSV 生成**业务评估报告**（Excel：回溯表/建模信息/KS/特征重要性/Lift+SWAP/打分分布 PSI+分桶+分段逾期率），支持新 vs 基线模型 SWAP 迁移与客群过滤，模板化输出 |
| `score-to-fico` | **概率分 → FICO 标准分转换**（LR 校准 + 标准分映射，范围约 [400,780]，分高险低）。两种入口：① development Stage 5 收口后总是询问调起，消费 top1 上线候选 run 的 predictions，产 `{run}/fico/`；② 独立调用（输入含概率分列+标签列样本 → coef.json + 打分 + 拟合方案）。触发词：转fico分、概率分转标准分、校准概率 |

> 模型上线（`model-publication`）、指标匹配（`metric-matching`）、演化方案（`classification-model-evolution-plan`）、分群建模（`classification-segment-model`）等为规划中的能力，**当前尚未实现**，不包含在本次交付内。

## 完整流程

```
用户建模诉求
   │
   ▼
model-task-routing（一次性提问 Q1/Q2/Q3 → 评估需求是否为 classification；本版本仅 classification）
   │
   ├── classification ──► classification-model-orchestration
   │                          │
   │                          ├─ 会话启动检查（扫描 runs/ → _manifest.json 推断进度）
   │                          ├─ classification-model-task-spec（需求确认 + 样本分析）
   │                          ├─ 创建任务目录 + 初始化 report.md
   │                          ├─ classification-model-recommend（历史模型推荐）
   │                          ├─ feature-matching（拉特征宽表 + 派生 feature-list.csv）
   │                          └─ 建模决策（询问用户）
   │                                  │
   │                                  ├── 是 ──► classification-model-development
   │                                  │            ├─ Stage 0: feature-analysis（一次性必跑）
   │                                  │            ├─ Stage 1: classification-model-training（baseline）
   │                                  │            ├─ Stage 2: 迭代（tuning 调参 / 特征筛选 / 换算法，loop）
   │                                  │            ├─ Stage 3: classification-model-comparison（session 级）
   │                                  │            └─ Stage 4: 收口 → report.md
   │                                  └── 否 ──► 流程结束
   │
   └── 拒绝（回归 / 多分类 / 聚类 / 时序 / NLP/CV）──► 给出转化建议
```

## 前置依赖

### 基础环境
参考仓库主[`model-evo/README.md`](../README.md)的前置依赖部分。


### 大数据取数（可选）

`feature-matching` / `classification-model-task-spec` / `classification-model-recommend` 的 **spark 模式**依赖 Spark 3.x + YARN + HDFS 集群、PySpark 与 Kerberos 认证，资源默认值见 [`_modelevo-shared/scripts/spark_defaults.template.yaml`](../_modelevo-shared/scripts/spark_defaults.template.yaml)（复制为 `spark_defaults.yaml` 并填本集群值，不入库）。**无集群时**用 `--mode local_file` 直接读本地 parquet/csv，可跑通除「Spark 取数」外的全部流程。

### 上下游数据前置

| 场景 | 前置数据 |
|---|---|
| 分类建模（local_file） | 一份含 `id + 特征列 + label`（可含日期列）的 parquet/csv |
| 分类建模（local_file 演示） | 内置演示数据 `data/demo/credit_risk_demo.csv`（1000 行 × 18 列，A 卡申请评分卡风格，含 `apply_date`/`is_bad`，坏率 9.6%，含少量缺失值），可先跑通全流程再替换真实数据 |
| 分类建模（spark） | 数仓样本表（含 `fuid/label/f_p_date`）+ 特征宽表 |
| 历史模型推荐 | `model-knowledge` 台账 `model_catalog.csv` 中有可检索的历史模型条目 |

## 使用说明

参考仓库[`model-evo/README.md`](../README.md)的**使用说明**部分。

## Session 产物结构

每个建模任务以 `runs/{YYYYMMDD-HHMMSS}-{model_name}/` 组织：

```
runs/20260624-114630-draw_willingness/
├── report.md                              # 项目总报告，各阶段逐步回填
├── task-spec/
│   ├── task-spec.md                       # 4 段式需求规格
│   └── _manifest.json                     # 结构化核心信息（含 routing 溯源字段）
├── data-profile/
│   ├── report.md / report.xlsx            # 样本分析报告
│   ├── _manifest.json
│   ├── _split_manifest.json               # 切分清单
│   └── {model_name}_sample_{YYYYMMDD}.parquet
├── model-recommend/                       # 历史模型推荐结果（local_file 模式下跳过）
├── sample-features/
│   ├── feature-matching/
│   │   ├── sample.parquet                 # 全量样本（id + features + label）
│   │   └── feature-list.csv
│   ├── feature-analysis/
│   │   └── analysis/{stats,iv_table,psi_table}.csv
│   ├── credit-data-analysis/            # 独立数据体检（可选，用户主动发起时）
│   │   └── 特征分析结果.xlsx + _manifest.json
│   └── splits/{train,test,oot}.parquet    # 三档切分（feature-analysis 产）
├── new-models/                            # 各次 run 的训练产物
│   └── {algo}-v{N}/                       # xgb-v1 / xgb-v1-feat / xgb-v1-tuned ...
│       ├── config/train_config.yaml
│       ├── features/ · model/ · evaluation/
│       ├── predictions/ · explainability/
│       ├── comparison/ · logs/
│       └── _manifest.json
├── model-comparison/                      # N-way 横向对比产物
└── scripts/                               # 各阶段执行命令 + 脚本源码快照（record_stage.py 落）
    ├── _manifest.json                     # 集中清单（按 stage 索引）
    └── {stage}/                           # task-spec / feature-matching / feature-analysis / training / tuning / comparison / fico / fill_report
        ├── <入口脚本>.py                  # 源码快照
        └── command.json                   # 执行命令详情（cmd/timestamp/sha256/python）
```

## 命名约定

| 类型 | 格式 | 示例 |
|---|---|---|
| Session 目录 | `YYYYMMDD-HHMMSS-{model_name}` | `20260624-114630-draw_willingness` |
| 模型简称 | 全小写英文 + 下划线，`{业务动作}_{预测目标}` | `draw_willingness`、`coupon_response` |
| 需求文档 | `task-spec.md` | 固定名称 |
| 元信息 | `_manifest.json` | 固定名称 |
| 项目报告 | `report.md` | 固定名称 |

命名前缀规则：仅 classification 专属 skill 加 `classification-` 前缀，跨流程共享 skill（`model-task-routing`、`feature-matching`、`feature-analysis`、`credit-data-analysis`、`model-knowledge`）不加前缀。每个 `SKILL.md` 的 `name` 字段必须等于其所在目录名。

## 公共代码

| 位置 | 作用 |
|---|---|
| `model-evo/_modelevo-shared/scripts/config_io.py` | yaml 配置读写 + 必填校验 + 数据安全红线（`load_config` / `validate_common` / `check_sensitive`，命中身份证/手机号即抛错） |
| `model-evo/_modelevo-shared/scripts/fetch_spark.py` | PySpark 集群取数 |
| `model-evo/_modelevo-shared/scripts/gen_fetch_command.py` | spark-submit wrapper 脚本生成 |
| `model-evo/_modelevo-shared/scripts/record_stage.py` | pipeline 阶段脚本快照记录：把「执行命令 + 入口脚本源码快照」落盘到 `<session>/scripts/<stage>/`（集中清单 `_manifest.json`，保证可复现） |
| `model-evo/_modelevo-shared/scripts/spark_defaults.template.yaml` | Spark 提交默认资源档模板（复制为 `spark_defaults.yaml` 后填本集群值，不入库） |
| `model-evo/_modelevo-shared/tests/` | 公共代码单元测试 |


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
4. **公共代码**：优先复用 `_modelevo-shared`（config_io 配置读写 + 数据安全红线），不重复造轮子
5. **校验 + 注册**：`validate_expert.py` → `register_expert.py`，禁止直接改 `marketplace.json`

### 接入 MCP 数据源 / 外部服务的契约

- **数据源类 MCP**（数仓、特征平台等）：拉取结果须符合 `feature-matching` 的样本契约——含 `id + 特征列 + label`（可含日期列），落盘 `sample.parquet` + `feature-list.csv`，或直接透传已有格式；同时在本 README「上下游数据前置」表格登记新数据来源
- **外部服务类 MCP**（模型服务、指标平台、监控告警等）：在对应 skill 的 SKILL.md 中声明调用方式与参数
- **数据安全红线**：任何 MCP 取数不得透出身份证 / 手机号等明文个人数据（`config_io.check_sensitive` 拦截）

### 关键决策确认门禁

所有 skill 执行过程中，凡影响建模结论的决策（场景/目标、训练窗口划分、特征筛选方案、算法与超参数、不平衡处理、模型选型交付）必须**先给方案、等用户确认、再执行**；默认值及门禁节点定义见 `agents/risk-control-modeling.md` 的「关键决策确认门禁」章节。

## 关键约束

- **评估在最前**：任何建模诉求先经 `model-task-routing` 做需求评估，下游不得直接承接未经评估的请求
- **信息透传不丢失**：routing_input JSON 中非 null 字段直接透传，下游不重复提问
- **文件落盘**：需求和分析结果必须保存为文件，不能只留在对话中
- **演化闭环**：evaluation/comparison → 回流 development/training 形成多轮迭代；完成后归档至 `model-knowledge`
- **数据安全红线**：`config_io.check_sensitive` 拦截配置中硬编码的身份证号 / 手机号；`where` / `sample_table` 等字段严禁写入明细个人数据
