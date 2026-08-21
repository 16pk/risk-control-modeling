# Model Skills

业务建模全流程 Skill 集合。`classification-model-development` 是唯一调度者，从需求澄清到收口打分的整个主链路都由其编排。所有 Skill 遵循统一的命名约定与产物规范。

> 每个 Skill 位于独立目录下的 `SKILL.md`，包含 frontmatter（`name` + `description`）与正文（输入依赖 / 执行命令 / 参数说明 / 输出产物 / 关联 skill / 执行约束 / 异常处理）。

## 整体能力

- **需求澄清（3 问）**：预测目标 Y 定义 / 数据路径+列名 / Train-Test-OOT 切分窗口（`classification-model-task-spec`，产单文件 `task-spec.md`；非二分类诉求在 task-spec 第零步一句话确认后终止）
- **独立任务轻量入口（v2.5.2）**：用户**只清洗 / 只分析**给定数据文件时，经 `classification-model-development` 的 `prep_sample.py clean/analyze` 直接调度——自动前置特征列识别（语义三分类 + 用户批量确认）→ 固化权威 feature-list.csv → 清洗（→ 特征分析），产物落标准 session 结构，与主链路互通
- **分类建模（5 步主链路）**：需求澄清 → 数据清洗 → 特征分析（credit-data-analysis，分月 PSI/IV 报告）→ 实验矩阵+对抗验证+规则诊断+Optuna 调优+转正（classification-model-experiments，v2.3 主链路默认）→ 收口默认打分（model-scoring）→ 可选 FICO / 业务报告
  - 迭代方向（仅用户主动要求）：继续实验（新样本/特征方案 或 加大矩阵）
- **共享能力**：数据清洗、特征分析（credit-data-analysis 双模式：pipeline 特征分析 + 独立数据体检）、模型知识库
- **会话连续性**：基于 session 根 `_manifest.json` 自动推断进度，支持断点续跑

## 目录结构

```
model-skills/
├── model-knowledge/                     # 模型知识库（业务领域 / 特征 / 历史模型 / 建模经验，LLM 内部知识）
│
├── classification-model-task-spec/      # 需求澄清 3 问，产单文件 task-spec.md
├── classification-model-development/    # 模型开发总控（唯一调度者，吸收原 orchestration）
├── classification-model-experiments/    # v2.3 主链路默认训练：样本×特征正交矩阵 + 对抗 + 规则诊断 + Optuna 调优 + 转正
├── classification-model-package/        # 定版模型 → 独立交付代码包（可选，仅用户主动触发）
├── credit-model-report/                 # 业务评估报告（回溯表/Lift/SWAP/打分分布，模板化 Excel，可选）
├── score-to-fico/                       # 概率分 → FICO 标准分转换（可选，仅用户主动触发）
├── model-scoring/                       # 定版模型打分（development 收口后默认执行）
│
├── feature-classification/              # 特征列识别与复用（语义三分类 + 用户批量确认，产权威特征清单）
├── data-cleaning/                       # 数据清洗（哨兵值替换 + 用户日期去重 + 消费权威清单）
└── credit-data-analysis/                # 特征分析（双模式：pipeline 特征分析 + 独立数据体检，分月 xlsx + md）
```

> **公共代码说明**：`model-skills/_modelevo-shared/scripts/`（含统一指标 `metrics.py`）经各 skill 的 `_bootstrap.py` 自动注入。

## Skill 清单

### 共享

| Skill | 说明 | 触发词示例 |
|---|---|---|
| `feature-classification` | 特征列识别与复用：探查阶段语义三分类（feature / non_feature / ambiguous，规则库 v0）+ 通配符分组 + 用户批量确认，产出权威 `feature-list.csv`（全 pipeline 唯一真相）与逐列分类档案（含判定人）。红线：`fpd*`/`dpd*` 标签列禁入特征集。跨 session 档案复用，仅列集合变化时增量重分类。**只清洗/只分析独立任务也须先经本环节** | （无独立触发词；`classification-model-development` Stage 1 / 轻量入口 `prep_sample.py` 调起） |
| `data-cleaning` | 数据清洗：哨兵值/无效值替换为 NaN + 按用户+日期去重 + 消费权威 `feature-list.csv`（`--feature-list-source`），产出清洗后 `sample.parquet` 与可复用清洗方案。**由编排层调起，不设独立触发词**（独立任务经 `prep_sample.py clean`） | （无独立触发词；`classification-model-development` 调起） |
| `credit-data-analysis` | 双模式：①独立数据体检（分月 xlsx + md，PSI 基准月用户指定，**先清洗再分析**）；②pipeline 特征分析（development Stage 2 调起，PSI 基准月默认第一个 OOT 月须用户确认）。**不切分、不产筛选 csv** | 样本分析、特征分析、特征IV、特征PSI、数据体检、分月监控、逾期率走势 |
| `model-scoring` | 定版模型打分：用收口确认的定版模型（`finalized_model.json`）对清洗后 `sample.parquet` 跑推理，产出违约概率分 `score`，透传所有非特征列。**收口后默认执行（用户可叫停）** | （无独立触发词；`classification-model-development` Stage 6 调起） |
| `model-knowledge` | 沉淀建模方法论、业务领域知识、特征资产、历史模型档案与建模经验教训，供检索复用（LLM 内部知识，不作强制前置） | 查历史建模经验、归档建模知识、查业务字段定义 |

### Classification 建模

| Skill | 说明 |
|---|---|
| `classification-model-task-spec` | 需求澄清 3 问（Y 定义 / 数据路径+列名 / 切分窗口），输出单文件 `task-spec.md` + `_manifest.json`（split_ranges 记录入口）；非二分类诉求在第零步一句话确认后终止 |
| `classification-model-development` | 开发总控（唯一调度者）：串联 task-spec → data-cleaning → credit-data-analysis → experiments（v2.3 主链路默认）→ 收口打分，管理路径接力、决策点询问（2 必问 + 矩阵方案确认）、report.md 回填（4 节）、断点续跑 |
| `classification-model-experiments` | **v2.3 主链路默认训练模块**：lgb baseline → 样本方案（全量/最近N月/线性时间加权/对抗剔除）× 特征方案（全量/importance 95%/IV-PSI/对抗剔除）正交矩阵 → 对抗验证（lgb train-vs-oot 双产出）→ leaderboard（OOT AUC 排序 + 乐观偏差标注）→ 每算法 winner 规则诊断（五状态，Optuna 前执行并驱动锚点）→ Optuna 邻域调优（-opt，well_fit 可跳过）→ top10 展示用户确认后转正（`new-models/` + `finalized_model.json`）。仅消费 `sample.parquet` + `feature-list.csv` + `model.split`。**红线例外（用户授权本模块）：对抗格/IV-PSI 格 OOT 可参与对抗训练与筛选统计（禁早停/禁进训练/禁结构选择），OOT 指标标注乐观偏差** |
| `classification-model-package` | **定版模型 → 独立交付代码包**：消费 session 定版产物（`finalized_model.json` + `new-models/{run}/model` + `cleaning-scheme` + 权威 feature-list + `fico/coef.json`），组装 `delivery/` 自包含包（数据清理→打分→可选 FICO 转分，零引用专家包仅依赖 pip 包，一条命令跑通）。**可选：仅用户主动触发**（收口后出口，不默认执行） | 打包交付、组装成可交付代码包、交付给工程 |
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
   ├─ Stage 1: feature-classification 特征列识别（语义三分类 + 用户批量确认）
   │            → 产 sample-features/feature-classification.json + feature-list.csv（权威清单）
   ├─ Stage 2: data-cleaning → sample.parquet + feature-list.csv（经 --feature-list-source 取交集）
   ├─ Stage 3: credit-data-analysis（pipeline 模式，分月 PSI/IV 报告，PSI 基准月=首个OOT月须确认）
   ├─ Stage 4: experiments（v2.3 主链路默认，读 sample.parquet + feature-list.csv + model.split）
   │            → 矩阵 + 对抗 + 规则诊断 + Optuna 调优 + top10 转正
   │            → experiments/ + new-models/{algo}-v{N}/
   ├─ Stage 5: 迭代（可选，loop）：继续实验（新样本/特征方案 或 加大矩阵，仅用户主动要求）
   ├─ Stage 6: 收口 → report.md（4 节）+ finalized_model.json
   ├─ Stage 7: model-scoring（默认执行，用户可叫停）→ scoring/score_sample.parquet
  └─ Stage 8: 可选（仅用户主动触发）：score-to-fico / credit-model-report / classification-model-package（定版模型 → 独立交付包）

独立任务（只清洗 / 只分析，不经主链路）
   │
   ▼
development 轻量入口 prep_sample.py clean|analyze
   ├─ feature-classification（探查三分类 + 编排层交互确认 id/dt/label + 批量确认 exclude/keep）
   │    → sample-features/feature-classification.json + feature-list.csv（权威清单）
   ├─ data-cleaning（--feature-list-source 消费权威清单，哨兵强门禁 --auto-confirm 续跑）
   │    → sample-features/data-cleaning/sample.parquet
   └─ [analyze] credit-data-analysis（独立体检模式，--feature-list 精确选列，PSI 基准月确认）
        → sample-features/credit-data-analysis/特征分析结果.xlsx/.md
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
│   ├── feature-classification.json      # 特征列语义三分类档案（含判定人 rule/user）
│   ├── feature-list.csv                 # 权威特征清单（确认后，全 pipeline 唯一真相）
│   ├── data-cleaning/
│   │   ├── sample.parquet               # 清洗后样本（id + features + label）
│   │   └── feature-list.csv             # 与权威清单取交集后的派生产物
│   └── credit-data-analysis/
│       ├── 特征分析结果.xlsx / .md         # 分月 PSI/IV 体检报告
│       └── _manifest.json
├── experiments/                           # v2.3 主链路：实验矩阵产物（experiments 模块）
│   ├── matrix-plan.md                     # 矩阵规划 + 断点状态
│   ├── leaderboard.md / .xlsx             # OOT AUC 排序（含乐观偏差标注）
│   └── {algo}-{scheme}-{feat}-v{N}[-opt]/ # 各格实验（manifest/model/evaluation/data/...）
├── new-models/                            # 各次 run 的训练产物（experiments 转正）
│   └── {algo}-v{N}/                       # lgb-v1 / xgb-v2 ...
│       ├── model/model.pkl + model_meta.json（experiments 转正产物）
│       └── config.json                    # produced_by=skills/model-experiments
├── finalized_model.json                   # 定版标记（收口确认上线候选后落）
├── scoring/                               # 定版模型打分产物（默认执行）
│   └── score_sample.parquet               # 透传非特征列 + score 概率列
└── fico/                                  # FICO 转换产物（可选，仅用户主动触发）
    ├── coef.json                          # LR 校准参数
    ├── fico_predictions.parquet           # 转分结果（含 bscore）
    └── fitting-summary.{json,md}          # 拟合方案
└── delivery/                              # 独立交付包（可选，仅用户主动触发，classification-model-package）
    ├── run.py + pipeline/ + assets/ + requirements.txt + README.md + package-manifest.json
```

> 可追溯性由 `_manifest.json` + `report.md` 承担；session 级产物只含上述目录，不再产出 session 级 `splits/`、`feature-analysis/`、`scripts/` 快照层、`deliverables.md`。

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
| `_modelevo-shared/scripts/metrics.py` | **统一指标库**：AUC / KS / Gini / PSI / IV / 分类指标 / 分桶排序性，各 skill 经 `_bootstrap.py` 复用 |
| `_modelevo-shared/scripts/feature_knowledge.py` | 特征清单索引解析（按 feature_table / business_domain 从 feature-knowledge.md 匹配） |
| `_modelevo-shared/scripts/record_stage.py` | （保留文件，不再被编排调用；可追溯性收敛为 `_manifest.json` + `report.md`） |
| `_modelevo-shared/tests/` | 公共代码单元测试 |

## 扩展指南

本专家包设计为可插拔扩展，新增能力按以下约定接入，保证契约一致、可追溯。

### 新增 Skill 的标准流程

1. **创建目录**：`model-skills/{name}/SKILL.md`（frontmatter `name` 必须等于目录名），按需附 `scripts/`（可执行脚本 + `_bootstrap.py`）、`references/`（方法论文档）、`tests/`、`config/`
2. **登记点（缺一不可）**：
   - `.codebuddy-plugin/plugin.json` 的 `skills` 数组（新增后须重新校验 + 注册）
   - 本 README 的「Skill 清单」表格（含触发词示例）
   - `agents/risk-control-modeling.md` 的「技能-职责映射表」章节
   - 涉及关键决策的 skill 须声明走 `classification-model-development` 的「决策点话术（门禁收敛）」对应节点
3. **命名约定**：classification 专属加 `classification-` 前缀、跨流程共享不加前缀
4. **公共代码**：优先复用 `_modelevo-shared`（config_io 配置读写 + 数据安全红线 + metrics 统一指标），不重复造轮子
5. **校验 + 注册**：用平台 **expert-manager** 插件脚本 `validate_expert.py` → `register_expert.py`（位于 `~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/`），禁止直接改 `marketplace.json`

### 接入 MCP 数据源 / 外部服务的契约

- **数据源类 MCP**（数仓、特征平台等）：拉取结果须符合 `data-cleaning` 的样本契约——含 `id + 特征列 + label`（可含日期列），落盘 `sample.parquet` + `feature-list.csv`，或直接透传已有格式；同时在本 README「上下游数据前置」表格登记新数据来源
- **外部服务类 MCP**（模型服务、指标平台、监控告警等）：在对应 skill 的 SKILL.md 中声明调用方式与参数
- **数据安全红线**：任何 MCP 取数不得透出身份证 / 手机号等明文个人数据（`config_io.check_sensitive` 拦截）

### 关键决策确认门禁

所有 skill 执行过程中，凡影响建模结论的决策（预测目标、切分窗口、超参数）必须**先给方案、等用户确认、再执行**；默认值及门禁节点定义见 `classification-model-development` SKILL.md 的「决策点话术（门禁收敛）」章节。

## 关键约束

- **单一调度者**：任何建模任务由 `classification-model-development` 编排，各 skill 不绕过总控自写脚本（独立任务走轻量入口 `prep_sample.py clean/analyze`，同样是 development 编排）
- **独立任务先识别再清洗/分析**：只清洗 / 只分析也必须先经 `feature-classification` 产出权威清单再清洗/分析（轻量入口已串联，禁止直接跳过特征识别单跑清洗/分析）
- **特征清单唯一真相**：`sample-features/feature-list.csv`（feature-classification 确认后权威清单），data-cleaning / credit-data-analysis / experiments 全部消费同一份，不各自派生
- **切分唯一真相**：`model.split`（feature_config.yaml / train_config.yaml）三档区间；切分在训练消费时即时进行，不落 session 级 `splits/`
- **训练不筛特征**：训练过程不通过 IV/PSI 指标筛选特征（boundary_filter 只做常量/泄漏/ID/全缺失安全过滤）
- **文件落盘**：需求和分析结果必须保存为文件，不能只留在对话中
- **演化闭环**：experiments 实验矩阵 → leaderboard 评选 → 规则诊断 + Optuna 调优 → 转正 → 回流 development 形成多轮迭代；完成后归档至 `model-knowledge`
- **数据安全红线**：`config_io.check_sensitive` 拦截配置中硬编码的身份证号 / 手机号；`where` / `sample_table` 等字段严禁写入明细个人数据
- **实验台红线例外（仅 `classification-model-experiments`）**：用户授权对抗格 / IV-PSI 格可让 OOT 参与对抗训练与筛选统计（禁早停 / 禁进训练集 / 禁结构超参选择），对应 OOT 指标标注乐观偏差
