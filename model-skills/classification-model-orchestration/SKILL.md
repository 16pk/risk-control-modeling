---
name: classification-model-orchestration
description: 分类建模流程编排（覆盖营销/增长/获客/运营/风控场景），作为 model-task-routing 的下游。当 task_type=classification 后拉起本 skill，接收 routing_input JSON，自动创建任务目录、管理 session 时间戳和命名规范，通过 report.md 和各目录 _manifest.json 追踪进度。
---

# 分类建模流程编排器

## 1. 角色定义

你是**分类建模流程**的总调度。职责不是做需求挖掘或方案设计，而是**确保分类流水线按顺序执行、文件落在正确的位置、命名符合规范**。本 skill 由 `model-task-routing` 在判定为 classification 方向后拉起，接收 routing_input JSON，不向用户重复询问已知字段。

核心原则：
- **下游身份**：所有请求先经 `model-task-routing` 路由后再进入
- **信息透传**：routing_input JSON 中的已知字段直接透传给下游，不重复提问
- **每个新需求独立对待**，不假设与历史需求有关联
- **文件命名规范化**，确保后续 skill 能自动定位
- **Session 组织**：每次任务以 `{timestamp}-{model_name}` 组织
- **进度透明**：通过 `report.md` 和各目录 `_manifest.json` 追踪

## 2. 输入依赖

### 2.1 routing_input JSON（从 model-task-routing 接力）

启动时**先验证** routing_input JSON 是否存在且 `task_type == "classification"`，缺关键字字段 → 报错并指明缺哪个。

| 字段 | 含义 |
|------|------|
| `task_type` | 必须为 `"classification"` |
| `routing_basis` | 路由判定依据 |
| `user_raw_request` | 用户最初诉求原话 |
| `routed_at` | 路由时间戳 |

### 2.2 路径约定

- `<session_dir>` = `runs/{timestamp}-{model_name}/`，timestamp 为 session 启动时间（`YYYYMMDD-HHMMSS`），model_name 全小写+下划线
- 本 skill 在 task-spec 完成后创建 `<session_dir>` 及子目录

### 2.3 触发条件

本 skill **不由用户直接触发**，由 `model-task-routing` 在判定 `task_type == "classification"` 后拉起。

## 3. 工作流程

### 3.1 会话启动检查

扫描 `runs/` 下所有 `{timestamp}-{model_name}` 命名的任务文件夹，按时间戳倒序取最近 5 个，对每个文件夹读 `task-spec/_manifest.json` 推断进度，主动询问用户继续历史或新建。**用户已表达"新建"/"继续某 session"等意图的，直接按其意图执行。**

进度推断规则（8 阶段：task-spec / 样本分析 / model-recommend / feature-matching / feature-analysis / Dev Stage 1~3）详见 [references/session-progress-inference.md](references/session-progress-inference.md)。

### 3.2 数据源与澄清模式

新建 session 时**不询问驾驶模式、不提供 spark 取数**。**建模所需数据文件默认在本地**（local_file），即用户需提供本地 parquet/csv（含 `id_cols + label_col + dt_col + features`）。本流程**所有需求维度与决策点都必须向用户澄清**，不使用默认值填充、不跳过任何建模决策询问。

> **切分规则**：
> 1. **OOT 必须按 `dt_col` 升序、严格晚于训练窗**（跨时间稳定性检验是上线前置）；禁止用训练期样本充当 OOT。
> 2. **train/val 开发集允许随机切分（保证同分布，建模常用）或按时间切分**，由用户选择；随机切分时记录 seed 保证可复现，并标注"val 偏乐观、以 OOT 为裁决集"。
> 3. **OOT 评估前必须剔除标签缺失/非法样本**（切分时自动剔除 + 评估时防御剔除双保险）。
> 用户输入二选一：(a) 显式时间区间；(b) 比例（如 7:2:1，OOT 段按时间顺序切到对应比例）。

### 3.3 需求确认 + 样本分析 + 创建目录 + report.md 初始化

调用 `classification-model-task-spec`，**跳过其问题类型判定**（上游已判定），将 routing_input JSON 透传，`mode=local_file`。task-spec 自身的 `fetch_sample_task_spec.py --mode local_file` 拉本地样本，`run_sample_analysis_task_spec.py` 做切分+分析。所有需求维度（WHO/WHAT/HOW GOOD/CONSTRAINTS/SAMPLE）及切分方式均须向用户逐一澄清，不得用默认值替代。

**完成标准**：需求成熟度 A 或 B 级，且样本分析通过。

**出口校验（强制）**：

```bash
[ -f <session_dir>/task-spec/task-spec.md ] && \
[ -f <session_dir>/task-spec/_manifest.json ] && \
[ -f <session_dir>/task-spec/.done ] || \
echo "ERROR: task-spec 三件套缺失"
```

task-spec 完成后立即创建 session 目录、保存 task-spec 三件套（含 routing 溯源）、初始化 `report.md`：

```bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p runs/${TIMESTAMP}-{model_name}/{task-spec,data-profile}
```

**report.md 章节结构**（7 节固定顺序 + 1 附录，编号统一汉字 `一、二、...七、`，与 `fill_report.py` 锚点对齐）：

```markdown
# 建模全流程报告 — {model_name}
> 模型简称 / 需求 / 源表 / session

## 一、需求        ← task-spec 完成后填
## 二、样本        ← data-profile 完成后填
## 三、历史模型推荐 ← recommend 完成后填
## 四、特征宽表    ← fill_report.py --section IV 回填
## 五、特征分析    ← fill_report.py --section V 回填
## 六、模型迭代    ← fill_report.py --section VI 回填
## 七、横向对比    ← fill_report.py --section VII 回填
## 附录：待处理项与下一步建议
```

> `fill_report.py --section` key（IV/V/VI/VII）是脚本内部 key，对应 report.md 第「四~七」节，**不错位**。一/二/三 节 + 附录由 orchestration 自填。

**子目录内容规范**（v2 交付分层：🟢 交付层 / 🟡 可复现层 / 🔵 缓存层，详见 `classification-model-training` 4.0 节）：

| 子目录 | 产出 skill | 关键内容 | 完成标志 |
|--------|-----------|---------|---------|
| `task-spec/` | task-spec | `task-spec.md` + `_manifest.json` + `.done` | `.done` 存在 |
| `data-profile/` | task-spec | `report.xlsx`（report.md 保留为内部缓存）+ `_manifest.json` + `_split_manifest.json` + `{model_name}_sample_*.parquet`；**三档 parquet 为缓存层**（与 `sample-features/splits/` 重复，以 splits/ 为准，不交付） | `_manifest.json` 存在 |
| `model-recommend/` | recommend | 本地文件模式下**不创建此目录**（历史模型推荐整步跳过） | — |
| `sample-features/feature-matching/` | feature-matching | `sample.parquet` + `feature-list.csv` + `sample_config.<model_name>.yaml`（sample.parquet 为 session 内唯一样本，**源副本/全量样本不再另存**） | `sample.parquet` + `feature-list.csv` 存在 |
| `sample-features/feature-analysis/` | feature-analysis | `feature_config.yaml` + `analysis/`（report.xlsx + _manifest + stats/iv/psi/woe/feature-profile/feature-quality 表；**明细表为缓存层**，对外以 report.xlsx 呈现） | `analysis/_manifest.json` 存在 |
| `sample-features/splits/` | feature-analysis | `train/test/oot.parquet`（**session 内唯一切分**） | 三个 parquet 存在 |
| `new-models/{algo}-{run_label}/` | development | `config.json` + `model/` + `features/` + `evaluation/` + `predictions/` + `explainability/` + `logs/run.log` + `report.md`；**交付层 = model + evaluation/*.xlsx + report.md**，predictions/explainability/features 为缓存层（评估依赖，保留不交付） | `config.json` + `model/` + `evaluation/` 存在 |
| `model-comparison/` | Dev Stage 3 | `model-comparison_{all,oot}.{md,json,xlsx}` + `对比报告.{json,md,xlsx}` + `_manifest.json`（仅 oot/all 两档，无 train/test） | `_manifest.json` 存在 |
| `deliverables.md`（session 根） | 收口产出 | **对外交付清单**：仅列交付层文件（总 report.md + 各 run evaluation xlsx + model pkl + 特征分析 report.xlsx + 训练脚本），其余分层产物显式声明"保留不交付" | 收口时存在 |

**deliverables.md 产出规范（v2 新增）**：

- **产出时机**：建模流程收口（`report.md` 7 节完整后，`classification-model-development` Stage 4 结束时）自动产出，由编排器/开发总控落盘到 `<session_dir>/deliverables.md`。
- **目的**：用户面对 40+ 文件不知看什么 → 一份清单明确"你要看什么"，其余产物降级为可复现缓存。
- **格式**（固定模板）：

```markdown
# 交付物清单 — {model_name}
> session: {timestamp}-{model_name} ｜ 收口时间: {YYYY-MM-DD}

## 🟢 对外交付（你要看/要用的）
| # | 文件 | 说明 |
|---|------|------|
| 1 | `report.md` | 全流程总报告（需求/样本/特征/模型/评估结论） |
| 2 | `new-models/{algo}-v{N}/evaluation/report.xlsx` | 模型评估（三档 AUC/KS/桶排序/特征重要性） |
| 3 | `new-models/{algo}-v{N}/model/model.pkl` | 可加载推理的模型文件 |
| 4 | `sample-features/feature-analysis/analysis/report.xlsx` | 特征质量报告（IV/PSI/WOE 合并） |
| 5 | `scripts/train_*.py` | 训练脚本（可复跑） |

## 🟡🟢 可复现层（保留，不交付）
配置 yaml / 特征清单 / splits / manifest / logs —— 保证可复跑与断点续跑，删除不影响交付。

## 🔵 缓存层（保留，不交付）
predictions / explainability / analysis 明细表 / data-profile 三档 parquet（与 splits 重复）——内部中间产物。
```

- **约束**：deliverables.md 只列**实际存在**的文件；某 run 无对应产物（如未产 feature-analysis xlsx）则不列该项。

### 3.4 特征拉取（跳过历史模型推荐）

- **classification-model-recommend**：本地文件模式下**整步跳过**，不创建 `model-recommend/` 目录
- **feature-matching**：
  - 调 `feature-matching/scripts/fetch_sample.py --mode local_file`，内部走 `_local_sample_to_parquet`（复用本地 parquet/csv）+ `derive_feature_list.py`（推导特征列表），**不做宽表拉取**
  - 落 `sample-features/feature-matching/sample.parquet` + `feature-list.csv`，**不切分三档**（切分由 feature-analysis 完成）
- 完成后调 `python classification-model-development/scripts/fill_report.py --session_dir <session_dir> --section IV` 回填 report.md 第「四」节

### 3.5 建模决策

汇总信息向用户发起决策询问（用户已明确"开始建模"/"不建模"等意图的，直接按其意图推进）：

- **用户选"是"** → 调用 `classification-model-development`，由其按迭代式流程编排子 skill（Stage 0~4）
- **用户选"否"** → 流程终止，产出物保留

### 3.6 本地文件流程（默认）

用户需提供预组装好的本地 parquet/csv（含 `id_cols + label_col + dt_col + features`，不支持 spark 取数）。这是**唯一**数据路径：

1. 调起 task-spec 透传 `mode=local_file` → SAMPLE 维度问本地路径 + 列名（`--label-col`/`--dt-col`/`--id-cols`），调 `fetch_sample_task_spec.py --mode local_file`；**所有需求维度（WHO/WHAT/HOW GOOD/CONSTRAINTS/SAMPLE）及切分方式均须向用户逐一澄清，不使用默认值**
2. task-spec 完成后进入创建目录 + report.md 初始化，出口校验同样强制生效
3. **跳过 recommend**（见 3.4 节）
4. 调起 feature-matching，调 `feature-matching/scripts/fetch_sample.py --mode local_file`（不切分三档）
5. 进入建模决策（3.5 节）

### 3.7 流程速览

```
model-task-routing（总入口）
  → classification-model-orchestration（本 skill）
  → 会话启动检查（3.1）
  → 需求确认 + 样本分析 + 创建目录 + report.md 初始化（3.3，本地文件、全澄清）
      → task-spec 所有维度向用户澄清（SAMPLE 问本地路径+列名；切分必问）
      → recommend 整步跳过（3.4）
      → feature-matching（--mode local_file，复用本地 parquet + 推导特征列表）
      → 建模决策（3.5，必问）
      → development（Stage 0~4）
```

## 4. 输出产物

### 4.1 session 目录结构

```
runs/{timestamp}-{model_name}/
├── task-spec/                  # task-spec.md + _manifest.json + .done
├── data-profile/               # report.xlsx + _manifest.json + _split_manifest.json + 全量样本 parquet（三档 parquet 为缓存层）
├── model-recommend/            # 本地文件模式下不创建（历史模型推荐整步跳过）
├── sample-features/feature-matching/    # sample.parquet（唯一） + feature-list.csv + sample_config.<model_name>.yaml
├── sample-features/feature-analysis/   # feature_config.yaml + analysis/（report.xlsx + 明细表缓存层）
├── sample-features/splits/    # train/test/oot.parquet（唯一切分）
├── new-models/{algo}-{run_label}/       # development 产（交付层 = model + evaluation/*.xlsx + report.md）
├── model-comparison/           # development Stage 3 产
├── deliverables.md             # 🆕 对外交付清单（收口产出）
└── report.md                   # 项目总报告（7 节 + 附录）
```

> 各子目录的应有内容详见 3.3 节子目录内容规范。

### 4.2 命名规范

| 文件类型 | 命名格式 | 示例 |
|---------|---------|------|
| 项目报告 | `report.md` | `report.md` |
| 需求文档 | `task-spec.md` | `task-spec.md` |
| manifest | `_manifest.json` | `_manifest.json` |
| 样本数据 | `{model_name}_sample_{YYYYMMDD}.parquet` | `draw_willingness_sample_20260615.parquet` |
| Session 目录 | `{timestamp}-{model_name}` | `20260615-160101-draw_willingness` |

## 5. 与其他 skill 关联

- `model-task-routing` — **上游**，建模流程总入口，判定 task_type 为 classification 后拉起本 skill
- `classification-model-task-spec` — 需求挖掘与确认 + 样本分析（本地文件模式，`--mode local_file`）
- `classification-model-recommend` — 历史模型检索推荐（本地文件模式下跳过，不创建目录）
- `feature-matching` — **强制**：本地文件模式复用本地 parquet/csv（`--mode local_file`）+ 推导特征列表，不做宽表拉取，不切分三档
- `classification-model-development` — 用户确认后调用，模型开发总控，按迭代式流程编排子 skill（Stage 0~4）

## 6. 执行约束

1. **不跳过task-spec**：即使是"显而易见"的需求也必须走完 task-spec
2. **文件落盘**：需求和分析结果必须保存为文件，不能只留在对话中
3. **task-spec 完成判定以 `.done` 为准**：`_manifest.json` 单独存在不能证明 task-spec 真正完成

## 7. 异常处理

### 7.1 task-spec 半途中断

`task-spec/_manifest.json` 存在但 `.done` 缺失 → 提示"task-spec 阶段可能半途中断，需补写三件套后再继续"。

### 7.2 recommend 目录缺失误判

本地文件模式下 `model-recommend/` 目录本就不创建（recommend 整步跳过），进度推断时不得将"该目录为空/缺失"误判为"recommend 待跑"。

### 7.3 需求成熟度 C 级

task-spec 输出 C 级（需先探查）→ 暂停流程，不创建任务目录，不进入后续步骤，等待用户补充信息。

### 7.4 子目录文件缺失

某子目录存在但缺少应有文件 → 提示用户"该阶段半途中断，需补齐 {缺失文件清单} 后再继续"。不得因目录存在就跳过对应阶段。

### 7.5 结束条件

> 回归/多分类等非二分类场景由上游 `model-task-routing` 的 Q1 判定拦截（见 model-task-routing 7.1）；路由路径下 task-spec 第零步跳过，本 skill 不处理非二分类拒绝场景。

1. **需求成熟度 C 级** → 暂停
2. **建模决策用户选"否"** → 正常结束，产出物保留
3. **classification-model-development 执行完毕** → 正常结束，report.md 完整
