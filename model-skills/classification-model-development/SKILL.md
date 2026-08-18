---
name: classification-model-development
description: 模型开发唯一调度者。从需求澄清到收口打分的整个建模主链路都由本 skill 编排：调度 task-spec / data-cleaning / credit-data-analysis / classification-model-training / model-scoring 等 sub-skill，管理路径接力、决策点询问、report.md 回填、断点续跑、session 决议。触发词：开发模型、跑 baseline、调参、多模型对比、列历史 session、建模。
---

# 模型开发总控（classification-model-development）

## 1. 角色与边界

你是建模任务的**唯一调度者**,**不是工程师**。从用户确认建模意图后，本 skill 串联整个主链路，管理"何时跑哪个 skill、读哪些产物、问用户什么决策、怎么写回 report.md、怎么断点续跑"。

主链路（5 步）：
```
需求澄清(task-spec 3 问) → 数据清洗(data-cleaning) → 特征分析(credit-data-analysis)
  → 训练+评估(training, 内嵌 evaluation, 切分后置即时切分) → 收口打分(model-scoring 默认执行)
```

## 2. 输入契约

启动时**先验证**以下产物存在，缺任一 → 报错并指明缺哪个、上游哪个 skill 该补：

| 输入 | 路径 | 来源 skill |
|------|------|-----------|
| `session_dir` | `runs/{timestamp}-{model_name}/` | 本 skill 创建（吸收 orchestration 职责）或沿用已有 |
| 需求文档 | `task-spec/task-spec.md` + `_manifest.json` | classification-model-task-spec |
| 清洗样本 | `sample-features/data-cleaning/sample.parquet` + `feature-list.csv` | data-cleaning |
| 特征分析 | `sample-features/credit-data-analysis/特征分析结果.{md,xlsx}` + `_manifest.json` | credit-data-analysis（pipeline 模式） |

### 2.1 Session 决议（本 skill 负责创建）

1. 调 `list_sessions.py` 列出历史 sessions（若用户想复用已有 session）。
2. 新建：追问 `task_name` + 可选 `description`，按 `ts=$(date +%Y%m%d-%H%M%S)` 建 `runs/${ts}-${task_name}/` 并写 `session.json`。
3. 初始化 `report.md`（4 节骨架 + 附录「待处理项」），作为会话内单一归档。
4. 同一会话内**不重复询问**。

## 3. 阶段流水

```
Stage 0: 需求澄清 (task-spec 3 问: Y 定义 / 数据路径+列名 / 切分窗口)
   ↓ 产 task-spec/task-spec.md + _manifest.json (split_ranges 记录入口)
Stage 1: 数据清洗 (data-cleaning)
   ↓ 产 sample-features/data-cleaning/sample.parquet + feature-list.csv
Stage 2: 特征分析 (credit-data-analysis pipeline 模式)
   ↓ 产 sample-features/credit-data-analysis/{特征分析结果.xlsx, 特征分析结果.md, _manifest.json}
   ↓ PSI 基准月 = 第一个 OOT 月 (读 feature_config.yaml 的 model.split.oot_range), 须用户确认
Stage 3: baseline 训练 (classification-model-training, 内嵌 evaluation)
   ↓ 产 new-models/{algo}-v{N}/ (切分在消费时按 model.split 即时切分, 写 run 内部 data/splits/)
   ↓ run_build 自动 chain: evaluation → per-run comparison → session-aggregate
Stage 4: 迭代决策点 (loop, 可重复 0~N 次, 仅用户主动要求时)
   ├─ 4a: 超参调优 (classification-model-tuning run_tuning, 可选)
   ├─ 4b: 换算法 (改 yaml model.algo 重跑 Stage 3, 可选)
   └─ 4c: 特征筛选 (select_features, 仅用户明确要求; 默认不做 IV/PSI 筛选)
Stage 5: 收口 — 回填 report.md, 落 finalized_model.json
Stage 6: model-scoring (定版模型打分, **默认执行**, 用户可叫停)
   ↓ 产 scoring/score_sample.parquet
Stage 7 (可选): score-to-fico / credit-model-report — 仅用户主动触发, 不默认询问
   ↓ credit-model-report 的 xlsx 报告落 <session_dir>/model-report/ (严禁落 scoring/ 子目录)
```

**关键**: Stage 4 是 loop。每个 run 完成后都问用户"继续迭代 / 停下收口"，不自动推进。

## 4. 路径接力契约

| 阶段 | 读 | 写 | 接力 CLI |
|------|----|----|---------|
| Stage 0 | 用户需求 | `task-spec/task-spec.md` + `_manifest.json` | 对话式（3 问），落盘单文件 task-spec |
| Stage 1 | task-spec + 用户样本 | `sample-features/data-cleaning/sample.parquet` + `feature-list.csv` | `data-cleaning/scripts/clean_data.py` |
| Stage 2 | `sample.parquet` + `feature_config.yaml` | `sample-features/credit-data-analysis/{特征分析结果.xlsx, 特征分析结果.md, _manifest.json}` | `credit-data-analysis/scripts/feature_analysis.py --split-config <feature_config.yaml> --base-month <确认的OOT首月>` |
| Stage 3 | `sample.parquet` + `train_config.yaml`（含 `model.split`） | `new-models/{algo}-v{N}/` | `classification-model-training/scripts/run_build.py --config <train_config.yaml> --output_dir <session_dir> --version v1` |
| Stage 4a | baseline run dir | `new-models/{algo}-tuned-v{N}/` | `classification-model-tuning/scripts/run_tuning.py --baseline_run <baseline_run_dir> [--method rule\|optuna]` |
| Stage 4c | baseline run dir | `new-models/{algo}-feat-v{N}/` | `classification-model-tuning/scripts/select_features.py --baseline_run <baseline_run_dir>`（数据直算） |
| Stage 5 | `new-models/{run}/model/model_meta.json` | `finalized_model.json` | `model-scoring/scripts/mark_finalized.py --session-dir <session_dir> --run-name {run}` |
| Stage 6 | `finalized_model.json` + `sample.parquet` | `scoring/score_sample.parquet` | `model-scoring/scripts/score_data.py --model-path <session_dir>/new-models/{run}/model --data <session_dir>/sample-features/data-cleaning/sample.parquet --out <session_dir>/scoring/score_sample.parquet` |

**切分唯一真相** = `feature_config.yaml` 的 `model.split`（train/test/oot 三档区间）。`train_config.yaml` 的 `model.split` 与之保持一致；`run_build` 在训练消费时按区间即时切分，不落盘 session 级 `splits/`。

**PSI 基准月确认（Stage 2 门禁）**：编排层读 `--split-config` 的 `model.split.oot_range`，取起始日所在月为默认基准月，向用户确认（"PSI 基准月默认取第一个 OOT 月 `{YYYY-MM}`，是否确认？"），确认后以 `--base-month` 显式传入。

## 5. 决策点话术（门禁收敛）

> 本 skill 是建模链路**唯一交互确认入口**：所有影响建模结论的关键决策，必须先给出**方案 + 理由 + 备选**，等待用户确认后再执行；仅当用户明确说"按默认 / 你定"时才可用默认值快速通过。**严禁在关键决策上擅自拍板执行。**

**必问 2 项 + 确认 1 项**，其余用默认值 + 报告展示：

| 门禁 | 时机 | 处理 |
|------|------|------|
| #1 预测目标 Y 定义 | Stage 0 | **必问**：预测什么行为 / 好坏标签定义 / 观察窗口 / 数据来源。标签列由特征名识别后请用户确认，识别不到时引导定义观察期/表现期/好坏标准 |
| #2 样本切分窗口 | Stage 0 | **必问**：Train/Test/OOT 三档起止 + 切分方式（时间/随机）。切分窗口定义是训练消费唯一真相，三档区间不强制时间递增（时序排布由业务侧保证）；train/test 开发集允许随机切分（记录 seed）；OOT 评估剔除标签缺失样本；无时间字段时退化为分层随机切分并显式说明局限 |
| 超参确认 | Stage 3 前 | **一次确认**：主动展示完整超参数表（参数/值/理由/备选），或 Optuna 搜索空间；用户说"按默认"即通过 |
| PSI 基准月 | Stage 2 | **确认**：默认第一个 OOT 月，可覆盖 |
| 特征筛选 | Stage 4 | **不问**：默认不筛（IV/PSI 筛选已从训练流程移除），报告展示 |
| 不平衡处理 | Stage 3 | **不问**：默认 `scale_pos_weight` 自动，正样本率 <1% 时提示一次 |
| 定版打分 | Stage 6 | **默认执行**：收口后直接打分，用户可叫停 |
| FICO / 业务报告 | Stage 7 | **不默认询问**：仅用户主动触发 |

**确认交互方式**：
- 每个门禁给出「推荐方案（默认值）」+ 简要理由 + 可选替代项，等待用户确认或指示调整；**用户未确认前不得推进关键执行**（跑训练、大规模调参等重操作）。
- 若用户提供的信息与门禁默认冲突，**以用户信息为准**（除非自相矛盾，需指出并要求澄清）。
- 同一会话内不重复询问已确认项。

**决策点必问（迭代决策点）**：Stage 4 每个 run 完成后都询问下一步，不自动推进：
```
> {run_name} 已落盘。三档 AUC: train={x} / test={y} / oot={z}
> 下一步?
>   A. 超参调优 (产 -tuned run)
>   B. 换算法 (产 dnn/lr run)
>   C. 特征筛选 (仅当你要做 IV/PSI 筛选, 产 -feat run)
>   D. 跑横向对比 (若已有 ≥2 个 run)
>   E. 停, 进入收口 (Stage 5)
```

**Stage 6 打分（默认执行）**：
```
> 已确认上线候选 {run_name}(oot AUC={x}), 已落定版标记 finalized_model.json。
> 默认对清洗后数据打分(产 scoring/score_sample.parquet)。如不需要打分请告知。
```

## 6. report.md 回填契约

每个阶段完成后**必须**用 `fill_report.py` 回填 `report.md` 对应段落，禁止留占位。

**回填 CLI**（脚本位置: `classification-model-development/scripts/fill_report.py`）:
```bash
python classification-model-development/scripts/fill_report.py \
    --session-dir <session_dir> --section {IV|V|VI|all}
```

`report.md` 4 节结构：

| 段落 | 何时写 | 内容来源 |
|------|--------|---------|
| 一、需求 | Stage 0 | task-spec（Y 定义 / 切分窗口） |
| 二、样本与特征 | Stage 2 | credit-data-analysis 报告摘要（分月 PSI / IV / 缺失率） |
| 三、模型迭代 | 每个 run 完成后 | `new-models/*/config.json.runtime`（run_name / algo / 三档 AUC） |
| 四、结论与交付 | Stage 5/6 | 上线候选 / 定版模型 / 打分结果 / 待处理项 |

数字一致性由 `fill_report.py` 直接读产物文件保证（不从对话手敲）。

## 7. 断点续跑

启动时扫描 `session_dir`，按 `_manifest.json` 推断当前阶段，向用户回显"当前进度: Stage X，下一步: Stage Y"，确认后从断点继续：

| 检查 | 推断阶段 |
|------|---------|
| `task-spec/_manifest.json` 不存在 | Stage 0 待跑 |
| `sample-features/data-cleaning/sample.parquet` 不存在 | Stage 1 待跑 |
| `sample-features/credit-data-analysis/_manifest.json` 不存在 | Stage 2 待跑 |
| `new-models/` 为空 | Stage 3 待跑 |
| `new-models/` 非空但 `finalized_model.json` 不存在 | Stage 4 迭代中 / Stage 5 收口待跑 |
| `finalized_model.json` 存在但 `scoring/score_sample.parquet` 不存在 | Stage 6 待跑（默认执行） |

## 8. 多算法切换

Stage 4b 换算法时：
1. 改 `train_config.yaml` 的 `model.algo: dnn|lr`
2. 重跑 `run_build.py`（回到 Stage 3 CLI）
3. 不同算法产物布局一致，`config.json.algo` 区分
4. comparison 自动跨算法 N-way 对比

## 9. 与其他 skill 的接口

- **输入**：用户建模意图（无上游编排器，本 skill 即起点）
- **输出**：`session_dir` 下完整包含 task-spec / sample-features(data-cleaning + credit-data-analysis) / new-models / finalized_model.json / scoring / report.md
- **结束条件**：用户在 Stage 4 选"停" 或 所有决策点选"停"，且 Stage 6 打分完成（或用户叫停）

## 10. 反模式

- ❌ **自动推进**（不问用户就跑下一阶段）— 迭代决策点必问
- ❌ **绕过总控自写脚本散落执行** — 所有 stage 走 sub-skill CLI
- ❌ **在 development 里写新 Python 脚本** — 复用 sub-skill CLI；本 skill 仅提供 `list_sessions.py` + `fill_report.py`
- ❌ **report.md 留占位** — 阶段完成必填实
- ❌ **把 development 当成"模型工程脚手架"** — 它是编排器，不是代码生成器
- ❌ **在训练流程中自动做 IV/PSI 特征筛选** — 训练不筛特征（boundary_filter 只做常量/泄漏/ID/全缺失的安全过滤）

## 11. 关联 skill

- 上游依赖: `classification-model-task-spec` / `data-cleaning` / `credit-data-analysis`
- 下游编排: `classification-model-training`（含 evaluation）/ `classification-model-tuning`（可选）/ `classification-model-comparison`（可选）/ `model-scoring`（Stage 6 默认）
- 可选触发: `score-to-fico` / `credit-model-report` / `model-knowledge`（仅用户主动要求）
