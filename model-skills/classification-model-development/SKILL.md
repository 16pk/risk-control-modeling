---
name: classification-model-development
description: 模型开发唯一调度者。从需求澄清到收口打分的整个建模主链路都由本 skill 编排：调度 task-spec / feature-classification / data-cleaning / credit-data-analysis / classification-model-experiments / model-scoring 等 sub-skill，管理路径接力、决策点询问、report.md 回填、断点续跑、session 决议；另提供轻量编排入口 prep_sample.py（clean/analyze），供「只清洗 / 只分析」独立任务直接调度。触发词：开发模型、跑 baseline、列历史 session、建模。
---

# 模型开发总控（classification-model-development）

## 1. 角色与边界

你是建模任务的**唯一调度者**,**不是工程师**。从用户确认建模意图后，本 skill 串联整个主链路，管理"何时跑哪个 skill、读哪些产物、问用户什么决策、怎么写回 report.md、怎么断点续跑"。

主链路（6 步）：
```
需求澄清(task-spec 3 问) → 特征列识别(feature-classification, 语义三分类 + 用户批量确认)
  → 数据清洗(data-cleaning) → 特征分析(credit-data-analysis)
  → 实验矩阵+对抗验证+规则诊断+Optuna 调优+转正(classification-model-experiments, v2.3 主链路)
  → 收口打分(model-scoring 默认执行)
```

### 1.1 轻量编排入口（独立任务）

用户**不经过完整建模主链路**、仅针对给定数据文件要求「只清洗 / 只分析」时，调度本 skill 的
`scripts/prep_sample.py`（复用 sub-skill CLI 的编排剧本，不新增 skill）：

```bash
# 只清洗: 特征列识别(探查三分类+批量确认) → 固化权威 feature-list.csv → 清洗
python <skill_dir>/scripts/prep_sample.py clean \
    --input <数据文件> --session-dir <session_dir> \
    --id-col fuid --dt-col ftrans_date --label-col fpd7_sx30 \
    --exclude fser_date,sx_order_id,ftrans_time,... [--keep flag_ok,...]

# 只分析: clean 基础上追加特征分析(credit-data-analysis 独立体检模式)
python <skill_dir>/scripts/prep_sample.py analyze \
    --input <数据文件> --session-dir <session_dir> \
    --id-col fuid --dt-col ftrans_date --label-col fpd7_sx30 \
    --exclude ... [--base-month 2025-04]
```

- **交互门禁沿用主链路**：探查 → 用户逐项确认 id/dt/label → 展示三分类报告 → 用户批量确认
  exclude/keep → 固化权威 `feature-list.csv`（红线 `fpd*`/`dpd*` 默认剔除）→ 清洗（哨兵强门禁
  由编排层确认后 `--auto-confirm` 续跑）→ analyze 再确认 PSI 基准月（`--base-month`）。
- 产物落标准 session 结构 `<session-dir>/sample-features/`，与主链路完全互通
  （清洗后 `sample.parquet` 可直接作为 Stage 4 输入，无需重跑前置阶段）；
  `--session-dir` 缺省按 `runs/{ts}-prep-*/` 自动建目录。
- 权威清单唯一真相：finalize 固化的 `sample-features/feature-list.csv` 经
  `--feature-list-source`（清洗）/ `--feature-list`（分析）消费，不各自派生。

## 2. 输入契约

启动时**先验证**以下产物存在，缺任一 → 报错并指明缺哪个、上游哪个 skill 该补：

| 输入 | 路径 | 来源 skill |
|------|------|-----------|
| `session_dir` | `runs/{timestamp}-{model_name}/` | 本 skill 创建（吸收 orchestration 职责）或沿用已有 |
| 需求文档 | `task-spec/task-spec.md` + `_manifest.json` | classification-model-task-spec |
| 清洗样本 | `sample-features/data-cleaning/sample.parquet` + `feature-list.csv` | data-cleaning（feature-list 为 feature-classification 确认后的权威清单，经 `--feature-list-source` 消费） |
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
Stage 1: 特征列识别 (feature-classification, 语义三分类 + 用户批量确认)
   ↓ 产 sample-features/feature-classification.json + feature-list.csv (权威清单, 全 pipeline 唯一真相)
Stage 2: 数据清洗 (data-cleaning)
   ↓ 产 sample-features/data-cleaning/sample.parquet + feature-list.csv (经 --feature-list-source 取交集)
Stage 3: 特征分析 (credit-data-analysis pipeline 模式)
   ↓ 产 sample-features/credit-data-analysis/{特征分析结果.xlsx, 特征分析结果.md, _manifest.json}
   ↓ PSI 基准月 = 第一个 OOT 月 (读 feature_config.yaml 的 model.split.oot_range), 须用户确认
Stage 4: 实验矩阵 (classification-model-experiments, v2.3 主链路默认)
   ↓ 读 sample.parquet + feature-list.csv + feature_config.yaml(model.split)
   ↓ 内部流程: 安全过滤 → 实验范围确认(v2.5: 算法三选一 + 样本/特征选择/对抗验证/Optuna 4 开关, 编排层不重复问)
   ↓          → 矩阵规划(样本×特征正交, 按开关收缩) → 波1 baseline 格 → 波2 importance/iv-psi 格
   ↓          → 波3 对抗格(幅度确认) → leaderboard(OOT AUC 评选) → winner 规则诊断
   ↓          → 诊断驱动 Optuna 锚点调整/调优(well_fit 可跳过; 关闭 Optuna 时整段跳过) → top10 转正确认
   ↓ 产 experiments/matrix-plan.json + experiments/{id}/ 各格 + new-models/{algo}-v{N}/ 转正 run
Stage 5: 迭代决策点 (loop, 可重复 0~N 次, 仅用户主动要求时)
   └─ 继续实验 (新样本/特征方案 或 加大矩阵, 重跑 experiments, 可选)
Stage 6: 收口 — 回填 report.md, 落 finalized_model.json
Stage 7: model-scoring (定版模型打分, **默认执行**, 用户可叫停)
   ↓ 产 scoring/score_sample.parquet (支持 experiments 转正的 model.pkl 加载)
Stage 8 (可选): score-to-fico / credit-model-report — 仅用户主动触发, 不默认询问
   ↓ credit-model-report 的 xlsx 报告落 <session_dir>/model-report/ (严禁落 scoring/ 子目录)
```

**关键**: Stage 5 是 loop。每个 run 完成后都问用户"继续实验 / 停下收口"，不自动推进。
experiments 内部每个确认点(实验范围 / 对抗幅度 / top10 转正)也是必须等待用户确认的决策点。

## 4. 路径接力契约

| 阶段 | 读 | 写 | 接力 CLI |
|------|----|----|---------|
| Stage 0 | 用户需求 | `task-spec/task-spec.md` + `_manifest.json` | 对话式（3 问），落盘单文件 task-spec |
| Stage 1 | task-spec + 用户样本 | `sample-features/feature-classification.json` + `feature-list.csv` + `_manifest.json` | `feature-classification/scripts/classify_features.py`（探查三分类）→ 用户批量确认 → `finalize_feature_list.py --exclude ...`（固化权威清单） |
| Stage 2 | task-spec + 用户样本 + `sample-features/feature-list.csv` | `sample-features/data-cleaning/sample.parquet` + `feature-list.csv` | `data-cleaning/scripts/clean_data.py --feature-list-source <session_dir>/sample-features/feature-list.csv` |
| Stage 3 | `sample.parquet` + `sample-features/feature-list.csv`（权威清单）+ `feature_config.yaml` | `sample-features/credit-data-analysis/{特征分析结果.xlsx, 特征分析结果.md, _manifest.json}` | `credit-data-analysis/scripts/feature_analysis.py --feature-list <session_dir>/sample-features/feature-list.csv --split-config <feature_config.yaml> --base-month <确认的OOT首月>` |
| Stage 4 | `sample.parquet` + `feature-list.csv` + `feature_config.yaml`（含 `model.split`） | `experiments/matrix-plan.json` + `experiments/{id}/` + `new-models/{algo}-v{N}/` | `classification-model-experiments/scripts/run_experiments.py --session-dir <session_dir> --sample sample-features/data-cleaning/sample.parquet --feature-list sample-features/feature-list.csv --config <feature_config.yaml> --until promote` |
| Stage 5 | `experiments/matrix-plan.json` + 源格 | `experiments/{新格}/` + `new-models/{新 algo}-v{N}/` | 同上 CLI（新矩阵/新方案重跑, `--resume` 可跳过 done 格） |
| Stage 6 | `new-models/{run}/model/model_meta.json` | `finalized_model.json` | `model-scoring/scripts/mark_finalized.py --session-dir <session_dir> --run-name {run}` |
| Stage 7 | `finalized_model.json` + `sample.parquet` | `scoring/score_sample.parquet` | `model-scoring/scripts/score_data.py --model-path <session_dir>/new-models/{run}/model --data <session_dir>/sample-features/data-cleaning/sample.parquet --out <session_dir>/scoring/score_sample.parquet` |

**切分唯一真相** = `feature_config.yaml` 的 `model.split`（train/test/oot 三档区间）。
- v2.3 主链路（experiments）：开发池 = train+test 区间合并，每格 seed=42 分层随机 70/30 切 train/val，OOT 纯榜单；**experiments 无独立 test 档（test=实验台 val，回填 report 时注明语义）**。

**PSI 基准月确认（Stage 3 门禁）**：编排层读 `--split-config` 的 `model.split.oot_range`，取起始日所在月为默认基准月，向用户确认（"PSI 基准月默认取第一个 OOT 月 `{YYYY-MM}`，是否确认？"），确认后以 `--base-month` 显式传入。

## 5. 决策点话术（门禁收敛）

> 本 skill 是建模链路**唯一交互确认入口**：所有影响建模结论的关键决策，必须先给出**方案 + 理由 + 备选**，等待用户确认后再执行；仅当用户明确说"按默认 / 你定"时才可用默认值快速通过。**严禁在关键决策上擅自拍板执行。**

**必问 2 项 + 确认 1 项**，其余用默认值 + 报告展示：

| 门禁 | 时机 | 处理 |
|------|------|------|
| #1 预测目标 Y 定义 | Stage 0 | **必问**：预测什么行为 / 好坏标签定义 / 观察窗口 / 数据来源。标签列由特征名识别后请用户确认，识别不到时引导定义观察期/表现期/好坏标准 |
| #2 样本切分窗口 | Stage 0 | **必问**：Train/Test/OOT 三档起止 + 切分方式（时间/随机）。切分窗口定义是训练消费唯一真相，三档区间不强制时间递增（时序排布由业务侧保证）；train/test 开发集允许随机切分（记录 seed）；OOT 评估剔除标签缺失样本；无时间字段时退化为分层随机切分并显式说明局限 |
| 特征列清单确认 | Stage 1 | **一次确认**（feature-classification）：探查分类报告（三分类计数 + 组级折叠）→ **用户批量确认**：剔除 non_feature 候选、判定 ambiguous 是否保留（默认保留）、可展开混合组逐列调整；红线 `fpd*`/`dpd*` 标签列默认剔除可跳过询问（但展示决策结果） |
| 矩阵方案确认 | Stage 4 前 | **一次确认**（替代原超参确认）：experiments 内部规划前自动询问（算法 lgb/xgb/两者 + 样本选择/特征选择/对抗验证/Optuna 调优 4 开关，对抗与 Optuna 附耗时提醒）；编排层**不重复询问**，仅提示用户按需回答（用户说"按默认"即通过）；对抗剔除幅度、Optuna trials（默认 25）仍由 experiments 内部逐确认点推进 |
| PSI 基准月 | Stage 3 | **确认**：默认第一个 OOT 月，可覆盖 |
| 对抗幅度 | Stage 4 波3 | **确认**：对抗样本剔除幅度（experiments 交互确认） |
| top10 转正 | Stage 4 尾 | **确认**：top10 候选逐条确认，**显式标注乐观偏差候选（对抗/IV-PSI 格）且不默认优先推荐** |
| 特征筛选 | Stage 5 | **不问**：默认不筛（IV/PSI 筛选不作为主链路特征筛选，experiments 的 iv-psi 格用于对比展示） |
| 不平衡处理 | Stage 4 | **不问**：默认 `scale_pos_weight` 自动（experiments 每格已含），正样本率 <1% 时提示一次 |
| 定版打分 | Stage 7 | **默认执行**：收口后直接打分，用户可叫停 |
| FICO / 业务报告 | Stage 8 | **不默认询问**：仅用户主动触发 |

**确认交互方式**：
- 每个门禁给出「推荐方案（默认值）」+ 简要理由 + 可选替代项，等待用户确认或指示调整；**用户未确认前不得推进关键执行**（跑训练、大规模调参等重操作）。
- 若用户提供的信息与门禁默认冲突，**以用户信息为准**（除非自相矛盾，需指出并要求澄清）。
- 同一会话内不重复询问已确认项。

**决策点必问（迭代决策点）**：Stage 5（矩阵转正 run 落盘后）询问下一步，不自动推进：
```
> {run_name} 已落盘(experiments 矩阵转正)。OOT AUC: {oot_auc} / val AUC: {val_auc}（乐观偏差标注见 leaderboard）
> 下一步?
>   A. 继续实验 (新样本/特征方案 或 加大矩阵, 重跑 experiments)
>   B. 停, 进入收口 (Stage 6)
```

**Stage 7 打分（默认执行）**：
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
| 二、样本与特征 | Stage 3 | credit-data-analysis 报告摘要（分月 PSI / IV / 缺失率） |
| 三、模型迭代 | 每个 run 完成后 | `new-models/*/config.json.runtime`（run_name / algo / 三档 AUC） |
| 四、结论与交付 | Stage 6/7 | 上线候选 / 定版模型 / 打分结果 / 待处理项 |

数字一致性由 `fill_report.py` 直接读产物文件保证（不从对话手敲）。

## 7. 断点续跑

启动时扫描 `session_dir`，按 `_manifest.json` 推断当前阶段，向用户回显"当前进度: Stage X，下一步: Stage Y"，确认后从断点继续：

| 检查 | 推断阶段 |
|------|---------|
| `task-spec/_manifest.json` 不存在 | Stage 0 待跑 |
| `sample-features/feature-classification.json` 不存在 | Stage 1 待跑（特征列识别） |
| `sample-features/data-cleaning/sample.parquet` 不存在 | Stage 2 待跑 |
| `sample-features/credit-data-analysis/_manifest.json` 不存在 | Stage 3 待跑 |
| `new-models/` 为空 且 `experiments/matrix-plan.json` 不存在 | Stage 4 待跑 |
| `experiments/matrix-plan.json` 存在但 `new-models/` 为空 | Stage 4 进行中（experiments 矩阵，`--resume` 续跑） |
| `new-models/` 非空但 `finalized_model.json` 不存在 | Stage 5 迭代中 / Stage 6 收口待跑 |
| `finalized_model.json` 存在但 `scoring/score_sample.parquet` 不存在 | Stage 7 待跑（默认执行） |

## 8. 算法与矩阵

experiments 主链路算法收敛到 **lgb/xgb**（专家包仅支持 lgb/xgb 树模型，不提供 DNN/LR 评分卡）：

1. 矩阵方案（样本×特征正交）由 `plan_matrix` 在 Stage 4 按 feature_config 生成，可经 Stage 5 加大/替换方案
2. 每算法 OOT AUC 榜首为 winner → 规则诊断 → Optuna 邻域调优(-opt)
3. 转正 top10 按 leaderboard 排序确认；不同算法 run 布局一致，`config.json.algo` 区分

## 9. 与其他 skill 的接口

- **输入**：用户建模意图（无上游编排器，本 skill 即起点）
- **输出**：`session_dir` 下完整包含 task-spec / sample-features(feature-classification + data-cleaning + credit-data-analysis) / experiments / new-models / finalized_model.json / scoring / report.md
- **结束条件**：用户在 Stage 5 选"停" 或 所有决策点选"停"，且 Stage 7 打分完成（或用户叫停）

## 10. 反模式

- ❌ **自动推进**（不问用户就跑下一阶段）— 迭代决策点必问；experiments 内部对抗幅度/top10 转正也必须等用户确认
- ❌ **绕过总控自写脚本散落执行** — 所有 stage 走 sub-skill CLI
- ❌ **在 development 里写新 Python 脚本** — 复用 sub-skill CLI；本 skill 仅提供 `list_sessions.py` + `fill_report.py` + `prep_sample.py`（**豁免：`prep_sample.py` 是复用 sub-skill CLI 的轻量编排剧本**，只做 subprocess 串联，不实现探查/清洗/分析逻辑）
- ❌ **report.md 留占位** — 阶段完成必填实
- ❌ **把 development 当成"模型工程脚手架"** — 它是编排器，不是代码生成器
- ❌ **在主链路训练流程中自动做 IV/PSI 特征筛选** — 训练不筛特征（experiments 的 iv-psi 格仅用于对比展示，不自动进训练/结构选择）
- ❌ **把 OOT 用于早停/结构选择** — 红线不变：OOT 禁早停/禁进训练/禁结构选择；对抗/IV-PSI 格 OOT 参与筛选统计须标注乐观偏差

## 11. 关联 skill

- 上游依赖: `classification-model-task-spec` / `feature-classification`（特征列识别，Stage 1）/ `data-cleaning` / `credit-data-analysis`
- 下游编排(v2.3 主链路): `classification-model-experiments`（矩阵+对抗+诊断+调优+转正）/ `model-scoring`（Stage 7 默认，支持 experiments 转正 pkl 加载）
- 可选触发: `score-to-fico` / `credit-model-report` / `classification-model-package`（定版模型 → 独立交付包）/ `model-knowledge`（仅用户主动要求）
