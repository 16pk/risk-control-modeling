# Changelog

本文件只记录专家包与建模框架的**版本迭代信息**（结构变更、职责迁移、产物调整），供包维护者参考。用户调用 skill 时无需阅读；各 skill 的 SKILL.md 只描述当前状态，不携带版本标记。

## v2.7.0（2026-08-21）

**完全移除** `classification-model-training` / `classification-model-tuning` / `classification-model-comparison` 三个备用模块（v2.3 起已不在默认编排，本版删除目录与注册）。专家包算法收敛为 **lgb/xgb**。

### 移除与能力下线

- 删除 `model-skills/classification-model-{training,tuning,comparison}/` 三个目录；`plugin.json` skills 数组 14 → 11 项，version 2.6.1 → 2.7.0。
- 永久下线能力：DNN/LR 评分卡训练、基于单模型 run 的调参/特征筛选（`run_tuning` / `select_features`）、N-way 深度对比（`compare_models` / `aggregate_session_comparison`）。规则诊断与 Optuna 调优由 experiments 承接（`diagnose_winner` / `recommend_winner` / `tune_winner`，逻辑早已移植）。

### model-scoring 收敛（仅 lgb/xgb）

- `score_data.py`：删除 `_locate_training_scripts()` 与 dnn/lr pickle 加载分支（原依赖 training 的 `trainers.train_dnn` / `trainers.train_lr` 反序列化）；`infer_algo` / `--algo` 收敛 `lgb|xgb`；`test_score_data.py` 同步删 lr/dnn 用例。
- SKILL.md 算法边界改"仅 lgb/xgb（含 xgb 历史 model.json）"，删备用路径上游行。

### development 收敛

- `SKILL.md`：frontmatter 触发词删"调参、多模型对比"；Stage 5 删 5b（comparison）/5c（备用路径）仅留继续实验；路径接力表删 5b/5c 行；切分说明删备用路径两行；决策点话术删深度对比选项与备用路径括注；§8 删 dnn/lr 出口；§11 关联 skill 删备用路径与 comparison。
- `fill_report.py`：删 `_latest_run_splits_dir`（training 即时切分定位）与 `_classify_run` 的 `model-tuning` 分支；§VII 横向对比段改纯占位（leaderboard 为准）；旧 session 的 training/tuning run 不再支持回填 report。

### 文档同步

- `agents/risk-control-modeling.md`：职责映射删 3 行，frontmatter/正文 LR/DNN/评分卡表述收敛为 lgb/xgb，删"可解释性优先 LR 评分卡"建议。
- `README.md` / `CODEBUDDY.md` / `model-skills/README.md`：目录树、清单表、流程图、session 结构、技术栈（删 optbinning/shap）同步。
- 周边 skill 文档去引用：task-spec / data-cleaning / credit-data-analysis / credit-model-report / score-to-fico / model-knowledge / experiments（切分消费方与评估口径改对 experiments，移植溯源标注"原模块已移除"）；`_modelevo-shared` 注释/示例同步；`classification-model-package` 算法边界文案更新（校验保留，防御旧 session）。

### 测试

- model-scoring 6 过 2 跳 / development 16 过 / experiments 87 过 / _modelevo-shared 73 过 / task-spec 9 / data-cleaning 26 / credit-data-analysis 7 / score-to-fico 3 / package 13 / feature-classification 35，全部通过。

## v2.6.1（2026-08-21）

`classification-model-experiments` 实验格**数据快照瘦身**（方案 A）：切分数据不再按格冗余落盘，保留 Optuna 调优轻量依赖。

### 背景与依据

- 每格 `data/` 原先落盘 `train/val/oot.parquet`（复现用），矩阵 N 格 × 3 份 parquet 存储冗余严重；且同一样本方案下不同特征方案的格子 parquet 内容完全相同。
- `split_dev`（`random_state=42` 分层切分）与对抗分类器（`seed=42`）均为纯确定性，同一 dev 输入切分逐字节可复现——数据无需落盘，运行时重切即可。
- 消费面核对：三份 parquet 仅被 `tune_winner`（-opt 调优）与复现契约引用；训练本身 / 转正 / 打分不读快照。

### 改动

- **`run_single_experiment.py`**：`_save_inputs` 签名精简为 `(exp_dir, features, params, weight)`，`data/` 只落 `features.json` + `params.json` + `weights.csv`（Optuna 依赖），**不再写 train/val/oot.parquet**。
- **`tune_winner.py`**：
  - `tune_winner` 新增 `train_df/val_df/oot_df` 透传参数（主流程运行时重切）；未透传时回退 `_load_winner_inputs_legacy` 读旧目录完整快照（v2.6.1 前生成的 run 兼容）。
  - `_load_winner_inputs` 改为只读 `features.json` + `weights.csv`；新增 `_load_winner_inputs_legacy` 保留旧快照读取。
- **`run_experiments.py`**：新增 `_resplit_for_optuna()`——按 winner 格样本方案运行时重切（full/recent/timeweight 由 `_scheme_for` 构造；**adversarial 从 `adversarial_meta.json` 取实际剔除比例，重训对抗分类器重建 drop_mask**，与 winner 完全同基线）；label 列统一改名 `label` 透传；重切失败（如缺 `data/features.json`）跳过调优并 log。
- **文档**：experiments SKILL.md（§4 `data/` 描述改为 Optuna 轻量依赖）、references/constraints-and-exceptions.md（复现契约改为「依赖上游 sample.parquet + model.split 重切」+ 旧目录兼容说明）。
- **测试**：`test_tune_winner_diagnosis_smoke.py` fixture 不再写三份 parquet（仅 features.json），冒烟测试透传重切数据；experiments 模块 87 例全过。

### 注意

- 对抗格 winner 的 Optuna 调优前会重训一次对抗分类器（seed=42 确定性重放，少量开销）。
- 旧 runs（v2.6.1 之前）若保留完整 `data/` 快照，调优自动走 legacy 路径，行为不变。
- plugin.json version 2.6.0 → 2.6.1。

## v2.6.0（2026-08-20）

新增 **`classification-model-package`**：把已定版训练任务组装为**可独立运行的交付代码包**（仅用户主动触发，主链路收口后可选出口）：

### 打包器 `package_model.py`

- 校验链：`finalized_model.json` → `new-models/{run}/model`（`model.pkl`/`model.json` + `model_meta.json`，`feature_names` 非空、algo ∈ {lgb,xgb}）→ `cleaning-scheme.json`（缺则默认哨兵集 + WARN）→ 权威 `feature-list.csv`（与 model feature_names 一致性 WARN）→ 探测 `fico/coef.json`
- 组装 `<session_dir>/delivery/` 交付包：`run.py` + `pipeline/`（clean/score/fico 自包含）+ `assets/`（模型 + model_meta + cleaning-scheme + feature-list + 可选 coef）+ `requirements.txt`（按 algo 渲染）+ `README.md`（渲染会话信息）+ `package-manifest.json`
- 设计：**资产驱动（零占位符）** + **纯拷贝模板**（`package_templates/` 与单测共享同一份源码）；打包器经 `_bootstrap.py` 注入共享代码读权威清单，**交付包零引用专家包与 `_modelevo-shared`**
- 算法边界：仅支持主链路 lgb/xgb（含 xgb 历史 `model.json`）；dnn/lr 打包器报错拒绝（自包含包无法携带其反序列化所需 training 脚本依赖）

### 交付包行为

- 数据清理：仅特征列哨兵值→NaN，非交互 + WARN；**不做样本去重**；不校验 id/dt/label 列（允许缺 label）
- 打分：特征缺失报错退出（列出缺失）→ 按 feature_names 重排推理 → 透传非特征列 + `score`
- FICO（条件包含）：打包时存在 `fico/coef.json` 才含转分模块；**纯应用模式**（不拟合，吃固化 coef/intc），同表追加 `bscore`，bscore 越界 [400,780] 仅 WARN 不中止
- 输出：`score.parquet` + `cleaning-report.json` + `fico-summary.json` + `run-manifest.json`

### 注册

- `.codebuddy-plugin/plugin.json` skills 数组登记 + version **2.5.2 → 2.6.0**；`model-skills/README.md` 目录/清单/流程/Session 结构；`agents/risk-control-modeling.md` 职责映射
- `classification-model-development` SKILL.md §11 可选触发补登记；§1.1 轻量入口、§10 反模式不涉及（本模块为新独立 skill）
- 测试：新增 `classification-model-package/tests/test_package_model.py`（13 例：校验链 + 组装 + 模板共享 + 打包→交付包端到端跑通（lgb 真实模型 + FICO）），通过

## v2.5.3（2026-08-20）

**`data-cleaning` label 列改为非必选**（支持无标签场景清洗）：

- `clean_data.py`：`--label-col` 由 `required=True` 改为 `default=None`；`clean_data()` / `_validate_columns()` 签名 `label_col: Optional[str]`，无标签时跳过该列存在性校验
- 行为：不传 `label_col` 时去重退化为「组内保首行」（`dedup_sample.py` 原有逻辑），哨兵值替换不做坏率统计；主体清洗（哨兵替换、派生特征清单、清洗方案）不受影响
- 语义提示：数据含 label 列但未声明 `--label-col` 时，该列会被当作普通特征列派生进 feature-list（排除列表仅 id/dt），需显式声明才能排除
- 文档：`data-cleaning/SKILL.md` + `references/constraints-and-exceptions.md` 输入依赖/参数表/约束/异常表同步更新
- 测试：新增 `data-cleaning/tests/test_clean_data.py`（3 例：数据含 label 但不传、数据无 label 列、传 label 回归），data-cleaning 全量 26 例通过

## v2.5.2（2026-08-20）

新增**轻量编排入口**，打通「只清洗 / 只分析」独立任务的特征识别前置链路（解决：单独清洗/单独分析时未触发 `feature-classification`，日期/订单号/标签列混入特征、哨兵值干扰 PSI/IV）：

### 新增 `classification-model-development/scripts/prep_sample.py`

- `clean` 子命令：`feature-classification`（classify_features 探查三分类）→ 编排层交互确认 → `finalize_feature_list.py` 固化权威 `feature-list.csv` → `clean_data.py --feature-list-source <权威清单>` 清洗
- `analyze` 子命令：在 clean 基础上追加 `feature_analysis.py --feature-list <权威清单> --time-col <dt> --iv-label <label>`（独立体检模式，`--base-month` 交互确认）
- 只做**编排链**（subprocess 绝对路径调各 sub-skill CLI，不跨 skill import）；交互门禁沿用主链路（id/dt/label 确认、三分类 exclude/keep 批量确认、PSI 基准月），非交互续跑显式传 `--auto-confirm / --exclude / --keep / --base-month`
- 产物沿用标准 session 结构 `<session-dir>/sample-features/`，`--session-dir` 缺省按 `runs/{ts}-prep-*/` 自动建；清洗后 `sample.parquet` 可直接作为主链路 Stage 4 输入

### 触发定位扩展（文档）

- `feature-classification` / `data-cleaning` / `credit-data-analysis` 三 skill 触发定位从「仅由 development 编排自动调起」扩展为「主链路 + 轻量入口 prep_sample.py clean/analyze 独立任务」，**不设独立触发词**；独立任务必须先特征识别再清洗/分析（禁止直接跳过）
- development SKILL.md 新增 §1.1 轻量编排入口说明；§10 反模式豁免 `prep_sample.py`（复用 sub-skill CLI 的编排剧本）
- `agents/risk-control-modeling.md` 职责映射补充轻量入口行；`model-skills/README.md` 流程/约束补充独立任务前置链路

### 测试

- 新增 `classification-model-development/tests/test_prep_sample.py`（7 例）：编排链参数顺序/契约（classify → finalize → clean → analysis）、`--keep`/`--base-month` 可选、非零返回码透传、自动建目录；mock subprocess 不跑真实数据

## v2.5.1（2026-08-20）

框架问题修复（依据 `runs/20260820-095729-ka-dpd30-2c-fwtest/框架问题修复方案-20260820.md`，实施问题 1/3/5；问题 2 已修复记录；问题 4 暂不纳入、方案留档）：

### 问题 1：credit-data-analysis 透传特征清单（区间法弃用）

- `feature_analysis.py` 新增 `--feature-list`（复用 `gen_feature_list.load_feature_list` 解析契约），特征选择的唯一真相 = 权威 `feature-list.csv`，消除"分析集 ≠ 建模集"的静默偏差（本次区间法曾夹带 8 个非清单列）；清单缺失列仅 WARN（容忍列漂移）；`--feature-start/--feature-end` 区间法降级 DEPRECATED（独立体检兼容）；两者均缺报错提示迁移
- manifest 记录 `feature_list` / `feature_source` / `feature_list_missing`（区间参数兼容保留）；报告头标注特征来源
- 新增 `credit-data-analysis/scripts/_bootstrap.py`（注入 `_modelevo-shared`，与其它 skill 一致）
- 插件目录 `clean_data.py` 覆盖同步为**全列落盘版**（消除与源仓库的不一致）
- development SKILL.md Stage 3 接力 CLI 更新为 `--feature-list <session_dir>/sample-features/feature-list.csv`

### 问题 5：tune 幂等（每算法最多一个 -opt 格）

- `run_experiments.py` tune 段：① 候选 winner 排除已调优格（`is_tuned`）② 已有 done 的 `-opt` 格跳过（防重放重复调优）
- `leaderboard.py`：行数据透传 `is_tuned`，表格标注 `/tuned`
- `tune_winner.py` 入口防御：`spec.is_tuned` 直接返回 None（三保险）

### 问题 3 治本：Optuna 锚点域与收窄域对齐（三件套）

- ① 锚点生成约束（`hyperparams.optuna_anchors`）：lgb `min_child_samples` 公式值**先 clip 进经验域 `[20,200]`** 再生成邻域（26 万样本：520 → clip 200 → 锚点 (120,280)）
- ② 收窄相对化（`recommend_winner.adjust_optuna_anchors`）：`min_child_samples`/`min_child_weight` 类跨数量级参数 overfit 收窄改**相对锚点**（lgb 下界 ×1.5 / xgb ×10），不再用绝对区间（防交集空 → 收窄静默失效）
- ③ fail-fast 校验（`hyperparams.validate_anchors`）：`optuna_anchors` / `adjust_optuna_anchors` 出口断言 `low<=high`，非法即 RuntimeError 启动暴露

### 验证

- 单测 112 passed（experiments 82 + credit-data-analysis 7 + data-cleaning 23），新增：`--feature-list` 清单/缺失列/双源缺失（4 例）、`validate_anchors` + clip 行为（4 例）、overfit 相对化（2 例）、`is_tuned` 透传标注、`tune_winner` 入口防御
- 端到端：真实 49 万样本 `feature_analysis --feature-list`（209 特征精确选列、无崩溃、manifest 溯源）；26 万样本模拟下锚点 clip + overfit 收窄有效交集 `(180,280)`；`--until tune --resume` 重放无 `-opt-opt` 残留、-opt spec 唯一

## v2.5（2026-08-20）

`classification-model-experiments` 交互优化：矩阵规划（生成实验计划）前新增**实验范围确认**，算法与 4 个开关由用户逐项选定，按回答收缩实验矩阵。

### 实验范围确认（规划前交互）

- **新增 `scripts/plan_scope.py`**（交互确认器）：先问算法（xgb / lgb / 两者都选），再依次问 4 个开关——样本选择（recent-N/时间加权）、特征选择（importance/iv-psi）、对抗验证（**耗时提醒**）、Optuna 调优（**耗时提醒**）。优先级：CLI 显式参数 > 交互询问 > 环境约束降级 > 默认值；`--auto-apply` 跳过交互全默认；EOF/非法输入回退默认；无 OOT → 对抗自动关闭、开发池月份 <2 → 样本选择自动关闭（均有理由落 `matrix-plan.json`）。
- **矩阵收缩语义**：不做样本选择 → 样本仅全量 full；不做特征选择 → 特征仅 all（安全过滤后全量）；不做对抗验证 → 跳过波3 对抗格；不做 Optuna → 跳过 winner 规则诊断与调优整段（仅出 leaderboard，转正链路不受影响）。
- **`plan_matrix.build_matrix`**：新增 `feat_select` / `adversarial` 开关参数（默认 True 向后兼容），关闭时跳过对应波次。
- **`parse_cli` 新增 4 个显式开关**：`--no-sample-select` / `--no-feat-select` / `--no-adversarial` / `--no-tune`；`--algos` 默认改为 None（缺省时交互询问）。`--resume` 断点续跑沿用已存矩阵，不重复询问。
- 新增 `tests/test_plan_scope.py`（9 例：算法三选一 / 开关 y-n / EOF 兜底 / CLI 覆盖 / 无 OOT 与月份降级 / 摘要）；`test_plan_matrix.py` 补 3 例关闭态矩阵结构断言。
- **leaderboard 结果透出**：`leaderboard.md/xlsx` 表格新增「诊断」「调优」两列（-opt 格显示五状态 / tuned(trials, best_val)；well_fit 跳过调优显示标注），新增「诊断与调优明细」小节（诊断状态+触发原因+关键信号、Optuna trials/best_value/best_params/search_space）；数据源 = specs 内 -opt 格 diagnosis/optuna 字段，manifest.json 兜底；`--no-tune` 时列显示 `-`、明细小节省略；`test_leaderboard.py` 补 5 例（列透出 / manifest 兜底 / well_fit 跳过标注 / 无明细）。
- 文档同步：experiments SKILL.md（§2 主流程 / §3 参数表 / §7 交互兜底）、development SKILL.md（Stage 4 流程 + 「矩阵方案确认」门禁话术，编排层不重复询问）、plugin.json（version 2.5.0）。

## v2.4（2026-08-20）

新增 `feature-classification` 特征列识别模块，把特征判定从「隐式排除法」前移为「语义三分类 + 用户批量确认」，产出权威 `feature-list.csv` 供全 pipeline 复用。

### 新增模块

- **`feature-classification`**（新 skill，四步注册完成：plugin.json / model-skills README / agents 职责映射 / 本 CHANGELOG）：
  - `scripts/classify_features.py`：探查扫描 → 语义三分类（feature / non_feature / ambiguous）→ 通配符分组（`pfx_*` 折叠 + 混合组展开）→ 落 `feature-classification.json` 探查档案
  - `scripts/rules.py`：**规则库 v0**（方案实证核心资产，直接复用）——日期/时间戳/订单号/ID/序号/分区列 → non_feature 候选；`if_/is_/has_/flag_` 纯 0/1 标识列 → non_feature 候选；匿名编码列（前缀+纯数字）→ ambiguous 默认保留；**用户红线 `fpd*`/`dpd*` 标签列置顶禁入特征集**
  - `scripts/finalize_feature_list.py`：应用用户批量确认名单（`--exclude` 剔除 / `--keep` 恢复）→ 固化 `decided_by(rule/user)` → 产出权威 `feature-list.csv`（全 pipeline 唯一真相）
  - 35 单测 + 真实数据端到端冒烟通过（190 列样本：探查即识别 19 non_feature 候选，finalize 后权威清单 170 列与实证逐列一致）
- **编排接入**（development）：主链路新增 Stage 1（task-spec 与 data-cleaning 之间）；门禁新增「特征列清单确认」（一次批量确认）；路径接力传 `--feature-list-source`；断点续跑新增 `feature-classification.json` 推断
- **data-cleaning 零改动**：已有 `--feature-list-source` 消费能力直接承接权威清单（哨兵替换天然只作用于数值列）
- 文档同步：development SKILL.md（门禁/接力/断点/流程）、agents 职责映射、model-skills README、plugin.json（13 项注册）、根 README、CODEBUDDY.md

## v2.3（2026-08-18）

主链路训练模块切换到 `classification-model-experiments`（实验台→主链路默认）；training/tuning/comparison 代码完整保留为备用路径。核心变化：

### 主链路切换

- **Stage 3 默认 = experiments**：`classification-model-development` 主链路由「training 单 baseline 训练」改为「experiments 实验矩阵（样本×特征正交 + 对抗验证 + 规则诊断 + Optuna 调优 + top10 转正）」，CLI = `run_experiments.py --until promote`；development SKILL.md 的门禁（超参确认→矩阵方案确认）、路径接力、断点续跑（新增 `experiments/matrix-plan.json` 推断）、迭代决策点同步重写。
- **training / tuning / comparison 完整保留**（不删目录、不注销注册），仅标注为备用路径 / 用户主动触发；DNN/LR 评分卡能力随之从主链路下线（experiments 原生仅 lgb/xgb）。

### 规则诊断移植（tuning → experiments）

- 新增 `classification-model-experiments/scripts/diagnose_winner.py` + `recommend_winner.py`：移植自 tuning 的 `diagnose.py` / `recommend_params.py`（阈值逐字对齐：gap 0.05/0.005、PSI 0.10、收敛比 0.95；五状态优先级 overfit > underfit > underconverged > unstable_psi > well_fit），**仅保留 lgb/xgb 策略**（lgb 按参数域新增等价策略表）。
- `tune_winner.py`：Optuna 前对 winner 执行规则诊断 → 按诊断状态调用 `adjust_optuna_anchors` 调整搜索锚点（overfit 收窄容量+抬正则、underfit 放宽、underconverged 拉高 n_estimators、unstable_psi 收窄采样率、well_fit 默认跳过调优）；诊断结果落 `-opt` 格 manifest.json["diagnosis"] 并在日志展示；新增 `--force-tune` 覆盖 well_fit 跳过。

### model-scoring 加载改造

- `score_data.py`：支持 joblib 加载 `model.pkl` 的 lgb/xgb（experiments 转正产物），二维概率输出取违约列（第 1 列）；保留 xgb `model.json`（Booster）历史路径与 dnn/lr pickle 分支；`infer_algo` 扩展 lgb/xgb 判定；`--algo` 覆盖范围扩为 lgb/xgb/dnn/lr。打分能力零损失（dnn/lr 保留）。

### fill_report 适配（消费端，promote 零改动）

- `fill_report.py`：新增 `_is_experiments_run` / `_experiments_source_dir` / `_experiments_split_manifest` / `_latest_run_config`；§IV 从 `experiments/{source_exp}/data/{train,val,oot}.parquet` 重建切分信息（test=实验台 val，注明语义）、§V 兜底读 `experiments/{id}/feature_importance.csv`、§VI 兼容 experiments 型 config.json 顶层 metrics（oot_auc/val_auc + 方案 + 乐观偏差标注，`_classify_run` 增 "experiments 矩阵转正"）、§VII 缺省提示可手动触发 comparison。

### 测试与文档

- 新增 `test_diagnose_winner.py`（五状态 / 策略表 / 锚点调整 / well_fit 跳过）、`test_score_data.py` 增补 lgb/xgb pkl 端到端打分、`test_fill_report_experiments.py`（experiments 型 run 的 §IV/§V/§VI/§VII 回填）。
- 文档同步：agents 职责映射 / model-skills README / CODEBUDDY.md / plugin.json（version 2.3.0，12 项注册保留）/ README.md / experiments·model-scoring·comparison·development SKILL.md；training/tuning SKILL.md 标注备用路径。

## v2.2（2026-08-18）

新增独立实验台模块 `classification-model-experiments`（与 training/tuning 完全解耦）：

- **模块形态**：新 skill + 独立 CLI（`run_experiments.py`），不挂 development Stage 4，任意 session 直接调用；仅消费 `sample.parquet` + `feature-list.csv` + `model.split`。
- **实验矩阵**：lgb/xgb 串行；样本方案（full / recent-N / 线性时间加权 / 对抗剔除）× 特征方案（all 安全过滤 / importance 95% 截断（依赖同样本 all 格，严格正交）/ IV-PSI 单格直算 / 对抗剔除）笛卡尔积；`lgb-full-all-v1` 兼任全局 baseline（不重复跑）。
- **对抗验证**：lgb train-vs-oot 对抗分类器，双产出（样本剔除 + 特征剔除），剔除幅度 AI 推荐 + 用户确认。
- **评选与调优**：leaderboard 按 OOT AUC 排序（乐观偏差标注 + 失败清单）；每算法 winner 1 组 Optuna 邻域调优（TPE seed=42 / 目标 val AUC / 100 轮早停 / 默认 25 trials）产 `-opt` run（复用 winner 数据快照）。
- **转正闭环**：top10 展示 + 用户确认/改选 → 复制 `new-models/{algo}-v{N}/` + 落 `finalized_model.json`（结构对齐 model-scoring 消费契约）。
- **架构纪律**：算法无关（仅 hyperparams/algo_factory/模板与算法相关）；精简产物（每格 = model/ + evaluation/ + feature_importance.csv + manifest.json + logs/ + scripts 代码快照 + data/ 输入快照）；仅复用 `_modelevo-shared`（metrics/config_io/date_utils/gen_feature_list），禁跨 skill import；训练代码权威模板自包含 + 每格快照 + `code_sha256`，可复现。
- **红线例外（用户授权本模块）**：对抗格 OOT 可参与对抗分类器训练与样本/特征筛选统计；IV-PSI 格 OOT 参与 PSI 统计；均禁 OOT 作早停集/进训练集/结构超参选择，OOT 指标标注乐观偏差。
- 四步注册完成：plugin.json / model-skills README / agents 职责映射表 / 本 CHANGELOG。

## v2.1（2026-08-17）

v2.0 精简重构的落地细化（当前版本）。

### 编排与流程

- 编排层合并为一层：删除 `model-task-routing`（二分类判定）与 `classification-model-orchestration`（session 创建 / 报告初始化 / 数据源澄清），职责并入 `classification-model-development`（唯一调度者，串联需求澄清 → 收口打分全链路）。
- 决策点门禁收敛为「必问 2 项（Y 定义、切分窗口）+ 确认 1 项（超参）」；特征筛选 / 不平衡处理 / 打分不再单独询问。
- `model-scoring` 收口后**默认执行**（用户可叫停）；`score-to-fico` / `credit-model-report` / `model-knowledge` 降级为**仅用户主动触发**。

### 切分与特征

- **切分后置**：删除 `feature-analysis`，不再有阶段产 `splits/`。切分唯一真相 = `model.split`（feature_config.yaml / train_config.yaml 三档区间），由 training 在训练消费时即时切分（写 run 内部 `data/splits/` 临时目录），不落 session 级 `splits/`。
- 特征分析由 `credit-data-analysis` 承接（pipeline 模式），新增 Markdown 报告（与 Excel 同源）；PSI 基准月默认第一个 OOT 月（须用户确认）。
- **训练不筛特征**：训练过程不通过 IV/PSI 指标筛选特征（`select_features` 降级为仅用户主动要求）。
- `select_features` 改为**数据直算**（读 baseline 的 train/oot parquet，用共享 `metrics.py` 计算 stats/IV/PSI），不再依赖 feature-analysis csv；`--analysis_dir` 仅兼容保留。
- boundary_filter 支持数据直算（`filter_boundary_features_from_df`）与 csv 读（`filter_boundary_features`）两种入口；csv 缺失时对应规则 warn-and-skip，不阻断训练。

### 评估与可追溯性

- 评估内嵌 `classification-model-training`：`eval_single.py` 从 `classification-model-evaluation` 迁入 `training/scripts/`，评估报告逻辑由 training 承担（不再委托独立 skill）。
- 可追溯性收敛：删除 `record_stage` 脚本快照链 / `render-check` / `deliverables.md` / session 级 `scripts/` 快照层；保留 run 的 `_manifest.json`（断点续跑）+ 单份 `report.md`（4 节）。
- task-spec 只做需求澄清：不产独立样本分析报告（`data-profile/report.md` + `report.xlsx`）、不做引擎裁决（本地文件唯一链路）、不做三档切分；输出收敛为 `task-spec.md`（单文件）+ `_manifest.json`。

## v2.0.0（2026-08-17 前）

- skill 数量 16 → 11：删除 `model-task-routing` / `classification-model-orchestration` / `feature-analysis` / `classification-model-evaluation` / `classification-model-report`（职责分别并入 task-spec / development / credit-data-analysis / training；更早已移除 `classification-model-recommend` 与 `feature-matching`）。
- 公共代码汇集至 `_modelevo-shared/scripts/`：`config_io.py`（配置读写 + 数据安全红线）、`date_utils.py`、`gen_feature_list.py`、`metrics.py`（**统一指标库**：AUC / KS / Gini / PSI / IV / 分桶）、`record_stage.py`（保留文件但不再被编排调用）。
- 全仓库废除 spark 取数：数据链路唯一 local_file（task-spec → data-cleaning → credit-data-analysis → training）。
- 产物调整：删除 session 级 `splits/{train,test,oot}.parquet`、`feature-analysis/`、`scripts/` 快照层、`deliverables.md`。