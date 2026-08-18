# Changelog

本文件只记录专家包与建模框架的**版本迭代信息**（结构变更、职责迁移、产物调整），供包维护者参考。用户调用 skill 时无需阅读；各 skill 的 SKILL.md 只描述当前状态，不携带版本标记。

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