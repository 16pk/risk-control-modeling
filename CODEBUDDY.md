# CODEBUDDY.md

信贷风控建模专家包（ModelEvo 框架）：一个 Agent 定义 + 16 个建模 Skill 的专家包，端到端交付可上线的风控分类模型（XGBoost / LightGBM / LR 评分卡 / DNN）。

## 怎么跑起来

本项目是**专家包源码**（非可执行应用），按 `.codebuddy-plugin/plugin.json` 组织，导入专家后由 Agent 调度 skill 脚本执行真实计算。

- 校验 / 注册 / 打包：走平台 `expert-manager` 插件（`~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/`），不在本仓库。
- 单测：`python -m pytest model-skills/<skill>/tests/ -q`（按 skill 单独跑，跨 skill 混跑会 collection 报错）。
- 运行态产物 `runs/` 是符号链接（指向外部运行态目录），不入库。

## 技术栈

Python 3.10+；建模依赖 xgboost / lightgbm / scikit-learn / optbinning / shap / pandas / numpy / pyarrow / matplotlib / joblib。

## 目录与约定

- `agents/risk-control-modeling.md` — 专家知识体（角色 / SOP / 关键决策确认门禁 / 红线）。
- `model-skills/{name}/` — 每个 skill：`SKILL.md`（frontmatter `name` == 目录名）+ `scripts/` + `references/` + `tests/`。
  - classification 专属 skill 加 `classification-` 前缀；跨流程共享 skill（`data-cleaning`、`feature-analysis`、`credit-data-analysis`、`model-knowledge`、`model-scoring`、`score-to-fico`、`model-task-routing`）不加前缀。
- `model-skills/_modelevo-shared/scripts/` — 公共代码（`config_io.py` 配置读写 + 数据安全红线、`date_utils.py` 日期归一化、`gen_feature_list.py`、`record_stage.py`），各 skill 经 `_bootstrap.py` 注入。
- 契约：样本 `id + 特征列 + label`（可含日期列），落盘 `sample.parquet` + `feature-list.csv`；默认 id=`fuid`、日期=`f_p_date`、格式 `YYYY-MM-DD`。
- 数据链路唯一 local_file：task-spec(local_file) → data-cleaning → feature-analysis → training。**全仓库已废除 spark 取数**。
- 建模纪律红线：OOT 必须按时间且晚于训练窗；PSI 红线 0.10、IV 0.02/1.0、缺失率 0.95；禁止硬编码身份证 / 手机号明文。

## 当前状态与下一步

- v1.5.0（plugin.json），仅支持 classification；已支持定版模型打分（model-scoring）与 FICO 转换（score-to-fico）。
- 规划中未实现：`model-publication`、`metric-matching`、`classification-model-evolution-plan`、`classification-segment-model`、特征衍生 / 样本调整 / loss 优化 / 网络结构优化。
