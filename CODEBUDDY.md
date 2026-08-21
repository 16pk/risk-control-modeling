# CODEBUDDY.md

信贷风控建模专家包（ModelEvo 框架）：一个 Agent 定义 + 精简后 11 个建模 Skill 的专家包，端到端交付可上线的风控分类模型（XGBoost / LightGBM / LR 评分卡 / DNN）。

## 怎么跑起来

本项目是**专家包源码**（非可执行应用），按 `.codebuddy-plugin/plugin.json` 组织，导入专家后由 Agent 调度 skill 脚本执行真实计算。

- 校验 / 注册 / 打包：走平台 `expert-manager` 插件（`~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/`），不在本仓库。
- 单测：`python -m pytest model-skills/<skill>/tests/ -q`（按 skill 单独跑，跨 skill 混跑会 collection 报错）。
- 运行态产物 `runs/` 是符号链接（指向外部运行态目录），不入库。

## 技术栈

Python 3.10+；建模依赖 xgboost / lightgbm / scikit-learn / optbinning / shap / pandas / numpy / pyarrow / matplotlib / joblib。

## 目录与约定

- `agents/risk-control-modeling.md` — 专家知识体（薄文档：角色 / 核心原则 / 技能职责映射 / 全局红线 / 错误恢复），只承载行为规则，实现细节以下沉 skill 为准。
- `model-skills/{name}/` — 每个 skill：`SKILL.md`（frontmatter `name` == 目录名）+ `scripts/` + `references/` + `tests/`。
  - classification 专属 skill 加 `classification-` 前缀；跨流程共享 skill（`data-cleaning`、`credit-data-analysis`、`model-knowledge`、`model-scoring`、`score-to-fico`）不加前缀。版本迭代信息见根目录 `CHANGELOG.md`。
  - `classification-model-experiments` — **v2.3 起主链路默认训练模块**（原独立实验台）：样本×特征正交矩阵 + 对抗验证 + **winner 规则诊断（移植自 tuning，置于 Optuna 前并驱动搜索锚点）** + Optuna 调优 + 转正；仅消费 `sample.parquet` + `feature-list.csv` + `model.split`。**红线例外（用户授权）：对抗格/IV-PSI 格 OOT 可参与对抗训练与筛选统计（禁早停/禁进训练/禁结构选择），OOT 指标标注乐观偏差**。`classification-model-training` / `classification-model-tuning` 保留为备用路径（代码与注册不删）。
- `model-skills/_modelevo-shared/scripts/` — 公共代码（`config_io.py` 配置读写 + 数据安全红线、`date_utils.py` 日期归一化、`gen_feature_list.py`、`metrics.py` 统一指标库（AUC/KS/Gini/PSI/IV/分桶）、`record_stage.py`（保留文件但不再被编排调用）），各 skill 经 `_bootstrap.py` 注入。
- 契约：样本 `id + 特征列 + label`（可含日期列），落盘 `sample.parquet` + `feature-list.csv`；默认 id=`fuid`、日期=`f_p_date`、格式 `YYYY-MM-DD`。
- 数据链路唯一 local_file：task-spec(local_file) → **feature-classification（v2.4 起：语义三分类 + 用户批量确认，产权威 feature-list.csv；红线 `fpd*`/`dpd*` 标签列禁入特征集）** → data-cleaning（零改动，经 `--feature-list-source` 消费权威清单） → credit-data-analysis → experiments（v2.3 主链路训练）。**全仓库已废除 spark 取数**。
- **切分唯一真相** = `feature_config.yaml` / `train_config.yaml` 的 `model.split`；experiments 主链路开发池=train+test 合并后 seed=42 随机 70/30 切 train/val（无独立 test 档）；备用路径 training 消费时即时切分（写 run 内部 `data/splits/` 临时目录），均不落 session 级 `splits/`。
- 建模纪律红线：PSI 红线 0.10、IV 泄漏红线 1.0、缺失率 0.95；禁止硬编码身份证 / 手机号明文；训练过程不通过 IV/PSI 指标筛选特征（boundary_filter 只做常量/泄漏/ID/全缺失安全过滤）。

## 专家扩展接入约定

本专家在设计上支持持续扩展。新增能力按以下约定接入，保证可插拔、可追溯：

### 1. 新增 Skill（建模 / 工具类能力）

1. 在 `model-skills/{name}/` 创建 `SKILL.md`（frontmatter 的 `name` 必须等于目录名），必要时附 `scripts/`、`references/`、`tests/`
2. 在 `plugin.json` 的 `skills` 数组登记路径（新增目录后需重新校验 + 注册）
3. 在 `model-skills/README.md` 的 Skill 清单登记（含触发词）
4. 在 `agents/risk-control-modeling.md`「技能-职责映射表」登记，说明触发场景
5. 需要公共能力时复用 `_modelevo-shared`（配置/安全红线/统一指标）
6. 新 skill 若涉及关键决策，须声明走 `classification-model-development` 的「决策点话术（门禁收敛）」对应节点

### 2. 接入 MCP（数据源 / 外部服务）

- **数据源类 MCP**（数仓、特征平台等）：拉取结果须符合下游 skill 契约——样本含 `id + 特征列 + label`（可含日期列），落盘 `sample.parquet` + `feature-list.csv`；在 `model-skills/README.md`「上下游数据前置」登记数据来源
- **外部服务类 MCP**（模型服务、指标平台、监控告警等）：在对应 skill 的 SKILL.md 中声明调用方式与参数
- 取数仍须遵守**数据安全红线**：不透出身份证 / 手机号等明文个人数据

### 3. 新增 bin/ 工具

- 通用 CLI 工具放 `bin/`，按规范在 plugin.json 声明

## 当前状态与下一步

- 当前（plugin.json `version`，v2.4.0）：单编排器（development）+ 主链路（需求澄清 → 特征列识别 feature-classification → 清洗 → 特征分析 → experiments 实验矩阵+对抗+规则诊断+Optuna+转正 → 默认打分）；training/tuning 保留为备用路径；已支持定版模型打分（model-scoring 默认执行，支持 experiments 转正 pkl 加载）与 FICO 转换（score-to-fico，可选）。
- 规划中未实现：`model-publication`、`metric-matching`、`classification-model-evolution-plan`、`classification-segment-model`、特征衍生 / 样本调整 / loss 优化 / 网络结构优化。
