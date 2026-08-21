# 风控建模专家 (Risk Control Modeling)

基于 ModelEvo 框架的信贷风控建模专家：分类建模（XGBoost / LightGBM）、特征工程（IV / PSI / WOE）与模型评估，端到端交付可直接上线的风险模型。

## 类型

Agent 型（单个 AI 专家）

## 功能

- **建模入口判定**：通过 Q1/Q2/Q3 判定需求是否属于 classification（预测"谁会发生结果"）。
- **分类建模**：实验矩阵（样本×特征正交 + 对抗验证 + 规则诊断 + Optuna 调优 + 转正，v2.3 主链路默认）→ 定版打分 → 归档。
- **特征工程**：IV / PSI / WOE 分析（credit-data-analysis 分月体检报告）。
- **风控纪律**：时间切分（Train<Test<OOT）、PSI 0.10 / IV 0.02·1.0 / 缺失率 0.95 红线、数据安全红线（身份证 / 手机号硬编码拦截）。

## 已挂载的建模技能（随包分发，开箱可用）

专家目录下的 `model-skills/` 包含完整的 ModelEvo 技能集，导入后专家可直接调度执行真实计算；`_modelevo-shared/` 为公共代码（配置读写 + 数据安全红线）。

- 知识库：`model-knowledge`
- 共享能力：`feature-classification` / `data-cleaning` / `credit-data-analysis` / `model-scoring`
- 分类建模：`classification-model-task-spec` / `classification-model-development` / `classification-model-experiments`（主链路默认训练）/ `classification-model-package` / `credit-model-report` / `score-to-fico`

## 使用示例

- 基于本地样本文件做一个信贷逾期风险分类模型（lgb/xgb 实验矩阵）
- 帮我做特征分析，按 IV / PSI / 缺失率 查看不达标变量（credit-data-analysis）
- 跑实验矩阵（含对抗验证、规则诊断 + Optuna 调优）并转正胜出模型

## 目录结构

```
risk-control-modeling/
├── .codebuddy-plugin/plugin.json   # 专家元数据（含 skills 挂载声明）
├── agents/risk-control-modeling.md # 专家知识体（角色/能力/SOP/红线）
├── avatars/expert.png              # 头像
├── CHANGELOG.md                    # 版本迭代信息（仅供包维护者）
├── model-skills/                   # 挂载的 ModelEvo 技能集（11 个 skill）
│   ├── _modelevo-shared/           # 公共代码（数据安全红线/配置读写/metrics）
│   └── classification-model-development/ ...  # 各 skill（均含 SKILL.md）
└── README.md
```

## 头像

头像已自动生成在 `avatars/` 目录下。如需替换：PNG / JPG、512×512 px、≤500KB。

## 安装 / 导入

本专家包按 WorkBuddy/CodeBuddy 专家规范（`.codebuddy-plugin/plugin.json`）组织。将其解压到专家目录后，用 **expert-manager 平台插件**（内置 `skill-expert-manager`）的脚本校验并注册：

```bash
# 平台插件脚本位于 ~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/
python3 ~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/register_expert.py <expert-dir> --session-id <session-id>
```

> 注：`register_expert.py` / `validate_expert.py` / `package_expert.py` 属于平台 expert-manager 插件，**不在本仓库**。需要完整工作流时调用 `expert-manager` skill（触发词：创建/导入/修改/校验/打包专家）。

`model-skills/` 内的技能会随专家一并被加载，无需单独安装。

## 打包分享

```bash
python3 ~/.workbuddy/plugins/marketplaces/workbuddy-builtin/skills/expert-manager/scripts/package_expert.py <expert-dir> [output-dir]
```
