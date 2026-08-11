# 风控建模专家 (Risk Control Modeling)

基于 ModelEvo 框架的信贷风控建模专家：分类建模（XGBoost / LightGBM / LR 评分卡 / DNN）、特征工程（IV / PSI / WOE）与模型评估对比，端到端交付可直接上线的风险模型。

## 类型

Agent 型（单个 AI 专家）

## 功能

- **建模入口判定**：通过 Q1/Q2/Q3 判定需求是否属于 classification（预测"谁会发生结果"）。
- **分类建模**：baseline 训练 → 调参 / 特征筛选 / 换算法迭代 → 多模型 N-way 对比 → 归档。
- **特征工程与评分卡**：IV / PSI / WOE 筛选，LR 评分卡（`Score = base_score − factor·ln(odds)`）。
- **风控纪律**：时间切分（Train<Test<OOT）、PSI 0.10 / IV 0.02·1.0 / 缺失率 0.95 红线、数据安全红线（身份证 / 手机号硬编码拦截）。

## 已挂载的建模技能（随包分发，开箱可用）

专家目录下的 `model-skills/` 包含完整的 ModelEvo 技能集，导入后专家可直接调度执行真实计算；`_modelevo-shared/` 为公共代码（配置读写 + 数据安全红线）。

- 路由与知识：`model-task-routing` / `model-knowledge`
- 共享能力：`feature-matching` / `feature-analysis`
- 分类建模：`classification-model-orchestration` / `classification-model-task-spec` / `classification-model-recommend` / `classification-model-development` / `classification-model-training` / `classification-model-tuning` / `classification-model-evaluation` / `classification-model-comparison` / `classification-model-report`

## 使用示例

- 基于本地样本文件做一个信贷逾期风险分类模型，并产出评分卡
- 帮我做特征分析，按 IV / PSI / 缺失率 筛选不达标的变量
- 对比多个候选模型的 AUC / KS 分桶排序性，给出推荐与上线建议

## 目录结构

```
risk-control-modeling/
├── .codebuddy-plugin/plugin.json   # 专家元数据（含 skills 挂载声明）
├── agents/risk-control-modeling.md # 专家知识体（角色/能力/SOP/红线）
├── avatars/expert.png              # 头像
├── model-skills/                   # 挂载的 ModelEvo 技能集（16 个 skill）
│   ├── _modelevo-shared/           # 公共代码（数据安全红线/配置读写），install 时置于此处
│   ├── model-task-routing/  ...    # 各 skill（均含 SKILL.md）
└── README.md
```

## 头像

头像已自动生成在 `avatars/` 目录下。如需替换：PNG / JPG、512×512 px、≤500KB。

## 安装 / 导入

将本专家包解压到专家目录后注册即可：

```bash
python3 scripts/register_expert.py <expert-dir> --session-id <session-id>
```

`model-skills/` 内的技能会随专家一并被加载，无需单独安装。

## 打包分享

```bash
python3 scripts/package_expert.py <expert-dir> [output-dir]
```
