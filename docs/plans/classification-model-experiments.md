# classification-model-experiments 实验台模块 — 执行计划

> 状态：**plan ready，未执行**（2026-08-18 确认，用户将在新建对话内执行）
> 保存位置：`docs/plans/classification-model-experiments.md`
> 执行方式：新建对话中告知 AI「读取 `docs/plans/classification-model-experiments.md` 并开始执行本计划」

---

## 1. 用户需求（原话）

> 我想要创建个新模块，该模块完全独立于model-training和model-tuning模块。模块被调用时，首先使用lightgbm跑一版baseline模型，作为后续所有训练任务的比较基准。然后模块开始使用不同模型框架，平行执行多个类别的建模调优任务，比如lightgbm任务，xgboost任务等。单个任务内，需要考虑多个样本筛选方案，eg：全量训练窗口内样本做训练集，训练窗口内最近N个月样本做训练集，训练集样本按时间加权等。需要考虑多个特征筛选方案，eg：全量特征，通过前一组实验的特征重要性筛选特征，通过iv/psi等指标筛选特征。对抗验证筛选样本+特征的方案也纳入。每组实验的参数要保留下来，确保可以复现实验。针对每一类任务中在oot上auc最佳的那组实验，再使用optuna调优。
> 设计调优方案时，可以参考model-tuning模块，择优复用。
> 由于不同实验的训练集样本并不相同，所以在每一组实验中从训练集中随机切出30%样本作为验证集。如果你有更好的建议，请给出。
> 推荐的超参数方案：n_estimators=1000，100轮早停；learning_rate=0.04；num_leaves=min(31, 2^(S/10))；max_depth=-1（lgb）或4（xgb）；min_child_samples / min_data_in_leaf=max(20, 0.2% * M)；min_child_weight / min_sum_hessian_in_leaf=1e-3；bagging_fraction=0.6；feature_fraction=max((num_leaves * 2) / S, 0.5)，其中M为样本数，S为特征维度。

---

## 2. 已确认决策记录（逐轮，全部由用户拍板）

### 2.1 第一轮分支确认（grill-me）

| 分支 | 决策 |
|---|---|
| A1 转正流程 | 展示 top10 指标让用户确认，允许改选其他实验作为转正实验 |
| A2 目录命名 | `experiments/{algo}-{sample_scheme}-{feat_scheme}-v{N}/`（如 `lgb-full-all-v1`） |
| 模块形态 | 完全独立新 skill `classification-model-experiments` + 独立 CLI；**不挂 development Stage 4** |
| 输入契约 | 仅消费 `sample.parquet` + `feature-list.csv` + `model.split`（train/test/oot 三档时间区间），不依赖任何既有 run |
| B1 baseline | 不重复跑，`lgb-full-all-v1` 兼任全局 baseline（矩阵第一格） |
| B2 随机种子 | 全部实验统一 seed=42（随机切 30% val） |
| B3 超参公式 | M/S 每组实验独立计算；xgb 侧 `min_child_weight=1e-3`（不映射 min_child_samples） |
| C1 算法 | 仅 lgb + xgb，**串行**（先 lgb 矩阵完再 xgb）；不纳入 dnn/lr |
| C2 组合方式 | 样本×特征**笛卡尔积全组合**；对抗验证独立 1 格（不与其余方案交叉） |
| C3 组数自决 | 样本/特征方案组数由 AI 按样本情况动态自决并记录理由 |
| C4 时间加权 | **线性衰减**（最近月 1.0 → 最远月 0.2：`w = 0.8*(t-t_min)/(t_max-t_min)+0.2`） |
| C5 重要性依赖 | 实验串行；importance 特征方案取**同样本方案下 all 特征格**的特征重要性（第二轮修改点 4 强化为正交） |
| C6 IV/PSI 阈值 | **放松**：PSI>0.2 / IV<0.015 / 缺失>0.95，单组实验内数据直算 |
| C7 对抗验证域 | train vs oot；**用户授权本模块例外：OOT 可用于对抗筛选样本/特征**（禁早停/禁进训练集/禁结构超参选择） |
| C8 对抗产出 | 双产出（样本侧 + 特征侧）；剔除幅度 AI 评估推荐 + 用户确认 |
| D1 开发池 | = `model.split` 的 **train + test 两档合并**，seed=42 随机切 70%/30%（训练/val）；OOT 纯榜单 |
| D2 评选+调优 | 每算法 OOT AUC 最优 1 组进 Optuna；TPE seed=42 / 目标 val AUC / 100 轮早停 / n_trials 默认 25 |
| D3 调优后落点 | 转正前展示 top10 + 推荐最优，用户确认/改选后写 `finalized_model.json` + 复制进 `new-models/` |
| D4 对抗幅度 | 剔除幅度 AI 运行时评估推荐 + 与用户确认后执行 |
| E1 对抗分类器 | lgb 小模型（num_leaves=31 / lr=0.05 / 100 轮早停 / train vs oot），total_gain 作特征剔除依据、predict_proba 作样本剔除依据 |
| F1 失败容错 | 单格失败 → 跳过 + 记录原因 + 继续其余，汇总报告含失败清单 |
| F2 汇总产物 | `leaderboard.{md,xlsx}`（全实验排序总表，top10 数据来源）+ 每格 manifest + 断点续跑 |
| F3 入口 | 独立 skill + 独立 CLI，任意 session 直接调用 |
| F4 预算 | 单算法实验数上限默认 12 格（可配 `--max-experiments-per-algo`） |

### 2.2 第二轮修改建议（全部确认采纳）

| # | 修改 | 落定内容 |
|---|---|---|
| 1 | 算法无关 + sklearn 兼容 | plan_matrix / sample_schemes / feature_schemes / leaderboard / promote / adversarial 全部算法无关；仅 `hyperparams.py` 与 `algo_factory.py`（统一 `fit(X,y,sample_weight)` / `predict_proba` / `feature_importances_` 接口）与算法相关；lgb=`LGBMClassifier`、xgb=`XGBClassifier`；未来加 dnn/lr 只扩展 factory |
| 2 | 精简目录（废除八阶段） | 每格仅保留：`model/`（pkl+meta）、`evaluation/`（eval.{json,md}）、`feature_importance.csv`、`manifest.json`、`logs/run.log`、`scripts/`（代码快照）；skill 自身也精简 |
| 3 | 仅复用共享层 | **禁止跨 skill import**；不 import training 的 `eval_single.py`/boundary_filter；评估自实现精简版四档（复用 `_modelevo-shared/metrics.py` 的 AUC/KS/IV/PSI/分桶）；安全过滤（常量/泄漏/ID/全缺失）内聚本模块 |
| 4 | 严格正交 | importance 特征方案 = 取**同样本方案交叉 all 特征方案（`{sample_scheme}-all`）**实验的 total_gain 累积 95% 截断；依赖为同算法内 sample_scheme 维度的 DAG |
| 5 | 训练代码进实验目录 | **采用推荐折中**：一份权威模板 `scripts/templates/train_template.py`（sklearn 风格、高参数化），每格实验开始时快照复制进 `<exp_dir>/scripts/` 并记 `code_sha256 + template_version` 到 manifest；默认全格同代码（可比），AI 需定制某格时逐格 fork 修改（manifest 记 `code_modified=true`）；复现 = 重跑实验目录代码 |

---

## 3. 产品概述

新增完全独立的建模实验台模块（新 skill `classification-model-experiments`），与 `classification-model-training`、`classification-model-tuning` 完全解耦。模块自跑 LightGBM baseline 作为比较基准，随后串行执行多算法（lgb / xgb）下「样本方案 × 特征方案」的正交实验矩阵，经 leaderboard 评选 + Optuna 调优，最终展示 top10 由用户确认后转正为定版模型。全程参数落盘、可复现、支持断点续跑。

## 4. 核心功能

- **Baseline 构建**：LightGBM + 全量开发期样本 + 安全过滤后全量特征跑 `lgb-full-all-v1`，兼任全局 baseline 与矩阵第一格（不重复跑）。
- **多算法实验矩阵**：lgb 与 xgb 串行；每类任务内样本方案（全量 / 最近 N 个月 / 时间加权 / 对抗剔除）× 特征方案（全量 / 特征重要性 / IV-PSI / 对抗剔除）正交笛卡尔积；方案组数由 AI 依据样本量、特征数、类别分布动态自决并记录理由。
- **对抗验证独立实验格**：train vs oot 训练 lgb 对抗分类器，双产出（剔除分布差异最大样本 + 剔除对抗 top-K 特征），剔除幅度 AI 推荐 + 用户确认。
- **验证集与评选**：开发池 = train + test 合并，每格 seed=42 随机切 70%/30%，OOT 纯榜单；leaderboard 按 OOT AUC 排序，每算法最优 1 组进 Optuna 邻域调优（目标 val AUC、100 轮早停、默认 25 trials）。
- **转正闭环**：展示 top10 + 推荐最优，用户确认/改选后复制进 `new-models/` + 写 `finalized_model.json`，供 model-scoring 无感消费。
- **可复现与容错**：每格完整 manifest（超参/方案/seed/依赖源/代码 hash）+ 代码快照；单格失败跳过并记录；重跑跳过已完成实验。

## 5. 技术方案要点

### 5.1 执行主流程

```
矩阵规划(AI 自决组数+理由落 matrix-plan.md)
  → 波1: {各 sample_scheme}-all-v1 串行（lgb-full-all 兼 baseline）
  → 波2: {sample_scheme}-importance-v1（依赖同样本 all 格 95% 截断）
         {sample_scheme}-iv-psi-v1（单格直算，无依赖）
  → 对抗格（lgb train-vs-oot 双产出，幅度确认）
  → 每算法 leaderboard（OOT AUC 排序 + 乐观偏差标注）
  → 每算法 winner Optuna 邻域调优（-opt run）
  → 汇总 top10 展示 + 推荐最优
  → 用户确认/改选 → 复制 new-models/{algo}-v{N}/ + finalized_model.json
  → model-scoring 打分（下游）
```

### 5.2 关键技术决策

1. **算法无关 + sklearn 兼容**：除 hyperparams/algo_factory 外全部与算法无关；统一 estimator 接口。
2. **精简产物**：每格 = model/ + evaluation/ + feature_importance.csv + manifest.json + logs/run.log + scripts/（代码快照）。
3. **仅复用共享层**（`_modelevo-shared/scripts/`：config_io 安全红线、date_utils、metrics、gen_feature_list），经 `_bootstrap.py` 注入；**禁止 import 任何其他 skill 脚本**。
4. **样本×特征严格正交**：importance 依赖同样本方案的 all 格（同算法内 DAG）。
5. **训练代码权威模板 + 快照**：`scripts/templates/train_template.py`，每格快照复制 + `code_sha256`；默认全格同代码可比，可逐格 fork（记 `code_modified=true`）。
6. **推荐超参公式**（每格按自身 M/S 独立计算）：
   - `n_estimators=1000`、100 轮早停（train 拟合、val 早停）、`learning_rate=0.04`
   - lgb：`num_leaves=min(31, 2^(S/10))`、`max_depth=-1`、`min_child_samples=max(20, 0.002*M)`、`min_sum_hessian_in_leaf=1e-3`
   - xgb：`max_depth=4`、`min_child_weight=1e-3`
   - 共用：`subsample=0.6`、`colsample_bytree=max((num_leaves*2)/S, 0.5)`、`scale_pos_weight=neg/pos` 自动、`seed=42`
7. **开发池与切分**：开发池 = train + test 合并；每格施加样本方案后 seed=42 随机切 70%/30%；OOT 纯榜单。
8. **红线例外（用户授权本模块特例，SKILL.md 显式声明）**：
   - ① 对抗格：OOT 可参与对抗分类器训练与样本/特征筛选统计；
   - ② IV-PSI 格：OOT 参与 PSI 统计；
   - 两者均**禁止 OOT 作早停集、进训练集、参与结构超参选择**；两类实验的 OOT 指标存在乐观偏差，leaderboard 显著标注。
9. **Optuna 邻域调优**：以 winner 格 M/S 推导超参为锚点收窄邻域（lr ±50%、num_leaves ±8、min_child_samples ±40% 等）；TPE seed=42、目标 val AUC、100 轮早停、默认 25 trials；产 `-opt` run。
10. **转正保持 model-scoring 消费契约**：`new-models/{run}/model/` + `config.json`（`mark_finalized.py` / `score_data.py` 消费）。
11. **性能与可靠性**：实验串行；每格完成落 `_manifest.json` 支持断点续跑；单格失败捕获继续；数据安全沿用 `config_io.check_sensitive`。

### 5.3 系统架构

```mermaid
flowchart TD
    subgraph 输入
        SP[sample.parquet + feature-list.csv + model.split]
    end
    SP --> MP[矩阵规划 AI 自决 + matrix-plan.md]
    MP --> W1[波1: 各样本方案 all 格 串行<br/>lgb-full-all 兼 baseline]
    W1 --> W2[波2: importance格 依赖同样本all格 + iv-psi格 单格直算]
    W2 --> ADV[对抗格 lgb train-vs-oot 双产出 幅度确认]
    ADV --> LB[leaderboard OOT AUC 排序 + 乐观偏差标注]
    LB --> TUNE[每算法 winner Optuna 邻域调优 -opt run]
    TUNE --> TOP[汇总 top10 展示 + 推荐最优]
    TOP -->|确认/改选| PROM[复制 new-models/{algo}-v{N} + finalized_model.json]
    PROM --> SC[model-scoring 打分]
```

## 6. 目录结构

### 6.1 新 skill

```
model-skills/classification-model-experiments/   # [NEW] 新 skill 根目录
├── SKILL.md                                     # 输入契约/执行命令/红线例外/产物规范/异常处理/测试
├── scripts/
│   ├── _bootstrap.py                            # 注入 _modelevo-shared 共享路径
│   ├── run_experiments.py                       # 主入口 CLI：规划→矩阵串行→评选→Optuna→转正确认
│   ├── plan_matrix.py                           # 矩阵规划器：AI 自决组数+理由、实验清单、断点状态
│   ├── sample_schemes.py                        # 样本方案：full / recent-N / 线性时间加权（算法无关）
│   ├── feature_schemes.py                       # 特征方案：all(安全过滤) / importance 95% / iv-psi 直算（算法无关）
│   ├── hyperparams.py                           # M/S 公式推导超参（lgb/xgb 两侧）
│   ├── algo_factory.py                          # sklearn 兼容 estimator 工厂（LGBM/XGB，可扩展）
│   ├── safety_filter.py                         # 常量/泄漏/ID/全缺失安全过滤（内聚本模块）
│   ├── evaluate.py                              # 精简四档评估（复用 metrics.py：AUC/KS/IV/PSI/分桶）
│   ├── adversarial.py                           # 对抗分类器 lgb train-vs-oot + 双产出幅度推荐（算法无关）
│   ├── run_single_experiment.py                 # 单格执行器：模板快照+hash、训练、评估、manifest、失败容错
│   ├── leaderboard.py                           # leaderboard.{md,xlsx} + top10 排序（含乐观偏差标注）
│   ├── tune_winner.py                           # winner Optuna 邻域调优产 -opt run（统一 objective 接口）
│   ├── promote.py                               # top10 展示确认/改选、复制 new-models、调 mark_finalized
│   └── templates/
│       └── train_template.py                    # 权威训练模板（sklearn 风格、高参数化，供每格快照复制）
├── references/
│   └── constraints-and-exceptions.md            # 红线例外声明/搜索空间/异常处理全表
└── tests/
    ├── test_plan_matrix.py                      # 矩阵规划/组数自决/断点续跑
    ├── test_sample_schemes.py                   # recent-N 切窗/线性加权/开发池合并
    ├── test_feature_schemes.py                  # importance 截断/iv-psi 直算/安全过滤
    ├── test_hyperparams.py                      # 公式推导与边界
    ├── test_adversarial.py                      # 对抗分类器/剔除幅度推荐
    └── test_leaderboard.py                      # leaderboard 排序/乐观偏差标注/代码 hash 比对
```

### 6.2 涉及修改（四步注册）

- `.codebuddy-plugin/plugin.json` — skills 数组新增一项
- `model-skills/README.md` — 目录结构 / Skill 清单 / 流程登记
- `agents/risk-control-modeling.md` — 技能-职责映射表新增一行
- `CHANGELOG.md` — 版本迭代记录

### 6.3 session 实验产物

```
<session_dir>/experiments/{algo}-{sample_scheme}-{feat_scheme}-v{N}/
├── manifest.json            # 全超参/方案/seed/依赖源/code_sha256/template_version/code_modified/status
├── model/model.pkl + model_meta.json
├── evaluation/eval.{json,md} # train/val/oot/all 四档
├── feature_importance.csv
├── scripts/train.py         # 训练代码快照（AI 可逐格 fork 修改）
└── logs/run.log
<session_dir>/experiments/leaderboard.{md,xlsx}  # 全实验排序总表 + 失败清单 + matrix-plan.md
```

## 7. 关键代码结构

```python
# hyperparams.py —— M/S 公式推导超参（lgb / xgb 两侧，每组实验独立计算）
def derive_params(algo: str, n_samples: int, n_features: int) -> dict:
    """按用户推荐公式计算基线超参，M=样本数 S=特征维度"""
    num_leaves = min(31, int(2 ** (n_features / 10)))
    base = {
        "objective": "binary", "metric": "auc", "n_estimators": 1000,
        "learning_rate": 0.04, "seed": 42, "verbosity": -1,
        "subsample": 0.6, "colsample_bytree": max((num_leaves * 2) / n_features, 0.5),
        "scale_pos_weight": "auto", "early_stopping": 100,
    }
    if algo == "lgb":
        base.update({"num_leaves": num_leaves, "max_depth": -1,
                     "min_child_samples": max(20, int(0.002 * n_samples)),
                     "min_sum_hessian_in_leaf": 1e-3})
    elif algo == "xgb":
        base.update({"max_depth": 4, "min_child_weight": 1e-3, "tree_method": "hist"})
    return base

# algo_factory.py —— 算法无关 sklearn 兼容工厂
def build_estimator(algo: str, params: dict):
    """统一 fit(X,y,sample_weight) / predict_proba / feature_importances_ 接口"""
    if algo == "lgb":
        return lgb.LGBMClassifier(**params)
    elif algo == "xgb":
        return xgb.XGBClassifier(**params)
    raise ValueError(f"unsupported algo: {algo}")

# plan_matrix.py —— 实验格契约（manifest 落盘核心）
# ExperimentSpec = {
#   "id": "lgb-full-all-v1", "algo": "lgb", "wave": 1,
#   "sample_scheme": "full", "feat_scheme": "all",
#   "depends_on": None,                      # importance 格 = 同(algo,sample_scheme) 的 all 格
#   "n_samples": 0, "n_features": 0, "params": {}, "seed": 42,
#   "code_sha256": "", "template_version": "v1", "code_modified": False,
#   "status": "pending|done|failed", "fail_reason": None,
# }
```

## 8. 执行待办（todo 顺序）

1. **skill-skeleton**：创建 skill 骨架（SKILL.md + scripts/references/tests 结构）并完成四步注册（plugin.json/README/agent/CHANGELOG）
2. **algo-core**：algo_factory 工厂、hyperparams M/S 公式推导、safety_filter 安全过滤、evaluate 精简四档评估（复用 metrics.py）
3. **matrix-planner**：plan_matrix.py 矩阵规划器（AI 自决组数、理由落 matrix-plan.md、实验清单、断点续跑）
4. **sample-feature-schemes**：sample_schemes.py（full/recent-N/线性时间加权）+ feature_schemes.py（all/importance 依赖同样本 all 格 95% 截断/iv-psi 直算）
5. **adversarial-runner**：adversarial.py（lgb train-vs-oot 双产出、幅度推荐确认）+ run_single_experiment.py（模板快照+hash、训练、评估、manifest、失败容错）
6. **leaderboard-tune**：leaderboard.py（排序 + 乐观偏差标注）+ tune_winner.py（每算法 OOT 最优 1 组 Optuna 邻域调优产 -opt run）
7. **promote-tests**：promote.py（top10 展示 + 确认/改选转正）+ 汇总报告 + 全量 pytest 用例

## 9. 环境与测试约定

- 环境：base = `/Users/jensenliu/miniforge3`（py3.12.10）；缺 optuna/shap；optuna 需 `pip install --user "optuna<4"`（项目既有约定），缺失时清晰报错、相关用例跳过
- 测试：`python -m pytest model-skills/classification-model-experiments/tests/ -q`（单 skill 运行）；真跑用小型合成数据冒烟
- 复用共享层：`_modelevo-shared/scripts/metrics.py`（AUC/KS/Gini/PSI/IV/分桶）、`config_io`（check_sensitive 数据安全红线）、`date_utils`（month_prefix 等）