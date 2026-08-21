---
name: classification-model-experiments
description: 建模主链路默认训练模块（v2.3）兼独立实验台。用 LightGBM 跑 baseline（lgb-full-all-v1 兼任全局基准），随后串行执行 lgb / xgb 下「样本方案 × 特征方案」正交实验矩阵（全量 / 最近N月 / 线性时间加权 / 对抗剔除 × 全量 / 特征重要性 / IV-PSI / 对抗剔除），含对抗验证独立格、leaderboard 评选（OOT AUC 排序 + 乐观偏差标注）、每算法 winner 规则诊断（overfit/underfit/underconverged/unstable_psi/well_fit，Optuna 前执行并驱动搜索锚点调整、well_fit 可跳过）+ Optuna 邻域调优（-opt run），最终展示 top10 由用户确认后转正（new-models/ + finalized_model.json）。当用户说"跑实验矩阵""实验台""多方案对比实验""baseline 实验""正交实验""对抗验证""实验调优"或主链路默认训练时使用。
---

# classification-model-experiments

建模实验台（v2.3 起为主链路默认训练模块）：矩阵规划 → 样本/特征方案正交实验 → 对抗验证 → leaderboard 评选 → **winner 规则诊断**（Optuna 前执行）→ Optuna 调优（诊断驱动锚点）→ 转正闭环。由 `classification-model-development` Stage 3 默认调度；也可任意 session 独立调用。

> ⚠️ **红线例外（仅本模块，用户已授权，见 §9）**：对抗格 OOT 可用于对抗分类器训练与样本/特征筛选统计；IV-PSI 格 OOT 参与 PSI 统计。两者均**禁止** OOT 作早停集、进训练集、参与结构超参选择；对应实验的 OOT 指标存在乐观偏差，leaderboard 显著标注。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `sample.parquet` | ✅ | `data-cleaning` 清洗后样本 | 含 `id + 特征列 + label`（可含日期列），格式 `YYYY-MM-DD`/`YYYYMMDD` |
| `feature-list.csv` | ✅ | `data-cleaning` 派生特征清单 | 单列 `feature_name` |
| `model.split` | ✅ | `feature_config.yaml` / `train_config.yaml` | train/test/oot 三档时间区间（切分唯一真相） |
| Optuna | 否 | `pip install --user "optuna<4"` | 仅 Optuna 调优需要，缺失时清晰报错、相关用例跳过 |

本模块**不依赖任何既有 run**，仅消费上述三件套。

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录（即本文件所在目录），执行时替换为实际绝对路径。

### 2.1 主入口（run_experiments.py）

```bash
python <skill_dir>/scripts/run_experiments.py \
    --session-dir <session_dir> \
    --sample <sample.parquet> \
    --feature-list <feature-list.csv> \
    --config <feature_config.yaml>            # 读 model.split；或直接传 --split-train/--split-test/--split-oot
    [--label-col <label>] [--id-col fuid] [--dt-col f_p_date] \
    [--algos lgb xgb] [--max-experiments-per-algo 12] \
    [--no-sample-select] [--no-feat-select] [--no-adversarial] [--no-tune] \
    [--n-trials 25] [--auto-apply] [--resume] [--force-tune]
```

主流程：**实验范围确认（v2.5：规划前逐项询问——算法三选一 lgb/xgb/两者，样本选择 / 特征选择 / 对抗验证 / Optuna 调优 4 个开关；对抗验证与 Optuna 附耗时提醒；选择理由落 matrix-plan.md）** → 矩阵规划（AI 自决组数 + 理由落 `matrix-plan.md`）→ 波1 各样本方案 all 格（`lgb-full-all-v1` 兼 baseline）→ 波2 importance / iv-psi 格 → 对抗格（lgb train-vs-oot 双产出，幅度确认）→ 每算法 leaderboard（OOT AUC 排序 + 乐观偏差标注）→ 每算法 winner **规则诊断**（`diagnose_winner`，五状态：overfit/underfit/underconverged/unstable_psi/well_fit）→ 按诊断调整 Optuna 搜索锚点（`adjust_optuna_anchors`）→ 邻域调优（`-opt` run；well_fit 默认跳过，`--force-tune` 强制；**关闭 Optuna 时诊断与调优整段跳过**）→ 汇总 top10 展示 + 用户确认/改选 → 复制 `new-models/{algo}-v{N}/` + `finalized_model.json` 转正。

### 2.2 断点续跑与阶段控制

```bash
# 断点续跑：跳过已 done 实验
python .../run_experiments.py ... --resume
# 只执行到矩阵完成（不调优不转正）
python .../run_experiments.py ... --until matrix
# 只做调优（跳过已完成矩阵）
python .../run_experiments.py ... --until tune
# 只做转正确认（跳过重跑）
python .../run_experiments.py ... --until promote
# 指定转正实验（跳过 top10 交互选择）
python .../run_experiments.py ... --until promote --promote-id lgb-full-all-v1
```

## 3. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--session-dir` | ✅ | - | session 根目录；实验产物落 `<session_dir>/experiments/` |
| `--sample` | ✅ | - | 样本 parquet 路径 |
| `--feature-list` | ✅ | - | 特征清单 csv 路径 |
| `--config` | 否 | - | 含 `model.split` 的 yaml（`feature_config.yaml`/`train_config.yaml`），与 `--split-*` 二选一 |
| `--split-train/--split-test/--split-oot` | 否 | - | 直接给三档区间（`["2026-01-01","2026-06-30"]` 或 `20260101,20260630`），与 `--config` 二选一 |
| `--label-col` | 否 | 从 config 读 | 标签列；config 缺 label_col 且未传时从 parquet 推断（列名含 label 者） |
| `--id-col` | 否 | `fuid` | id 列（安全过滤排除） |
| `--dt-col` | 否 | `f_p_date` | 日期列（样本方案 / 时间加权用） |
| `--algos` | 否 | 交互询问 | 实验算法（lgb/xgb/两者都选），串行执行；缺省时规划前交互询问；`--resume` 断点续跑时沿用已存矩阵算法 |
| `--no-sample-select` | 否 | `False` | 不做样本选择：仅全量 full 方案（跳过 recent-N / 时间加权） |
| `--no-feat-select` | 否 | `False` | 不做特征选择：仅 all 方案（跳过 importance / iv-psi 格） |
| `--no-adversarial` | 否 | `False` | 不做对抗验证：跳过波3 对抗格 |
| `--no-tune` | 否 | `False` | 不做 Optuna 调优：winner 规则诊断一并跳过，仅出 leaderboard |
| `--max-experiments-per-algo` | 否 | `12` | 单算法实验格数上限（含 -opt；超出报错提示收敛方案组数） |
| `--n-trials` | 否 | `25` | Optuna 调优 trials（默认 25） |
| `--auto-apply` | 否 | `False` | 跳过所有交互确认（对抗剔除幅度 / 转正选择用默认推荐） |
| `--resume` | 否 | `False` | 断点续跑：跳过 status=done 的实验 |
| `--until` | 否 | `promote` | 执行到阶段：`matrix`（矩阵全格）/ `tune`（+调优）/ `promote`（+转正） |
| `--promote-id` | 否 | - | 指定转正实验 id（跳过 top10 交互选择） |
| `--force-tune` | 否 | `False` | v2.3: winner 规则诊断为 well_fit 时仍强制 Optuna 调优（默认跳过） |

## 4. 输出产物

```text
<session_dir>/experiments/
├── matrix-plan.md                    # 矩阵规划：方案组数自决理由 + 全实验清单 + 断点状态
├── leaderboard.md                    # 全实验 OOT AUC 排序总表（含乐观偏差标注 + 诊断/调优列 + 诊断与调优明细小节 + 失败清单）
├── leaderboard.xlsx                  # 同上（缺 openpyxl 时仅 md；含 诊断/调优 两列）
├── {algo}-{sample_scheme}-{feat_scheme}-v{N}/    # 每格实验（例：lgb-full-all-v1）
│   ├── manifest.json                 # 全超参/方案/seed/依赖源/code_sha256/template_version/code_modified/status
│   ├── model/model.pkl + model_meta.json
│   ├── evaluation/eval.{json,md}     # train/val/oot/all 四档
│   ├── feature_importance.csv        # feature,total_gain,split_count,gain_pct
│   ├── logs/run.log
│   ├── scripts/train.py              # 训练代码快照（train_template.py 副本 + code_sha256）
│   └── data/                         # Optuna 依赖（轻量，无 train/val/oot.parquet）：
                                      # features.json + params.json + weights.json
                                      # （v2.6.1 起切分数据不落盘，调优时运行时重切）
└── {algo}-...-v{N}-opt/              # Optuna 调优格（每算法 winner）
```

转正产物（保持 model-scoring 消费契约）：

```text
<session_dir>/new-models/{algo}-v{N}/model/model.pkl + model_meta.json + config.json
<session_dir>/finalized_model.json
```

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `data-cleaning` | 提供 `sample.parquet` + `feature-list.csv` |
| 上游 | `classification-model-task-spec` / 任意含 `model.split` 的 yaml | 提供三档切分区间（切分唯一真相 = `model.split`） |
| 编排 | `classification-model-development` | **v2.3 主链路 Stage 3 默认调度本模块**（`run_experiments.py --until promote`）；本模块内部对抗幅度/top10 转正仍逐点确认 |
| 下游 | `model-scoring` | 转正后 `finalized_model.json` 无感消费（`mark_finalized.py` / `score_data.py` 支持 joblib 加载 `model.pkl` 的 lgb/xgb） |
| 依赖 | `_modelevo-shared` | 仅复用共享层：`metrics.py`（AUC/KS/Gini/PSI/IV/分桶）、`config_io.check_sensitive`、`date_utils`；**禁止 import 其他 skill 脚本** |

> 规则诊断（`diagnose_winner` / `recommend_winner`）逻辑移植自原 `classification-model-tuning` 的
> `diagnose.py` / `recommend_params.py`（仅保留 lgb/xgb 策略，lgb 为新增等价策略表）；
> 原 tuning 模块已于 v2.7 移除，本模块为规则诊断的唯一承载。

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 算法无关 + sklearn 兼容 | 除 `hyperparams.py` / `algo_factory.py` 外全部算法无关；统一 `fit(X,y,sample_weight)` / `predict_proba` / `feature_importances_` 接口；未来加 dnn/lr 只扩展 factory |
| ⚠️ 规则诊断（v2.3） | `diagnose_winner.py` / `recommend_winner.py` 仅支持 lgb/xgb；诊断输入全部来自实验格自身产物（eval.json + manifest.json + model_meta.json），禁跨 skill import；五状态优先级 overfit > underfit > underconverged > unstable_psi > well_fit；阈值对齐 tuning 源（gap 0.05 / 0.005、PSI 0.10、收敛比 0.95） |
| ⚠️ 严格正交 | importance 特征方案 = 取**同样本方案 all 格**的 total_gain 累积 95% 截断；依赖为同算法内 sample_scheme 维度的 DAG |
| ⚠️ 开发池与切分 | 开发池 = train + test 合并；每格施加样本方案后 seed=42 随机切 70%/30%（训练/val，分层随机）；OOT 纯榜单 |
| ⚠️ 超参公式 | 每格按自身 M/S 独立计算：`n_estimators=1000`、100 轮早停、`learning_rate=0.04`、`num_leaves=min(31, 2^(S/10))`、lgb `max_depth=-1` + `min_child_samples=max(20,0.002*M)` + `min_sum_hessian_in_leaf=1e-3`、xgb `max_depth=4` + `min_child_weight=1e-3`、共用 `subsample/bagging_fraction=0.6`、`colsample_bytree/feature_fraction=max((num_leaves*2)/S,0.5)`、`scale_pos_weight=neg/pos` 自动、`seed=42` |
| ⚠️ 锚点治理（v2.5 修复） | Optuna 搜索锚点 = 样本量公式**先 clip 进经验域**（lgb `min_child_samples` clip 到 `[20,200]`）再生成邻域，防 0.002×M 在 26 万样本下算出 520 这类与经验域脱节的值；overfit 收窄对 `min_child_samples`/`min_child_weight` 类跨数量级参数改**相对锚点**（lgb 下界×1.5 / xgb ×10），容量/正则/采样参数保留绝对收窄；`hyperparams.validate_anchors` 在 optuna_anchors / adjust_optuna_anchors 出口 fail-fast 校验 `low<=high`（非法即 RuntimeError 启动暴露） |
| ⚠️ tune 幂等（v2.5 修复） | 每算法最多一个 `-opt` 格：run_experiments tune 段排除已调优 winner（`is_tuned`）+ 已有 done 的 `-opt` 格跳过；leaderboard 标注 `/tuned`；tune_winner 入口防御（`spec.is_tuned` 直接返回 None，三保险防任何调用路径重复调优） |
| ⚠️ 训练代码快照 | 权威模板 `scripts/templates/train_template.py`（sklearn 风格、高参数化），每格快照复制进 `scripts/train.py` + 记 `code_sha256 + template_version`；默认全格同代码可比，可逐格 fork（记 `code_modified=true`）；复现 = 重跑实验目录代码 |
| ⚠️ 失败容错 | 单格失败 → 跳过 + 记录原因 + 继续其余；汇总报告含失败清单；重跑跳过已完成实验 |
| ⚠️ 数据安全红线 | 复用 `config_io.check_sensitive`；不透出身份证 / 手机号明文 |

## 7. 异常处理

| 异常 | 处理方式 |
|---|---|
| `--config` 与 `--split-*` 均缺 / 三档不全 | 报错退出，提示提供 `model.split` |
| sample/feature-list 文件不存在 | 报错退出 |
| 交互输入非法 / EOF（无 tty） | 回退默认值并提示；`--auto-apply` 跳过全部交互用默认 |
| 无 OOT 且未显式关闭对抗 | 对抗验证自动置 False 并记录理由（train-vs-oot 无样本可训） |
| 开发池月份数 < 2 且未显式关闭样本选择 | 样本选择自动置 False 并记录理由（无 recent-N/时间加权衍生方案） |
| 开发池正负样本任一侧为 0 | 该格跳过并记录 `fail_reason=no_positive_or_negative` |
| 某格训练异常 | 捕获 → `status=failed` + `fail_reason` → 继续其余格 |
| Optuna 未安装 | 清晰报错，跳过调优并记入失败清单；相关测试 `@pytest.mark.skipif` |
| 单算法实验格数超 `--max-experiments-per-algo` | 报错，提示收敛样本/特征方案组数 |
| 对抗分类器无法训练（如 OOT 样本不足） | 跳过对抗格并记录原因 |

## 8. 测试

```bash
python -m pytest <skill_dir>/tests/ -q                    # 全量
python -m pytest <skill_dir>/tests/ -q -m "not slow"      # 仅快测（不训练）
```

新增（v2.3）：`test_diagnose_winner.py` — 五状态判定 / lgb+xgb 策略表 / Optuna 锚点调整 / well_fit 跳过调优。

## 9. 红线例外声明（本模块特有，用户已授权）

| # | 例外 | 授权范围 | 仍禁止 |
|---|---|---|---|
| ① | 对抗格 | OOT 可参与对抗分类器训练（train vs oot）与样本/特征筛选统计 | OOT 作早停集 / 进训练集 / 结构超参选择 |
| ② | IV-PSI 格 | OOT 参与 PSI 统计（train vs oot 分布漂移） | 同上 |

两类实验的 OOT AUC 存在乐观偏差，leaderboard 以 `⚠ 乐观偏差` 标注。其余所有实验 OOT 仅作评估榜（纯榜单，不参与任何训练/统计）。

---

数据来源：`sample.parquet`（清洗后）+ `feature-list.csv`（派生特征）+ `model.split`（切分唯一真相）。完整方法论文档见 `references/constraints-and-exceptions.md`。
