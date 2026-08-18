# classification-model-experiments — 约束与例外（全表）

本文档是 SKILL.md §6/§7/§9 的展开版：红线例外声明、Optuna 搜索空间、异常处理全表、方案自决规则。

---

## 1. 红线例外声明（仅本模块，用户已授权）

全局 OOT 红线（agent 文档）：OOT 仅可参与实验比较 / 方向指引；**禁止**参与早停、
特征工程统计（插补/分箱/归一化）、训练集、结构超参选择。

本模块经用户明确授权存在 **2 处例外**（plan §5.2.8）：

| # | 例外格 | OOT 参与范围 | 仍禁止（绝对） |
|---|---|---|---|
| ① | 对抗格 `-adversarial-adversarial-v1` | OOT 作为对抗分类器的正样本（train vs OOT）训练；OOT 的对抗概率参与样本剔除；OOT 特征判别分数（对抗 importance）参与特征剔除 | OOT 作早停集 / 进最终训练集 / 结构超参选择 |
| ② | IV-PSI 格 `*-iv-psi-v1` | OOT 参与每特征 PSI 统计（dev→oot 分布漂移） | 同上 |

**后果声明**：① ② 两格训练的模型未见过 OOT 的标签，但其**特征选择/样本选择过程消费了 OOT 信息**，
OOT AUC 存在乐观偏差。leaderboard 对这两类格以 `⚠ 乐观偏差` 标注，转正时向用户显著提示。

## 2. 方案组数自决规则（plan §2.1 C3）

样本方案（由 `sample_schemes.decide_sample_schemes` 决定，理由落 matrix-plan.md）：

| 条件 | 方案 | 说明 |
|---|---|---|
| 恒有 | `full` | 全量开发池；`lgb-full-all-v1` 兼任全局 baseline |
| 开发池月份数 >= 2 | `recent{N}` | N = max(3, 月份数//2)，最近 N 月窗口 |
| 开发池月份数 >= 3 | `timeweight` | 线性时间衰减加权（最近月 1.0 → 最远月 0.2） |
| 月份数 < 2 | 仅 `full` | 日期信息不足以支撑窗口/加权方案，跳过并在理由中记录 |

特征方案（每组样本方案正交）：

| 方案 | 依赖 | 说明 |
|---|---|---|
| `all` | 无 | 安全过滤后全量特征 |
| `importance` | 同 (algo, sample_scheme) 的 `all` 格（DAG） | 依赖格 `feature_importance.csv` total_gain 累积 95% 截断（严格正交） |
| `iv-psi` | 无 | 单格直算：缺失 > 0.95 / IV < 0.015 / PSI(dev→oot) > 0.2（阈值放松，plan §2.1 C6） |
| `adversarial` | 对抗格自身 | 对抗 total_gain top-K 剔除特征（K = max(3, 15% 特征数)），与样本剔除合并应用 |

单算法实验数默认上限 12（`--max-experiments-per-algo`，含 -opt）。

## 3. 推荐超参公式（每格按自身 M/S 独立推导，plan §5.2.6）

- `n_estimators=1000`、100 轮早停（train 拟合、**val 早停**）、`learning_rate=0.04`
- `seed=42`（全部实验统一）
- lgb：`num_leaves=min(31, 2^(S/10))`、`max_depth=-1`、
  `min_child_samples=max(20, 0.002*M)`、`min_sum_hessian_in_leaf=1e-3`
- xgb：`max_depth=4`、`min_child_weight=1e-3`
- 共用：`subsample/bagging_fraction=0.6`、
  `colsample_bytree/feature_fraction=max((num_leaves*2)/S, 0.5)`、
  `scale_pos_weight=neg/pos` 自动（训练时按训练段计算）

M = 施加样本方案后切分出的训练段样本数，S = 安全过滤+特征方案后的特征维度。

## 4. Optuna 邻域搜索空间（plan §5.2.9）

以 winner 格 M/S 推导超参为锚点收窄；TPE seed=42；目标 **val AUC**；100 轮早停；默认 25 trials。

| 超参 | lgb | xgb |
|---|---|---|
| `learning_rate` | 锚点 × [0.5, 1.5] | 锚点 × [0.5, 1.5] |
| `num_leaves` | [锚点-8, 锚点+8]（下限 8） | - |
| `min_child_samples` | [锚点×0.6, 锚点×1.4]（下限 5） | - |
| `max_depth` | - | [3, 6] |
| `min_child_weight` | - | [1e-4, 1e-2]（log） |
| `feature_fraction` / `colsample_bytree` | [0.5, 0.9] | [0.5, 0.9] |
| `bagging_fraction` / `subsample` | [0.5, 0.9] | [0.5, 0.9] |

-opt run 复用 winner 格 `data/` 快照（同一基线可比），产物规范与单格一致。

## 5. 异常处理全表（plan §2.1 F1）

| 条件 | 处理 |
|---|---|
| `--config` 与 `--split-*` 均缺 / 三档不全 | 报错退出 |
| 依赖格（importance 引用的 all 格）未完成 | 该格 `failed`，fail_reason=依赖未完成 |
| 样本/特征方案过滤后为空 | 该格 `failed`，跳过继续 |
| 训练/评估/落盘异常 | 捕获 → `failed` + fail_reason → 继续其余格 |
| Optuna 未安装 | 跳过 -opt，打印 `pip install --user "optuna<4"`；测试 skipif |
| 单算法格数超上限 | 报错，提示收敛方案组数 |
| 对抗分类器训练失败（OOT 样本不足等） | 对抗格 `failed`，继续 |
| 转正候选为空 | 提示无 done 实验，返回非 0 |
| `new-models/{algo}-v{N}` 自增冲突 | 自动扫描 max+1 |

## 6. 断点续跑（plan §2.1 F2）

- `matrix-plan.json` 内嵌 specs（含 status），`--resume` 跳过 `status=done` 且有 manifest 的格。
- 每格完成即 `save_state`，中断后可续。
- `--until matrix|tune|promote` 控制执行到哪个阶段（跳过已完成阶段不重跑）。

## 7. 算法无关与扩展约定（plan §2.2 修改 1）

- 仅 `hyperparams.py` / `algo_factory.py` 与算法相关；其余模块（plan/sample/feature/leaderboard/
  promote/adversarial/evaluate/safety_filter）一律算法无关。
- 未来新增 dnn/lr：在 `algo_factory.build_estimator` + `hyperparams.derive_params` + 模板
  `feature_importances` 扩展，其余不动。
- **禁止跨 skill import**：不 import training 的 eval_single/boundary_filter，不 import tuning 任何脚本。
- 仅复用 `_modelevo-shared`（metrics / config_io / date_utils / gen_feature_list），经 `_bootstrap.py` 注入。

## 8. 训练代码快照与可复现（plan §2.2 修改 5）

- 权威模板 `scripts/templates/train_template.py`（**自包含**：不依赖本 skill 其他模块）。
- 每格开始时复制进 `<exp_dir>/scripts/train.py`，manifest 记 `code_sha256` + `template_version`。
- 默认全格同代码；AI 定制某格 → 逐格 fork `scripts/train.py`，manifest 记 `code_modified=true`。
- 复现 = 重跑实验目录代码（快照目录自带 `data/` 输入快照，全录入 `train/val/oot.parquet` +
  `features.json` + `params.json` + `weights.csv`）。