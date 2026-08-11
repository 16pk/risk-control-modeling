---
name: score-to-fico
description: 风控模型分 → FICO 标准分转换。对已训练好、能输出违约概率(prob)的分类模型，做 LR 校准 + 标准分映射，产出 FICO 量纲标准分（范围约 [400,780]，分高险低）。两种入口：① 嵌入分类建模 pipeline（classification-model-development Stage 5 收口后自动调起，消费 new-models/{run}/predictions/*.parquet，产 {run}/fico/）；② 独立调用（输入含概率分列+标签列的 CSV/parquet，输出校准参数 coef.json + 打分 + 拟合方案 fitting-summary）。当用户说"转fico分""概率分转标准分""校准概率""转成fico""概率校准"或收口后要 FICO 分时使用。
---

# 概率分 → FICO 标准分转换（score-to-fico）

> 本 skill 只做 **校准 + 转分** 一步，消费上游训练产物或独立样本，产出 FICO 量纲标准分。
> 前置：模型已训练好，能输出违约概率（XGBoost / LightGBM / LR / DNN 均可）。

## 一、两步转换原理

**Step 1 — LR 校准**
模型原始概率直接转分往往量级漂移（不同模型概率分布不同）。先把概率映射到 `odds`，再用 logistic 回归对 `odds → 真实逾期` 校准：

```
odds           = ln(p / (1 - p))                        # 取对数 odds（概率边界裁剪 1e-6 防 log(0)）
lr             = LogisticRegression(C=20).fit(odds, y)  # 拟合集为 train（仅用 y∈{0,1}，剔除标签缺失）
logistic_prob  = sigmoid(coef * odds + intc)            # 校准后的概率
```

**Step 2 — 标准分映射**
把校准后概率转成 FICO 量纲标准分：

```
bscore = 400 - 35/ln(2) * ln(logistic_prob / (1 - logistic_prob))
```

- `bscore` 范围约 **[400, 780]**，**分越高风险越低**（与 FICO 习惯一致）
- `coef` / `intc` 是校准关键参数，必须随模型一起保存、透传，生产跨批次用 `--apply` 复用保证口径一致

## 二、两种入口

### 入口 1：嵌入分类建模 pipeline（development Stage 5 调起）

在 `classification-model-development` 收口（Stage 4 上线候选确认）之后，由编排**总是询问**用户是否对 top1 上线候选 run 做 FICO 转换；用户确认后调起本 skill：

```bash
python <skill_dir>/scripts/score_to_fico.py --from-run --run-dir <session_dir>/new-models/{run}
```

消费 `new-models/{run}/predictions/{train,test,oot}_predictions.parquet`（schema: `[id_cols..., label, score, bucket]`，`score` 即违约概率），产出 `new-models/{run}/fico/`。

### 入口 2：独立调用（用户主动发起）

输入含**概率分列 + 标签列**的样本 → 拟合 + 转分 + 拟合方案；或复用已有校准参数仅转分：

```bash
# 拟合 + 转分（带标签训练集）
python <skill_dir>/scripts/score_to_fico.py \
  --fit --data train.csv --prob_col pred_proba --label_col y \
  --uid_col fuid --date_col f_p_date \
  --out result_score.parquet --coef_out coef.json \
  --summary_out fitting-summary.json --summary_md fitting-summary.md

# 仅转分（复用已保存 coef/intc，无需标签；生产批量打分 / 离线回溯）
python <skill_dir>/scripts/score_to_fico.py \
  --apply --data new_sample.csv --prob_col pred_proba \
  --uid_col fuid --date_col f_p_date --coef coef.json --out result_score.parquet
```

## 三、参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--fit` / `--apply` / `--from-run` | ✅ 三选一 | - | 模式开关 |
| `--data` | fit/apply 必填 | - | 输入 CSV / parquet（含概率列；fit 还需标签列） |
| `--prob_col` | 否 | fit/apply: `pred_proba`；from-run: `score` | 概率列名 |
| `--label_col` | fit 必填 | from-run: `label` | 标签列名（`y∈{0,1}`，其余剔除） |
| `--uid_col` | 否 | `fuid` | 用户ID列名（透传） |
| `--date_col` | 否 | `f_p_date` | 日期分区列名（透传） |
| `--out` | 否 | `result_score.parquet` | 打分输出路径（按扩展名自动 CSV/parquet） |
| `--coef_out` / `--coef` | 否 | `coef.json` | 校准参数写出 / 读入 |
| `--summary_out` / `--summary_md` | 否 | `fitting-summary.{json,md}` | 拟合方案输出（fit 模式自动生成） |
| `--run-dir` | from-run 必填 | - | run 目录，如 `new-models/lgb-v1` |
| `--fico-dir` | 否 | `<run-dir>/fico` | from-run 输出目录 |

## 四、输出产物

### from-run（pipeline 嵌入）：`new-models/{run}/fico/`

```text
new-models/{run}/fico/
├── coef.json                        # {"coef": ..., "intc": ...} 校准参数（生产 --apply 复用）
├── fico_train_predictions.parquet   # train 转分: id_cols + label + score + bucket + odds + logistic_prob + bscore
├── fico_test_predictions.parquet    # test 转分（评估用，未参与拟合）
├── fico_oot_predictions.parquet     # oot 转分（OOT 稳定性/上线裁决用）
└── fitting-summary.{json,md}        # 拟合方案: coef/intc + 三档 bscore 范围/均值 + 分位表
```

### 独立调用（fit/apply）

- `coef.json`：校准参数
- `result_score.{csv,parquet}`：`[uid, date, prob, odds, logistic_prob, bscore]`
- `fitting-summary.{json,md}`：拟合方案（校准参数 / 拟合样本量 / bscore 范围与分位表）

## 五、与其他 skill 的关系

| skill / 模块 | 关系 | 说明 |
|---|---|---|
| `classification-model-development` | **编排调起（Stage 5）** | 收口后总是询问用户是否对 top1 上线候选转换（用户确认的决策） |
| `classification-model-training` | 上游 | 产 `predictions/{train,test,oot}_predictions.parquet`（本 skill from-run 的输入，`score` 列即违约概率） |
| `classification-model-evaluation` | 评估口径 | 转分后可用 `bscore` 替换 `score` 重跑评估（分桶/lift 口径随之变为 FICO 分） |
| `classification-model-report` / `credit-model-report` | 下游可选 | FICO 产物独立落盘；报告纳入 fico 为后续版本规划（当前不消费） |
| `model-knowledge` | 归档 | `coef.json` + fitting-summary 随 run 归档，台账记录模型已校准参数 |

## 六、执行约束

| 约束 | 说明 |
|---|---|
| 校准纪律 | **校准参数只在 train 拟合，严禁用 OOT/test 拟合**（否则高估泛化）；test/oot 仅 `--apply` 转分 |
| 标签过滤 | 拟合时仅用 `y ∈ {0,1}`，剔除未成熟样本（OOT 表现期未到导致标签缺失） |
| 概率边界 | 概率裁剪至 `[1e-6, 1-1e-6]` 防 `log(0)` |
| bscore 越界 | 正常区间约 `[400, 780]`；显著越界 → `[WARN]` 提示 `coef/intc` 不适用当前分布，需重新 `--fit` |
| `coef<=0` | 概率与真实逾期方向相反或量级异常 → `[WARN]` 提示检查概率列是否为违约概率 |
| 数据安全红线 | 透传 id 列（用户ID）合规；不写入身份证（`\d{17}[\dxX]`）/ 手机号（`1[3-9]\d{9}`）明文 |
| 依赖 | `pandas / numpy / scikit-learn`；`pyarrow` 读 parquet。默认 `python` 无依赖时用 `miniforge3` / 项目环境 |
| 不动上游 | 只读 `predictions/`，唯一写目录是 `fico/`（from-run）或用户指定的 `--out`（独立调用） |

## 七、异常处理

| 异常 | 处理方式 |
|---|---|
| `predictions/` 目录或三档文件缺失 | 立即退出，提示先跑完 training 的 predictions 阶段 |
| 预测文件无 `score`/`label` 列 | 提示用 `--prob-col` / `--label-col` 覆盖 |
| 拟合集无 `y∈{0,1}` 样本 | 报错退出，提示标签列取值异常 |
| `--apply` 的 `coef.json` 不存在 | 报错退出，提示先 `--fit` |
| 模式参数多选/漏选 | 报错提示必须三选一 |
