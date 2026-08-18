---
name: score-to-fico
description: 风控概率分 → FICO 标准分转换。对定版模型打分结果（含违约概率 score + 标签列）做 LR 校准 + 标准分映射，产出 FICO 量纲标准分（范围约 [400,780]，分高险低）。**仅用户主动触发**（收口后不再默认询问），消费 model-scoring 的打分结果，默认用全量样本 + Y 标签拟合校准参数。当用户明确要求"转 FICO 分 / 要 FICO 分"时由编排层调起。
---

# 概率分 → FICO 标准分转换（score-to-fico）

> 本 skill 只做 **校准 + 转分** 一步，消费上游 `model-scoring` 的打分结果，产出 FICO 量纲标准分。
> 前置：`model-scoring` 已用定版模型产出违约概率分 `score`。

## 1. 触发定位与职责边界

| 项 | 说明 |
|---|---|
| 触发 | **仅用户主动触发**（明确说"转 FICO 分"），由 `classification-model-development` 编排调起；收口后**不再默认询问**（FICO 是业务展示层格式，非训练二分类模型的核心目标） |
| 输入 | `model-scoring` 打分结果（`<session_dir>/scoring/score_sample.parquet`，含 `score` 概率列 + 透传的 `label` 列） |
| 输出 | FICO 标准分产物，落 `<session_dir>/fico/` |
| 不做 | 不再提供独立 `--fit` / `--apply` / `--from-run` 入口；不重新对原始数据推理（推理由 model-scoring 完成） |

## 2. 两步转换原理

**Step 1 — LR 校准**
模型原始概率直接转分往往量级漂移（不同模型概率分布不同）。先把概率映射到 `odds`，再用 logistic 回归对 `odds → 真实逾期` 校准：

```
odds           = ln(p / (1 - p))                        # 取对数 odds（概率边界裁剪 1e-6 防 log(0)）
lr             = LogisticRegression(C=20).fit(odds, y)  # 拟合集默认全量样本（仅用 y∈{0,1}，剔除标签缺失）
logistic_prob  = sigmoid(coef * odds + intc)            # 校准后的概率
```

**Step 2 — 标准分映射**
把校准后概率转成 FICO 量纲标准分：

```
bscore = 400 - 35/ln(2) * ln(logistic_prob / (1 - logistic_prob))
```

- `bscore` 范围约 **[400, 780]**，**分越高风险越低**（与 FICO 习惯一致）
- `coef` / `intc` 是校准关键参数，随模型一起保存，保证跨批次口径一致

## 3. 执行命令

```bash
python <skill_dir>/scripts/score_to_fico.py \
    --data <session_dir>/scoring/score_sample.parquet \
    --out-dir <session_dir>/fico \
    [--prob-col score] [--label-col label] \
    [--fit-label-col label] [--fit-date-range 20260101,20261231] \
    [--date-col f_p_date] [--uid-col fuid]
```

- **拟合参数确认**：默认用全量样本 + `label` 列拟合校准。编排层在执行前用 `AskUserQuestion` 询问用户，允许修改**参与拟合的样本时间范围**（`--fit-date-range`）与**拟合标签**（`--fit-label-col`）；脚本本身不做 `input()` 交互，只接收已确认的参数。

## 4. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--data` | ✅ | - | model-scoring 打分结果 parquet（含 `score` 概率列 + `label` 标签列） |
| `--out-dir` | ✅ | - | FICO 产物输出目录（建议 `<session_dir>/fico`） |
| `--prob-col` | 否 | `score` | 概率列名 |
| `--label-col` | 否 | `label` | 标签列名（透传/输出用） |
| `--fit-label-col` | 否 | 同 `--label-col` | 拟合用标签列名（`y∈{0,1}`，其余剔除） |
| `--fit-date-range` | 否 | 全量 | 拟合样本时间范围 `start,end`（YYYY-MM-DD / YYYYMMDD） |
| `--date-col` | 否 | `f_p_date` | 日期分区列名（用于 `--fit-date-range` 过滤） |
| `--uid-col` | 否 | `fuid` | 用户ID列名（透传） |

## 5. 输出产物

产物落 `<session_dir>/fico/`（与 `scoring/` 平级）：

```text
<session_dir>/fico/
├── coef.json                   # {"coef": ..., "intc": ...} 校准参数（生产复用）
├── fico_predictions.parquet    # 转分结果: 全部输入列 + odds + logistic_prob + bscore
└── fitting-summary.{json,md}   # 拟合方案: coef/intc + bscore 范围/均值 + 分位表
```

## 6. 与其他 skill 的关系

| skill / 模块 | 关系 | 说明 |
|---|---|---|
| `classification-model-development` | **编排调起（Stage 6，可选）** | 收口后仅用户主动要求时转 FICO |
| `model-scoring` | 上游（Stage 5） | 产 `scoring/score_sample.parquet`（本 skill 输入，含 `score` + `label`） |
| `classification-model-training`（内嵌评估） | 评估口径 | 转分后可用 `bscore` 替换 `score` 重跑评估 |
| `credit-model-report` | 下游可选 | FICO 产物独立落盘；报告纳入 fico 为后续版本规划（当前不消费） |
| `model-knowledge` | 归档 | `coef.json` + fitting-summary 随 session 归档，台账记录模型已校准参数 |

## 7. 执行约束

| 约束 | 说明 |
|---|---|
| 校准纪律 | 校准参数只用拟合集拟合；拟合集默认全量样本，可经用户确认改为指定时间范围 |
| 标签过滤 | 拟合时仅用 `y ∈ {0,1}`，剔除未成熟样本（OOT 表现期未到导致标签缺失） |
| 概率边界 | 概率裁剪至 `[1e-6, 1-1e-6]` 防 `log(0)` |
| bscore 越界 | 正常区间约 `[400, 780]`；显著越界 → `[WARN]` 提示 `coef/intc` 不适用当前分布 |
| `coef<=0` | 概率与真实逾期方向相反或量级异常 → `[WARN]` 提示检查概率列是否为违约概率 |
| 数据安全红线 | 透传 id 列（用户ID）合规；不写入身份证（`\d{17}[\dxX]`）/ 手机号（`1[3-9]\d{9}`）明文 |
| 依赖 | `pandas / numpy / scikit-learn`；`pyarrow` 读 parquet |
| 不动上游 | 只读 `scoring/`；唯一写目录是 `fico/`（`--out-dir`） |

## 8. 异常处理

| 异常 | 处理方式 |
|---|---|
| `--data` 缺失概率列 / 标签列 | 报错退出，提示用 `--prob-col` / `--label-col` 覆盖 |
| 拟合集无 `y∈{0,1}` 样本 | 报错退出，提示标签列取值异常 |
| `--fit-date-range` 格式非法 / 未命中样本 | 报错退出，提示检查日期格式与范围 |
| 数据缺 `--date-col` 但传了 `--fit-date-range` | 报错退出，提示用 `--date-col` 覆盖 |
