---
name: feature-classification
description: 建模前探的特征列识别与复用 skill。在数据探查/清洗阶段对样本列做语义三分类（feature / non_feature / ambiguous，规则库 v0：日期/时间戳/订单号/ID/序号/纯标识列/标签列(用户红线) → non_feature 候选；匿名编码列/疑似标签列 → ambiguous 默认保留），通配符分组展示 + 用户批量确认（非特征剔除须用户确认，不自动剔），产出权威 feature-list.csv 供 data-cleaning / credit-data-analysis / classification-model-experiments 全 pipeline 复用。跨 session 档案复用，仅列集合变化时增量重分类。触发场景：建模前确认特征列、特征清单确认、排除日期/订单号/标签等非特征列；以及「只清洗 / 只分析」独立任务（经 development 轻量入口 prep_sample.py clean/analyze 自动前置本 skill）。
---

# feature-classification（特征列识别与复用）

建模链路的第一道「特征语义识别」工序（v2.3 起在 task-spec 与 data-cleaning 之间编排）：把特征列判定从隐式排除法（所有非 id/dt/label 列一律视为特征）**前移并显式化**，探查阶段完成语义三分类 + 用户批量确认，产出**权威 `feature-list.csv`** 供全 pipeline 复用。

> ⚠️ **触发定位**：由 `classification-model-development` 编排自动调起（主链路 Stage 1 前，或轻量入口 `prep_sample.py clean/analyze` 的独立任务场景），**不设独立触发词**；也可由用户主动要求（"确认特征列 / 特征清单"）时触发。**「只清洗 / 只分析」独立任务必须先经本 skill 产出权威 `feature-list.csv` 再进入清洗/分析**（轻量入口已串联，禁止跳过本环节直接清洗/分析原始数据）。

## 1. 核心原则（方案 §二）

1. **误剔真特征的代价 > 误留非特征**：规则只做"候选标记"，不自动剔除；非特征有特征分析（IV/PSI）环节兜底，真特征被剔则信息永久丢失。
2. **批量确认，不做逐列确认**：通配符分组展示，组级确认 + 可展开明细。
3. **展示层通配符，存储层逐列**：交互用 `pfx_*` 折叠，档案永远逐列精确记录，保证可复现、可审计、跨 session 复用。
4. **清单唯一真相**：确认后的 `feature-list.csv` 成为唯一权威，清洗/分析/训练全部消费同一份，不各自派生。
5. **不自动剔除 ambiguous**：`ambiguous` 默认保留，仅报数量；规则判定的 `non_feature` 也必须经用户批量确认后才剔除。

## 2. 流程

```
探查扫描(缺失率/dtype/唯一值)
   → 列语义三分类: feature / non_feature / ambiguous (rules.py classify_column)
   → 通配符分组 + 用户批量确认(exclude 剔除 / keep 保留)
   → 落档 feature-classification.json(逐列+判定人) + feature-list.csv(权威)
   → data-cleaning 经 --feature-list-source 消费(现有能力, 零改动)
   → credit-data-analysis / classification-model-experiments 复用同一清单
```

## 3. 语义规则与三分类（规则库 v0）

| 类别 | 判定方式 | 处置 |
|---|---|---|
| `feature` | 默认类别（含匿名编码列中的业务词列） | 保留，进入特征清单 |
| `non_feature` | 规则命中：日期/时间戳/订单号/ID/序号/纯标识列/**标签列（用户红线 `fpd*`,`dpd*`）** | 列出候选，用户批量确认后才剔除 |
| `ambiguous` | 列名无语义信息（`i_30`、`m_1221` 等匿名编码列）或疑似标签列 | 默认保留，仅报数量 |

优先级：**用户红线 > 用户自定义规则 > 默认规则 > 启发式**。红线必须置顶（先于其他所有规则），否则会被更具体的模式抢先匹配而失效。

内置默认规则（`rules.py` 可配置扩展）：
- non_feature 模式：`(^|_)date$`（日期列）、`(^|_)time$|(_|^)time_`（时间戳列）、`order_?id$|order_?id_`（订单号列）、`(^|_)(id|uid)$`（ID列）、`(^|_)(rn|seq|no)$`（序号列）、`^f_p_`（分区日期列）
- 标识前缀 `if_/is_/has_/flag_`：数值且唯一值 ⊆ {0,1} → 纯标识列 non_feature；否则 ambiguous 值域待确认
- 匿名编码列 `^([a-zA-Z]+)_(\d+)$` → ambiguous（可能误伤 `score_80002` 等评分列——归 ambiguous 而非剔除，安全）
- `fst_rn/last_rn` → ambiguous（可能是排名特征，须用户确认）

**规则局限（诚实记录）**：
- `score_80002` 等匿名列可能为评分列，默认保留（ambiguous）
- 含标签语义 token 但前缀不同（如 `v12_fpd7_v2`）不在红线路面上，默认保留，提示用户确认
- 规则库支持 `--label-prefixes` / `--extra-patterns` 用户自定义扩展；用户清单可覆盖规则判定

## 4. 执行命令

`<skill_dir>` 指本 skill 所在目录，执行时替换为实际绝对路径，**不要复制脚本到 cwd**。

### 4.1 探查扫描（classify_features.py）

```bash
python <skill_dir>/scripts/classify_features.py \
    --input <本地样本文件(.parquet/.csv/.feather/.xlsx/.json)> \
    --out-dir <session_dir>/sample-features \
    --id-col fuid \
    --dt-col f_p_date \
    --label-col label \
    [--label-prefixes fpd,dpd] \
    [--extra-patterns "pat1,pat2"] \
    [--min-group 2]
```

- `--label-prefixes`：标签列红线前缀（默认 `fpd,dpd`），命中即 non_feature 候选（置顶规则）
- `--extra-patterns`：追加 non_feature 正则（用户自定义规则，优先级高于默认）
- 产物：`<session_dir>/sample-features/feature-classification.json`（探查版）+ `_manifest.json`
- 打印分类报告（三分类计数 + 组级折叠 + 混合组展开），编排层据此向用户批量确认

### 4.2 固化权威清单（finalize_feature_list.py）

```bash
python <skill_dir>/scripts/finalize_feature_list.py \
    --classification <session_dir>/sample-features/feature-classification.json \
    --out-dir <session_dir>/sample-features \
    --exclude if_tf,if_ka,fser_date,sx_order_id,jy_order_id,ftrans_time,fst_rn,last_rn \
    [--keep flag_ok,...]
```

- `--exclude`：用户确认剔除的列 → 固化 `decided_by=user` + 类别 non_feature
- `--keep`：用户确认保留的列（如恢复规则误判）→ 固化 `decided_by=user`
- 权威 `feature-list.csv` = 全部非 non_feature 列（逐列，无通配符进入存储层）；`feature-classification.json` 原地固化（`generated_as=final` + `user_confirmed_exclude` + `current_counts`）
- **校验**：剔除/保留名单不在档案中 → 抛错（防手滑）

## 5. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--input` | ✅ | - | 本地样本文件路径 |
| `--out-dir` | ✅ | - | 产物目录（建议 `<session_dir>/sample-features/`） |
| `--id-col` / `--dt-col` / `--label-col` | ✅ | `fuid` / `f_p_date` / `label` | 关键列（不参与分类；脚本校验均存在） |
| `--label-prefixes` | 否 | `fpd,dpd` | 标签列前缀红线（置顶规则，命中即 non_feature 候选） |
| `--extra-patterns` | 否 | 空 | 追加 non_feature 正则（用户自定义规则） |
| `--min-group` | 否 | `2` | 组内列数 ≥ N 折叠为 `pfx_*` |
| `--classification` | ✅(finalize) | - | 探查 json 路径（原地固化） |
| `--exclude` | ✅(finalize) | - | 用户确认剔除列（逗号分隔；缺失列报错） |
| `--keep` | 否(finalize) | 空 | 用户确认保留列（逗号分隔） |

## 6. 输出产物（方案 §六）

```text
<session_dir>/sample-features/
├── feature-classification.json   # 逐列三分类档案(探查版/固化版): {列名 → category → reason → decided_by(rule/user)}
├── feature-list.csv              # 权威特征清单(确认后, 已剔除 non_feature; 全 pipeline 唯一真相)
└── _manifest.json                # 产出清单
```

`feature-classification.json`（final 版结构）：

```json
{
  "schema_version": 1, "generated_as": "final", "rulebook": "v0",
  "id_col": "fuid", "dt_col": "ftrans_date", "label_col": "dpd30_3c",
  "counts": {"feature": 158, "ambiguous": 20, "non_feature": 11},
  "current_counts": {"feature": 155, "ambiguous": 15, "non_feature": 19},
  "user_confirmed_exclude": ["if_tf", "...", "last_rn"],
  "user_confirmed_at": "2026-08-20T11:46+08:00",
  "columns": {
    "fals_d15_cell_nbank_else_orgnum": {"category": "feature", "decided_by": "rule", "dtype": "float64", "null_ratio": 0.231},
    "if_tf": {"category": "non_feature", "decided_by": "user", "reason": "用户确认非特征", "dtype": "int64", "null_ratio": 0.0}
  }
}
```

**复用约定**：下次 session 读档案直接复用，**不重复询问**；仅列集合变化时触发增量重分类（重跑 classify_features 增量扫描）。

## 7. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `classification-model-task-spec` | 提供 id/dt/label 列名（task-spec 3 问已确认） |
| 编排 | `classification-model-development` | **v2.3 起在 Stage 0 与 Stage 1 之间编排本 skill**：探查扫描 → 用户批量确认 → 固化权威清单；确认属关键决策门禁（先方案+理由+备选，等用户确认） |
| 下游 | `data-cleaning` | **零改动**消费：经 `--feature-list-source <session_dir>/sample-features/feature-list.csv` 取交集清洗（哨兵替换天然只作用于数值列） |
| 下游 | `credit-data-analysis` / `classification-model-experiments` | 复用同一权威 `feature-list.csv`，不各自派生 |
| 资产 | `model-knowledge` | 特征清单资产优先复用/回写（本 skill 产物即权威清单源） |
| 依赖 | `_modelevo-shared` | 复用 `gen_feature_list` 清单解析约定 |

## 8. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 不自动剔除 | 任何规则命中仅是候选标记；non_feature 必须经用户批量确认后才剔除（信任边界：真特征被剔则永久丢失） |
| ⚠️ ambiguous 默认保留 | 仅报数量；用户可自行调整（`--keep` / `--exclude` 均可作用于 ambiguous 列） |
| ⚠️ 红线置顶 | `label-prefixes` 先于其他规则匹配，否则会被更具体模式抢先而失效 |
| ⚠️ 关键决策门禁 | 展示分类报告 → 用户**批量确认**（组级）→ 才固化；不做逐列轰炸式询问 |
| ⚠️ 存储层逐列 | 通配符只存在于交互展示，档案与清单永远逐列精确 |
| ⚠️ 数据安全红线 | 复用 `config_io.check_sensitive` 纪律：不透出身份证 / 手机号明文 |
| ⚠️ 架构纪律 | 本 skill 只读探样本 + 落清单档案，不产 splits、不判特征好坏（IV/PSI 有效性由 credit-data-analysis 分析，experiments 不做特征筛选） |

## 9. 异常处理

| 异常 | 处理 |
|---|---|
| 样本文件不存在 / 格式不支持 | 报错并列出支持的格式 |
| id/dt/label 列缺失 | 报错指明缺哪个列 |
| 除 id/dt/label 外无特征列 | 报错，提示数据契约问题 |
| 剔除/保留名单含未知列 | 抛错（防手滑），提示先重跑探查 |
| 全列均为 ambiguous/non_feature 候选 | 正常展示，仍等用户批量确认 |

## 10. 测试

```bash
python -m pytest model-skills/feature-classification/tests/ -q
```

覆盖：三分类语义（日期/订单号/ID/序号/标签红线/匿名列/标识列）、红线置顶、用户自定义规则、通配符分组折叠/混合组、finalize 固化（exclude/keep/decided_by/current_counts/未知列校验）、权威清单内容。