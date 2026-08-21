---
name: credit-model-report
description: 从模型打分 CSV 生成风控模型**业务评估报告**（Excel，6-sheet：回溯表 / 建模信息 / KS / 特征重要性 / 模型效果 Lift+SWAP / 打分分布 PSI+分桶+分段逾期率），支持新模型 vs 基线模型对比、客群过滤、模板化输出。**移出建模主流程**：仅用户主动要求业务评估报告时触发（"评估报告""业务评估报告""回溯表""Lift""SWAP""打分分布""分段逾期率""模型报告模板"）。标准化指标（AUC/KS/分桶排序性）由 classification-model-experiments 的评估产物（evaluation/eval.json/.md）承载。
compatibility: python3, pandas, openpyxl, numpy
---

# 风控模型评估报告生成

从模型打分 CSV 文件生成完整的 Excel 评估报告，包含回溯表、建模信息、KS、特征重要性、模型效果（Lift + SWAP）、打分分布（PSI + 分桶 + 分段逾期率）共 6 个 sheet。

## 定位与分工

| | credit-model-report（本 skill） | classification-model-experiments 评估 |
|---|---|---|
| 触发场景 | **用户主动要求**业务评估报告（回溯表 / Lift / SWAP / 打分分布 / 模板输出），不进入 development 主流程 | 建模 pipeline 内标准化评估（AUC / KS / 分桶排序性，evaluation/eval.json/.md 自动产出） |
| 输出 | `{model_name}评估报告_YYYYMMDD.xlsx`（模板化，6-sheet） | JSON + MD 评估产物（标准化评估） |
| 特色能力 | **新旧模型 SWAP 迁移分析**、分段逾期率、回溯表、模板复用 | 标准评估指标 + 客群拆分 |

> **触发关键词**：评估报告 / 业务评估报告 / 回溯表 / Lift / SWAP / 打分分布 / 分段逾期率 / 模型报告模板 → 本 skill（仅用户主动要求时）。仅要 AUC / KS 等标准化指标 → experiments 评估产物 `evaluation/eval.json/.md` 已含。

## 执行流程

### Step 1: 确认输入数据

找到或请求用户提供打分 CSV 文件。扫描前 0 行（`pd.read_csv(nrows=0)`）获取所有列名，推断：

| 推断 | 规则 |
|---|---|
| **标签列** | 列名以 `fpd` 或 `dpd` 开头 |
| **评分列** | 列名含 `score` 且不含 `label` |
| **日期列** | 列名含 `date` 或 `ser`（默认 `fser_date`） |
| **用户ID列** | 默认 `fuid` |
| **月份范围** | 读日期列前 50 万行，提取 YYYY-MM 去重排序 |

> ⚠️ **构造打分 CSV 的常见坑（务必注意）**：
> 1. **多模型打分宽表不要按 uid join**：同一 uid 可能跨多个观察月出现（如按月采样），按 `uid` merge 会 1:N 放大行数（实测 41.8 万行 join 后炸到 124 万）。**正确做法**：若打分表与样本表行序一一对应（如由同一份 splits 逐行预测产出），直接**按行序横向拼接**（`pd.concat([sp, pr[['score']]], axis=1)`），或按 `(uid, 日期列)` 联合键 join。
> 2. **日期列格式**：脚本按 `YYYY-MM` 聚合，日期列建议给 `YYYY-MM-DD` 字符串（如 `fser_date`），脚本内部可提取月份；不要用纯 int `YYYYMMDD` 且不写日期列名映射（`date_col` 必须显式指定）。
> 3. **标签列命名**：脚本默认按 `fpd`/`dpd` 前缀推断标签列；自定义标签名（如 `label`）必须显式传 `primary_label`/`ks_labels`/`lift_labels`/`seg_labels`。

### Step 2: 交互确认配置（关键决策确认门禁）

根据推断结果和用户需求，确认以下 config 参数。**有默认值的不要问，没有的才问**。

**必填（必须用户提供）：**
- `csv_path`：数据文件路径（可从上一步推断）
- `models`：`[[主模型名, 评分列], [基线模型名, 评分列]]` —— 从评分列推断后让用户确认
- `model_name`：报告/输出文件名（如用户未指定，用主模型名）

**选填（有默认值，仅当与默认不同才确认）：**
- `filter_conds`：`{列名: 值}` 客群过滤，如 `{"if_tf": 1}` —— 询问用户是否需要过滤
- `primary_label`：主标签列（默认第一个 `dpd` 开头的列）
- `train_range`：`["YYYY-MM", "YYYY-MM"]` —— 默认取月份范围前半段
- `oot_range`：默认取月份范围后半段
- `base_month`：PSI 基准月，默认 OOT 第一个月
- `ks_labels`：所有标签列（自动推断，通常无需确认）
- `lift_labels`：Lift/SWAP 标签列表（默认前两个标签，如 `["fpd7_sx30", "dpd30_3c"]`）
- `seg_labels`：分段逾期率标签（默认同 lift_labels）
- `lift_bins`：Lift 等频桶数，默认 10
- `swap_bins`：SWAP 等频桶数，默认 5
- `template_path`：Excel 模板路径

> 门禁对齐：`models`（主/基线模型与评分列）、`train_range` / `oot_range`（时间窗）、`filter_conds`（客群）属「关键决策确认门禁」范畴（门禁 #1/#2/#6），**先给方案等用户确认再跑**；有默认值的项不重复提问。

> **⚠️ 生成前必确认（硬要求，不得静默按默认跑）**：
> 1. **比较基准（基线模型/分数列）**：`models` 里谁是主模型、谁是基线，必须让用户明确指定——报告里的 SWAP/Lift 对比、PSI 口径都依赖这个选择；推断出的基线列要先展示给用户确认，不能直接采用
> 2. **train_range / oot_range**：必须与用户确认（通常与建模切分方案一致，但需用户点头）
> 3. **filter_conds**：是否需要客群过滤、过滤条件是什么
> 确认方式：用文本表格展示「主模型 / 基线模型 / train_range / oot_range / filter_conds」方案，用户确认或修改后再运行。**用户未确认前不得运行脚本。**

### Step 2.5 结果异常自检（生成后自动执行，异常必反馈）

脚本生成报告后自动运行「结果异常自检」，检查以下指标，**发现异常打印 WARN 并向用户反馈，不静默出报告**：

| 检查项 | 异常判据 | 提示 |
|---|---|---|
| 分桶退化 | 打分分布桶数 ≤ 2 | 分数列口径问题（概率 0~1 应自动用 0.1 步长；若仍退化请检查是否打分列选错） |
| PSI 全 0 | 各月 PSI 均 ≈ 0（桶数 ≥ 3 时） | 分布无漂移信息，检查分桶/基准月是否正确 |
| KS 越界 | OOT KS 不在 (0,1) | 打分与标签方向可能反了或数据异常 |
| 标签全无正样本 | 全量标签 0 正样本 | 报告逾期率/KS/Lift 无意义 |
| 打分列全空 | 该模型打分列全 NaN | 报告各表无有效内容 |

> 自检通过打印「>>> 全部指标在合理范围」；有异常时逐条列 WARN + 「>>> ⚠️ 检测到异常指标，请先核实原因再使用报告」。**LLM 侧拿到自检输出后，若有 WARN 必须先与用户核实，不得直接把报告当作有效交付。**

### Step 3: 复制脚本和模板

将 skill 自带的脚本和模板复制到**输出目录**。`WORK_DIR` 即报告输出目录（也是脚本工作目录）：

- **建模 session 内**：`WORK_DIR=<session_dir>/model-report`（先 `mkdir -p`），与 `scoring/`、`fico/` 平级。**严禁把 xlsx 报告输出到 `<session_dir>/scoring/` 子目录**。
- **独立使用**：`WORK_DIR=<用户数据文件所在目录>`。

```bash
WORK_DIR="<session_dir>/model-report"     # 建模 session 内
# WORK_DIR="<用户数据文件所在目录>"         # 独立使用
mkdir -p "$WORK_DIR"
cp "<skill_dir>/scripts/generate_report.py" "$WORK_DIR/"
cp "<skill_dir>/scripts/metric.py" "$WORK_DIR/"
cp "<skill_dir>/assets/模型报告模板.xlsx" "$WORK_DIR/" 2>/dev/null || true
```

`<skill_dir>` 是当前 skill 所在的目录路径（即 `SKILL.md` 所在目录）。

### Step 4: 生成 config JSON 并运行

在工作目录创建 config JSON 文件（含必填项 + 与默认不同的选填项），然后命令行执行：

```bash
cat > "$WORK_DIR/config.json" <<'JSON'
{
  "csv_path": "<csv_path>",
  "template_path": "<WORK_DIR>/模型报告模板.xlsx",
  "model_name": "<model_name>",
  "primary_label": "<primary_label>",
  "uid_col": "fuid",
  "date_col": "fser_date",
  "models": [["<主模型名>", "<评分列>"], ["<基线模型名>", "<评分列>"]],
  "filter_conds": {"if_tf": 1},
  "train_range": ["2025-08", "2025-11"],
  "oot_range": ["2025-12", "2026-04"],
  "base_month": "2025-12",
  "ks_labels": ["fpd7_sx30", "fpd15_sx30", "fpd30_sx30",
                "dpd30_1c", "dpd30_2c", "dpd30_3c", "dpd30_4c", "dpd30_5c", "dpd30_6c"],
  "lift_labels": ["fpd7_sx30", "dpd30_3c"],
  "seg_labels": ["fpd7_sx30", "dpd30_3c"],
  "lift_bins": 10,
  "swap_bins": 5
}
JSON
cd "$WORK_DIR" && python3 generate_report.py config.json --dir "$WORK_DIR"
```

也可不落盘 config，直接在 Python 中调用 `main_with_config(config_dict)`（脚本会自动校验列名、补全 OOT/训练集切分等默认值）。两种方式等价。
**输出目录**：命令行用 `--dir <目录>`、Python 调用用 `main_with_config(config_dict, base_dir=<目录>)` 显式指定 xlsx 输出目录；config 里也可直接给 `out_path` 全路径。三者均未给时，脚本默认输出到 CSV 同目录（若 CSV 在 `scoring/` 子目录，则自动落到其父目录，即 session 根）。
- **CSV 列名校验**：`uid_col` / `date_col` / 各模型评分列 / 所有标签列 / `filter_conds` 列必须在 CSV 中存在，否则脚本报错列出可用列。

### Step 5: 报告输出

脚本输出 Excel 文件到 `--dir` / `out_path` 指定目录（缺省规则见 Step 4），文件名格式 `{model_name}评估报告_YYYYMMDD.xlsx`。**建模 session 内 xlsx 必须落在 `<session_dir>/model-report/`，不得落在 `scoring/` 子目录**。自检打印包含：
- OOT 合并各模型各标签 KS
- Lift 汇总样本数和逾期率
- SWAP 样本数
- 交叉校验

向用户展示自检摘要和输出文件路径。

## 关键口径（脚本内部逻辑）

- **去重**：全表按 `(fmth, uid_col)` 去重（`groupby.first()`，自动跳过 NaN 取非空值）
- **评分 NaN**：不参与该模型评估；**标签 NaN**：不算有表现
- **有表现 < 30**：逾期率标记为 `-`，不参与 Lift/PSI
- **逾期率 / 占比**：百分数格式 `0.00%`（Excel 单元格格式）
- **KS / Lift / PSI**：保留 4 位小数
- **打分分布分桶**：10 分等距，边界为 10 的整数倍，覆盖分数 `[min, max]`，按各月最差占比判据聚合头尾 <1% 桶
- **Lift 排序**：低分到高分升序（高风险桶在上、Lift > 1），10 等频桶
- **SWAP**：旧模型行 × 新模型列，5 等频桶，含样本数/坏样本数/逾期率三子表
- **SWAP 标签**：与 Lift 标签一致
- **PSI**：仅新模型，基准月为 OOT 首月

## 两种使用模式

**模式 A（全自动）**：用户只说"帮我评估一下"。脚本自动推断所有参数，打印推断结果后直接跑。

**模式 B（半自动）**：用户指定部分参数（如"用 v12_mob4_score 当新模型，过滤 if_tf=1"）。未指定参数用默认值推断。

两种模式在 Step 2 的交互深度不同，但底层执行流程（Step 3-5）完全一致。

## 输出产物

```text
<WORK_DIR>/
├── {model_name}评估报告_YYYYMMDD.xlsx   # 主交付：6-sheet 业务评估报告
├── config.json                          # 参数溯源（可复现）
├── generate_report.py / metric.py       # 保留脚本（可复现）
└── 模型报告模板.xlsx                     # 模板（若用户目录已有则用用户的）
```

> **产物目录约定（硬性）**：建模 session 内，xlsx 报告一律落 `<session_dir>/model-report/`（与 `scoring/`、`fico/` 平级），**严禁落到 `<session_dir>/scoring/` 子目录**；独立使用时落用户数据目录即可。

## 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 平行 | `classification-model-experiments`（评估） | 分工不重叠：experiments 评估出标准化指标（pipeline 内），本 skill 出业务模板报告（回溯表/Lift/SWAP/打分分布） |
| 独立 | 无强制依赖 | 纯 pandas / numpy / openpyxl，不依赖 `_modelevo-shared` / Spark |

## 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 关键决策先确认 | `models`（主/基线评分列）、`train_range`/`oot_range`、`filter_conds` 属门禁范畴，先列方案等确认再跑 |
| ⚠️ 数据安全红线 | 报告不得透出身份证 / 手机号等明文个人数据；CSV 含敏感列时先排除再评估 |
| ⚠️ 有表现样本门槛 | 有表现 <30 时逾期率标记 `-`，不参与 Lift/PSI，避免小样本噪声 |
| ⚠️ 模板优先用户版 | 用户目录已有 `模型报告模板.xlsx` 时优先用用户的（可能含历史格式） |

## 异常处理

| 异常 | 处理 |
|---|---|
| CSV 列名缺失（uid/date/评分/标签/filter 列） | 脚本报错并列出可用列名，让用户修正 config |
| 无基线模型（单模型模式） | 只生成 Lift，跳过 SWAP sheet |
| 评分 / 标签全 NaN | 该模型不参与评估 / 不算有表现 |
| 分桶边界不足 | 打分分布按 10 分等距 + <1% 桶聚合兜底 |
