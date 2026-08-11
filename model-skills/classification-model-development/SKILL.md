---
name: classification-model-development
description: 模型开发总控。在 orchestration 用户确认建模后调起，按迭代式流程串联 classification-model-training / classification-model-tuning / classification-model-comparison 三个 sub-skill，管理路径接力、决策点询问、report.md 回填、断点续跑、无上游 session 时的历史 session 列举与新建。触发词：开发模型、跑 baseline、调参、多模型对比、列历史 session。
---

# 模型开发总控（classification-model-development）

## 1. 角色与边界

你是开发阶段的总调度,**不是工程师**。本 skill 只编排 4 个 sub-skill 已有的 CLI,外加一个 session 决议工具 `list_sessions.py`(扫描 `runs/` 下历史 sessions 供用户选择):
- 上游: `classification-model-orchestration` Step 5 在用户选"是"后调起本 skill,传入 `session_dir` 及各上游产物路径。
- 下游: 5 个 sub-skill
  0. `feature-analysis` — 特征质量分析(IV/PSI/基础统计,一次性必跑)
  1. `classification-model-training` — baseline 训练(8-stage 产物管线)
  2. `classification-model-tuning` — 特征筛选 / 超参调优迭代
  3. `classification-model-comparison` — session 级 N-way 横向对比
  4. `score-to-fico` — 收口后概率分 → FICO 标准分转换(Stage 5,对 top1 上线候选 run 转分)

你不替代任何 sub-skill 的内部逻辑,只负责:**何时跑哪个、读哪些产物、问用户什么决策、怎么写回 report.md、怎么断点续跑**。

> `feature-analysis` 由本 skill Stage 0 调起,产 `sample-features/feature-analysis/analysis/*.csv` + `sample-features/splits/{train,test,oot}.parquet`,Stage 2a feat 筛选直接消费。

## 2. 输入契约(从 orchestration 接力)

启动时**先验证**这 6 项都存在,缺任一 → 报错并指明缺哪个、上游哪个 skill 该补。
(feature-analysis 产物不在此校验 — 它是本 skill Stage 0 自产,见 3. 节)

| 输入 | 路径 | 来源 skill |
|------|------|-----------|
| `session_dir` | `runs/{timestamp}-{model_name}/` | orchestration Step 2(或本 skill 2.1 节 session 决议新建) |
| 需求文档 | `task-spec/task-spec.md` + `_manifest.json` | classification-model-task-spec |
| 样本分析 | `data-profile/_manifest.json` | classification-model-task-spec |
| 特征宽表 | `sample-features/feature-matching/sample.parquet` + `feature-list.csv` | feature-matching (spark 模式 / local_file 模式均产, 路径一致) |
| 历史推荐 | `model-recommend/{model_id}/`(含 `evaluation/` 三档 eval JSON) | classification-model-recommend (**local_file 模式下无此产物, 跳过 recommend**; Stage 1 baseline 不对齐历史 baseline) |
| base 对比列 | `model-recommend/{model_id}/evaluation/` 或 yaml `model.baseline_eval_dir` | classification-model-recommend (**local_file 模式下无此产物**; Stage 1 baseline 不做 base 对比) |

### 2.1 Session 决议(无上游 session 时由本 skill 负责)

当 orchestration 已传入 `session_dir` 时**直接复用,不询问**。当用户新启动一个会话直接调本 skill(无上游 session 上下文)时,本 skill 负责 session 决议:

1. 调本 skill 的 `list_sessions.py` 列出历史 sessions(输出 markdown 表:序号/session_id/task_name/created_at/description):

   ```bash
   python <skill_dir>/scripts/list_sessions.py
   ```

2. 用 `AskUserQuestion` 问用户:选**历史**(给编号)则 `session_dir` 拼该 session 目录;**新建**则追问 `task_name` + 可选 `description`,按 `ts=$(date +%Y%m%d-%H%M%S)` 建 `runs/${ts}-${task_name}/` 并写 `session.json`(字段同根目录 CLAUDE.md 约定)
3. 同一会话内**不重复询问**

`list_sessions.py` 参数:

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--outputs_dir` | 否 | 自动查找 | sessions 根目录;默认从脚本位置向上查找 `runs/` |

无历史 session(输出"无历史 session")时走新建流程,追问 `task_name`。

## 3. 阶段流水(迭代式,不是线性)

> **Gateway（入口先裁决，硬门禁）**：进入本 skill 后、执行任何 Stage 前，先读 `<session_dir>/task-spec/_manifest.json` 的 `engine.ruling`：
> - `local`（或缺省）→ 走下方正常本地流程：Stage 0(feature-analysis)→ Stage 1(training)…
> - `distributed` → **跳过 Stage 0 + splits 生产**，直连 ray-distributed-train（复用 task-spec 固化的 source_table/label_col/dt_col/f_p_date 三窗采参），收集集群返回的 AUC/KS+模型产物评估；特征分析段落写「分布式特征分析留待未来开发」（这不是未完成占位，而是有意义的工程记录）。
> - Gate P0 判定时机与测量方式见 task-spec §3.5.1；判据单一真相 = `config_io.LOCAL_BYTES_LIMIT`(字节)，已彻底废弃 R×C 元素数口径。

```
Stage 0: feature-analysis (一次性, 必跑 — ruling=local 时; distributed 时整段跳过)
   ↓ 产 sample-features/feature-analysis/analysis/{stats,iv_table,psi_table}.csv + report.md
   ↓ 产 sample-features/splits/{train,test,oot}.parquet (feature-analysis 内部切分)
   ↓ fill_report.py --section V 回填 report.md 第五段
Stage 1: baseline training (一次性, 必跑)
   ↓ 产 new-models/{algo}-v{N}/  (8 stages)
   ↓ run_build 自动 chain: evaluation → per-run comparison → session-aggregate
Stage 2: 迭代决策点 (loop, 可重复 0~N 次)
   ├─ 2a: feature selection → -feat 新 run (消费 feature-analysis 的 csv)
   ├─ 2b: hyperparameter tuning → -tuned 新 run
   └─ 2c: 换算法 → 新 algo run (回到 Stage 1, 用不同 model.algo)
Stage 3: session-level comparison (auto, 由 run_build 触发)
   ↓ 产 model-comparison/model-comparison_{split}.{json,md,xlsx} × 2 (oot, all) + 对比报告.{json,md,xlsx} + _manifest.json
Stage 4: 收口 — 回填 report.md 第七段, 询问用户是否上线候选; 上线候选填入附录「待处理项」
Stage 5: FICO 转换 (score-to-fico, 收口后**总是询问**)
   ↓ 用户确认后对 top1 上线候选 run 做概率分 → FICO 标准分转换
   ↓ 产 new-models/{run}/fico/{coef.json, fico_{train,test,oot}_predictions.parquet, fitting-summary.{json,md}}
   ↓ FICO 转换结果追加 report.md 附录「待处理项」
```

**关键: Stage 2 是 loop,不是单次。** 每个 run 完成后都问用户"继续迭代 / 换路子 / 停下对比",不自动推进。

## 4. 路径接力契约(强制)

每个阶段的"读什么 / 写什么 / 跑哪条 CLI / 跑完调哪条回填"如下表钉死,避免 agent 跑错路径:

| 阶段 | 读 | 写 | 接力 CLI | report.md 回填 |
|------|----|----|---------|---------------|
| Stage 0 | `sample-features/feature-matching/sample.parquet` + `feature-list.csv` | `sample-features/feature-analysis/analysis/{report.md,stats,iv_table,psi_table}.csv` + `_manifest.json` + `sample-features/splits/{train,test,oot}.parquet` | `python feature-analysis/scripts/run_analysis.py --config <session_dir>/sample-features/feature-analysis/feature_config.yaml --data_path <session_dir>/sample-features/feature-matching/sample.parquet --output_dir <session_dir>/sample-features/feature-analysis/analysis` | `--section V` |
| Stage 1 | `sample-features/feature-matching/sample.parquet` + `feature-list.csv`(全量) | `new-models/{algo}-v{N}/{config,features,model,evaluation,predictions,explainability,comparison,logs}/` | `python classification-model-training/scripts/run_build.py --config <train_config.yaml> --output_dir <session_dir> --version v1` | `--section VI`(追加新 run 行) |
| Stage 2a | baseline run dir + `sample-features/feature-analysis/analysis/{stats,iv_table,psi_table}.csv` | `new-models/{algo}-feat-v{N}/` | `python classification-model-tuning/scripts/select_features.py --baseline_run <baseline_run_dir> --analysis_dir <session_dir>/sample-features/feature-analysis/analysis` | `--section VI` |
| Stage 2b | baseline run dir | `new-models/{algo}-tuned-v{N}/` | `python classification-model-tuning/scripts/run_tuning.py --baseline_run <baseline_run_dir> [--method rule\|optuna]` | `--section VI` |
| Stage 2c | `sample-features/feature-matching/` + `feature-list.csv` | `new-models/{new_algo}-v{N}/` | 改 yaml `model.algo: dnn\|lr\|seg` 后重跑 Stage 1 的 `run_build.py` | `--section VI` |
| Stage 3 | `new-models/*/evaluation/*_{split}_eval.json` + `model-recommend/*/evaluation/` | `model-comparison/model-comparison_{split}.{json,md,xlsx}` × 2 (oot, all) + `对比报告.{json,md,xlsx}` + `_manifest.json` | **自动**: `run_build.py` 末尾调 `invoke.session_aggregate.invoke_session_aggregate(output_dir)`; 也可手动跑 `python classification-model-comparison/scripts/aggregate_session_comparison.py --session-dir <session_dir>` | `--section VII` |
| Stage 4 | (无新产物) | `report.md` 附录「待处理项」(人工填写上线候选) | (人工填写上线候选) | — (脚本不回填附录) |
| Stage 5 | `new-models/{run}/predictions/{train,test,oot}_predictions.parquet`(top1 上线候选 run) | `new-models/{run}/fico/{coef.json, fico_{split}_predictions.parquet, fitting-summary.{json,md}}` | `python score-to-fico/scripts/score_to_fico.py --from-run --run-dir <run_dir>` | FICO 结果追加附录「待处理项」(coef/intc + bscore 范围 + 产物路径) |

**baseline run dir 决议**: Stage 2a/2b 默认用 `new-models/` 下最新一版 run(按 timestamp 排序)。若用户指定其他 baseline,以用户指定为准。

**回填机制**: `fill_report.py` 是幂等的 — 同一段落多次调用结果一致,用 H2 锚点(`## 四、` 等)切分替换。Stage 2 每次新 run 后调 `--section VI` 会重新扫 `new-models/*/config.json` 重建整张表,不会重复 append。

## 5. 决策点话术(强制,每个阶段后必问)

> **决策点必问**: 每个 Stage 完成后都向用户询问下一步(Stage 0~4 决策点 + Stage 5 FICO 转换),不自动推进;下方话术在每个阶段后触发。

> **distributed 守卫**：`engine.ruling=="distributed"` 时本小节（以及下述 Stage 0 / 2a 相关话术）整体跳过 —— 无本地特征分析产物可汇报，直接走 ray-distributed-train。

**Stage 0 完成后**(feature-analysis 落盘, 仅 ruling=local 触发):
```
> feature-analysis 已落盘。n_features={N}, 三档样本量: train={x} / test={y} / oot={z}
> IV Top5: {列表} | PSI>0.10 特征数: {K} | 高缺失率(>50%)特征数: {M}
> 下一步?
>   A. 进 Stage 1 训 baseline (用全量 features, 后续再 -feat 筛)
>   B. 我先人工筛 features, 再进 Stage 1 (用户自定义 features 列表后重跑)
>   C. 停 (特征质量太差, 回去补特征)
```

**Stage 1 完成后**(以及 Stage 2 每次迭代后):
```
> {run_name} 已落盘。三档 AUC: train={x} / test={y} / oot={z}
> 下一步?
>   A. 特征筛选 (基于 IV/PSI 剔特征, 产 -feat run)
>   B. 超参调优 (基于规则/Optuna, 产 -tuned run)
>   C. 换算法 (产 dnn/lr/seg run)
>   D. 跑横向对比 (若已有 ≥2 个 run, 重跑 session-aggregate)
>   E. 停, 进入收口 (Stage 4)
```

**Stage 4 收口**:
```
> 共跑 {N} 个 run, session-level 对比在 model-comparison/。
> Top 候选(按 oot AUC): {列表}
> 是否标记上线候选? 或继续迭代?
```

**Stage 5 FICO 转换**(收口后, **总是询问**):
```
> 已确认上线候选 {run_name}(oot AUC={x})。是否将概率分转换为 FICO 标准分(范围约 [400,780], 分高险低)?
>   A. 是, 转换 (对 {run_name} 跑 score-to-fico: train 拟合校准, test/oot 转分, 产 {run}/fico/)
>   B. 否, 跳过 (产物保留概率分; 后续随时可手动补转)
```

话术是示例,不是脚本 — agent 可根据上下文调整措辞,但**必问的决策点不能省**(Stage 5 总是询问)。

## 6. report.md 回填契约(强制)

每个阶段完成后,**必须**用 `fill_report.py` 回填 `runs/{timestamp}-{model_name}/report.md` 对应段落。**禁止**留 `(待...执行)` 占位 — 阶段完成了就必须填实。

**回填 CLI**(脚本位置: `classification-model-development/scripts/fill_report.py`):
```bash
python classification-model-development/scripts/fill_report.py \
    --session-dir <session_dir> --section {IV|V|VI|VII|all}
```

脚本从 `session_dir` 下各 sub-skill 产出的 manifest/JSON/CSV 中提取信息,幂等更新段落(用 H2 锚点切分替换,多次调用结果一致)。`--section` 仅接受 IV/V/VI/VII/all,不支持「八、建模决策」(该内容归附录「待处理项」,由 Stage 4 人工填写)。

> 第四段(特征宽表)由 `classification-model-orchestration` Step 4B 回填;第五段(特征分析)/六~七段由本 skill 触发回填;附录「待处理项」由 Stage 4 人工填写(脚本不接管)。

**最终校验(收口门禁, 必跑)**: 所有回填(`fill_report.py` + Stage 4/Stage 5 手动填写)完成后、宣布收口交付前,**必须**运行 render-check 做最后一道数字一致性校验 —— 把「进报告的数字必须源自产物文件、绝不手敲」固化成机器判据(复盘事故 B 的结构化护栏)。任一 FAIL/WARN 未清零一律视为未完成收口:
```bash
# 受保护解释器按本机安全钩子要求走变量间接调用(如 PY=/path/to/python3; "$PY" ...)
python ../_modevo-shared/bin/render-check/render_check.py \
    --session <session_dir> \
    [--expect-kava-dev <dev AUC>]      # ka_v4 统一口径重测 dev 期望值,报告含该对照表时才传
    [--expect-kava-oot <OOT AUC>]      # 同上,OOT 期望值
```
退出码:`0`=一致可交付 / `1`=FAIL(WARN 默认亦算失败,可用 `--no-fail-on-warn` 降级) / `2`=缺 report.md 无法自检。三族白名单断言(其余自由文本不作数值比对,避免误报淹没真实冲突):①迭代/N-way 表的每行 train/test AUC·KS ↔ `new-models/{run}/evaluation/*_{split}_eval.json` → `metric_by_segment.全量`;②标题含 `ka_v4` 且带「统一口径」的对照表 dev/OOT 两行 ↔ `--expect-kava-*`(容差 ±0.005,吸收跨批次口径微差);③FICO bscore 摘要行(`- \`train\`: n=… | bscore 范围=[a,b] | 均值=c`)↔ `fico/fitting-summary.json`。**为让第③族能自动核对,FICO 摘要必须采用上述固定行格式**书写(见 `score-to-fico`, Stage 5)。

| 段落 | 何时写 | 内容来源 |
|------|--------|---------|
| 四、特征宽表 | **orchestration Step 4B 回填**,本 skill 不触发 | 特征数 / 三档样本量(从 `sample-features/splits/{train,test,oot}.parquet`(主) 或 `data-profile/_split_manifest.json`(fallback) 取) |
| 五、特征分析 | **本 skill Stage 0 回填**,orchestration 不触发 | IV Top10 / PSI>0.10 列表 / 高缺失率列表(从 `feature-analysis/analysis/{iv_table,psi_table,stats}.csv` 取)。**ruling=distributed 时此段写「分布式特征分析留待未来开发」** —— 这是有意义的工程记录（非未完成占位），与「禁止 (待…执行) 占位」不冲突 |
| 六、模型迭代 | 每个 run 完成后 | run_name / algo / 三档 AUC / 关键变更(从 `new-models/*/config.json.runtime` 取,自动识别 baseline / -feat / -tuned) |
| 七、横向对比 | Stage 3 完成后 | 对比表摘要(从 `model-comparison/model-comparison_oot.json` 的 `auc_comparison["全量"]` 取,按 AUC 降序) |
| 附录「待处理项」 | Stage 4 完成后 | 上线候选 / 下一步建议(**人工填写**,脚本不接管) |

## 7. 断点续跑

启动时扫描 `session_dir`,按以下顺序检查 `_manifest.json` 推断当前阶段:

| 检查 | 推断阶段 |
|------|---------|
| task-spec `_manifest.json.engine.ruling == "distributed"` | **Gate P0 已裁定分布式**：Stage 0 / splits 记为「已跳过」，直接续接 ray-distributed-train（见 §3 Gateway），不得导回补跑本地 Stage0 |
| 非 distributed 且 `sample-features/feature-analysis/analysis/_manifest.json` 不存在 | Stage 0 待跑 |
| 非 distributed 且 feature-analysis manifest 存在但 `new-models/` 为空 | Stage 1 待跑 |
| `new-models/` 非空但 `model-comparison/_manifest.json` 不存在 | Stage 2 迭代中,问用户继续迭代还是收口 |
| `model-comparison/_manifest.json` 存在 | Stage 3 已完成,问用户是否收口 |
| 收口已完成,但 top1 上线候选 run 无 `fico/` 产物(缺 `fico/fitting-summary.json`) | Stage 5 待跑,问用户是否 FICO 转换 |

向用户回显"当前进度: Stage X,下一步: Stage Y",确认后从断点继续。**不要**因为 manifest 存在就跳过对应阶段 — 只在用户明确说"从 Stage X 继续"时跳过。

## 8. 多算法切换

Stage 2c 换算法时:
1. 改 `train_config.yaml` 的 `model.algo: dnn|lr|seg`
2. 重跑 `run_build.py`(回到 Stage 1 的 CLI)
3. 不同算法的产物布局一致(8 stages),`config.json.algo` 区分
4. comparison 阶段自动跨算法 N-way 对比(`aggregate_session_comparison.py` 扫 `new-models/*/evaluation/` 不区分 algo)

**注意**: dnn/lr/seg 算法的 `run_build.py` 路径与 xgb 相同,内部按 `model.algo` 分流到不同 trainer。无需手动指定脚本。

## 9. 与 orchestration 的接口

- **输入**: orchestration Step 5 传入 `session_dir` + 各上游产物路径(不含 `feature-analysis/` 产物 — 由本 skill Stage 0 自产)
- **输出**: development 结束时, `session_dir` 下完整包含:
  - `task-spec/` + `data-profile/`(上游已产,本 skill 不动)
  - `model-recommend/`(上游已产,本 skill 不动)
  - `sample-features/feature-matching/`(上游已产,本 skill 不动)
  - `sample-features/feature-analysis/`(本 skill Stage 0 产)
  - `sample-features/splits/`(本 skill Stage 0 产)
  - `new-models/`(Stage 1 + Stage 2 产; Stage 5 FICO 转换产物 `{run}/fico/` 也在其中)
  - `model-comparison/`(Stage 3 产)
  - `report.md`(全段填充 — 四由 orchestration 回填, 五/六/七由本 skill 回填, 附录「待处理项」由 Stage 4 上线候选 + Stage 5 FICO 结果填写)
- **结束条件**: 用户在 Stage 4 选"停" 或 所有决策点选"停",且 Stage 5 FICO 询问完成(是→转分完成; 否→明确跳过)

## 10. 反模式

- ❌ **自动推进**(不问用户就跑下一阶段)— 决策点必问
- ❌ **在 development 里写新 Python 脚本** — 复用 sub-skill CLI;本 skill 仅提供 `list_sessions.py`(session 决议工具),不产其他代码
- ❌ **report.md 留占位** — 阶段完成必填实
- ❌ **把 development 当成"模型工程脚手架"** — 它是编排器,不是代码生成器
- ❌ **把 Stage 2 当成单次** — 它是 loop,每个 run 后都问"继续迭代吗"

## 11. 关联 skill

- 上游: `classification-model-orchestration`(Step 5 调起本 skill)
- 上游依赖: `classification-model-task-spec` / `feature-matching` / `classification-model-recommend`(产 2. 节输入契约中的产物)
- 下游编排: `feature-analysis`(Stage 0) / `classification-model-training`(Stage 1) / `classification-model-tuning`(Stage 2) / `classification-model-comparison`(Stage 3) / `score-to-fico`(Stage 5, 收口后总是询问)
- 下游可选: `classification-model-recommend`(run 完成后人工登记到台账)
