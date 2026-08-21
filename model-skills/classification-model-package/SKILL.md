---
name: classification-model-package
description: 把已定版训练任务组装为可独立运行的交付代码包。消费 session 定版产物（finalized_model.json + new-models/{run}/model + cleaning-scheme + 权威 feature-list + fico/coef.json），产出 delivery/ 交付包：包内自包含「数据清理(哨兵值→NaN) → 定版模型打分 →（若存在转 FICO 模块）同时输出模型分与 FICO 分」，零引用专家包，仅依赖 pip 包，一条命令跑通。**仅用户主动触发**（收口后可选出口，不默认执行）。
---

# 独立交付包组装（classification-model-package）

把**已完成定版（`finalized_model.json`）的训练任务**组装为一个**可独立运行的交付代码包**。消费方为工程/业务团队：给定一份数据文件，包内一条命令完成「数据清理 → 定版模型打分 →（若含 FICO 模块）同时输出模型分与 FICO 分」全链路。

> ⚠️ **触发定位**：**仅用户主动触发**（用户明确说"打包交付 / 组装成可交付代码包 / 交付给工程"），作为 `classification-model-development` 主链路收口后的**可选出口**，不默认执行、不默认询问。

## 1. 职责边界

| 做 | 不做 |
|---|---|
| 校验定版产物完整性（finalized_model.json / model 资产 / 清洗方案 / 权威特征清单） | 不做任何训练 / 评估 / 拟合（只消费已定版产物） |
| 组装 `delivery/` 独立交付包（自包含脚本 + 资产 + requirements + README + manifest） | 不改变 session 既有产物；唯一写目录是 `delivery/` |
| 打包器仅支持主链路产物 **lgb/xgb**（含 xgb 历史 `model.json`） | **不支持 dnn/lr**（自包含包无法携带其反序列化所需 training 脚本依赖，报错拒绝并提示走主链路） |
| 交付包含 FICO 模块当且仅当 session 存在 `fico/coef.json`（纯应用模式） | 包内不做 FICO 拟合（生产数据可能无 label，校准参数打包时固化） |

## 2. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `finalized_model.json` | ✅ | session 根（development Stage 6 落） | 含 `run_name` / `algo` / `model_path` |
| 定版模型 | ✅ | `new-models/{run}/model/` | `model.pkl`(lgb/xgb joblib) 或 `model.json`(xgb 历史) + `model_meta.json`(`feature_names`/`algo`) |
| 清洗方案 | ✅(缺则默认+WARN) | `sample-features/data-cleaning/cleaning-scheme.json` | 哨兵集 `invalid_values`；缺失回退默认 `[-1,-2,-9,-99,-999,-9999,-99999]` |
| 权威特征清单 | ✅(缺则回退 feature_names) | `sample-features/feature-list.csv` | 打包作 `assets/feature-list.csv`；与 `model_meta.feature_names` 一致性 WARN |
| FICO 校准参数 | 条件 | `fico/coef.json` | 存在 → 交付含 FICO 转分模块与 `assets/coef.json` |

## 3. 执行命令

`<skill_dir>` 指本 skill 所在目录，执行时替换为实际绝对路径。

```bash
python <skill_dir>/scripts/package_model.py \
    --session-dir <session_dir> \
    [--out-dir <delivery 输出目录>]
```

`--out-dir` 缺省 = `<session_dir>/delivery/`。

## 4. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--session-dir` | ✅ | - | 已定版 session 目录（须含 `finalized_model.json`） |
| `--out-dir` | 否 | `<session_dir>` | 交付包输出目录（交付包落 `<out-dir>/delivery/`） |

## 5. 输出产物

```text
<session_dir>/delivery/
├── run.py                        # 主入口: 清理 → 打分 →(可选)FICO 转分
├── pipeline/                     # 自包含实现（clean.py / score.py / fico.py / __init__.py）
│   └── fico.py                   # 仅当含 FICO 模块时（与模板统一，运行期 try-import 跳过）
├── assets/
│   ├── model.pkl 或 model.json   # 定版模型
│   ├── model_meta.json           # feature_names / algo（驱动打分）
│   ├── cleaning-scheme.json      # 哨兵集（驱动清理）
│   ├── feature-list.csv          # 权威特征清单（打包资产）
│   └── coef.json                 # FICO 校准参数（仅含 FICO 模块时）
├── requirements.txt              # 最小依赖（按 algo 渲染 lightgbm/xgboost）
├── README.md                     # 用法 / 输入 schema / 特征列表 / 输出说明（渲染会话信息）
└── package-manifest.json         # 打包元信息（来源 session / run / algo / 特征 / 是否含 FICO）
```

### 包内运行（交付侧）

```bash
cd delivery
pip install -r requirements.txt
python run.py --input <数据文件 parquet/csv> --output-dir <输出目录>
```

输出（`--output-dir` 下）：

```text
score.parquet          # 透传非特征列 + score（含 FICO 时追加 odds/logistic_prob/bscore）
cleaning-report.json   # 哨兵命中统计（命中即替换 NaN，仅 WARN）
fico-summary.json      # FICO 校准参数 + bscore 分布（含 FICO 模块时）
run-manifest.json      # 本次运行元信息
```

## 6. 与其他 skill 的关联

| skill / 模块 | 关系 | 说明 |
|---|---|---|
| `classification-model-development` | **编排调起（可选出口，用户主动触发）** | 主链路收口 + 打分（Stage 7）之后；本模块只读 session 定版产物，不改编排链路 |
| `model-scoring` / `score-to-fico` | 上游（产物来源） | 分别产 `scoring/` 与 `fico/coef.json`；本模块消费其定版与校准产物 |
| `data-cleaning` | 上游（清洗方案来源） | 消费 `cleaning-scheme.json` 哨兵集，**包内清洗裁剪去重/强门禁**（仅哨兵值→NaN + WARN） |
| `_modelevo-shared` | 依赖（仅打包器） | 打包器经 `_bootstrap.py` 注入复用 `gen_feature_list` 读权威清单；**交付包内零依赖** |

## 7. 执行约束

| 约束 | 说明 |
|---|---|
| 独立运行红线 | 交付包零引用专家包目录与 `_modelevo-shared`，仅依赖 `requirements.txt` 的 pip 包；包内脚本由 `package_templates/` 纯拷贝（模板与单测共享同一份源码） |
| 资产驱动 | 包内行为全部读 `assets/` 既有资产文件（feature_names / algo / invalid_values / coef-intc），零占位符、可审计 |
| 算法边界 | 仅支持 lgb/xgb（含 xgb 历史 model.json）；dnn/lr 打包器报错拒绝 |
| FICO 条件包含 | 打包时 `fico/coef.json` 存在才含转分模块；纯应用模式（不拟合），bscore 越界 [400,780] 仅 WARN 不中止 |
| 包内清洗 | 仅特征列哨兵值→NaN，非交互；**不做样本去重**；不校验 id/dt/label 列是否存在（允许缺 label） |
| 包内打分 | 特征缺失报错退出（列出缺失）；按 feature_names 重排后推理；输出透传非特征列 + score，不含特征列 |
| 数据安全红线 | 打包器只读 session、唯一写目录 `delivery/`；不透出身份证/手机号明文；README 只描述 schema 不含样本 |

## 8. 异常处理

| 异常 | 处理方式 |
|---|---|
| session 无 `finalized_model.json` | 报错退出，提示先走主链路收口（development Stage 6） |
| 定版模型缺 `model_meta.json` / 缺 `feature_names` | 报错退出 |
| algo 为 dnn/lr | 报错退出，提示仅支持主链路 lgb/xgb（自包含包无法反序列化 dnn/lr） |
| 模型目录缺 `model.pkl` / `model.json` | 报错退出 |
| `cleaning-scheme.json` / `feature-list.csv` 缺失 | WARN + 回退默认（默认哨兵集 / model feature_names），不阻断打包 |
| `fico/coef.json` 存在但解析失败 | WARN，交付包不含 FICO 模块 |
| 交付包内缺特征 / 缺模型文件 / 推理长度不一致 | 包内 `SystemExit([ERROR] ...)` 报错退出，不产半成品 |

## 9. 测试

```bash
python -m pytest model-skills/classification-model-package/tests/ -q
```
