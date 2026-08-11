# 分类模型建模经验

二分类/多分类建模专属经验条目。条目模板与归档规范见 `modeling-experience-knowledge.md`，编号前缀 `EXP-C`。

## 索引

| 编号 | 标题 | 业务域 |
|---|---|---|
| EXP-C-001 | 正负样本比与下采样 | 通用 |
| EXP-C-002 | 外部征信评分为主的样本集：IV 天花板与 OOT 漂移 | 信贷风控-贷后逾期 |
| EXP-C-003 | 面板快照宽表勘探要点：标签粘性 / 幸存者偏差 / 当期直读列泄漏红线（xj_dz_pd_mob12） | 信贷风控-贷中逾期 |
| EXP-C-004 | markdown 表格校验器表头解析：elif 猜词顺序会静默错配列，须用位置优先 + 顺序守卫 + 双向验证 | 通用（工具侧） |

## 经验条目

### EXP-C-001｜正负样本比与下采样
- 日期：2026-07-04
- 任务类型：分类｜业务域：通用（种子条目）
- 关联模型：—｜关联 session：—
- 背景：正样本率极低（如 T7 动支率 <5%）时，全量负样本训练慢且收益有限。
- 做法：负样本下采样至正负比约 1:8~1:10，训练后用真实分布的 test/OOT 评估；报告中登记「正负样本比」。
- 结论：KS/AUC 基本无损，训练时间显著缩短；模型分绝对值有偏移，分档使用时按分位数切档不受影响。
- 教训/建议：若下游要用概率绝对值（如期望收益计算），需做概率校准或不下采样；下采样比例写入 config.json 便于复现。


### EXP-C-002｜外部征信评分为主的样本集：IV 天花板与 OOT 漂移
- 日期：2026-08-05
- 任务类型：分类｜业务域：信贷风控-贷后逾期（dpd30_3c）
- 关联模型：credit_dpd30_lgb_v1_20260805｜关联 session：20260805-141943-credit_dpd30
- 背景：样本 209 特征几乎全是外部征信评分（ascore/sj_score/tx_model 等）与衍生变量，单变量 IV 天花板低（仅 2 特征 IV≥0.1）。
- 做法：① 明确排除内部模型分（xgb_5c/mtl_/fusion/ka_v*）防泄漏共线；② 原分布训练（正样本率 5.91%>5% 不欠采样，概率即真实违约率）；③ OOT 按用户方案合并（2025-08~10）并剔除标签缺失样本（表现期截断 17.6%）。
- 结论：LightGBM 默认表 OOT AUC 0.636 / KS 0.195，为数据集天花板；OOT 十分桶单调（lift 1.98），排序性成立。
- 教训/建议：① 外部评分为主的样本集不要期望高 AUC，模型价值在排序与分档而非绝对值；② 39 特征 PSI>0.1（tx_model_2_score PSI=2.37）须按月监控漂移，线上分布漂移需重训或降权；③ scale_pos_weight 从 16.2（自动）改为 1.0（原分布）对 OOT AUC 影响 <0.002，属噪声，但概率绝对值可直接用于额度定价，无需校准。


### EXP-C-003｜面板快照宽表勘探要点：标签粘性 / 幸存者偏差 / 当期直读列泄漏红线
- 日期：2026-08-05
- 任务类型：分类｜业务域：信贷风控-贷中逾期（xj_dz_pd_mob12）
- 关联模型：xj_dz_pd_lightgbm_v1｜关联 session：20260805-194322-xj_dz_pd_mob12
- 背景：`dp_jckx_mart.xj_dz_pd_sample_0623_feats_level2_ovd_bhvr_lhj` 是用户×月末的**面板快照**表（60,732,286 行 × 452 列含 fuid/label/fetl_time/f_p_date + 448 特征；源引擎 hive/ORC、f_p_date 分区）。目标 `mob12_ever30_bmg`（MOB12 内曾≥30天逾期，1=坏）。
- 做法（勘探三步走）：① 因 Presto 限制 SHOW COLUMNS/CREATE，改用 `SELECT * WHERE 1=0` + 抽样拉取获得权威 schema，避免 RAG top_n_cols 截断误导；② 识别角色列并锁定元信息（row_id=fuid、分区键=f_p_date、快照时刻=fetl_time）；③ 特征按命名族归类——user_overdue_stat 194 / order_sub_order 240 / order_level 14，其中 `fuser_covd_dpd / fodr_covd_rto` 等**当期直读状态列**被用户确认暂不做白名单过滤，以 IV>1.0 事后检测兜底警示。
- 关键结构发现：
  ① **标签高度粘性**：单户平均重复 ~9.3 个月，83.4% 多月用户在全部月份 label 恒定 → 同户跨月样本几乎不提供独立信号；
  ② bad rate≈16%（存量累加口径）高于传统时序不良率 → **存在幸存者/重入组偏差**，评分绝对值不可当违约概率用，评估以排序能力为主；
  ③ 无放款日期，无法严格做「观察起点→未来 MOB12」事件计数，近似采用「当月状态→未来12月结局」口径（已声明）；
  ④ baseline LightGBM 全量446特征原分布训练：train AUC .6599 / val .6591 / OOT .6544 KS .2181（窗口 train=24-10~25-02, val=25-03, oot=25-04~25-05），Top 特征为还款额类 `fsodr_rpy_amt_{12m,1m}` 与逾期极值 `fuser_ovd_max_1m` —— 未见明显穿越迹象（无 IV>1.0）。
- 教训/建议：① 面板数据务必先查「户均期数 + 标签稳定性」，否则会误以为有效样本很多而实际独立信息远少于行数；② 存量口径 bad rate 偏乐观，报告必标偏差假设；③ 租户实时直读列优先「保留+高IV监控」而非一刀切过滤，避免误伤真实时效特征，但对最终选型保持警觉。


### EXP-C-004｜markdown 表格校验器表头解析：elif 猜词顺序静默错配列（render-check）
- 日期：2026-08-06
- 任务类型：分类（工具侧）｜业务域：通用
- 关联模型：—｜关联 session：20260806-110918-dpd30_3c_overdue（render_check.py）
- 背景：为收口门禁写 report.md ↔ `evaluation/*_{split}_eval.json` 一致性校验器。迭代表头形如 `# | run_name | train AUC | test(val) AUC | oot AUC | oot KS | …`，需把四格数值按 train/test/oot_AUC/oot_KS 对齐到 eval JSON。
- 做法（演进 + 修复）：先用 if/elif 按 train→test→oot 猜关键词定位列 → **fail**；改用「位置优先 + 首见即得」：从左到右扫描每列的规范名（去空格转小写），按后缀判别命中 `train / test / oot_auc / oot_ks` 记首次下标；再强制要求四个目标列下标严格递增，倒挂即整表拒绝。
- 结论：修正后真 session `exit=0`、三族损坏夹具各自单点 FAIL（N-way train .76364→.77777 / ka_v4 OOT .59998→.62222 / FICO mean 560.45→980）。
- 关键坑：
  ① **elif 分支的顺序就是取舍规则**：`trainauc` 不含 `oo` 但含 `tran`，若条件写成 `"oo" in n and "auc"` 在前会把它误判成 oot —— 具体错误是 `test(val)`(idx3) 与 `oot AUC`(idx4) 因共享「后段特征」被 elif 互相抢占，col_map 里 test↔oot_auc 对调，每个值都落到相邻列上；
  ② **仅靠"真样本跑绿"证明不了正确性**：同一份代码曾出现"打点显示 col_map 正确、CLI 却逐格错位"，根源是被 import/in-process exec 污染了 module-level fails/warns 全局状态（脚本无 `if __name__=='__main__'` 守卫时尤其危险）；此后一律以纯终端 CLI + AST 编译双检为准，调试用环境变量开关且用完即删；
  ③ header 与 data row 必须共用同一切分函数（`_cells()`），一边剥首尾空 cell、另一边不剥会整体 off-by-one。
- 教训/建议：① 凡解析 markdown/CSV 表头做索引映射的工具，禁用"包含子串猜词"的裸 elif 链，改为主键前缀优先级显式排序 + 单调序守卫；② 校验器类工具必须**双向验证**——既能抓篡改坏值（corrupt fixtures），也须在真产物上全绿，二者缺一不可，否则必然出假阳或假阴；③ 收集器（fails/warns 列表）设计要考虑可重复执行与进程隔离，避免跨 exec 累积污染结果。

### EXP-C-005｜Ray 分布式打分汇总：作用域坑 / emit-first 设计 / 流式单趟 IV
- 日期：2026-08-06
- 任务类型：分类（分布式收口）｜业务域：信贷风控-贷中逾期（dz_ovd30_cllct）
- 关联模型：dz_ovd30_cllct_lgb_v1｜关联 session：runs/20260806-103945-dz_ovd30_cllct
- 背景：全量宽表 33M×236(≈13GB) ≫ 本地预算，设计为「集群侧打分+小体积聚合回传」——只把 {fuid,f_p_date,label,pred} 留在内存，metrics/buckets/monthly/psi/sens-IV/FICO 全部在 driver 内联 base64 打点进日志尾部拉取。
- 做法：score_and_summarize.py 分块 emit；多轮排障见 agent_memory §二~四：
  ① `iter_batches` 不接受 include_columns → 去掉该参数按批预测后裁列；
  ② carry_cols=("fuid","f_p_date") 走 data_utils load_and_split_data 透传；
  ③ **顶层类 StreamIvAudit 的方法用 np/pd 而 import 只在 main() 局部** → NameError: 'np' is not defined。根治=把 numpy/pandas 提到模块级唯一一次导入，不再下沉到函数内重复 import；
  ④ 敏感列 IV 审计不物化全列：P1/P99 定边界的 reservoir→lock 两阶段直方图累计（cap=400k、lock_minrows=1.2M），finalize 折 IV。
- 结论：核心四块(metrics/buckets/monthly/psi)任何后续失败前都已先行落盘 → 每次重跑都不浪费已完成部分；最终 PASS 8 SUCCEEDED 拿到全部 9 个 SUMMARY block。
- 关键结果：train/val/OOT AUC=0.6377/0.6274/0.6234 KS=0.2026/0.1877/0.1819；pred-PSI(nb20) train_vs_oot=0.00240 极稳；OOT Top-decile Lift 2.077；FICO LR(C=20) coef=1.049/intc=-1.462 仅在 train 拟合(SRP)，bscore≈[285,575]。
- cllct 泄漏核查：streamed IV top = fuser_cllct_contact_answer_rto_12m (IV=1.00121, ≥1.0 flag)——用户已裁决**保留入模不剔除**（R1=A 早警口径合法前瞻变量；仅声明不得用于终态归因）。
- 教训/建议：① 大数据集上收口评估优先「聚合内联 + emit-first」，比反复搬全量表高效一个量级；② 模块级公共类的命名空间与 main 局部 import 冲突是隐蔽雷区，import 一律置顶且全局唯一；③ 幂等报告生成脚本(build_report/build_deliverables/build_xlsx_and_fico)+可复现 summaries.json 使 re-run 不破坏产物一致性。
