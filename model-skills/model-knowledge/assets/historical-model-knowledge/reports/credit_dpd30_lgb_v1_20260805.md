# 模型档案 — credit_dpd30_lgb_v1（2026-08-05 重跑）

> 模型ID：credit_dpd30_lgb_v1（20260805-141943 session）｜ 算法：LightGBM ｜ 负责人：jensenliu

## 业务定义
- 业务线：信贷风控-贷后逾期
- 预测目标：是否发生滚动30天逾期3期（dpd30_3c），二分类
- 正样本定义：dpd30_3c=1（第3期逾期30天+）；训练段 bad rate 5.81%
- 数据源：test1/ka_df.feather（实际 Parquet 格式，local_file，547,966×259）

## 样本与切分
- 窗口内样本 448,223；有标签 418,762（正样本 24,751，5.91%）
- Train/Val = 2025-01~07 随机分层 7:3（seed=42）；OOT = 2025-08~10（时间外）
- train 196,823 / val(test档) 84,353 / oot 137,586（OOT 剔除标签缺失样本）

## 特征
- 209 个：列2~209 + ascore_fpd7_v3；不含内部模型分（xgb_5c/mtl_/fusion/ka_v 均排除）
- 哨兵值清洗 3 特征（-1/-2→NaN）；39 特征 PSI>0.1 漂移预警
- 单变量 IV 天花板低：仅 2 特征 IV≥0.1（tx_model_2_score 0.108 / ascore_fpd7_v3 0.106）

## 模型与超参
- LightGBM 默认表：lr=0.02, n_estimators=300, num_leaves=31, max_depth=6, min_child_samples=50,
  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, seed=42, 早停50(val)
- 不平衡：原分布训练（scale_pos_weight=1.0，正样本率 5.91%>5%，概率即真实违约率）
- best_iteration=300（未触发早停）

## 评估
| 档 | 样本量 | bad rate | AUC | KS |
|----|--------|----------|-----|-----|
| train | 196,823 | 5.81% | 0.7619 | 0.3758 |
| val | 84,353 | 5.81% | 0.6745 | 0.2512 |
| OOT | 137,586 | 6.11% | 0.6359 | 0.1949 |
| all | 418,762 | 5.91% | 0.7027 | 0.2858 |

OOT 十分桶：最高风险档 lift 1.98（覆盖 19.8% 坏样本），10 档单调，排序性成立。
特征重要性 Top5：score_80002 / tx_model_2_score / ascore_mob4_v6 / function1 / yzx_score

## FICO 标准分（Stage 5）
- 校准：logistic_prob = sigmoid(1.774955·ln(p/(1-p)) + 1.972376)，仅 train 拟合
- bscore = 400 − 35/ln2·ln(odds')；均值约 564；约 0.2% 极端样本 <400（概率饱和）
- 产物：`new-models/lgb-v1/fico/`（coef.json 生产 --apply 复用）

## 结论与风险
- OOT AUC 0.636 为数据集天花板（外部征信评分为主）；与昨日同口径持平（0.638）
- 39 特征 PSI>0.1（tx_model_2_score PSI=2.37），上线需按月监控漂移
- OOT 10月表现期不足（缺失 17.6%），补足后可复评
