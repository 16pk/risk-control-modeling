# 特征知识库

登记各业务域可复用的特征宽表与特征清单，供 data-cleaning / credit-data-analysis 选特征时检索。特征清单 csv 落 `feature-list/` 目录，新增特征表时在下表追加一行。

## 常用特征列表

| 分场景(sub domain) | 触发方式（trigger） | 特征表（feature table) | 可用特征清单(feature list) |
|------|---------|---------|-------|
| 贷前场景 | 贷前建模任务时（申请评分/授信审批/新客欺诈/新客准入） | 5 张外部三方特征表（互金/同盾/百融/百行洞侦/在网时长），特征清单见 feature-list/feature-list-loan-pre-v1.csv | feature-list/feature-list-loan-pre-v1.csv |
| 贷中场景 | 贷中建模任务时（贷中风险/还款逾期/催收/额度管理/复借） | 63 张自有行为特征表（还款/借款/逾期/电商/三方分/模型分等），特征清单见 feature-list/feature-list-loan-mid-v1.csv | feature-list/feature-list-loan-mid-v1.csv |
