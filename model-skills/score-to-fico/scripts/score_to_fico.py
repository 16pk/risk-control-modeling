#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score-to-fico: 概率分 → FICO 标准分转换（分类建模 pipeline 内可选 Stage 6）。

仅可在分类建模 pipeline 内调用（classification-model-development Stage 6），
消费上游 model-scoring（Stage 5）的打分结果（含 label + score 概率列，因
model-scoring 透传了所有非特征列），默认用全量样本 + Y 标签拟合 LR 校准参数
（coef/intc），再对全量打分结果转 FICO 标准分。拟合样本时间范围 / 拟合标签
由编排层在执行前与用户确认后作为参数传入（脚本内不做 input() 交互）。

两步转换:
  Step 1 — LR 校准: odds = ln(p/(1-p)); logistic_prob = sigmoid(coef * odds + intc)
  Step 2 — 标准分:  bscore = 400 - 35/ln2 * ln(logistic_prob / (1 - logistic_prob))  # 范围约 [400,780], 分高险低

用法示例(pipeline Stage 6 调起):
  python score_to_fico.py \
      --data <session_dir>/scoring/score_sample.parquet \
      --out-dir <session_dir>/fico \
      [--prob-col score] [--label-col label] \
      [--fit-label-col label] [--fit-date-range 20260101,20261231] \
      [--date-col f_p_date] [--uid-col fuid]

依赖: pandas / numpy / scikit-learn（LogisticRegression）
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---- 常量 ----
DEFAULT_BASE = 400          # 标准分基数（与 FICO 习惯一致, 分高险低）
DEFAULT_PDO_FACTOR = 35.0 / np.log(2.0)  # 每翻一倍 odds 扣 35 分
BSCORE_SANE_MIN, BSCORE_SANE_MAX = 400.0, 780.0  # 合理区间, 显著越界提示重新拟合
PROB_EPS = 1e-6             # 概率边界裁剪, 防止 log(0)


def calc_odds(pred_proba):
    p = np.clip(np.asarray(pred_proba, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p / (1 - p))


def logistic_prob(odds, coef, intc):
    z = coef * np.asarray(odds, dtype=float) + intc
    z = np.clip(z, -30, 30)  # 数值稳定
    return np.exp(z) / (1 + np.exp(z))


def calc_bscore(logistic_prob):
    """标准分: bscore = base - 35/ln2 * ln(p/(1-p)), 范围约 [400, 780]"""
    p = np.clip(np.asarray(logistic_prob, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return DEFAULT_BASE - DEFAULT_PDO_FACTOR * np.log(p / (1 - p))


def fit_calibration(data, prob_col, label_col):
    """在拟合集上拟合 LR 校准, 返回 coef / intc。仅用 y ∈ {0,1}（剔除标签缺失/未成熟样本）。"""
    valid = data[data[label_col].isin([0, 1])].copy()
    if len(valid) == 0:
        raise ValueError("[ERROR] 无有效标签样本 (y in {0,1}), 无法拟合校准")
    odds = calc_odds(valid[prob_col].values)
    lr = LogisticRegression(C=20)
    lr.fit(odds.reshape(-1, 1), valid[label_col].values)
    coef = float(lr.coef_[0][0])
    intc = float(lr.intercept_[0])
    pos_rate = float(valid[label_col].mean())
    print(f"[FIT] 拟合样本: {len(valid)} | 正样本率: {pos_rate:.4f}")
    print(f"[FIT] LR coef: {coef:.6f} | intercept: {intc:.6f}")
    if coef <= 0:
        print("[WARN] coef<=0: 概率与真实逾期方向相反或概率量级异常, 请检查概率列是否为违约概率")
    if abs(coef) > 8:
        print(f"[WARN] |coef|={abs(coef):.1f} 偏大: 概率分布可能过于集中(区分度低), 校准后 bscore 易越界, 请检查模型概率质量")
    return coef, intc


def apply_scores(data, prob_col, coef, intc):
    """对概率列做 校准 + 转分, 返回带过程列(odds/logistic_prob/bscore)的 DataFrame"""
    out = data.copy()
    out['odds'] = calc_odds(out[prob_col])
    out['logistic_prob'] = logistic_prob(out['odds'], coef, intc)
    out['bscore'] = calc_bscore(out['logistic_prob'])
    return out


def load_data(path, prob_col):
    """加载 CSV / parquet, 校验概率列存在。"""
    if str(path).endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if prob_col not in df.columns:
        raise ValueError(f"[ERROR] 数据缺失概率列: {prob_col}（可用 --prob-col 覆盖）")
    return df


def _to_yyyymmdd(value) -> str:
    """把日期值(YYYYMMDD 整数 / 'YYYY-MM-DD' / 'YYYYMMDD' 字符串)归一化为 8 位字符串。"""
    s = str(value).strip().replace('-', '')
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"[ERROR] 日期格式无法归一化为 YYYYMMDD: {value!r}")
    return s


def _parse_date_range(value):
    """解析拟合时间范围: 'start,end'(或两元素列表), 返回 (start, end) 8 位字符串。"""
    parts = [p.strip() for p in str(value).split(',')]
    if len(parts) != 2:
        raise ValueError(f"[ERROR] --fit-date-range 须为 'start,end': {value!r}")
    return _to_yyyymmdd(parts[0]), _to_yyyymmdd(parts[1])


def filter_fit_range(df, date_col, fit_date_range):
    """按日期列过滤拟合样本; 返回过滤后的 DataFrame。"""
    if not fit_date_range:
        return df
    if date_col not in df.columns:
        raise ValueError(f"[ERROR] --fit-date-range 需要日期列 {date_col!r}, 但数据中不存在（可用 --date-col 覆盖）")
    start, end = _parse_date_range(fit_date_range)
    dcol = df[date_col].astype(str).str.replace('-', '').str[:8]
    mask = (dcol >= start) & (dcol <= end)
    n_in = int(mask.sum())
    print(f"[FIT] 拟合时间范围 [{start}, {end}] 命中 {n_in}/{len(df)} 行")
    if n_in == 0:
        raise ValueError(f"[ERROR] 拟合时间范围 [{start}, {end}] 未命中任何样本")
    return df[mask]


def summarize(prob_col, coef, intc, fit_df=None):
    """组装拟合方案 summary（dict）: 校准参数 + bscore 分布 + 分位表。"""
    summ = {
        'method': 'LR_calibration + FICO_mapping',
        'formula': {
            'odds': 'ln(p/(1-p))',
            'logistic_prob': 'sigmoid(coef*odds+intc)',
            'bscore': '400 - 35/ln2 * ln(logistic_prob/(1-logistic_prob))',
            'range': '[400, 780] (分高险低)',
        },
        'params': {'coef': coef, 'intc': intc, 'C': 20, 'base': DEFAULT_BASE,
                   'factor': round(float(DEFAULT_PDO_FACTOR), 6)},
    }
    if fit_df is not None and 'bscore' in fit_df.columns:
        s = fit_df['bscore']
        summ['fit'] = {
            'n': int(len(fit_df)),
            'bscore_min': round(float(s.min()), 2),
            'bscore_max': round(float(s.max()), 2),
            'bscore_mean': round(float(s.mean()), 2),
        }
        # 分位表
        pcts = [10, 25, 50, 75, 90]
        summ['quantiles'] = {f'p{p}': round(float(np.percentile(s, p)), 2) for p in pcts}
    return summ


def write_summary(summary, json_path, md_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 拟合方案: {json_path}")
    if md_path:
        lines = ['# FICO 校准拟合方案（score-to-fico）', '']
        lines.append('**Step 1 — LR 校准**: `odds = ln(p/(1-p))` → `logistic_prob = sigmoid(coef*odds+intc)`（`LogisticRegression(C=20)`）')
        lines.append('')
        lines.append('**Step 2 — 标准分映射**: `bscore = 400 - 35/ln2 * ln(logistic_prob/(1-logistic_prob))`，范围约 **[400, 780]**，**分高险低**')
        lines.append('')
        lines.append('## 校准参数')
        p = summary['params']
        lines.append(f"- `coef` = {p['coef']:.6f} | `intc` = {p['intc']:.6f} | `C` = {p['C']} | `factor` = {p['factor']}")
        if 'fit' in summary:
            ft = summary['fit']
            lines.append(f"- 拟合集: n = {ft['n']} | bscore 范围 = [{ft['bscore_min']}, {ft['bscore_max']}] | 均值 = {ft['bscore_mean']}")
        if 'quantiles' in summary:
            q = summary['quantiles']
            lines.append(f"- bscore 分位: " + " | ".join(f"{k}={v}" for k, v in q.items()))
        lines.append('')
        lines.append('> 生产复用: 同一模型后续批次用 `--coef <coef.json>` 转分, 保证跨批次分数口径一致。')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"[SAVE] 拟合方案: {md_path}")


def check_sanity(result, prob_col):
    """bscore 越界检查 + 过程列非有限值检查"""
    bs = result['bscore']
    bad = int((~np.isfinite(bs)).sum())
    lo = float(bs[bs < BSCORE_SANE_MIN].count()) if len(bs) else 0
    hi = float(bs[bs > BSCORE_SANE_MAX].count()) if len(bs) else 0
    if bad:
        print(f"[WARN] bscore 非有限值 {bad} 个")
    if lo or hi:
        print(f"[WARN] bscore 越界 {int(lo)} 个(<{BSCORE_SANE_MIN}) / {int(hi)} 个(>{BSCORE_SANE_MAX}) — "
              f"正常区间 [{BSCORE_SANE_MIN}, {BSCORE_SANE_MAX}]; 显著越界说明 coef/intc 不适用当前分布, 需重新拟合")
    print(f"[INFO] {prob_col} → bscore 范围: [{float(bs.min()):.1f}, {float(bs.max()):.1f}] | 均值: {float(bs.mean()):.1f}")


def main():
    parser = argparse.ArgumentParser(description='概率分 → FICO 标准分转换（score-to-fico, pipeline 内可选 Stage 6）')
    parser.add_argument('--data', required=True,
                        help='model-scoring 打分结果 parquet（含 label + score 概率列）')
    parser.add_argument('--out-dir', required=True,
                        help='FICO 产物输出目录（建议 <session_dir>/fico）')
    parser.add_argument('--prob-col', type=str, default='score', help='概率列名(默认 score)')
    parser.add_argument('--label-col', type=str, default='label', help='标签列名(默认 label)')
    parser.add_argument('--fit-label-col', type=str, default=None,
                        help='拟合用标签列名(默认同 --label-col)')
    parser.add_argument('--fit-date-range', type=str, default=None,
                        help='拟合样本时间范围 "start,end"(YYYY-MM-DD/YYYYMMDD), 默认全量')
    parser.add_argument('--date-col', type=str, default='f_p_date', help='日期分区列名(用于 fit-date-range 过滤)')
    parser.add_argument('--uid-col', type=str, default='fuid', help='用户ID列名(透传)')
    args = parser.parse_args()

    df = load_data(args.data, args.prob_col)
    print(f"[FICO] 输入打分数据: {args.data} | shape={df.shape} | 列: {list(df.columns)}")

    label_col = args.fit_label_col or args.label_col
    if label_col not in df.columns:
        raise SystemExit(f"[ERROR] 数据缺失标签列: {label_col}（可用 --label-col / --fit-label-col 覆盖）")

    # 1) 拟合集: 默认全量, 支持按时间范围过滤(由编排层确认后传入)
    fit_df = filter_fit_range(df, args.date_col, args.fit_date_range)

    # 2) 拟合校准参数(仅用 y ∈ {0,1})
    coef, intc = fit_calibration(fit_df, args.prob_col, label_col)

    # 3) 对全量打分结果转分
    result = apply_scores(df, args.prob_col, coef, intc)
    check_sanity(result, args.prob_col)

    os.makedirs(args.out_dir, exist_ok=True)

    # 4) 保存校准参数
    coef_path = os.path.join(args.out_dir, 'coef.json')
    with open(coef_path, 'w', encoding='utf-8') as f:
        json.dump({'coef': coef, 'intc': intc}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 校准参数: {coef_path}")

    # 5) 输出转分结果: 全部输入列 + odds/logistic_prob/bscore
    out_cols = [c for c in df.columns] + ['odds', 'logistic_prob', 'bscore']
    out_path = os.path.join(args.out_dir, 'fico_predictions.parquet')
    result[out_cols].to_parquet(out_path, index=False)
    print(f"[SAVE] 转分结果: {out_path} | n={len(result)}")

    # 6) 拟合方案
    summ = summarize(args.prob_col, coef, intc, fit_df=result)
    summ['data'] = args.data
    summ['prob_col'] = args.prob_col
    summ['fit_label_col'] = label_col
    summ['fit_date_range'] = args.fit_date_range
    write_summary(summ, os.path.join(args.out_dir, 'fitting-summary.json'),
                  os.path.join(args.out_dir, 'fitting-summary.md'))

    print(f"[DONE] FICO 转换完成: {args.out_dir}")
    print(f"  - 校准参数: {coef_path}")
    print(f"  - 转分结果: {out_path}")
    print(f"  - 拟合方案: {os.path.join(args.out_dir, 'fitting-summary.md')}")


if __name__ == '__main__':
    main()
