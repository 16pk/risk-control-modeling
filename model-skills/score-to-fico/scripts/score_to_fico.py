#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风控模型分 → FICO 标准分转换（ModelEvo 专家 skill: score-to-fico）

两步转换:
  Step 1 — LR 校准: odds = ln(p/(1-p)); logistic_prob = sigmoid(coef * odds + intc)
  Step 2 — 标准分:  bscore = 400 - 35/ln2 * ln(logistic_prob / (1 - logistic_prob))  # 范围约 [400,780], 分高险低

三种模式:
  1. --fit      独立拟合: 输入含 概率列 + 标签列 的 CSV/parquet → coef.json + 打分文件 + 拟合方案
  2. --apply    独立转分: 输入 概率列 + 已保存 coef.json → 打分文件（无需标签, 批量/生产复用）
  3. --from-run pipeline 嵌入: 输入 new-models/{run}/predictions/{train,test,oot}_predictions.parquet
               → {run}/fico/ 下 coef.json + fico_{split}_predictions.parquet + fitting-summary.{json,md}

用法示例:
  # 拟合 + 转分（独立调用, 输入带标签训练集）
  python score_to_fico.py --fit --data train.csv --prob_col pred_proba --label_col y \
      --uid_col user_no --date_col pday --out result_score.parquet --coef_out coef.json

  # 仅转分（复用已保存校准参数, 无需标签）
  python score_to_fico.py --apply --data new.csv --prob_col pred_proba \
      --uid_col user_no --date_col pday --coef coef.json --out result_score.parquet

  # pipeline 嵌入（development Stage 5 调起, 消费 training 的 predictions 产物）
  python score_to_fico.py --from-run --run-dir new-models/lgb-v1

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
    """在训练集上拟合 LR 校准, 返回 coef / intc。仅用 y ∈ {0,1}（剔除标签缺失/未成熟样本）。"""
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


def load_data(path, prob_col, label_col=None, uid_col=None, date_col=None):
    """加载 CSV / parquet, 校验必需列（prob + label）；uid/date 为透传列, 缺失不报错"""
    if str(path).endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = [prob_col] + ([label_col] if label_col else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] 数据缺失必需列: {missing}")
    return df


def save_data(df, path, cols):
    """按扩展名保存 CSV / parquet, 只保留指定列"""
    out = df[cols].copy()
    if str(path).endswith('.parquet'):
        out.to_parquet(path, index=False)
    else:
        out.to_csv(path, index=False, sep='|')
    print(f"[SAVE] 输出: {path} | shape={out.shape} | 列: {list(out.columns)}")


def summarize(prob_col, coef, intc, fit_df=None, splits=None):
    """组装拟合方案 summary（dict）: 校准参数 + bscore 分布 + 分位表"""
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
    if splits:
        summ['splits'] = splits
    return summ


def write_summary(summary, json_path, md_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 拟合方案: {json_path}")
    if md_path:
        lines = ['# FICO 校准拟合方案（score-to-fico）', '']
        lines.append('**Step 1 — LR 校准**: `odds = ln(p/(1-p))` → `logistic_prob = sigmoid(coef*odds+intc)`（`LogisticRegression(C=20)`，拟合集为 train）')
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
        if 'splits' in summary:
            lines.append('')
            lines.append('## 三档转分')
            for sp, meta in summary['splits'].items():
                lines.append(f"- `{sp}`: n={meta['n']} | bscore 范围=[{meta['bscore_min']}, {meta['bscore_max']}] | 均值={meta['bscore_mean']}")
        lines.append('')
        lines.append('> 生产复用: 同一模型后续批次用 `--apply --coef <coef.json>` 转分, 保证跨批次分数口径一致。')
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
              f"正常区间 [{BSCORE_SANE_MIN}, {BSCORE_SANE_MAX}]; 显著越界说明 coef/intc 不适用当前分布, 需重新 --fit")
    print(f"[INFO] {prob_col} → bscore 范围: [{float(bs.min()):.1f}, {float(bs.max()):.1f}] | 均值: {float(bs.mean()):.1f}")


def _run_fit(args):
    df = load_data(args.data, args.prob_col, args.label_col, args.uid_col, args.date_col)
    coef, intc = fit_calibration(df, args.prob_col, args.label_col)
    result = apply_scores(df, args.prob_col, coef, intc)
    check_sanity(result, args.prob_col)
    # 保存校准参数
    with open(args.coef_out, 'w', encoding='utf-8') as f:
        json.dump({'coef': coef, 'intc': intc}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 校准参数: {args.coef_out}")
    # 保存打分（uid/date 透传, 存在才输出）
    cols = [c for c in [args.uid_col, args.date_col, args.prob_col, 'odds', 'logistic_prob', 'bscore'] if c and c in result.columns]
    save_data(result, args.out, cols)
    # 拟合方案
    if args.summary_out or args.summary_md:
        summ = summarize(args.prob_col, coef, intc, fit_df=result)
        write_summary(summ, args.summary_out, args.summary_md)


def _run_apply(args):
    if not os.path.exists(args.coef):
        raise SystemExit(f"[ERROR] 校准参数文件不存在: {args.coef}")
    with open(args.coef, 'r', encoding='utf-8') as f:
        params = json.load(f)
    coef, intc = params['coef'], params['intc']
    df = load_data(args.data, args.prob_col, None, args.uid_col, args.date_col)
    result = apply_scores(df, args.prob_col, coef, intc)
    check_sanity(result, args.prob_col)
    cols = [c for c in [args.uid_col, args.date_col, args.prob_col, 'odds', 'logistic_prob', 'bscore'] if c and c in result.columns]
    save_data(result, args.out, cols)


def _run_from_run(args):
    """pipeline 嵌入: 消费 new-models/{run}/predictions/*.parquet, 产出 {run}/fico/"""
    run_dir = args.run_dir
    pred_dir = os.path.join(run_dir, 'predictions')
    if not os.path.isdir(pred_dir):
        raise SystemExit(f"[ERROR] predictions 目录不存在: {pred_dir}（请确认已跑完 training 的 predictions 阶段）")
    fnames = {sp: os.path.join(pred_dir, f'{sp}_predictions.parquet') for sp in ('train', 'test', 'oot')}
    missing = [sp for sp, p in fnames.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[ERROR] 缺少预测文件: {missing}（score-to-fico 需要 train/test/oot 三档）")

    fico_dir = args.fico_dir or os.path.join(run_dir, 'fico')
    os.makedirs(fico_dir, exist_ok=True)

    # 1) train 拟合
    train = pd.read_parquet(fnames['train'])
    prob_col = args.prob_col or ('score' if 'score' in train.columns else None)
    label_col = args.label_col or ('label' if 'label' in train.columns else None)
    if not prob_col or not label_col:
        raise SystemExit(f"[ERROR] 预测文件中未找到概率列/标签列: {list(train.columns)}（可用 --prob-col/--label-col 覆盖）")
    coef, intc = fit_calibration(train, prob_col, label_col)

    # 2) 保存校准参数（生产 --apply 复用）
    coef_path = os.path.join(fico_dir, 'coef.json')
    with open(coef_path, 'w', encoding='utf-8') as f:
        json.dump({'coef': coef, 'intc': intc}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 校准参数: {coef_path}")

    # 3) 三档转分
    splits_meta = {}
    for sp, p in fnames.items():
        df = pd.read_parquet(p)
        res = apply_scores(df, prob_col, coef, intc)
        check_sanity(res, prob_col)
        # 输出列: id_cols(非 score/label/bucket 的列) + label + score + bucket + odds + logistic_prob + bscore
        id_cols = [c for c in df.columns if c not in (prob_col, label_col, 'bucket')]
        out_cols = id_cols + ([label_col] if label_col else []) + [prob_col, 'bucket', 'odds', 'logistic_prob', 'bscore']
        out_cols = [c for c in out_cols if c in res.columns]
        out_path = os.path.join(fico_dir, f'fico_{sp}_predictions.parquet')
        res[out_cols].to_parquet(out_path, index=False)
        print(f"[SAVE] {sp} 转分: {out_path} | n={len(res)}")
        splits_meta[sp] = {
            'n': int(len(res)),
            'bscore_min': round(float(res['bscore'].min()), 2),
            'bscore_max': round(float(res['bscore'].max()), 2),
            'bscore_mean': round(float(res['bscore'].mean()), 2),
            'file': os.path.basename(out_path),
        }

    # 3) 拟合方案
    summ = summarize(prob_col, coef, intc, fit_df=None, splits=splits_meta)
    summ['run_dir'] = run_dir
    summ['prob_col'] = prob_col
    summ['label_col'] = label_col
    write_summary(summ, os.path.join(fico_dir, 'fitting-summary.json'), os.path.join(fico_dir, 'fitting-summary.md'))

    # 4) 输出清单
    print(f"[DONE] FICO 转换完成: {fico_dir}")
    print(f"  - 校准参数: {os.path.join(fico_dir, 'coef.json')}")
    print(f"  - 拟合方案: {os.path.join(fico_dir, 'fitting-summary.md')}")


def main():
    parser = argparse.ArgumentParser(description='风控模型分 → FICO 标准分转换（score-to-fico）')
    parser.add_argument('--fit', action='store_true', help='拟合模式：输入带标签集, 产出 coef/intc + 转分 + 拟合方案')
    parser.add_argument('--apply', action='store_true', help='转分模式：输入概率 + 已保存 coef/intc, 产出标准分')
    parser.add_argument('--from-run', action='store_true', help='pipeline 嵌入：读 new-models/{run}/predictions/*.parquet, 产 {run}/fico/')
    # 通用
    parser.add_argument('--data', type=str, help='输入数据 CSV/parquet (fit/apply 模式)')
    parser.add_argument('--prob_col', type=str, help='概率列名 (fit/apply 默认 pred_proba; from-run 默认 score)')
    parser.add_argument('--label_col', type=str, help='标签列名 (fit 必填; from-run 默认 label)')
    parser.add_argument('--uid_col', type=str, default='fuid', help='用户ID列名')
    parser.add_argument('--date_col', type=str, default='f_p_date', help='日期分区列名')
    parser.add_argument('--out', type=str, default='result_score.parquet', help='打分输出路径 (fit/apply)')
    parser.add_argument('--coef_out', type=str, default='coef.json', help='校准参数输出 (fit)')
    parser.add_argument('--coef', type=str, default='coef.json', help='校准参数输入 (apply)')
    parser.add_argument('--summary_out', type=str, help='拟合方案 JSON 输出 (fit)')
    parser.add_argument('--summary_md', type=str, help='拟合方案 MD 输出 (fit)')
    # from-run
    parser.add_argument('--run-dir', type=str, help='run 目录 (from-run): 如 new-models/lgb-v1')
    parser.add_argument('--fico-dir', type=str, help='FICO 输出目录 (from-run, 默认 <run_dir>/fico)')
    args = parser.parse_args()

    n_modes = sum([args.fit, args.apply, args.from_run])
    if n_modes != 1:
        raise SystemExit("[ERROR] 必须三选一: --fit / --apply / --from-run")

    if args.from_run:
        if not args.run_dir:
            raise SystemExit("[ERROR] --from-run 模式需要 --run-dir")
        _run_from_run(args)
    elif args.fit:
        if not args.data or not args.label_col:
            raise SystemExit("[ERROR] --fit 模式需要 --data 和 --label_col")
        # summary 默认跟随 --out 所在目录, 避免误落 cwd
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if not args.summary_out:
            args.summary_out = os.path.join(out_dir, 'fitting-summary.json')
        if not args.summary_md:
            args.summary_md = os.path.join(out_dir, 'fitting-summary.md')
        _run_fit(args)
    else:
        if not args.data:
            raise SystemExit("[ERROR] --apply 模式需要 --data")
        _run_apply(args)


if __name__ == '__main__':
    main()
