#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score-to-fico 冒烟测试：合成数据验证 --fit / --apply / --from-run 三模式。
运行: python -m pytest tests/ -v   （需 pandas / numpy / scikit-learn / pyarrow）
"""
import os
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'score_to_fico.py')
SCRIPT = os.path.abspath(SCRIPT)


def make_synthetic(n=2000, seed=42, pos_rate=0.1):
    """合成数据: 概率列与标签正相关（真实逾期概率越高越好）"""
    rng = np.random.default_rng(seed)
    # 模拟真实风控违约概率分布（均值 ~12%, 范围 ~[0.01, 0.6]）
    true_p = np.clip(rng.beta(2, 8, n) * 0.6, 0.01, 0.6)
    y = (rng.random(n) < true_p).astype(float)
    # 故意让少数标签缺失（模拟 OOT 未成熟）
    y[0] = np.nan
    df = pd.DataFrame({
        'user_no': [f'U{i:06d}' for i in range(n)],
        'pday': 20260101 + (np.arange(n) % 30),
        'pred_proba': np.clip(true_p + rng.normal(0, 0.03, n), 0.001, 0.999),
        'label': y,
    })
    return df


@pytest.fixture(scope='module')
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _py():
    return sys.executable


def test_fit_mode(tmpdir):
    """--fit: 产 coef.json + 打分 parquet + fitting-summary"""
    data = make_synthetic()
    in_path = os.path.join(tmpdir, 'train.csv')
    data.to_csv(in_path, index=False)
    out = os.path.join(tmpdir, 'fit_out.parquet')
    coef_out = os.path.join(tmpdir, 'coef.json')
    summ = os.path.join(tmpdir, 'summary.json')

    r = subprocess.run(
        [_py(), SCRIPT, '--fit', '--data', in_path, '--prob_col', 'pred_proba',
         '--label_col', 'label', '--uid_col', 'user_no', '--date_col', 'pday',
         '--out', out, '--coef_out', coef_out,
         '--summary_out', summ, '--summary_md', os.path.join(tmpdir, 'summary.md')],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(coef_out)
    params = json.load(open(coef_out))
    assert params['coef'] > 0, '违约概率越高分数应越低, coef 应为正'
    assert os.path.exists(out)
    res = pd.read_parquet(out)
    assert 'bscore' in res.columns
    assert res['bscore'].between(400, 780).mean() > 0.9, 'bscore 应基本落在 [400,780]'
    assert os.path.exists(summ)
    s = json.load(open(summ))
    assert 'params' in s and 'quantiles' in s


def test_apply_mode(tmpdir):
    """--apply: 复用 coef 对无标签新样本转分"""
    data = make_synthetic()
    in_path = os.path.join(tmpdir, 'train.csv')
    data.to_csv(in_path, index=False)
    coef_out = os.path.join(tmpdir, 'coef.json')

    subprocess.run(
        [_py(), SCRIPT, '--fit', '--data', in_path, '--prob_col', 'pred_proba',
         '--label_col', 'label', '--uid_col', 'user_no', '--date_col', 'pday',
         '--out', os.path.join(tmpdir, 'fit_out.parquet'), '--coef_out', coef_out,
         '--summary_out', os.path.join(tmpdir, 's.json')],
        capture_output=True, text=True, check=True)

    # 新样本去掉标签
    new = data.drop(columns=['label']).head(100)
    new_path = os.path.join(tmpdir, 'new.csv')
    new.to_csv(new_path, index=False)
    out = os.path.join(tmpdir, 'apply_out.parquet')

    r = subprocess.run(
        [_py(), SCRIPT, '--apply', '--data', new_path, '--prob_col', 'pred_proba',
         '--uid_col', 'user_no', '--date_col', 'pday', '--coef', coef_out, '--out', out],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = pd.read_parquet(out)
    assert len(res) == 100 and 'bscore' in res.columns


def test_from_run_mode(tmpdir):
    """--from-run: 模拟 training 的 predictions 目录结构"""
    data = make_synthetic()
    run_dir = os.path.join(tmpdir, 'lgb-v1')
    pred_dir = os.path.join(run_dir, 'predictions')
    os.makedirs(pred_dir)
    for sp, n in [('train', 1200), ('test', 400), ('oot', 400)]:
        df = make_synthetic(n=n, seed=hash(sp) % 1000)
        df.rename(columns={'pred_proba': 'score'}, inplace=True)
        df['bucket'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
        df[['user_no', 'pday', 'score', 'label', 'bucket']].to_parquet(
            os.path.join(pred_dir, f'{sp}_predictions.parquet'), index=False)

    r = subprocess.run(
        [_py(), SCRIPT, '--from-run', '--run-dir', run_dir],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    fico_dir = os.path.join(run_dir, 'fico')
    assert os.path.exists(os.path.join(fico_dir, 'coef.json'))
    for sp in ('train', 'test', 'oot'):
        p = os.path.join(fico_dir, f'fico_{sp}_predictions.parquet')
        assert os.path.exists(p), f'缺 {sp} 转分产物'
        res = pd.read_parquet(p)
        assert {'score', 'label', 'bscore', 'odds', 'logistic_prob', 'bucket'}.issubset(res.columns)
    assert os.path.exists(os.path.join(fico_dir, 'fitting-summary.md'))
    summ = json.load(open(os.path.join(fico_dir, 'fitting-summary.json')))
    assert set(summ['splits'].keys()) == {'train', 'test', 'oot'}


if __name__ == '__main__':
    sys.exit(pytest.main([os.path.dirname(__file__), '-v']))
