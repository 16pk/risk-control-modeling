#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score-to-fico 冒烟测试：合成打分数据验证单一 pipeline 模式（Stage 6）。
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


def make_scoring(n=2000, seed=42, pos_rate=0.1):
    """合成 model-scoring 打分结果: 概率列与标签正相关, 含透传的 fuid/pday。"""
    rng = np.random.default_rng(seed)
    true_p = np.clip(rng.beta(2, 8, n) * 0.6, 0.01, 0.6)
    y = (rng.random(n) < true_p).astype(float)
    y[0] = np.nan  # 故意让少数标签缺失(模拟未成熟样本)
    df = pd.DataFrame({
        'fuid': [f'U{i:06d}' for i in range(n)],
        'f_p_date': 20260101 + (np.arange(n) % 30),
        'score': np.clip(true_p + rng.normal(0, 0.03, n), 0.001, 0.999),
        'label': y,
    })
    return df


@pytest.fixture(scope='module')
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _py():
    return sys.executable


def test_pipeline_mode(tmpdir):
    """单一 pipeline 模式: 消费打分结果, 拟合校准 + 转分, 产 coef.json + fico_predictions + summary"""
    data = make_scoring()
    in_path = os.path.join(tmpdir, 'scoring.parquet')
    data.to_parquet(in_path, index=False)
    out_dir = os.path.join(tmpdir, 'fico')

    r = subprocess.run(
        [_py(), SCRIPT, '--data', in_path, '--out-dir', out_dir],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    coef_path = os.path.join(out_dir, 'coef.json')
    assert os.path.exists(coef_path)
    params = json.load(open(coef_path))
    assert params['coef'] > 0, '违约概率越高分数应越低, coef 应为正'

    res = pd.read_parquet(os.path.join(out_dir, 'fico_predictions.parquet'))
    assert {'score', 'label', 'bscore', 'odds', 'logistic_prob'}.issubset(res.columns)
    assert res['bscore'].between(400, 780).mean() > 0.9, 'bscore 应基本落在 [400,780]'

    summ = json.load(open(os.path.join(out_dir, 'fitting-summary.json')))
    assert 'fit' in summ and 'bscore_mean' in summ['fit']
    assert 'splits' not in summ, '新格式应为单一全量样本, 无 splits'
    assert os.path.exists(os.path.join(out_dir, 'fitting-summary.md'))


def test_fit_date_range(tmpdir):
    """--fit-date-range 仅影响拟合样本, 转分仍作用于全量"""
    data = make_scoring()
    in_path = os.path.join(tmpdir, 'scoring.parquet')
    data.to_parquet(in_path, index=False)
    out_dir = os.path.join(tmpdir, 'fico')

    r = subprocess.run(
        [_py(), SCRIPT, '--data', in_path, '--out-dir', out_dir,
         '--fit-date-range', '20260101,20260115'],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    # 转分结果仍是全量
    res = pd.read_parquet(os.path.join(out_dir, 'fico_predictions.parquet'))
    assert len(res) == len(data)


def test_missing_label_error(tmpdir):
    """缺标签列时报错"""
    data = make_scoring().drop(columns=['label'])
    in_path = os.path.join(tmpdir, 'scoring.parquet')
    data.to_parquet(in_path, index=False)
    out_dir = os.path.join(tmpdir, 'fico')

    r = subprocess.run(
        [_py(), SCRIPT, '--data', in_path, '--out-dir', out_dir],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert 'label' in r.stderr


if __name__ == '__main__':
    sys.exit(pytest.main([os.path.dirname(__file__), '-v']))
