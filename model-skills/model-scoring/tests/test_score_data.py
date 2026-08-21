#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model-scoring 冒烟测试:
- 单元: read_model_meta / infer_algo / resolve_model_dir(不依赖重依赖)
- 端到端: xgb 定版模型打分(需 xgboost, 缺失则 skip), 验证特征重排对齐、
  透传非特征列、缺失特征报错。
运行: python -m pytest tests/ -q
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'score_data.py'))
SCRIPTS_DIR = os.path.dirname(SCRIPT)
sys.path.insert(0, SCRIPTS_DIR)

import score_data  # noqa: E402


def _py():
    return sys.executable


@pytest.fixture()
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# 单元测试(无重依赖)
# ---------------------------------------------------------------------------
def test_resolve_model_dir_accepts_dir_and_file(tmpdir):
    model_dir = os.path.join(tmpdir, 'model')
    os.makedirs(model_dir)
    (lambda p: open(p, 'w').close())(os.path.join(model_dir, 'model.json'))
    # 传目录
    assert score_data.resolve_model_dir(model_dir) == __import__('pathlib').Path(model_dir)
    # 传文件 → 取其父目录
    assert score_data.resolve_model_dir(os.path.join(model_dir, 'model.json')).name == 'model'


def test_infer_algo(tmpdir):
    model_dir = os.path.join(tmpdir, 'model')
    os.makedirs(model_dir)
    # xgb: model.json 存在
    open(os.path.join(model_dir, 'model.json'), 'w').close()
    assert score_data.infer_algo(__import__('pathlib').Path(model_dir), {}, None) == 'xgb'

    # v2.3: lgb/xgb 转正产物(model.pkl + meta.algo)
    os.remove(os.path.join(model_dir, 'model.json'))
    open(os.path.join(model_dir, 'model.pkl'), 'wb').close()
    assert score_data.infer_algo(__import__('pathlib').Path(model_dir), {'algo': 'lgb'}, None) == 'lgb'
    assert score_data.infer_algo(__import__('pathlib').Path(model_dir), {'algo': 'xgb'}, None) == 'xgb'

    # 显式 --algo 覆盖
    assert score_data.infer_algo(__import__('pathlib').Path(model_dir), {'algo': 'lgb'}, 'xgb') == 'xgb'


def test_infer_algo_pkl_missing_algo_error(tmpdir):
    """model.pkl 但 meta 缺 algo → 报错提示。"""
    model_dir = os.path.join(tmpdir, 'model')
    os.makedirs(model_dir)
    open(os.path.join(model_dir, 'model.pkl'), 'wb').close()
    with pytest.raises(SystemExit) as ei:
        score_data.infer_algo(__import__('pathlib').Path(model_dir), {}, None)
    assert '--algo' in str(ei.value)


def test_read_model_meta(tmpdir):
    model_dir = os.path.join(tmpdir, 'model')
    os.makedirs(model_dir)
    meta = {'algo': 'xgb', 'feature_names': ['f0', 'f1']}
    with open(os.path.join(model_dir, 'model_meta.json'), 'w') as f:
        json.dump(meta, f)
    assert score_data.read_model_meta(__import__('pathlib').Path(model_dir))['feature_names'] == ['f0', 'f1']


# ---------------------------------------------------------------------------
# xgb 端到端(需 xgboost)
# ---------------------------------------------------------------------------
def _make_xgb_model(model_dir, feature_names=('f0', 'f1', 'f2')):
    """训练一个微型 xgb booster 落 model.json + model_meta.json, 返回 model_dir。"""
    import xgboost as xgb

    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(feature_names)))
    y = (X[:, 0] + rng.normal(0, 0.3, 200) > 0).astype(int)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(feature_names))
    booster = xgb.train({'objective': 'binary:logistic', 'max_depth': 2},
                        dtrain, num_boost_round=4)
    booster.save_model(os.path.join(model_dir, 'model.json'))
    with open(os.path.join(model_dir, 'model_meta.json'), 'w') as f:
        json.dump({'feature_names': list(feature_names)}, f)
    return model_dir


def _make_data(path, feature_names=('f0', 'f1', 'f2')):
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame({
        'fuid': [f'U{i:05d}' for i in range(n)],
        'f_p_date': 20260101 + (np.arange(n) % 20),
        'label': rng.integers(0, 2, n),
    })
    for f in feature_names:
        df[f] = rng.normal(size=n)
    df.to_parquet(path, index=False)
    return df


def test_xgb_scoring_end_to_end(tmpdir):
    xgb = pytest.importorskip('xgboost')
    feature_names = ('f0', 'f1', 'f2')
    model_dir = _make_xgb_model(os.path.join(tmpdir, 'model'), feature_names)
    data_path = os.path.join(tmpdir, 'sample.parquet')
    src = _make_data(data_path, feature_names)
    out_path = os.path.join(tmpdir, 'scoring', 'score_sample.parquet')

    r = subprocess.run(
        [_py(), SCRIPT, '--model-path', model_dir, '--data', data_path, '--out', out_path],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    res = pd.read_parquet(out_path)
    # 输出 = 非特征列 + score, 不含原特征列
    assert 'score' in res.columns
    assert {'fuid', 'f_p_date', 'label'}.issubset(res.columns)
    for f in feature_names:
        assert f not in res.columns, '特征列不应透传'
    assert len(res) == len(src)
    # score 为概率, 落在 (0,1)
    assert res['score'].between(0, 1).all()
    # 行序保持(透传非特征列与 score 对齐)
    assert list(res['fuid']) == list(src['fuid'])


def test_missing_feature_error(tmpdir):
    pytest.importorskip('xgboost')
    feature_names = ('f0', 'f1', 'f2')
    model_dir = _make_xgb_model(os.path.join(tmpdir, 'model'), feature_names)
    # 数据缺 f2
    data_path = os.path.join(tmpdir, 'sample.parquet')
    _make_data(data_path, ('f0', 'f1'))
    out_path = os.path.join(tmpdir, 'out.parquet')

    r = subprocess.run(
        [_py(), SCRIPT, '--model-path', model_dir, '--data', data_path, '--out', out_path],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert '缺失' in r.stderr and 'f2' in r.stderr


# ---------------------------------------------------------------------------
# v2.3: lgb/xgb pkl(joblib) 打分(experiments 转正产物)
# ---------------------------------------------------------------------------
def _make_pkl_model(model_dir, algo, feature_names=('f0', 'f1', 'f2')):
    """训练微型 sklearn 分类器并 joblib.dump 落 model.pkl(experiments 转正形态)。"""
    import joblib

    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(feature_names)))
    y = (X[:, 0] + rng.normal(0, 0.3, 200) > 0).astype(int)
    if algo == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=20, learning_rate=0.1, verbosity=-1)
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=20, max_depth=2, verbosity=0)
    m.fit(pd.DataFrame(X, columns=feature_names), y)
    joblib.dump(m, os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(model_dir, 'model_meta.json'), 'w') as f:
        json.dump({'algo': algo, 'feature_names': list(feature_names)}, f)
    return m


@pytest.mark.parametrize('algo', ['lgb', 'xgb'])
def test_pkl_scoring_end_to_end(tmpdir, algo):
    pytest.importorskip(algo)
    feature_names = ('f0', 'f1', 'f2')
    model_dir = _make_pkl_model(os.path.join(tmpdir, f'model-{algo}'), algo, feature_names)
    data_path = os.path.join(tmpdir, 'sample.parquet')
    src = _make_data(data_path, feature_names)
    out_path = os.path.join(tmpdir, 'scoring', 'score_sample.parquet')

    r = subprocess.run(
        [_py(), SCRIPT, '--model-path', model_dir, '--data', data_path, '--out', out_path],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    res = pd.read_parquet(out_path)
    assert 'score' in res.columns
    assert {'fuid', 'f_p_date', 'label'}.issubset(res.columns)
    for f in feature_names:
        assert f not in res.columns
    assert len(res) == len(src)
    assert res['score'].between(0, 1).all()

    # 与直接 predict_proba 违约列一致
    import joblib
    m = joblib.load(os.path.join(model_dir, 'model.pkl'))
    X = src[list(feature_names)]
    expected = np.asarray(m.predict_proba(X))[:, 1]
    assert np.allclose(res['score'].values, expected, atol=1e-9)


if __name__ == '__main__':
    sys.exit(pytest.main([os.path.dirname(__file__), '-v']))
