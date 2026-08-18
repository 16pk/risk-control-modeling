# -*- coding: utf-8 -*-
"""plan_matrix.py：矩阵规划/组数自决/断点续跑。"""
import json
import os

import pytest

import plan_matrix as pm


def _sample_plans():
    return [
        {"name": "full", "n_months": None, "reason": "r1"},
        {"name": "recent4", "n_months": 4, "reason": "r2"},
        {"name": "timeweight", "n_months": None, "reason": "r3"},
    ]


def test_exp_id():
    assert pm.exp_id("lgb", "full", "all", "v1") == "lgb-full-all-v1"


def test_build_matrix_structure():
    specs = pm.build_matrix(["lgb", "xgb"], _sample_plans(), oot_available=True)
    ids = [s["id"] for s in specs]
    # 波1：3 样本方案 × 2 算法 = 6 all 格
    assert "lgb-full-all-v1" in ids
    assert "lgb-recent4-all-v1" in ids
    assert "lgb-timeweight-all-v1" in ids
    assert "xgb-full-all-v1" in ids
    # 波2：importance（3×2）+ iv-psi（3×2）= 12 格
    assert "lgb-full-importance-v1" in ids
    assert "lgb-full-iv-psi-v1" in ids
    # 对抗格仅 lgb
    assert "lgb-adversarial-adversarial-v1" in ids
    assert not any(s["algo"] == "xgb" and s["wave"] == 3 for s in specs)


def test_importance_dependency():
    specs = pm.build_matrix(["lgb"], _sample_plans(), oot_available=True)
    imp = [s for s in specs if s["feat_scheme"] == "importance"]
    for s in imp:
        assert s["depends_on"] == pm.exp_id(s["algo"], s["sample_scheme"], "all", "v1")


def test_no_oot_skips_iv_psi_and_adversarial():
    specs = pm.build_matrix(["lgb"], _sample_plans(), oot_available=False)
    assert not any(s["feat_scheme"] == "iv-psi" for s in specs)
    assert not any(s["wave"] == 3 for s in specs)


def test_budget_limit():
    with pytest.raises(ValueError):
        pm.build_matrix(["lgb"], _sample_plans(), oot_available=True, max_experiments=2)


def test_state_save_load(tmp_path):
    specs = pm.build_matrix(["lgb"], [{"name": "full", "n_months": None, "reason": "r"}],
                            oot_available=True)
    plan_json = os.path.join(str(tmp_path), "matrix-plan.json")
    pm.save_state(plan_json, specs, ["reason1"])
    assert os.path.exists(os.path.join(str(tmp_path), "matrix-plan.md"))
    loaded = pm.load_state(plan_json)
    assert [s["id"] for s in loaded] == [s["id"] for s in specs]
    # 状态更新
    pm.update_spec(specs, "lgb-full-all-v1", status="done")
    assert pm.get_spec(specs, "lgb-full-all-v1")["status"] == "done"


def test_update_spec_missing():
    specs = pm.build_matrix(["lgb"], [{"name": "full", "n_months": None, "reason": "r"}],
                            oot_available=True)
    with pytest.raises(KeyError):
        pm.update_spec(specs, "nope", status="done")