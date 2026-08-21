# -*- coding: utf-8 -*-
"""plan_scope.py：实验范围交互确认器（算法三选一 + 4 开关 + 兜底/降级）。"""
import plan_scope as ps


def _fake_ask(answers):
    """按序消费 answers 的 fake ask；耗尽后抛 EOFError（模拟无 tty 兜底）。"""
    it = iter(answers)

    def ask(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return ask


def test_default_scope_no_ask():
    scope = ps.resolve_scope(auto=True, oot_available=True, n_months=6)
    assert scope["algos"] == ["lgb", "xgb"]
    assert scope["sample_select"] and scope["feat_select"]
    assert scope["adversarial"] and scope["optuna"]
    assert any("--auto-apply" in r for r in scope["reasons"])


def test_algo_choices():
    for raw, expect in [("1", ["lgb"]), ("2", ["xgb"]), ("3", ["lgb", "xgb"]),
                        ("lgb", ["lgb"]), ("xgb", ["xgb"]), ("both", ["lgb", "xgb"]),
                        ("l", ["lgb"]), ("x", ["xgb"]), ("b", ["lgb", "xgb"])]:
        ask = _fake_ask([raw, "y", "y", "y", "y"])
        scope = ps.resolve_scope(ask=ask, oot_available=True, n_months=6)
        assert scope["algos"] == expect, raw
    # 非法输入 → 默认 both
    ask = _fake_ask(["??", "y", "y", "y", "y"])
    scope = ps.resolve_scope(ask=ask, oot_available=True, n_months=6)
    assert scope["algos"] == ["lgb", "xgb"]


def test_switches_y_n():
    ask = _fake_ask(["3", "n", "n", "n", "n"])
    scope = ps.resolve_scope(ask=ask, oot_available=True, n_months=6)
    assert scope["algos"] == ["lgb", "xgb"]
    assert scope["sample_select"] is False
    assert scope["feat_select"] is False
    assert scope["adversarial"] is False
    assert scope["optuna"] is False
    assert any("不做" in r and "样本" in r for r in scope["reasons"])
    assert any("不做" in r and "Optuna" in r for r in scope["reasons"])


def test_eof_falls_back_defaults():
    # 全部 EOF → 各开关默认；算法默认 both
    scope = ps.resolve_scope(ask=_fake_ask([]), oot_available=True, n_months=6)
    assert scope["algos"] == ["lgb", "xgb"]
    assert scope["sample_select"] and scope["feat_select"]
    assert scope["adversarial"] and scope["optuna"]


def test_cli_overrides_ask():
    # CLI 显式关闭优先，即使 fake ask 回答 y
    scope = ps.resolve_scope(ask=_fake_ask(["3", "y", "y", "y", "y"]),
                             no_sample_select=True, no_feat_select=True,
                             no_adversarial=True, no_tune=True,
                             oot_available=True, n_months=6)
    assert scope["sample_select"] is False
    assert scope["feat_select"] is False
    assert scope["adversarial"] is False
    assert scope["optuna"] is False
    assert any("--no-sample-select" in r for r in scope["reasons"])


def test_cli_algos_skips_ask():
    # CLI 显式算法 → 不再问算法
    scope = ps.resolve_scope(algos_cli=["xgb"], ask=_fake_ask(["y", "y", "y", "y"]),
                             oot_available=True, n_months=6)
    assert scope["algos"] == ["xgb"]
    assert any("--algos" in r for r in scope["reasons"])


def test_no_oot_forces_adversarial_false():
    scope = ps.resolve_scope(ask=_fake_ask(["3", "y", "y", "y", "y"]),
                             oot_available=False, n_months=6)
    assert scope["adversarial"] is False
    assert any("无 OOT" in r for r in scope["reasons"])


def test_few_months_forces_sample_select_false():
    # 月份数 <2：样本选择直接 False，但其余开关仍可问
    ask = _fake_ask(["3", "y", "y", "y"])
    scope = ps.resolve_scope(ask=ask, oot_available=True, n_months=1)
    assert scope["sample_select"] is False
    assert scope["feat_select"] and scope["adversarial"] and scope["optuna"]
    assert any("月份数" in r for r in scope["reasons"])


def test_scope_summary():
    scope = {"algos": ["lgb"], "sample_select": False, "feat_select": True,
             "adversarial": False, "optuna": True}
    s = ps.scope_summary(scope)
    assert "algos=[lgb]" in s and "optuna=True" in s