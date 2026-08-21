# -*- coding: utf-8 -*-
"""实验范围交互确认器：先问算法，再依次问 4 个开关（样本选择/特征选择/对抗验证/Optuna）。

合并优先级：CLI 显式参数 > 交互询问 > 环境约束降级 > 默认值。
- 交互只在非 --auto-apply 且非 --resume(有现存矩阵) 的新规划路径执行；
- --auto-apply 跳过全部交互直接使用默认（保持现有语义）；
- EOFError（无 tty）/ 非法输入回退默认值并提示；
- 环境约束：无 OOT → 对抗验证不可做；开发池月份数 <2 → 样本选择无衍生方案可选（仅 full）。

返回 scope dict（落 planning_reasons 可追溯）：
  {"algos": [...], "sample_select": bool, "feat_select": bool,
   "adversarial": bool, "optuna": bool, "reasons": [str, ...]}
"""
from __future__ import annotations

from typing import Callable, List

# 默认实验范围（--auto-apply / EOF / 非法输入时的兜底值）
DEFAULT_SCOPE = {
    "algos": ["lgb", "xgb"],
    "sample_select": True,
    "feat_select": True,
    "adversarial": True,
    "optuna": True,
}

_ALGO_CHOICES = {"1": "lgb", "l": "lgb", "lgb": "lgb",
                 "2": "xgb", "x": "xgb", "xgb": "xgb",
                 "3": "both", "b": "both", "both": "both"}


def _ask_bool(question: str, default: bool, ask: Callable[[str], str],
              warn: str = "") -> bool:
    """问一个 y/n 开关；非法输入/EOF 回退默认值。耗时提醒放 question 尾部。"""
    hint = "y/n" + ("（回车默认 y）" if default else "（回车默认 n）")
    if warn:
        print(f"  ⚠ {warn}")
    try:
        raw = ask(f"[范围] {question} [{hint}]：").strip().lower()
    except EOFError:
        raw = ""
    if raw in ("y", "yes", "是", "1"):
        return True
    if raw in ("n", "no", "否", "0"):
        return False
    print(f"  （输入无法识别，按默认 {'做' if default else '不做'}）")
    return default


def resolve_scope(algos_cli: List[str] | None = None,
                  no_sample_select: bool = False,
                  no_feat_select: bool = False,
                  no_adversarial: bool = False,
                  no_tune: bool = False,
                  n_months: int = 0,
                  oot_available: bool = False,
                  auto: bool = False,
                  ask: Callable[[str], str] = input) -> dict:
    """合并 CLI 显式参数 + 交互询问 + 环境约束 → 最终实验范围。

    Args:
        algos_cli: --algos 显式值（None 表示未传，需询问）
        no_sample_select / no_feat_select / no_adversarial / no_tune:
            对应 --no-* CLI 开关，显式关闭
        n_months: 开发池有效月份数（<2 → 样本选择无衍生方案，降级 False）
        oot_available: 是否有 OOT（无 OOT → 对抗验证降级 False）
        auto: --auto-apply，跳过交互全默认
        ask: 交互输入函数（测试注入 fake）

    Returns:
        scope dict：algos / sample_select / feat_select / adversarial / optuna / reasons
    """
    scope = dict(DEFAULT_SCOPE)
    scope["reasons"] = []

    # 1) 算法：CLI 显式 > 交互询问 > 默认 both
    if algos_cli:
        scope["algos"] = [a.lower() for a in algos_cli if a.lower() in ("lgb", "xgb")]
        scope["reasons"].append("算法由 CLI --algos 显式指定: %s" % ", ".join(scope["algos"]))
    elif not auto:
        try:
            raw = ask("[范围] 本次实验使用哪个算法训练？\n"
                      "  1 = lgb / 2 = xgb / 3 = 两者都选（默认 3）\n"
                      "  输入 1/2/3 或 lgb/xgb/both：").strip().lower()
        except EOFError:
            raw = ""
        pick = _ALGO_CHOICES.get(raw)
        if pick is None:
            print("  （输入无法识别，默认两者都选）")
            pick = "both"
        scope["algos"] = ["lgb", "xgb"] if pick == "both" else [pick]
        scope["reasons"].append("算法由用户选择: %s" % ", ".join(scope["algos"]))
    else:
        scope["reasons"].append("算法默认 both（--auto-apply）: lgb, xgb")

    # 2) 样本选择：CLI 显式关闭 > 环境降级 > 询问
    if no_sample_select:
        scope["sample_select"] = False
        scope["reasons"].append("样本选择: 不做（CLI --no-sample-select）")
    elif n_months < 2:
        scope["sample_select"] = False
        scope["reasons"].append(
            "样本选择: 不做（开发池月份数 %d < 2，无 recent-N/时间加权衍生方案可选）" % n_months)
    elif not auto:
        scope["sample_select"] = _ask_bool(
            "是否做样本选择（recent-N 最近窗口 / 线性时间加权等衍生样本方案）？", True, ask)
        scope["reasons"].append("样本选择: %s" % ("做" if scope["sample_select"] else "不做"))
    else:
        scope["reasons"].append("样本选择: 做（--auto-apply 默认）")

    # 3) 特征选择：CLI 显式关闭 > 询问（无环境降级，重要性/IV-PSI 均可算）
    if no_feat_select:
        scope["feat_select"] = False
        scope["reasons"].append("特征选择: 不做（CLI --no-feat-select）")
    elif not auto:
        scope["feat_select"] = _ask_bool(
            "是否做特征选择（importance 95% 截断 / IV-PSI 直算等衍生特征方案）？", True, ask)
        scope["reasons"].append("特征选择: %s" % ("做" if scope["feat_select"] else "不做"))
    else:
        scope["reasons"].append("特征选择: 做（--auto-apply 默认）")

    # 4) 对抗验证：CLI 显式关闭 > 环境降级（无 OOT 不可做）> 询问（耗时提醒）
    if no_adversarial:
        scope["adversarial"] = False
        scope["reasons"].append("对抗验证: 不做（CLI --no-adversarial）")
    elif not oot_available:
        scope["adversarial"] = False
        scope["reasons"].append("对抗验证: 不做（无 OOT 样本，train-vs-oot 对抗分类器无法训练）")
    elif not auto:
        scope["adversarial"] = _ask_bool(
            "是否做对抗验证（train-vs-oot 对抗分类器，剔除分布差异最大样本/特征）？", True, ask,
            warn="对抗验证耗时较久（需训练额外对抗分类器）")
        scope["reasons"].append("对抗验证: %s" % ("做" if scope["adversarial"] else "不做"))
    else:
        scope["reasons"].append("对抗验证: 做（--auto-apply 默认）")

    # 5) Optuna：CLI 显式关闭 > 询问（耗时提醒）；不做时规则诊断一并跳过
    if no_tune:
        scope["optuna"] = False
        scope["reasons"].append("Optuna 调优: 不做（CLI --no-tune，规则诊断一并跳过）")
    elif not auto:
        scope["optuna"] = _ask_bool(
            "是否做 Optuna 调优（对每算法 winner 邻域搜索超参，默认 25 trials）？", True, ask,
            warn="Optuna 调优耗时较久（每算法多轮训练）")
        scope["reasons"].append("Optuna 调优: %s" % ("做" if scope["optuna"] else "不做"))
    else:
        scope["reasons"].append("Optuna 调优: 做（--auto-apply 默认）")

    return scope


def scope_summary(scope: dict) -> str:
    """单行摘要（矩阵规划段日志展示）。"""
    return ("algos=[%s] sample_select=%s feat_select=%s adversarial=%s optuna=%s" % (
        ", ".join(scope["algos"]),
        scope["sample_select"], scope["feat_select"],
        scope["adversarial"], scope["optuna"]))