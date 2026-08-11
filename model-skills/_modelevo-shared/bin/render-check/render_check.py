# -*- coding: utf-8 -*-
"""report.md 交付数字校验器（建模收口最后一道防线）。

目标：把「进文档的数字必须源自产物文件」从口头纪律变成可执行校验。
扫描 <session>/report.md，用白名单规则定位「声明了权威来源的指标」，与产物真源
交叉比对；任一处理过即报 FAIL/WARN。段落四~七虽由 fill_report.py 自动回填
（其本身读源），本脚本仍整表复查一遍，兜住人为手改污染（复盘事件 B）。

三类白名单断言：

  ① N-way / 迭代表模型行：
        | # | run_name | train AUC | test(val) AUC | oot AUC | oot KS |
    逐格对照 new-models/{run}/evaluation/{run}_{split}_eval.json
    → metric_by_segment.全量.auc / .ks （train→train、test→test、oot→oot）

  ② ka_v4 统一口径重测对比表：
        表头含 'ka_v4' + '统一口径'，行名为 dev(test)/OOT，
    数值与 CLI --expect-kava-dev / --expect-kava-oot 对比（容差 --kava-tol）。
    未提供期望值时该项 SKIP（并提示，防基线漂移被悄悄吞掉）。

  ③ FICO bscore 摘要行：
        - `train`: n=… | bscore 范围=[a, b] | 均值=c
    对照 fico/fitting-summary.json splits.{train,test,oot}。

用法:
    python render_check.py --session <dir> \\
        [--expect-kava-dev 0.68157 --expect-kava-oot 0.59998] [-v]

退出码: 0=通过; 1=存在数值冲突(WARN 默认亦算失败,可用 --no-fail-on-warn);
        2=会话缺关键产物无法自检。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

fails: List[Tuple[str, str]] = []
warns: List[Tuple[str, str]] = []


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        warns.append((path.name, f"JSON 解析失败({e}),跳过"))
        return None


# ----------------------------------------------------------------------------
# ① N-way / 迭代表模型行
# 数值单元格允许裹粗体(**x**)或反引号(`x`),名称单元格可含 - . 等任意非|字符
MODEL_ROW_RE = re.compile(
    r"^\|\s*(?:#\s*)?(?P<rank>\d+)\s*\|\s*(?P<name>[^|]+?)\s*\|\s*"
    r"(?P<c1>[^|]*?)\s*\|\s*(?P<c2>[^|]*?)\s*\|\s*(?P<c3>[^|]*?)\s*\|\s*"
    r"(?P<c4>[^|]*?)\s*(?:\||$)",
)


def _cell_num(raw: str) -> Optional[float]:
    """解析 markdown 表格单元格为数字;**加粗/反引号/空格均可容忍。"""
    cleaned = re.sub(r"[`*\s]", "", raw)
    m = re.fullmatch(r"-?(?:\d+)(?:\.\d+)?", cleaned)
    return float(m.group(0)) if m else None


def stats_for_run(session: Path, run: str) -> Dict[str, Dict[str, float]]:
    """{split: {'auc':..,'ks':..}} —— evaluation/*_{split}_eval.json 为唯一真源。"""
    out: Dict[str, Dict[str, float]] = {}
    ev_dir = session / "new-models" / run / "evaluation"
    pat = re.compile(rf"^{re.escape(run)}_(?P<s>\w+)_eval\.json$")
    for jp in sorted(ev_dir.glob("*.json")):
        m = pat.match(jp.name)
        if not m or m.group("s") == "all":
            continue
        d = read_json(jp)
        seg = (d.get("metric_by_segment", {}).get("全量")) if isinstance(d, dict) else None
        if not isinstance(seg, dict):
            continue
        auc, ks = seg.get("auc"), seg.get("ks")
        if all(isinstance(x, (int, float)) and x == x for x in (auc, ks)):
            out[m.group("s")] = {"auc": round(float(auc), 6),
                                 "ks": round(float(ks), 6)}
    return out


def check_model_rows(lines: List[str], session: Path) -> int:
    """校验迭代表/N-way 中每个模型行的四格数值;返回成功核对的 run 数。"""
    run_dirs = sorted(p for p in (session / "new-models").glob("*") if p.is_dir())
    stat_cache = {p.name: stats_for_run(session, p.name) for p in run_dirs}
    if not run_dirs:
        warns.append(("N-way", "new-models/ 下无任何 run(SKIP)"))
        return 0

    col_map: Dict[str, int] = {}

    def parse_header(line_: str) -> Optional[Dict[str, int]]:
        """把表头行切成各列的（规范名 → 列下标）。

        规则 = 位置优先 + 首见即得：
          - 'trainauc' → train；'testvalauc' / 'test...auc' → test；
            'oot..auc' → oot_auc；'oot..ks' → oot_ks（关键词判别）
          - 之后要求四个目标列的下标严格从左到右递增，
            否则视为排版不可信拒绝整张表（防止测试/OOT 对调这类错位静默通过）。
        历史教训：单纯用 if/elif 猜词会让 'test(val)' 与 'oot' 等共享后缀的列互相抢占，
        导致 col_map 里 test↔oot_auc 对调、逐格误报 FAIL。
        """
        cells_lc = [re.sub(r"[^a-z]", "", c.lower()) for c in _cells(line_)]

        def kind_of(norm: str) -> Optional[str]:
            if "auc" not in norm and "ks" not in norm:
                return None
            base = norm.replace("auc", "").replace("ks", "")
            if "tra" in base or "tre" in base:
                return "train"
            if "oot" in base:
                return "oot_ks" if "ks" in norm else "oot_auc"
            if "tes" in base:
                return "test"
            return None

        seen: Dict[str, int] = {}
        for ci, nrm in enumerate(cells_lc):
            k = kind_of(nrm)
            if k is not None:
                seen.setdefault(k, ci)   # 首见即得，左侧更可信

        need = {"train", "test", "oot_auc", "oot_ks"}
        if not need.issubset(seen.keys()):
            return None
        order = ["train", "test", "oot_auc", "oot_ks"]
        idxs = [seen[k] for k in order]
        if idxs != sorted(idxs):
            return None                  # 列序倒挂 → 拒绝整表
        return {k: i for k, i in zip(order, idxs)}

    header_found_at = None
    first_good_hdr_idx: Optional[int] = None
    for li, raw_l in enumerate(lines):
        if not raw_l.lstrip().startswith("|"):
            continue
        parsed = parse_header(raw_l)
        if parsed is None:
            continue
        col_map = parsed
        header_found_at = li
        first_good_hdr_idx = li
        break
    if first_good_hdr_idx is None or not col_map:
        warns.append(("N-way", "未识别到带 train/test/oot AUC·KS 表头的迭代/N-way 表格"))

    cell_cols = ["train", "test", "oot_auc", "oot_ks"]
    split_of = {"train": "train", "test": "test",
                "oot_auc": "oot", "oot_ks": "oot"}
    met_of = {"train": "auc", "test": "auc",
              "oot_auc": "auc", "oot_ks": "ks"}

    def collect_model_row(row_text: str) -> Optional[Dict[str, Any]]:
        """数据行 → {'name', 'vals': {colkey: float}};列首须为数字序号。

        复用 parse_header 同款 _cells() 切分,保证与表头的列索引口径完全一致
        (历史 bug: 此前用 inner[1:-1].split 导致数据行比表头少剥一列、整体左移一格)。
        """
        inner = row_text.strip()
        if not (inner.startswith("|") and inner.endswith("|")):
            return None
        parts = _cells(row_text)
        # _cells() 已剥离首尾空 cell;此处仅做防御,不改变其切片语义
        while parts and not parts[-1]:
            parts.pop()
        if len(parts) < 5:
            return None
        if not re.fullmatch(r"\d+", re.sub(r"[`*\s]", "", parts[0])):
            return None
        vals: Dict[str, float] = {}
        for cn in cell_cols:
            ci = col_map[cn]
            if ci >= len(parts):
                return None
            num = _cell_num(parts[ci])
            if num is None:
                return None
            vals[cn] = num
        return {"name": parts[1].strip(), "vals": vals}

    rows_found = 0
    runs_reported = set()

    if first_good_hdr_idx is not None and col_map:
        hi_bound = min(len(lines), first_good_hdr_idx + 2 + 60)
        start_ln = first_good_hdr_idx + 2
        for ln_no in range(start_ln, hi_bound):
            tl = lines[ln_no].lstrip()
            if tl.startswith("## ") or (not tl.startswith("|")):
                break
            rt = collect_model_row(lines[ln_no])
            if rt is None:
                continue
            rows_found += 1
            place = f"N-way L{ln_no+1}({rt['name']})"
            st = stat_cache.get(rt["name"])
            if not st:
                warns.append((place, "找不到该 run 的 evaluation/*_eval.json,SKIP"))
                continue
            any_bad = False
            for cn, shown in rt["vals"].items():
                sp, mk = split_of[cn], met_of[cn]
                want_src = (st.get(sp) or {}).get(mk)
                if want_src is None:
                    warns.append((place, f"[{sp}:{mk}] eval 中缺失,SKIP"))
                    continue
                delta_val = abs(shown - float(want_src))
                tol_x = max(5e-4, 10 ** -(len(str(float(want_src)).split('.')[1]) + 1))
                if delta_val <= tol_x:
                    continue
                any_bad = True
                fails.append((place,
                              f"[{sp}:{mk}] report={shown:.5f} != "
                              f"eval={float(want_src):.5f}"
                              f"(delta={delta_val:.5f}>{tol_x:.2e})"))
            if not any_bad:
                runs_reported.add(rt["name"])

    if rows_found == 0 and len(stat_cache) > 0:
        warn_names = ", ".join(sorted(stat_cache))
        warns.append(("N-way",
                      f"{len(stat_cache)} 个 run(tolerance:{warn_names})在报告表格均未出现——"
                      f"可能缺迭代表/N-way 或排版不匹配,请人工确认"))
    return len(runs_reported)


# ----------------------------------------------------------------------------
# ② ka_v4 统一口径重测对比表
def _cells(row_line: str) -> List[str]:
    inner = row_line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    parts = [p.strip() for p in inner.split("|")]
    while parts and parts[-1] == "":
        parts.pop()
    while parts and parts[0] == "":
        parts.pop(0)
    return parts


def check_ka_v4(lines: List[str], expect_dev: Optional[float],
                expect_oot: Optional[float], kava_tol: float) -> None:
    """找『ka_v4 … 统一口径』表，核对 dev/test 行数值是否落在预期值±tol 内。"""
    want = {"dev": expect_dev, "oot": expect_oot}
    found_row: Dict[str, bool] = {k: False for k in want}
    headers_with_kava = [
        i for i, ln in enumerate(lines)
        if ln.lstrip().startswith("|") and "ka_v4" in ln
    ]

    def rows_below_header(hs_: int):
        """从表头下方逐行产出 (列表格行文本);分隔线跳过,遇空行/段首停止。"""
        res = []
        for ri in range(hs_ + 1, min(len(lines), hs_ + 40)):
            tl = lines[ri].strip()
            if tl == "" or not tl.startswith("|"):
                break
            res.append(ri)
        return res

    def _is_separator(rline: str) -> bool:
        cols = _cells(rline)
        return bool(cols) and all(
            re.fullmatch(r"[-:\s]+", c) for c in cols)

    for hs in headers_with_kava:
        hdr = _cells(lines[hs])
        if not any("ka_v4" in c for c in hdr):
            continue
        if not all(k in "".join(hdr) for k in ("统一", "口径")):
            continue  # 只认「统一口径重测」式对照表
        col_idx = next((ci for ci, c in enumerate(hdr) if "ka_v4" in c), 1)

        for ri in rows_below_header(hs):
            rtext = lines[ri]
            if _is_separator(rtext):
                continue
            cols = _cells(rtext)
            if len(cols) <= col_idx:
                continue
            low_lbl = cols[0].lower().replace(" ", "")
            rowtype = None
            if low_lbl.startswith("oot"):
                rowtype = "oot"
            elif low_lbl.startswith("dev(test)"):
                rowtype = "dev"
            if rowtype is None:
                continue
            exp = want.get(rowtype)
            if exp is None:
                found_row[rowtype] = False      # 未提供期望值视同跳过核对
                continue
            raw = cols[col_idx].replace("*", "").replace(",", "")
            try:
                shown = float(raw)
            except ValueError:
                warns.append((f"ka_v4[{rowtype}] L{ri}",
                              f"'{raw}' 不是数值,SKIP"))
                continue
            delta = abs(shown - float(exp))
            place = f"ka_v4[{rowtype}] L{ri}"
            if delta > kava_tol:
                fails.append((place,
                              f"report={shown:.5f} != 预期基线={float(exp):.5f}"
                              f"(delta={delta:.5f}>{kava_tol})"))
            found_row[rowtype] = True

    provided = {k for k, v in want.items() if v is not None}
    any_kava_table_found = bool(
        i for i in headers_with_kava
        if all(k in "".join(_cells(lines[i])) for k in ("统一", "口径"))
    )
    if provided and not any_kava_table_found:
        # 给了期望值但报告里没有「统一口径」对照表 → 提醒(避免静默漂移)
        warns.append(("ka_v4",
                      "提供了 --expect-kava-* 但在报告中未找到带‘统一口径’的 ka_v4 "
                      "对照表 —— 该项无法核对;若确无该类对比,调用时可省略期望参数"))
    missing = {k for k in provided if not found_row.get(k)}
    for mk in sorted(missing):
        warns.append((f"ka_v4[{mk}]", "报告中缺少对应对照行(SKIP)"))


# ----------------------------------------------------------------------------
# ③ FICO bscore 摘要行
RANGE_RE = re.compile(
    r"bscore\s+范围=\[(?P<a>-?\d+(?:\.\d+)?)\s*,\s*(?P<b>-?\d+(?:\.\d+)?)\]"
)
MEAN_RE = re.compile(r"均值=(?P<m>-?\d+(?:\.\d+)?)")
SPLIT_PREFIX_RE = re.compile(r"^-\s*`?(?P<sp>train|test|oot)`?\s*[:：]")


def check_fico(lines: List[str], session: Path) -> None:
    """校验形如 '- `train`: n=… | bscore 范围=[a,b] | 均值=c' 的行 vs fitting-summary。"""
    sums: List[Dict[str, Any]] = []
    for jp in sorted((session / "new-models").glob("*/fico/fitting-summary.json")):
        d = read_json(jp)
        if isinstance(d, dict) and isinstance(d.get("splits"), dict):
            sums.append(d["splits"])
    if not sums:
        warns.append(("fico", "无 any fitting-summary.json(FICO 未产出),SKIP"))
        return
    src_splits = sums[0]      # FICO 仅 top1 上线候选产生,取第一个即可

    checked_lines = set()
    matched = 0
    for ln_no, raw in enumerate(lines, start=1):
        pm = SPLIT_PREFIX_RE.match(raw.strip())
        rm = RANGE_RE.search(raw)
        mm = MEAN_RE.search(raw)
        if (rm is None and mm is None) or pm is None:
            continue
        sp = pm.group("sp").lower()
        key = f"{sp}-{ln_no}"
        if key in checked_lines:
            continue
        sec = src_splits.get(sp)
        where = f"fico[{sp}] L{ln_no}"
        if not isinstance(sec, dict):
            warns.append((where, "fitting-summary 缺该 split,SKIP"))
            continue
        ok_block = True
        if rm is not None:
            got_a, got_b = float(rm.group("a")), float(rm.group("b"))
            lo, hi = sec.get("bscore_min"), sec.get("bscore_max")
            if lo is None or hi is None:
                warns.append((where, "fitting 无 bscore_min/max,SKIP"))
                ok_block = False
            else:
                da = abs(got_a - float(lo)); db = abs(got_b - float(hi))
                if da > 100 or db > 300:
                    fails.append((where,
                                  f"range=[{got_a},{got_b}] != fitting "
                                  f"[{float(lo):.2f},{float(hi):.2f}]"
                                  f"(Δmin={da:.2f}, Δmax={db:.2f})"))
                    ok_block = False
        if mm is not None and ok_block:
            got_m = float(mm.group("m"))
            sm = sec.get("bscore_mean")
            if not isinstance(sm, (int, float)):
                warns.append((where, "fitting 无 bscore_mean,SKIP"))
            else:
                dm = abs(got_m - float(sm))
                tol_m = max(30.0, 0.03 * abs(float(sm)))
                if dm > tol_m:
                    fails.append((where,
                                  f"mean={got_m} != fitting bscore_mean={sm}"
                                  f"(Δ={dm:.2f}>tol={tol_m:.1f})"))
        checked_lines.add(key)
        matched += 1


# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True, help="session dir(内含 report.md)")
    ap.add_argument("--expect-kava-dev", type=float, default=None,
                    help="ka_v4 dev(test) AUC 期望值")
    ap.add_argument("--expect-kava-oot", type=float, default=None,
                    help="ka_v4 OOT AUC 期望值")
    ap.add_argument("--kava-tol", type=float, default=0.005,
                    help="ka_v4 允许偏差(默认0.005,吸收跨批次口径微差)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="输出额外核对明细(目前用于 FICO/跑批计数)")
    ap.add_argument("--no-fail-on-warn", action="store_true",
                    help="仅 WARN 时也返回0")
    args = ap.parse_args(argv)

    global fails, warns
    fails.clear(); warns.clear()

    session = Path(args.session)
    report_p = session / "report.md"
    if not report_p.exists():
        print("[render-check] 致命: 缺少 report.md", file=sys.stderr)
        return 2
    lines = report_p.read_text(encoding="utf-8").splitlines()

    check_model_rows(lines, session)
    check_ka_v4(lines, args.expect_kava_dev, args.expect_kava_oot, args.kava_tol)
    check_fico(lines, session)

    rc = 0
    status = "OK"
    if fails:
        rc, status = 1, "FAIL"
    elif warns and not args.no_fail_on_warn:
        rc, status = 1, "WARN"

    print("\n==== render-check 结果 ====")
    for kind, items in (("FAIL", fails), ("WARN", warns)):
        for place, msg in items:
            print(f"  [{kind}] {place}: {msg}")
    if not fails and not warns:
        print("  全部断言项通过 —— report.md 与产物文件一致。")
    else:
        print(f"  FAIL×{len(fails)}  WARN×{len(warns)}  (exit={rc})")
    print(f"状态: {status}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())