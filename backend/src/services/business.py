"""同一作业员路线改造前后的配对复测，不涉及库存业务。"""
import math
import random
import numpy as np
from scipy import stats


def metadata():
    return {"name": "拣选路线改造复测", "short": "pick-paired", "tagline": "让同一个人的前后差，回答路线是否缩短。", "client": "北岭履约 · 80 名作业员的流程改善试点组", "workflow": "改善员把两周作业均时直接比较，忽略人员结构和熟练度差异。", "pains": ["不同人员混在前后平均中", "缺失配对无法识别", "改善秒数缺少区间与个体差异"], "upstream": "匿名作业员配对测时 BL_route_retest.csv", "downstream": "流程改善组决定路线方案是否扩大复测",
            "entities": ["配对复测样本", "路线改善核验", "班组复测简报"], "dataset_name": "80 名作业员 · 原路线与新路线配对",
            "columns": [{"key": "operator", "label": "匿名作业员", "type": "string"}, {"key": "round", "label": "复测编号", "type": "string", "optional": True}, {"key": "team", "label": "试点小组", "type": "string"}, {"key": "before", "label": "原路线/秒", "type": "number"}, {"key": "after", "label": "新路线/秒", "type": "number"}],
            "parameters": [{"key": "team", "label": "试点小组", "type": "select", "default": "全部", "options": [{"value": x, "label": x} for x in ["全部", "一组", "二组", "三组", "四组"]]}, {"key": "confidence", "label": "置信度", "type": "select", "default": "0.95", "options": [{"value": str(x), "label": f"{x*100:g}%"} for x in [.9, .95, .99]]}],
            "workbench_title": "人员配对改善核验", "action_label": "计算配对差与区间", "unit": "秒", "accent": "oklch(47% 0.075 248)", "style": "blueprint",
            "seed_story": ["07:50 改善员导入 BL_route_retest.csv，共 80 人。", "08:10 核对 OP-001 至 OP-080 的前后测时完整。", "08:30 选择全部小组、95% 置信区间。", "08:35 执行配对检验，查看平均节省秒数。", "08:50 打开逐人差值，核对反向改善样本。", "09:10 单独选择四组，比较改善差异。", "09:30 导出班组简报，安排负改善人员的路线走查。"],
            "method": "逐人差值 d=before−after；配对 t=mean(d)/(sd(d)/√n)，自由度 n−1；置信区间由 t 分布给出；标准化配对效应 dz=mean(d)/sd(d)。", "limitations": ["顺序、学习效应可能混入路线效果", "不对个人绩效作评价", "需保持复测订单难度和测时口径一致"]}


def seed_records():
    rng = random.Random(1006)
    rows = []
    for i in range(80):
        j = i % 4
        before = rng.gauss(410 + 8 * j, 38)
        improvement = rng.gauss([31, 22, 16, 6][j], 12)
        rows.append({"operator": f"OP-{i+1:03}", "team": ["一组", "二组", "三组", "四组"][j], "before": round(before, 3), "after": round(before - improvement, 3)})
    return rows


def round_label(value):
    """复测编号统一为字符串口径；布尔、容器、非有限数等非法值按缺失处理。"""
    if value is None or isinstance(value, bool) or isinstance(value, (dict, list)):
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value).strip()


def has_rounds(records):
    """批次中任意记录带复测编号即按多次测时口径配对；否则走旧口径，结果与历史一致。"""
    return any(round_label(r.get("round")) for r in records if isinstance(r, dict))


def _measurement(value):
    """测时必须为正有限数；缺失或非法返回 None，由配对环节标记缺测。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _round_sort(label):
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def _params(params):
    team = str(params.get("team", "全部"))
    try:
        confidence = float(params.get("confidence", .95))
    except (ValueError, TypeError):
        raise ValueError("置信度只接受 0.90/0.95/0.99")
    if confidence not in (.9, .95, .99) or team not in ("全部", "一组", "二组", "三组", "四组"):
        raise ValueError("未知小组或置信度")
    return team, confidence


def analyze(records, params):
    team, confidence = _params(params)
    if has_rounds(records):
        return _analyze_rounds(records, team, confidence)
    return _analyze_legacy(records, team, confidence)


def _analyze_legacy(records, team, confidence):
    """无复测编号的旧批次：保持原口径，结果与历史完全一致。"""
    clean, seen = [], set()
    for r in records:
        try:
            name, group = str(r["operator"]), str(r["team"])
            before, after = float(r["before"]), float(r["after"])
            if name in seen or not all(math.isfinite(x) and x > 0 for x in (before, after)):
                raise ValueError()
            seen.add(name)
            if team == "全部" or team == group:
                clean.append({"operator": name, "team": group, "before": before, "after": after, "difference": before - after})
        except (KeyError, ValueError, TypeError):
            raise ValueError("每人仅能有一条完整配对，测时必须为正有限数")
    if len(clean) < 3:
        raise ValueError("至少需要 3 组完整配对")
    diffs = np.array([r["difference"] for r in clean])
    n, mean, sd = len(diffs), float(diffs.mean()), float(diffs.std(ddof=1))
    if sd <= 1e-12:
        raise ValueError("配对差完全相同，无法估计标准误")
    se, df = sd / math.sqrt(n), n - 1
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), df))
    margin = float(stats.t.ppf((1 + confidence) / 2, df)) * se
    low, high = mean - margin, mean + margin
    rows = [{**r, "before": round(r["before"], 3), "after": round(r["after"], 3), "difference": round(r["difference"], 3), "review": "缩短" if r["difference"] > 0 else "需走查"} for r in sorted(clean, key=lambda r: r["difference"])]
    teams = sorted(set(r["team"] for r in clean))
    return {"summary": f"{n} 人完整配对，平均节省 {mean:.3f} 秒；{confidence*100:g}% 区间 [{low:.3f}, {high:.3f}] 秒。",
            "metrics": [{"label": "平均节省", "value": round(mean, 5), "unit": "秒"}, {"label": "配对 p 值", "value": float(f"{p:.6g}")}, {"label": "配对效应 dz", "value": round(mean / sd, 5)}, {"label": "需走查人员", "value": int(sum(diffs <= 0)), "unit": "人"}],
            "table": {"columns": [{"key": "operator", "label": "匿名人员"}, {"key": "team", "label": "小组"}, {"key": "before", "label": "原路线/秒"}, {"key": "after", "label": "新路线/秒"}, {"key": "difference", "label": "节省/秒"}, {"key": "review", "label": "复测建议"}], "rows": rows},
            "chart": {"type": "bar", "labels": teams, "series": [{"name": "平均节省秒数", "values": [round(float(np.mean([r["difference"] for r in clean if r["team"] == x])), 4) for x in teams]}]},
            "findings": [f"区间下界 {low:.4f} 秒，上界 {high:.4f} 秒；配对差标准差 {sd:.4f} 秒。", f"t={t:.4f}，df={df}，分析小组：{team}。", f"{int(sum(diffs > 0))} 人路线缩短，{int(sum(diffs <= 0))} 人需要走查，不能仅依据总均值推断所有人受益。"], "method": metadata()["method"], "warnings": ["前后顺序固定，不能排除学习效应；结果用于下一轮复测设计。"]}


def _analyze_rounds(records, team, confidence):
    """同批多次测时：按作业员+复测编号配对，重复投递去重，冲突配对隔离，按作业员聚合。"""
    groups = {}
    for r in records:
        operator = str(r.get("operator") or "").strip()
        key = (operator, round_label(r.get("round")))
        groups.setdefault(key, []).append({"team": str(r.get("team") or "").strip(),
                                           "before": _measurement(r.get("before")), "after": _measurement(r.get("after"))})
    pairs, missing, isolated = [], [], []
    for (operator, round_no), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], _round_sort(kv[0][1]))):
        teams = sorted({row["team"] for row in rows if row["team"]})
        if not round_no:
            isolated.append({"operator": operator, "round": None, "team": "／".join(teams), "reason": "缺少复测编号，无法配对", "deliveries": len(rows)})
            continue
        complete = [row for row in rows if row["before"] is not None and row["after"] is not None]
        if not complete:
            miss_before = any(row["before"] is None for row in rows)
            miss_after = any(row["after"] is None for row in rows)
            reason = "缺少有效前后测时" if miss_before and miss_after else ("缺少有效原路线测时" if miss_before else "缺少有效新路线测时")
            missing.append({"operator": operator, "round": round_no, "team": "／".join(teams), "reason": reason})
            continue
        values = {(row["before"], row["after"]) for row in complete}
        complete_teams = sorted({row["team"] for row in complete})
        reasons = []
        if len(values) > 1:
            reasons.append("同一复测编号前后测时冲突（" + "、".join(f"前 {b:g} / 后 {a:g}" for b, a in sorted(values)) + "）")
        if len(complete_teams) > 1:
            reasons.append("同一复测编号小组冲突（" + "、".join(complete_teams) + "）")
        if reasons:
            isolated.append({"operator": operator, "round": round_no, "team": "／".join(complete_teams), "reason": "；".join(reasons), "deliveries": len(rows)})
            continue
        row = complete[0]
        pairs.append({"operator": operator, "round": round_no, "team": row["team"], "before": row["before"], "after": row["after"],
                      "difference": row["before"] - row["after"], "deliveries": len(rows)})
    by_operator = {}
    for pair in pairs:
        by_operator.setdefault(pair["operator"], []).append(pair)
    included, operators = [], []
    for name in sorted(by_operator):
        own = by_operator[name]
        teams = sorted({p["team"] for p in own})
        if len(teams) > 1:
            for p in own:
                isolated.append({"operator": name, "round": p["round"], "team": "／".join(teams),
                                 "reason": "同一作业员跨复测小组冲突（" + "、".join(teams) + "）", "deliveries": p["deliveries"]})
            continue
        included.extend(own)
        k = len(own)
        operators.append({"operator": name, "team": own[0]["team"], "rounds": [p["round"] for p in own],
                          "before": sum(p["before"] for p in own) / k, "after": sum(p["after"] for p in own) / k,
                          "difference": sum(p["difference"] for p in own) / k})
    isolated.sort(key=lambda x: (x["operator"], _round_sort(x["round"] or "")))
    deduplicated = sum(p["deliveries"] - 1 for p in included)
    scoped = [o for o in operators if team == "全部" or o["team"] == team]
    if len(scoped) < 3:
        raise ValueError(f"有效配对作业员不足 3 名（当前 {len(scoped)} 名；缺测与隔离不计入）")
    diffs = np.array([o["difference"] for o in scoped])
    n, mean, sd = len(diffs), float(diffs.mean()), float(diffs.std(ddof=1))
    if sd <= 1e-12:
        raise ValueError("配对差完全相同，无法估计标准误")
    se, df = sd / math.sqrt(n), n - 1
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), df))
    margin = float(stats.t.ppf((1 + confidence) / 2, df)) * se
    low, high = mean - margin, mean + margin
    rows = [{"operator": o["operator"], "team": o["team"], "rounds": "、".join(o["rounds"]),
             "before": round(o["before"], 3), "after": round(o["after"], 3), "difference": round(o["difference"], 3),
             "review": "缩短" if o["difference"] > 0 else "需走查"} for o in sorted(scoped, key=lambda o: o["difference"])]
    team_rows = []
    for name in sorted({o["team"] for o in scoped}):
        ds = np.array([o["difference"] for o in scoped if o["team"] == name])
        tn, tmean = len(ds), float(ds.mean())
        if tn >= 2:
            tsd = float(ds.std(ddof=1))
            tmargin = float(stats.t.ppf((1 + confidence) / 2, tn - 1)) * tsd / math.sqrt(tn)
            team_rows.append({"team": name, "operators": tn, "mean": round(tmean, 4), "sd": round(tsd, 4), "low": round(tmean - tmargin, 4), "high": round(tmean + tmargin, 4)})
        else:
            team_rows.append({"team": name, "operators": tn, "mean": round(tmean, 4), "sd": None, "low": None, "high": None})
    pairs_used = sum(len(o["rounds"]) for o in scoped)
    audit = f"全批 {len(included)} 对配对纳入（去重 {deduplicated} 次重复投递），缺测 {len(missing)} 项、隔离 {len(isolated)} 对未计入；多测人员按作业员聚合，不重复加权。"
    return {"summary": f"{n} 名作业员纳入分析（{pairs_used} 次复测配对按人聚合），平均节省 {mean:.3f} 秒；{confidence*100:g}% 区间 [{low:.3f}, {high:.3f}] 秒。全批缺测 {len(missing)} 项、隔离 {len(isolated)} 对，明细见配对核对。",
            "metrics": [{"label": "平均节省", "value": round(mean, 5), "unit": "秒"}, {"label": "配对 p 值", "value": float(f"{p:.6g}")}, {"label": "配对效应 dz", "value": round(mean / sd, 5)}, {"label": "需走查人员", "value": int(sum(diffs <= 0)), "unit": "人"}, {"label": "纳入作业员", "value": n, "unit": "人"}, {"label": "缺测", "value": len(missing), "unit": "项"}, {"label": "隔离配对", "value": len(isolated), "unit": "对"}],
            "table": {"columns": [{"key": "operator", "label": "匿名人员"}, {"key": "team", "label": "小组"}, {"key": "rounds", "label": "计入复测"}, {"key": "before", "label": "原路线均值/秒"}, {"key": "after", "label": "新路线均值/秒"}, {"key": "difference", "label": "节省/秒"}, {"key": "review", "label": "复测建议"}], "rows": rows},
            "chart": {"type": "bar", "labels": [r["team"] for r in team_rows], "series": [{"name": "平均节省秒数", "values": [r["mean"] for r in team_rows]}]},
            "teams": team_rows,
            "pairing": {"included": [{**q, "before": round(q["before"], 3), "after": round(q["after"], 3), "difference": round(q["difference"], 3)} for q in included], "missing": missing, "isolated": isolated, "deduplicated": deduplicated},
            "findings": [f"区间下界 {low:.4f} 秒，上界 {high:.4f} 秒；配对差标准差 {sd:.4f} 秒。", f"t={t:.4f}，df={df}，分析小组：{team}。", f"{int(sum(diffs > 0))} 人路线缩短，{int(sum(diffs <= 0))} 人需要走查，不能仅依据总均值推断所有人受益。", audit],
            "method": metadata()["method"] + " 同批多次测时按作业员与复测编号配对，逐人平均后再检验；整体与小组区间均按作业员聚合，多测人员不重复加权。",
            "warnings": ["前后顺序固定，不能排除学习效应；结果用于下一轮复测设计。", "缺测与隔离记录未参与统计；补齐或更正后请新建批次重新分析，本批次结果保持不变。"]}
