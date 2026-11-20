"""备件保障分析应用命令行入口。

完整季度演练流程：
  forecast   截点预测（故障与备件需求区间、遮挡与可信度说明）
  risk       某站风险下钻（来源设备、时限、供给交期）
  plan       预算内方案选优（常规/加急/改港/互借的对比与贪心结果）
  commit     把选定动作落实为采购/互借承诺（校验库存、两端扣减）
  simulate-actuals 生成演练用“后续实际资料”（新故障/延期/验收回库）
  replay     用实际资料回看原预测，归因需求误差/兼容判断/履约变化
  report     输出单文件 HTML 演练报告
所有命令默认截点 2025-04-21（资料第110日，在途批次最早到日）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python3 -I spares/__main__.py` 直接执行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "spares"

import pandas as pd

from .commit import commit_actions, load_ledger, save_snapshot
from .dataio import TZ, DataPack, load_pack, point_in_time
from .explain import (marginal_budget_value, occlusion_report,
                      station_risk_detail)
from .forecast import consumption_vs_demand, forecast
from .planning import Action, optimize
from .replay import attribute_gaps
from .report import write_report
from .supply import ARRIVAL_UNIFORM_ASSUMPTION, expected_eta_table

DEFAULT_CUTOFF = "2025-04-21T08:00:00+08:00"


def _print_df(df: pd.DataFrame, max_rows: int = 60) -> None:
    if df.empty:
        print("（无记录）")
        return
    with pd.option_context("display.max_columns", None,
                           "display.width", 200, "display.max_rows", max_rows):
        print(df.to_string(index=False))


def _cutoff(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    return ts if ts.tzinfo else ts.tz_localize(TZ)


def _load(args) -> DataPack:
    return load_pack(args.data)


def cmd_forecast(args) -> None:
    pack = point_in_time(_load(args), _cutoff(args.cutoff))
    fc = forecast(pack, _cutoff(args.cutoff), horizon_days=args.horizon,
                  scenario_id=args.scenario, n_sims=args.sims)
    print(f"# 截点 {args.cutoff} 后 {args.horizon} 天故障预测"
          + (f"（情景 {args.scenario}）" if args.scenario else ""))
    print("\n## 站×代际×环境 故障数（均值与 50%/90% 区间）")
    _print_df(fc.failure_forecast.round(2))
    print("\n## 站×物料 备件需求（含现有积压，均值与区间）")
    _print_df(fc.part_demand.round(2))
    print("\n## 截点未结工单积压（确定的待满足需求）")
    _print_df(fc.backlog)
    print("\n## 账面消耗 vs 真实故障需求（替代抬高/缺货遮挡偏差）")
    _print_df(consumption_vs_demand(pack))
    print("\n## 缺货遮挡清单")
    _print_df(occlusion_report(pack, _cutoff(args.cutoff)))
    print("\n## 可信度与假设登记")
    for a in fc.assumptions:
        print("- " + a)
    print("- " + ARRIVAL_UNIFORM_ASSUMPTION)
    _maybe_state(args, fc, pack)


def _maybe_state(args, fc, pack) -> None:
    if not args.state:
        return
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    fc.failure_forecast.to_csv(state / "forecast_failures.csv", index=False)
    fc.part_demand.to_csv(state / "forecast_parts.csv", index=False)
    fc.backlog.to_csv(state / "backlog.csv", index=False)
    occlusion_report(pack, _cutoff(args.cutoff)).to_csv(
        state / "occlusion.csv", index=False)
    consumption_vs_demand(pack).to_csv(
        state / "consumption_vs_demand.csv", index=False)


def cmd_risk(args) -> None:
    pack = point_in_time(_load(args), _cutoff(args.cutoff))
    ledger = load_ledger(args.state) if args.state else None
    route_delay = 0
    if args.scenario:
        route_delay = int(pack.plans[
            pack.plans.scenario_id == args.scenario].route_delay_days.iloc[0])
    fc = forecast(pack, _cutoff(args.cutoff), scenario_id=args.scenario,
                  n_sims=args.sims)
    print(f"# 站点 {args.station} 风险下钻（截点 {args.cutoff}）")
    _print_df(station_risk_detail(pack, args.station, fc,
                                  _cutoff(args.cutoff), ledger, route_delay),
              max_rows=80)


def cmd_plan(args) -> None:
    raw = _load(args)
    pack = point_in_time(raw, _cutoff(args.cutoff))
    budget = args.budget
    if budget is None:
        budget = float(pack.plans[
            pack.plans.scenario_id == (args.scenario or "normal")].budget.iloc[0])
    chosen, ev, trace = optimize(pack, _cutoff(args.cutoff), budget,
                                 args.scenario, n_sims=args.sims)
    print(f"# 保障方案（情景 {args.scenario or '基准'}，预算 {budget:.0f} 元）")
    print("\n## 选优过程（每步的加权按时恢复与积压）")
    _print_df(pd.DataFrame(trace))
    print("\n## 选定动作")
    if not chosen:
        print("（预算内无正增益动作——基线供给已覆盖预测需求）")
    for a in chosen:
        print(f"- {a.kind:8s} {a.part_id} x{a.quantity} @ {a.station_id} "
              f"ref={a.ref} 资金={a.cash:,.0f} 到达={pd.Timestamp(a.eta_latest).date()} | {a.detail}")
    print("\n## 方案评估")
    print(f"加权按时恢复概率: {ev.on_time_weighted:.3f}")
    print(f"单量按时恢复概率: {ev.on_time_unweighted:.3f}")
    print(f"加权 SLA 违约期望: {ev.sla_breach_weighted:.3f}")
    print(f"窗末期望积压(件): {ev.expected_backlog_units:.2f}")
    print(f"资金占用: {ev.cash_committed:,.0f} / 预算 {budget:,.0f} "
          f"（{'预算内' if ev.within_budget else '超预算'}）")
    print("\n## 分站指标")
    _print_df(ev.station_metrics.round(3))
    print("\n## 边际预算价值（追加预算改善哪里）")
    fc = forecast(pack, _cutoff(args.cutoff), scenario_id=args.scenario,
                  n_sims=args.sims)
    route_delay = int(pack.plans[
        pack.plans.scenario_id == args.scenario].route_delay_days.iloc[0]) \
        if args.scenario else 0
    _print_df(marginal_budget_value(
        pack, fc, chosen, _cutoff(args.cutoff), budget, args.scenario,
        route_delay, n_sims=min(args.sims, 600)).round(3))
    print("\n## 假设登记")
    for a in ev.assumptions:
        print("- " + a)
    if args.state:
        snapshot = {
            "cutoff": str(_cutoff(args.cutoff)),
            "scenario_id": args.scenario, "route_delay_days": route_delay,
            "budget": budget, "horizon_days": fc.horizon_days,
            "part_demand": fc.part_demand.to_dict("records"),
            "actions": [a.__dict__ | {
                "eta_earliest": str(a.eta_earliest),
                "eta_latest": str(a.eta_latest)} for a in chosen],
            "evaluation": {
                "on_time_weighted": ev.on_time_weighted,
                "on_time_unweighted": ev.on_time_unweighted,
                "expected_backlog_units": ev.expected_backlog_units,
                "cash_committed": ev.cash_committed},
        }
        save_snapshot(snapshot, args.state, "plan_snapshot.json")
        print(f"\n快照已保存：{Path(args.state) / 'plan_snapshot.json'}")


def cmd_commit(args) -> None:
    raw = _load(args)
    pack = point_in_time(raw, _cutoff(args.cutoff))
    snapshot = json.loads((Path(args.state) / "plan_snapshot.json").read_text(
        encoding="utf-8"))
    actions = [Action(
        kind=a["kind"], station_id=a["station_id"], part_id=a["part_id"],
        quantity=a["quantity"], unit_cost=a["unit_cost"], fee=a["fee"],
        eta_earliest=pd.Timestamp(a["eta_earliest"]),
        eta_latest=pd.Timestamp(a["eta_latest"]), ref=a["ref"],
        detail=a["detail"]) for a in snapshot["actions"]]
    ledger = load_ledger(args.state)
    before_b, before_p = len(ledger.borrows), len(ledger.purchases)
    ledger = commit_actions(pack, actions, args.state,
                            _cutoff(args.cutoff), ledger)
    print(f"# 承诺落实完成：新增互借 {len(ledger.borrows)-before_b} 笔，"
          f"新增采购/改港 {len(ledger.purchases)-before_p} 笔")
    bf, pf = ledger.to_frames()
    print("\n## 互借承诺（出借站保障能力已扣减）")
    _print_df(bf)
    print("\n## 采购/改港承诺")
    _print_df(pf)
    rejected = Path(args.state) / "rejected_commitments.json"
    if rejected.exists():
        print("\n## 被拒绝的承诺（库存不足，未占实物）")
        print(rejected.read_text(encoding="utf-8"))
    print("\n## 落实后各站可保障库存")
    rows = []
    for s in pack.stations.station_id:
        for p in pack.parts.part_id:
            rows.append({"station_id": s, "part_id": p,
                         "guarantee_stock": ledger.guarantee_stock(
                             pack, s, p, _cutoff(args.cutoff))})
    _print_df(pd.DataFrame(rows))


def cmd_simulate_actuals(args) -> None:
    """生成演练用后续实际资料目录（新故障/批次延期/返修验收）。

    合成资料的延伸，结构与输入契约一致，供 replay 回看；
    真实使用时以业务系统导出的同结构文件替换。
    """
    from .actuals import generate_actuals
    out = generate_actuals(_load(args), _cutoff(args.cutoff),
                           args.out, days=args.horizon, seed=args.seed,
                           delay_days=args.delay_days,
                           returns=args.returns, new_failures=args.new_failures)
    print(f"演练实际资料已生成：{out}")
    print("文件：work_orders_extra.csv / shipments_revised.csv / "
          "stock_events_extra.csv / part_uses_extra.csv")


def cmd_replay(args) -> None:
    from .actuals import merge_actuals
    base = _load(args)
    review_at = _cutoff(args.review_at)
    extended = merge_actuals(base, args.actuals, review_at)
    snapshot = json.loads((Path(args.state) / "plan_snapshot.json").read_text(
        encoding="utf-8"))
    ledger = load_ledger(args.state)
    result = attribute_gaps(snapshot, extended, review_at,
                            snapshot["horizon_days"], ledger)
    print(f"# 季度回看（回看时点 {args.review_at}）")
    print("\n## 新观测概览")
    print(json.dumps(result.actuals, ensure_ascii=False, indent=2)[:1500])
    print("\n## 缺口归因（需求误差/兼容判断/履约变化）")
    _print_df(result.gap_attribution, max_rows=80)
    print("\n## 承诺兑现状态（两端保障变化）")
    _print_df(result.commitment_status)
    print("\n## 用新资料重评原方案（原预测快照保留不变）")
    print(json.dumps(result.new_evaluation, ensure_ascii=False, indent=2))
    print("\n## 回看口径假设")
    for a in result.assumptions:
        print("- " + a)
    if args.state:
        result.gap_attribution.to_csv(
            Path(args.state) / "replay_gap_attribution.csv", index=False)
        result.commitment_status.to_csv(
            Path(args.state) / "commitment_status.csv", index=False)
        save_snapshot({"review_at": str(review_at),
                       "new_evaluation": result.new_evaluation,
                       "new_part_demand": result.new_part_demand.to_dict("records")},
                      args.state, "replay_snapshot.json")


def cmd_report(args) -> None:
    pack = point_in_time(_load(args), _cutoff(args.cutoff))
    fc = forecast(pack, _cutoff(args.cutoff), scenario_id=args.scenario,
                  n_sims=args.sims)
    chosen, ev, trace = optimize(
        pack, _cutoff(args.cutoff),
        float(pack.plans[pack.plans.scenario_id
              == (args.scenario or "normal")].budget.iloc[0]),
        args.scenario, n_sims=args.sims)
    path = write_report(args.out, pack, fc, chosen, ev, trace,
                        _cutoff(args.cutoff), args.scenario)
    print(f"HTML 演练报告已生成：{path}")


def cmd_eta(args) -> None:
    pack = _load(args)
    delay = int(pack.plans[pack.plans.scenario_id
              == (args.scenario or "normal")].route_delay_days.iloc[0]) \
        if args.scenario else 0
    print(f"# 在途批次到达区间（情景迟延 {delay} 天；不按最早日确定入账）")
    _print_df(expected_eta_table(pack, _cutoff(args.cutoff), delay))
    print("- " + ARRIVAL_UNIFORM_ASSUMPTION)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="海外储能备件保障分析应用")
    p.add_argument("--data", default="data", help="资料目录")
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="分析截点（ISO 时刻）")
    p.add_argument("--state", default=None, help="演练状态目录（承诺/快照）")
    p.add_argument("--sims", type=int, default=2000, help="蒙特卡洛次数")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("forecast", help="故障与备件需求预测")
    sp.add_argument("--horizon", type=int, default=90)
    sp.add_argument("--scenario", default=None)
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("risk", help="站点风险下钻")
    sp.add_argument("station")
    sp.add_argument("--scenario", default=None)
    sp.set_defaults(func=cmd_risk)

    sp = sub.add_parser("plan", help="预算内保障方案选优")
    sp.add_argument("--scenario", default=None)
    sp.add_argument("--budget", type=float, default=None)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("commit", help="落实采购与互借承诺")
    sp.set_defaults(func=cmd_commit)

    sp = sub.add_parser("simulate-actuals", help="生成演练用后续实际资料")
    sp.add_argument("--out", required=True)
    sp.add_argument("--horizon", type=int, default=45)
    sp.add_argument("--seed", type=int, default=11)
    sp.add_argument("--delay-days", type=int, default=10)
    sp.add_argument("--returns", type=int, default=3)
    sp.add_argument("--new-failures", type=int, default=None,
                    help="不填则按历史速率与经过天数推导")
    sp.set_defaults(func=cmd_simulate_actuals)

    sp = sub.add_parser("replay", help="实际恢复回看与缺口归因")
    sp.add_argument("--actuals", required=True)
    sp.add_argument("--review-at", required=True)
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("report", help="输出 HTML 演练报告")
    sp.add_argument("--out", default="drill_report.html")
    sp.add_argument("--scenario", default=None)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("eta", help="在途批次到达区间")
    sp.add_argument("--scenario", default=None)
    sp.set_defaults(func=cmd_eta)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
