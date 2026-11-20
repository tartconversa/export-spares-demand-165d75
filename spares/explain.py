"""面向最终用户的可追溯解释：风险从哪些设备与交期来、预算改善哪里。

所有结论都下钻到可查的工单、设备与供给批次，避免黑箱数字：
- 站点/工单类型风险＝积压工单＋预测新故障，逐条列出来源设备与时限；
- 每个风险关联最早可用供给（现存/在途区间/承诺），说明交期来源；
- 追加预算的边际改善通过同一套仿真重新评估，给出改善位置而非总数。
"""
from __future__ import annotations

import pandas as pd

from .dataio import DataPack, point_in_time
from .forecast import ForecastResult, forecast, occlusion_summary
from .inventory import CommitmentLedger
from .planning import Action, evaluate
from .supply import effective_shipment_window


def station_risk_detail(pack: DataPack, station_id: str,
                        fc: ForecastResult, cutoff: pd.Timestamp,
                        ledger: CommitmentLedger | None = None,
                        route_delay: int = 0) -> pd.DataFrame:
    """某站风险逐单展开：来源设备、故障时刻、服务时限、最早可用供给。"""
    ledger = ledger or CommitmentLedger()
    cutoff = pd.Timestamp(cutoff)
    st = pack.stations.set_index("station_id")
    service_h = int(st.loc[station_id, "service_hours"])
    wo = pack.work_orders.merge(
        pack.installed_base[["unit_id", "station_id", "generation",
                             "firmware", "environment", "commissioned"]],
        on="unit_id")
    rows = []
    # 1) 已发生未恢复（waiting）——风险是已经违约的待修复工单
    for _, o in wo[(wo.station_id == station_id)
                   & (wo.state == "waiting")].sort_values("failed_at").iterrows():
        deadline = o.failed_at + pd.Timedelta(hours=service_h)
        rows.append({
            "risk_source": "backlog", "order_id": o.order_id,
            "unit_id": o.unit_id, "generation": o.generation,
            "environment": o.environment,
            "failed_at": o.failed_at, "service_deadline": deadline,
            "consequence": int(o.consequence),
            "already_breached": bool(cutoff > deadline),
            "supply_note": _supply_note(pack, ledger, station_id,
                                        o.generation, cutoff, route_delay),
        })
    # 2) 预测新故障——来源设备按龄期分层（龄期越大故障率越高）
    new = fc.failure_forecast[fc.failure_forecast.station_id == station_id]
    ib = pack.installed_base[pack.installed_base.station_id == station_id]
    for _, r in new.iterrows():
        units = ib[(ib.generation == r.generation)
                   & (ib.environment == r.environment)]
        oldest = units.commissioned.min()
        rows.append({
            "risk_source": "forecast_new_failure",
            "order_id": f"EXP:{station_id}:{r.generation}:{r.environment}",
            "unit_id": f"{len(units)} 台（最早投运 {oldest}）",
            "generation": r.generation, "environment": r.environment,
            "failed_at": None,
            "service_deadline": f"故障后 {service_h} 小时",
            "consequence": None,
            "already_breached": False,
            "supply_note": (f"90天故障均值 {r.failures_mean:.2f}，"
                            f"90%区间 [{r.failures_p05:.1f}, {r.failures_p95:.1f}]；"
                            + _supply_note(pack, ledger, station_id,
                                           r.generation, cutoff, route_delay)),
        })
    return pd.DataFrame(rows)


def _supply_note(pack: DataPack, ledger: CommitmentLedger, station: str,
                 generation: str, cutoff: pd.Timestamp, route_delay: int) -> str:
    """该代际设备所需各物料的最早可用供给文字说明。"""
    firmware = pack.installed_base[
        pack.installed_base.generation == generation].firmware.iloc[0]
    qual = pack.stations.set_index("station_id").loc[station, "qualification"]
    from .compat import repair_requirements
    reqs = repair_requirements(pack, generation, firmware, qual)
    notes = []
    for group, options in reqs.items():
        for opt in options:
            p = opt.part_id
            oh = ledger.guarantee_stock(pack, station, p, cutoff)
            ships = pack.shipments[
                (pack.shipments.station_id == station)
                & (pack.shipments.part_id == p)]
            ship_txt = ""
            if len(ships):
                e, l = effective_shipment_window(ships.iloc[0], route_delay)
                ship_txt = f"，在途 {ships.quantity.sum()} 件 {e.date()}~{l.date()}"
            notes.append(f"{group}:{p} 现存保障 {oh}{ship_txt}")
    return "；".join(notes)


def occlusion_report(pack: DataPack, cutoff: pd.Timestamp) -> pd.DataFrame:
    """缺货遮挡清单（可核对到工单与登记区间）。"""
    pit = point_in_time(pack, cutoff)
    return occlusion_summary(pit)


def marginal_budget_value(pack: DataPack, fc: ForecastResult,
                          chosen: list[Action], cutoff: pd.Timestamp,
                          budget: float, scenario_id: str | None,
                          route_delay: int, increments=(5000, 10000, 20000),
                          n_sims: int = 600) -> pd.DataFrame:
    """追加预算能改善哪里：在每个更高预算下重跑选优并对比分站指标。"""
    base = evaluate(pack, fc, chosen, cutoff, budget, scenario_id,
                    route_delay, n_sims=n_sims)
    rows = [{
        "budget": budget, "on_time_weighted": base.on_time_weighted,
        "expected_backlog_units": base.expected_backlog_units,
        "cash_committed": base.cash_committed,
        "worst_station": base.station_metrics.sort_values(
            "on_time_weighted").iloc[0].station_id,
    }]
    from .planning import optimize
    for extra in increments:
        _, ev, _ = optimize(pack, cutoff, budget + extra, scenario_id,
                            n_sims=min(n_sims, 800))
        rows.append({
            "budget": budget + extra,
            "on_time_weighted": ev.on_time_weighted,
            "expected_backlog_units": ev.expected_backlog_units,
            "cash_committed": ev.cash_committed,
            "worst_station": ev.station_metrics.sort_values(
                "on_time_weighted").iloc[0].station_id,
        })
    return pd.DataFrame(rows)


def assumptions_registry(*assumption_lists: list[str]) -> pd.DataFrame:
    """把各模块的显式假设集中登记，逐条编号，便于用户检查风险情景。"""
    rows = []
    for i, text in enumerate([a for lst in assumption_lists for a in lst], 1):
        rows.append({"assumption_id": f"A{i:02d}", "assumption": text,
                     "supported_by_data": "否-建模假设" if "假设" in text
                     else "是-资料事实"})
    return pd.DataFrame(rows)
