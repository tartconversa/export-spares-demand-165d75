"""季度演练回看：新事件触发重解释，并把缺口归因到三类来源。

触发事件（业务要求）：
- 新故障涌入：更新工单与暴露，重新预测；
- 航运批次延期：更新在途到达区间，重算按时恢复概率；
- 返修件验收合格：return 事件回库，更新可保障库存。

归因三分法（每个未兑现缺口必须且只能归入一类）：
- demand_error      需求误差：实际故障/需求落在原预测区间之外；
- compat_judgement  兼容性判断：需求数量在区间内，但固件/资质/配套
                     判断导致原方案的物料不能实际修复；
- fulfillment_change 履约变化：需求与兼容判断均成立，但供给时刻/数量
                     （延期、改港、互借未达、验收结果）偏离原假设。

原方案当时使用的预测以快照形式保留，回看只对比、不覆写。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataio import DataPack, point_in_time
from .forecast import forecast, order_demand_lines
from .inventory import CommitmentLedger
from .planning import Action, evaluate


@dataclass
class ReplayResult:
    review_at: pd.Timestamp
    old_snapshot: dict
    new_part_demand: pd.DataFrame
    actuals: dict
    gap_attribution: pd.DataFrame
    commitment_status: pd.DataFrame
    new_evaluation: dict
    assumptions: list[str]


def _demand_actuals(pack: DataPack, since: pd.Timestamp,
                    until: pd.Timestamp) -> pd.DataFrame:
    """统计 since 之后新发生故障按 BOM 展开的真实备件需求（与领用无关）。"""
    wo = pack.work_orders[
        (pack.work_orders.failed_at > since)
        & (pack.work_orders.failed_at <= until)].merge(
        pack.installed_base[["unit_id", "station_id", "generation"]],
        on="unit_id")
    pref = (pack.compatibility.sort_values(["generation", "firmware",
                                            "required_group", "part_id"])
            .groupby(["generation", "required_group"]).first().reset_index())
    rows = []
    for _, o in wo.iterrows():
        for _, b in pref[pref.generation == o.generation].iterrows():
            rows.append({"station_id": o.station_id, "part_id": b.part_id,
                         "qty": int(b.parts_per_repair),
                         "order_id": o.order_id})
    if not rows:
        return pd.DataFrame(columns=["station_id", "part_id", "qty", "order_id"])
    return pd.DataFrame(rows)


def attribute_gaps(old_snapshot: dict, new_pack: DataPack,
                   review_at: pd.Timestamp, horizon_days: int,
                   ledger: CommitmentLedger) -> ReplayResult:
    """对照原预测快照与 review_at 时点的新资料，完成缺口归因与重评估。"""
    review_at = pd.Timestamp(review_at)
    cutoff = pd.Timestamp(old_snapshot["cutoff"])
    scenario_id = old_snapshot.get("scenario_id")
    route_delay = old_snapshot.get("route_delay_days", 0)

    # 重预测只用 review_at 可知资料；新预测单独保存，不动旧快照
    new_pack_pit = point_in_time(new_pack, review_at)
    new_fc = forecast(new_pack_pit, cutoff, horizon_days=horizon_days,
                      scenario_id=scenario_id)

    actual_df = _demand_actuals(new_pack_pit, cutoff, review_at)
    actual = (actual_df.groupby(["station_id", "part_id"]).qty.sum()
              .reset_index() if len(actual_df)
              else pd.DataFrame(columns=["station_id", "part_id", "qty"]))

    old_demand = pd.DataFrame(old_snapshot["part_demand"])
    elapsed_days = max(
        1, (review_at.tz_localize(None) - cutoff.tz_localize(None)).days)
    horizon = int(old_snapshot.get("horizon_days", 90))
    scale = min(1.0, elapsed_days / horizon)
    rows = []
    keys = set(old_demand.set_index(["station_id", "part_id"]).index)
    keys |= set(actual.set_index(["station_id", "part_id"]).index) if len(actual) else set()
    for station, part in sorted(keys):
        old = old_demand[(old_demand.station_id == station)
                         & (old_demand.part_id == part)]
        old_mean = float(old.demand_mean.iloc[0]) if len(old) else 0.0
        # 泊松过程可加性：经过 elapsed_days 的累计量均值按 scale 缩放，
        # 其随机波动按 sqrt(scale) 缩放（而非分位数线性缩放）
        if len(old):
            full_lo = float(old.demand_p05.iloc[0])
            full_hi = float(old.demand_p95.iloc[0])
            old_p05 = max(0.0, old_mean * scale
                          + (full_lo - old_mean) * np.sqrt(scale))
            old_p95 = old_mean * scale + (
                full_hi - old_mean) * np.sqrt(scale)
        else:
            old_p05 = old_p95 = 0.0
        act = actual[(actual.station_id == station) & (actual.part_id == part)]
        actual_qty = int(act.qty.iloc[0]) if len(act) else 0

        if actual_qty > old_p95 or actual_qty < old_p05:
            category = "demand_error"
            reason = (f"截至回看点 {elapsed_days} 天实际新增需求 {actual_qty}，"
                      f"落在按时长折算后的区间 [{old_p05:.1f}, {old_p95:.1f}] "
                      f"之外（原 90 天区间均值 {old_mean:.1f}）")
        else:
            cat, reason = _check_fulfillment_and_compat(
                new_pack_pit, station, part, ledger, review_at, old_snapshot)
            category, reason = cat, reason
        rows.append({
            "station_id": station, "part_id": part,
            "old_mean": round(old_mean, 2),
            "scaled_p05": round(old_p05, 2),
            "scaled_p95": round(old_p95, 2),
            "elapsed_days": elapsed_days,
            "actual_new_demand": actual_qty, "gap_category": category,
            "reason": reason,
        })
    gap = pd.DataFrame(rows)

    # 承诺兑现状态
    commitment_rows = []
    for b in ledger.borrows:
        arrived = b["eta"] <= review_at
        commitment_rows.append({
            "commitment_id": b["commitment_id"], "kind": "borrow",
            "from": b["from_station"], "to": b["to_station"],
            "part_id": b["part_id"], "quantity": b["quantity"],
            "eta": b["eta"], "arrived_by_review": bool(arrived),
        })
    for p in ledger.purchases:
        arrived = p["eta"] <= review_at
        commitment_rows.append({
            "commitment_id": p["commitment_id"], "kind": p["kind"],
            "from": "-", "to": p["station_id"], "part_id": p["part_id"],
            "quantity": p["quantity"], "eta": p["eta"],
            "arrived_by_review": bool(arrived),
        })
    commitment_status = pd.DataFrame(commitment_rows)

    # 用新资料重新评估原动作（动作从快照重建）
    old_actions = [Action(**a) for a in old_snapshot.get("actions", [])]
    new_eval = evaluate(new_pack_pit, new_fc, old_actions, cutoff,
                        old_snapshot.get("budget", np.inf), scenario_id,
                        route_delay)
    assumptions = [
        "回看只使用 review_at 时点 available_at/occurred_at 不晚于该时点的资料；",
        "返修件以 return_accepted 对应的 stock_events(return) 作为验收合格回库依据，"
        "未验收的在修件不回库；",
        "区间内缺口先判需求误差，再判兼容/履约，避免同一缺口重复归因；",
        "原预测快照不可变，重预测结果另行保存，差异即模型/履约学习信号。",
    ]
    return ReplayResult(
        review_at=review_at, old_snapshot=old_snapshot,
        new_part_demand=new_fc.part_demand,
        actuals={"new_demand": actual_df.to_dict("records"),
                 "new_orders": int(len(new_pack_pit.work_orders[
                     new_pack_pit.work_orders.failed_at > cutoff]))},
        gap_attribution=gap, commitment_status=commitment_status,
        new_evaluation={
            "on_time_weighted": new_eval.on_time_weighted,
            "on_time_unweighted": new_eval.on_time_unweighted,
            "expected_backlog_units": new_eval.expected_backlog_units,
            "cash_committed": new_eval.cash_committed,
            "station_metrics": new_eval.station_metrics.to_dict("records"),
        }, assumptions=assumptions)


def _check_fulfillment_and_compat(pack: DataPack, station: str, part: str,
                                  ledger: CommitmentLedger, review_at: pd.Timestamp,
                                  old_snapshot: dict) -> tuple[str, str]:
    """需求在预测区间内时，区分兼容性判断与履约变化。"""
    review_naive = review_at.tz_localize(None) if review_at.tzinfo else review_at
    cutoff = pd.Timestamp(old_snapshot["cutoff"])
    cutoff_n = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    service_h = int(pack.stations.set_index("station_id").loc[
        station, "service_hours"])
    # 兼容性证据必须同时满足：截点后新工单、候选含该物料、已超服务时限、
    # 物料有库存却零领用。未超时的等待工单不构成兼容判断证据。
    lines = order_demand_lines(pack)
    blocked = lines[(lines.station_id == station)
                    & (lines.candidate_parts.str.contains(part))
                    & (lines.state == "waiting")
                    & (lines.failed_at > cutoff)]
    overdue = []
    for _, ln in blocked.iterrows():
        deadline = ln.failed_at + pd.Timedelta(hours=service_h)
        if review_at > deadline and ln.demand_occluded:
            overdue.append(ln)
    if overdue:
        from .inventory import on_hand_at
        if on_hand_at(pack, station, part, review_at) > 0:
            return ("compat_judgement",
                    f"{len(overdue)} 个已超服务时限工单在 {part} 有库存时"
                    "仍未领用，指向固件/资质/配套兼容判断偏差")
    # 履约：在途批次延期或承诺未达
    late_ship = pack.shipments[
        (pack.shipments.station_id == station)
        & (pack.shipments.part_id == part)
        & (pd.to_datetime(pack.shipments.earliest) > review_naive)]
    late_commits = [p for p in ledger.purchases
                    if p["station_id"] == station and p["part_id"] == part
                    and (p["eta"].tz_localize(None) if p["eta"].tzinfo
                         else p["eta"]) > review_naive]
    parts_txt = []
    if len(late_ship):
        parts_txt.append("在途批次晚于回看时点："
                         + ",".join(late_ship.shipment_id))
    if late_commits:
        parts_txt.append("采购承诺未到："
                         + ",".join(p["commitment_id"] for p in late_commits))
    if parts_txt:
        return "fulfillment_change", "；".join(parts_txt)
    if overdue:
        return ("fulfillment_change",
                "工单已超时限且零领用，但该物料无库存，属供给履约缺口"
                "（非兼容判断）")
    return ("fulfillment_change",
            "需求在区间内；等待工单尚未超服务时限或已有处理，"
            "按履约变化登记，待下一轮资料确认")
