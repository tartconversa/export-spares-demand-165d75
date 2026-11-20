"""在途供给、改港与跨站互借的时点建模。

口径：
- 在途货有预计到达区间 [earliest, latest]，不按最早到达日当成确定库存；
- 同批共享路线风险：情景 route_delay_days 整体平移该路线批次；
- 改港产生 redirect_cost，改港后到达估计是用户输入假设，资料只给费用；
- 互借在承诺后从出借站可保障库存扣减，运输到达前借入站不算到库；
  互借运输时长资料未提供，统一用显式假设参数，默认值在假设登记中声明。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dataio import DataPack
from .inventory import CommitmentLedger

# 资料未提供跨站互借运输时长，采用显式默认假设（天），可由用户覆盖。
DEFAULT_BORROW_DAYS = 3
# 在到达区间内采用均匀分布表达“最早-最迟”的不确定性（显式建模假设）。
ARRIVAL_UNIFORM_ASSUMPTION = (
    "在途批次在 [earliest, latest] 上按均匀分布建模；资料只给区间未给分布，"
    "这是可由用户检查/替换的假设，不代表已观测概率。")


@dataclass
class SupplyArrival:
    source: str            # shipment_id / on_hand / borrow / purchase
    station_id: str
    part_id: str
    quantity: int
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None
    route: str | None
    redirect_cost: float
    note: str


def effective_shipment_window(shipment: pd.Series, route_delay_days: int = 0,
                              redirect_to_station: str | None = None
                              ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """应用情景迟延（及改港）后的到达区间。"""
    shift = pd.Timedelta(days=int(route_delay_days))
    earliest = pd.Timestamp(shipment.earliest) + shift
    latest = pd.Timestamp(shipment.latest) + shift
    if redirect_to_station is not None and redirect_to_station != shipment.station_id:
        # 改港额外耗时资料未提供：保守取区间宽度作为额外迟延的显式假设
        shift2 = pd.Timestamp(shipment.latest) - pd.Timestamp(shipment.earliest)
        earliest = earliest + shift2
        latest = latest + shift2
    tz = "Asia/Shanghai"
    return earliest.tz_localize(tz), latest.tz_localize(tz)


def arrival_probability_by(earliest: pd.Timestamp, latest: pd.Timestamp,
                           required_by: pd.Timestamp) -> float:
    """均匀到达假设下，批次在 required_by 之前（含）到达的概率。"""
    required_by = pd.Timestamp(required_by)
    if required_by >= latest:
        return 1.0
    if required_by < earliest:
        return 0.0
    span = (latest - earliest).total_seconds()
    if span <= 0:
        return 1.0
    return (required_by - earliest).total_seconds() / span


def supply_options(pack: DataPack, station_id: str, part_id: str,
                   cutoff: pd.Timestamp, ledger: CommitmentLedger | None = None,
                   route_delay_days: int = 0,
                   borrow_days: int = DEFAULT_BORROW_DAYS) -> list[SupplyArrival]:
    """列出某站某物料在截点后可调用的全部供给来源（不含新采购建议）。"""
    cutoff = pd.Timestamp(cutoff)
    ledger = ledger or CommitmentLedger()
    opts: list[SupplyArrival] = []

    guarantee = ledger.guarantee_stock(pack, station_id, part_id, cutoff)
    if guarantee > 0:
        opts.append(SupplyArrival(
            "on_hand", station_id, part_id, guarantee, cutoff, cutoff,
            None, 0.0, "现存可保障库存（已扣外借承诺）"))

    for _, s in pack.shipments[
            (pack.shipments.station_id == station_id)
            & (pack.shipments.part_id == part_id)].iterrows():
        e, l = effective_shipment_window(s, route_delay_days)
        opts.append(SupplyArrival(
            s.shipment_id, station_id, part_id, int(s.quantity), e, l,
            s.route, float(s.redirect_cost), "在途批次（可改港）"))

    # 跨站互借：他站可保障库存（含其在途不能借，未到库不是实物）
    for other in pack.stations.station_id:
        if other == station_id:
            continue
        qty = ledger.guarantee_stock(pack, other, part_id, cutoff)
        if qty <= 0:
            continue
        eta = cutoff + pd.Timedelta(days=borrow_days)
        opts.append(SupplyArrival(
            f"borrow:{other}", other, part_id, qty, eta, eta, None, 0.0,
            f"从 {other} 互借（运输 {borrow_days} 天为显式假设，承诺即扣减 {other} 保障能力）"))
    return opts


def purchase_lead_time(part: pd.Series, mode: str,
                       route_delay_days: int = 0) -> int:
    """采购交期（天）：normal/expedite，情景共同迟延同样作用于常规订货。"""
    base = int(part.normal_days if mode == "normal" else part.expedite_days)
    # 加急通常走优先通道，情景迟延减半作用（显式假设，登记于假设清单）
    delay = route_delay_days if mode == "normal" else max(
        0, route_delay_days // 2)
    return base + delay


def expected_eta_table(pack: DataPack, cutoff: pd.Timestamp,
                       route_delay_days: int = 0) -> pd.DataFrame:
    """全部在途批次应用情景迟延后的到达区间与期望到达日。"""
    rows = []
    for _, s in pack.shipments.iterrows():
        e, l = effective_shipment_window(s, route_delay_days)
        rows.append({
            "shipment_id": s.shipment_id, "part_id": s.part_id,
            "station_id": s.station_id, "route": s.route,
            "quantity": int(s.quantity),
            "earliest_adjusted": e.date(), "latest_adjusted": l.date(),
            "expected_eta": (e + (l - e) / 2).date(),
            "redirect_cost": float(s.redirect_cost),
        })
    return pd.DataFrame(rows)
