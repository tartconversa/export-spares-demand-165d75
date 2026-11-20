"""站点库存过程：现存库存、缺货遮挡与承诺扣减。

口径要点（docs/资料口径.md）：
- 本包起始库存为零，stock_events 的累计和即现存库存；
- receipt/return 为正、issue 为负；返修件验收合格（return_accepted 对应的
  stock_events return）才回库，退回不等于原工单需求消失；
- 已确认互借在承诺后从出借站可保障库存中扣除，但收件方在到达前不算到库；
- 尚未落实的建议不占实物。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .dataio import DataPack


def on_hand_at(pack: DataPack, station_id: str, part_id: str,
               when: pd.Timestamp) -> int:
    """某站某物料在 when 时点的现存库存（不含在途、不含已承诺未出库）。"""
    events = pack.stock_events[
        (pack.stock_events.station_id == station_id)
        & (pack.stock_events.part_id == part_id)
        & (pack.stock_events.occurred_at <= when)
    ]
    return int(events.quantity.sum())


def on_hand_table(pack: DataPack, when: pd.Timestamp | None = None) -> pd.DataFrame:
    """全部站点×物料的现存库存表。when 为空时取资料中最晚事件时点。"""
    if when is None:
        when = pack.stock_events.occurred_at.max()
    rows = []
    for s in pack.stations.station_id:
        for p in pack.parts.part_id:
            rows.append((s, p, on_hand_at(pack, s, p, when)))
    return pd.DataFrame(rows, columns=["station_id", "part_id", "on_hand"])


def stock_timeline(pack: DataPack, station_id: str, part_id: str) -> pd.DataFrame:
    """单站单物料的库存变化时间线（累计现存库存）。"""
    events = pack.stock_events[
        (pack.stock_events.station_id == station_id)
        & (pack.stock_events.part_id == part_id)
    ].sort_values("occurred_at")
    out = events[["occurred_at", "kind", "quantity"]].copy()
    out["running_on_hand"] = events.quantity.cumsum().values
    return out.reset_index(drop=True)


def zero_stock_intervals(pack: DataPack) -> pd.DataFrame:
    """由库存事件推出的现存库存为零的时段（左闭右开）。

    与独立登记的 stockouts 互相印证：登记缺货保留原始业务原因，
    本函数只反映库存事件本身，不修改任何原始数量。
    """
    rows = []
    for (station, part), grp in pack.stock_events.sort_values("occurred_at").groupby(
            ["station_id", "part_id"]):
        running = 0
        zero_since = None
        for _, e in grp.iterrows():
            running += int(e.quantity)
            if running == 0 and zero_since is None:
                zero_since = e.occurred_at
            elif running > 0 and zero_since is not None:
                rows.append((station, part, zero_since, e.occurred_at))
                zero_since = None
        if zero_since is not None:
            rows.append((station, part, zero_since, None))
    return pd.DataFrame(rows, columns=["station_id", "part_id", "zero_from", "zero_to"])


@dataclass
class CommitmentLedger:
    """已落实的采购与互借承诺。

    只有进入本台账的动作才占用实物或资金：
    - borrow 承诺立即从出借站可保障库存扣减，到达后才加入借入站库存；
    - purchase 承诺在到货入库时增加库存，承诺时先记资金占用；
    - 建议（未确认）不得写入本台账。
    """
    borrows: list[dict] = field(default_factory=list)
    purchases: list[dict] = field(default_factory=list)

    def add_borrow(self, borrow_id: str, from_station: str, to_station: str,
                   part_id: str, quantity: int, commit_at: pd.Timestamp,
                   eta: pd.Timestamp) -> None:
        if quantity <= 0:
            raise ValueError("互借数量必须为正")
        self.borrows.append({
            "commitment_id": borrow_id, "kind": "borrow",
            "from_station": from_station, "to_station": to_station,
            "part_id": part_id, "quantity": int(quantity),
            "commit_at": pd.Timestamp(commit_at), "eta": pd.Timestamp(eta),
        })

    def add_purchase(self, commitment_id: str, station_id: str, part_id: str,
                     quantity: int, mode: str, commit_at: pd.Timestamp,
                     unit_cost: float, fee: float, eta: pd.Timestamp) -> None:
        if mode not in {"normal", "expedite"}:
            raise ValueError("采购模式必须是 normal 或 expedite")
        if quantity <= 0:
            raise ValueError("采购数量必须为正")
        self.purchases.append({
            "commitment_id": commitment_id, "kind": "purchase", "mode": mode,
            "station_id": station_id, "part_id": part_id, "quantity": int(quantity),
            "commit_at": pd.Timestamp(commit_at), "eta": pd.Timestamp(eta),
            "unit_cost": float(unit_cost), "fee": float(fee),
        })

    def lender_reserved(self, station_id: str, part_id: str,
                        when: pd.Timestamp) -> int:
        """出借站在 when 时点已承诺转出、尚未到达借入站的数量。"""
        total = 0
        for b in self.borrows:
            if (b["from_station"] == station_id and b["part_id"] == part_id
                    and b["commit_at"] <= when < b["eta"]):
                total += b["quantity"]
        return total

    def guarantee_stock(self, pack: DataPack, station_id: str, part_id: str,
                        when: pd.Timestamp) -> int:
        """可保障库存＝现存库存 − 本站已承诺外借未达部分。"""
        return on_hand_at(pack, station_id, part_id, when) - self.lender_reserved(
            station_id, part_id, when)

    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (pd.DataFrame(self.borrows), pd.DataFrame(self.purchases))
