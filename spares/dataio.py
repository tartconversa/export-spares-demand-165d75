"""合成资料装载与类型解析。

所有读入的数据均为合成业务资料。空值按口径保留为 NA（不解释为零），
时刻统一解析为带 +08:00 时区的 pandas Timestamp，日期为 date。
本模块只做装载与类型转换，不做业务解释。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TZ = "Asia/Shanghai"

TABLES = [
    "stations", "parts", "installed_base", "work_orders", "stock_events",
    "part_uses", "compatibility", "shipments", "stockouts", "plans",
]


def _read_csv(path: Path, dates=(), datetimes=()) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    for col in dates:
        frame[col] = pd.to_datetime(frame[col], errors="raise").dt.date
    for col in datetimes:
        frame[col] = pd.to_datetime(frame[col], errors="raise", utc=True).dt.tz_convert(TZ)
    return frame


@dataclass(frozen=True)
class DataPack:
    """一次装载的全部资料表。"""
    root: Path
    stations: pd.DataFrame
    parts: pd.DataFrame
    installed_base: pd.DataFrame
    work_orders: pd.DataFrame
    stock_events: pd.DataFrame
    part_uses: pd.DataFrame
    compatibility: pd.DataFrame
    shipments: pd.DataFrame
    stockouts: pd.DataFrame
    plans: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in TABLES}


def load_pack(root: str | Path) -> DataPack:
    """从资料目录读取全部 CSV 并完成类型解析。"""
    root = Path(root)
    stations = _read_csv(root / "stations.csv")
    parts = _read_csv(root / "parts.csv")
    installed_base = _read_csv(
        root / "installed_base.csv", dates=("commissioned", "warranty_end"))
    work_orders = _read_csv(
        root / "work_orders.csv",
        datetimes=("failed_at", "opened_at", "repair_at", "available_at"))
    stock_events = _read_csv(root / "stock_events.csv", datetimes=("occurred_at",))
    part_uses = _read_csv(root / "part_uses.csv", datetimes=("occurred_at",))
    compatibility = _read_csv(root / "compatibility.csv")
    shipments = _read_csv(root / "shipments.csv", dates=("earliest", "latest"))
    stockouts = _read_csv(root / "stockouts.csv", dates=("start", "end"))
    plans = _read_csv(root / "plans.csv")

    for col in ("service_hours",):
        stations[col] = stations[col].astype(int)
    for col in ("unit_cost", "cash_days", "normal_days", "expedite_days", "expedite_cost"):
        parts[col] = parts[col].astype(float if col in ("unit_cost", "expedite_cost") else int)
    for col in ("parts_per_repair",):
        compatibility[col] = compatibility[col].astype(int)
    for col in ("quantity",):
        stock_events[col] = stock_events[col].astype(int)
        part_uses[col] = part_uses[col].astype(int)
        shipments[col] = shipments[col].astype(int)
    for col in ("new_units", "route_delay_days"):
        plans[col] = plans[col].astype(int)
    for col in ("failure_multiplier", "budget"):
        plans[col] = plans[col].astype(float)
    work_orders["consequence"] = work_orders["consequence"].astype(int)

    return DataPack(
        root=root, stations=stations, parts=parts, installed_base=installed_base,
        work_orders=work_orders, stock_events=stock_events, part_uses=part_uses,
        compatibility=compatibility, shipments=shipments, stockouts=stockouts,
        plans=plans,
    )


def point_in_time(pack: DataPack, cutoff: pd.Timestamp) -> DataPack:
    """返回截点 cutoff 时可知的资料切片。

    口径：work_orders 按 available_at 过滤（事实最早可知时刻）；
    stock_events / part_uses 按 occurred_at 过滤；
    stockouts 保留在截点前已开始的登记区间（区间末端可晚于截点，
    因为缺货本身在开始时即已可知）；
    设备档案、站点、物料、兼容关系、在途批次属于主数据/已下单事实，
    截点时整体可知；plans 是情景假设而非观测，原样保留。
    严禁使用晚于 cutoff 才可知的修复结果解释截点前的状态。
    """
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(TZ)
    work_orders = pack.work_orders[pack.work_orders.available_at <= cutoff].copy()
    stock_events = pack.stock_events[pack.stock_events.occurred_at <= cutoff].copy()
    part_uses = pack.part_uses[pack.part_uses.occurred_at <= cutoff].copy()
    stockouts = pack.stockouts[
        pd.to_datetime(pack.stockouts.start).dt.tz_localize(TZ) <= cutoff
    ].copy()
    return DataPack(
        root=pack.root, stations=pack.stations, parts=pack.parts,
        installed_base=pack.installed_base, work_orders=work_orders,
        stock_events=stock_events, part_uses=part_uses,
        compatibility=pack.compatibility, shipments=pack.shipments,
        stockouts=stockouts, plans=pack.plans,
    )
