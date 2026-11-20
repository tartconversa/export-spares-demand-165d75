"""演练用“后续实际资料”生成与合并。

真实使用时，业务系统导出的同结构文件直接放到 actuals 目录即可；
本生成器只用于在合成资料上跑通完整季度演练，明确标注为模拟实际，
不参与任何预测估计（预测只看截点前资料）。

生成三类事件，对应业务要求的三种触发：
- 新故障涌入：work_orders_extra.csv（available_at 晚于原截点）；
- 航运批次延期：shipments_revised.csv（覆盖原批次到达区间）；
- 返修验收合格：stock_events_extra.csv(return) + part_uses_extra.csv。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import TZ, DataPack, point_in_time

EXTRA_WO_COLUMNS = ["order_id", "unit_id", "failed_at", "opened_at",
                    "repair_at", "state", "consequence", "available_at"]
EXTRA_SE_COLUMNS = ["stock_event_id", "station_id", "part_id",
                    "occurred_at", "kind", "quantity"]
EXTRA_PU_COLUMNS = ["use_id", "order_id", "station_id", "part_id",
                    "quantity", "occurred_at", "kind"]


def generate_actuals(pack: DataPack, cutoff: pd.Timestamp, out_dir: str | Path,
                     days: int = 45, seed: int = 11, delay_days: int = 10,
                     returns: int = 3, new_failures: int | None = None,
                     expected_90d_failures: float | None = None) -> Path:
    """生成演练用后续实际资料。

    new_failures 为 None 时按 expected_90d_failures（缺省取历史观测速率
    外推）和经过天数推导，避免演练样本与模型量级系统性脱节；
    用户仍可显式传入任意数量以构造高/低需求压力情景。
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(cutoff)
    ib = pack.installed_base
    existing_ids = set(pack.work_orders.order_id)

    if new_failures is None:
        if expected_90d_failures is None:
            # 历史速率：观测期故障数 / 总台日 × 90 × 当前在保台数
            ib_dt = ib.copy()
            ib_dt["commissioned"] = pd.to_datetime(ib_dt["commissioned"])
            obs_end = pack.work_orders.failed_at.max().tz_localize(None)
            unit_days = ((obs_end - ib_dt.commissioned).dt.days + 1).sum()
            rate = len(pack.work_orders) / max(1.0, unit_days)
            expected_90d_failures = rate * 90 * len(ib)
        new_failures = int(round(expected_90d_failures * days / 90.0))

    # 1) 新故障：从未故障设备中抽样，部分设为 waiting（制造履约压力）
    failed_units = set(pack.work_orders.unit_id)
    pool = ib[~ib.unit_id.isin(failed_units)]
    picks = pool.sample(n=min(new_failures, len(pool)), random_state=seed)
    wo_rows = []
    for i, (_, u) in enumerate(picks.iterrows(), 1):
        oid = f"XW{i}"
        existing_ids.add(oid)
        failed = cutoff + pd.Timedelta(days=int(rng.integers(2, days - 5)),
                                       hours=8)
        opened = failed + pd.Timedelta(days=1)
        is_waiting = i % 3 == 0
        repair = "" if is_waiting else (failed + pd.Timedelta(days=2))
        wo_rows.append({
            "order_id": oid, "unit_id": u.unit_id,
            "failed_at": failed.isoformat(), "opened_at": opened.isoformat(),
            "repair_at": repair.isoformat() if repair else "",
            "state": "waiting" if is_waiting else "closed",
            "consequence": int(rng.choice([1, 1, 1, 3])),
            "available_at": (opened + pd.Timedelta(days=2)).isoformat(),
        })
    pd.DataFrame(wo_rows, columns=EXTRA_WO_COLUMNS).to_csv(
        out / "work_orders_extra.csv", index=False)

    # 2) 批次延期：每个批次的最早/最迟整体后移 delay_days
    rev = pack.shipments.copy()
    rev["earliest"] = (pd.to_datetime(rev.earliest)
                       + pd.Timedelta(days=delay_days)).dt.date
    rev["latest"] = (pd.to_datetime(rev.latest)
                     + pd.Timedelta(days=delay_days)).dt.date
    rev.to_csv(out / "shipments_revised.csv", index=False)

    # 3) 返修验收合格：从历史 issue 过 P1 的站随机回库
    p1_stations = (pack.part_uses[(pack.part_uses.part_id == "P1")
                                  & (pack.part_uses.kind == "issue")]
                   .station_id.unique())
    se_rows, pu_rows = [], []
    for i in range(returns):
        station = str(rng.choice(p1_stations))
        t = cutoff + pd.Timedelta(days=int(rng.integers(3, days - 3)), hours=8)
        se_rows.append({
            "stock_event_id": f"XSE{i+1}", "station_id": station,
            "part_id": "P1", "occurred_at": t.isoformat(),
            "kind": "return", "quantity": 1,
        })
        # 返修件来自一张已关闭 G2 工单（回库记录，不取消原需求）
        g2_order = pack.work_orders.merge(
            ib[["unit_id", "generation", "station_id"]], on="unit_id")
        g2_order = g2_order[(g2_order.generation == "G2")
                            & (g2_order.station_id == station)
                            & (g2_order.state == "closed")]
        oid = g2_order.order_id.iloc[i % len(g2_order)]
        pu_rows.append({
            "use_id": f"XPU{i+1}", "order_id": oid, "station_id": station,
            "part_id": "P1", "quantity": 1,
            "occurred_at": t.isoformat(), "kind": "return_accepted",
        })
    pd.DataFrame(se_rows, columns=EXTRA_SE_COLUMNS).to_csv(
        out / "stock_events_extra.csv", index=False)
    pd.DataFrame(pu_rows, columns=EXTRA_PU_COLUMNS).to_csv(
        out / "part_uses_extra.csv", index=False)

    manifest = {"note": "演练模拟实际资料，不用于截点前估计",
                "cutoff": str(cutoff), "delay_days": delay_days,
                "new_failures": new_failures, "returns": returns}
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return out


def merge_actuals(base: DataPack, actuals_dir: str | Path,
                  review_at: pd.Timestamp) -> DataPack:
    """把演练实际资料合并进资料包，并按回看时点做可知性切片。"""
    d = Path(actuals_dir)
    review_at = pd.Timestamp(review_at)

    def read(name):
        return pd.read_csv(d / name, dtype=str, keep_default_na=False,
                           na_values=[""])

    wo_extra = read("work_orders_extra.csv")
    for c in ("failed_at", "opened_at", "repair_at", "available_at"):
        wo_extra[c] = pd.to_datetime(wo_extra[c], errors="coerce",
                                     utc=True).dt.tz_convert(TZ)
    wo_extra["consequence"] = wo_extra["consequence"].astype(int)
    wo = pd.concat([base.work_orders, wo_extra], ignore_index=True)

    se_extra = read("stock_events_extra.csv")
    se_extra["occurred_at"] = pd.to_datetime(
        se_extra.occurred_at, utc=True).dt.tz_convert(TZ)
    se_extra["quantity"] = se_extra.quantity.astype(int)
    se = pd.concat([base.stock_events, se_extra], ignore_index=True)

    pu_extra = read("part_uses_extra.csv")
    pu_extra["occurred_at"] = pd.to_datetime(
        pu_extra.occurred_at, utc=True).dt.tz_convert(TZ)
    pu_extra["quantity"] = pu_extra.quantity.astype(int)
    pu = pd.concat([base.part_uses, pu_extra], ignore_index=True)

    ships = base.shipments
    rev_path = d / "shipments_revised.csv"
    if rev_path.exists():
        rev = pd.read_csv(rev_path, parse_dates=["earliest", "latest"])
        ships = rev.copy()
        ships["earliest"] = rev.earliest.dt.date
        ships["latest"] = rev.latest.dt.date
        for c in ("quantity",):
            ships[c] = ships[c].astype(int)
        for c in ("redirect_cost",):
            ships[c] = ships[c].astype(float)

    extended = DataPack(
        root=base.root, stations=base.stations, parts=base.parts,
        installed_base=base.installed_base, work_orders=wo,
        stock_events=se, part_uses=pu, compatibility=base.compatibility,
        shipments=ships, stockouts=base.stockouts, plans=base.plans)
    return point_in_time(extended, review_at)
