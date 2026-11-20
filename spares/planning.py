"""保障方案的动作建模、履约模拟与预算内选优。

评价对象是“设备能否按服务承诺恢复”，不是库存排行榜：
- 以服务后果权重 consequence 加权的按时恢复概率为首要指标；
- 同时报告未满足需求（积压风险）与资金占用；
- waiting 工单始终计入待履行需求，不允许通过拒收困难工单美化履约率。

四种动作：
- normal   常规订货：交期长、无加急费；
- expedite 加急订货：交期短、单件 expedite_cost；
- redirect 改港：把在途批次改到别站，付 redirect_cost；
- borrow   跨站互借：承诺即扣出借站可保障库存，到达前不入借入站。

履约判定按工单的完整配套做：不同 required_group 的物料必须在服务时限内
同时到齐才算按时恢复；同组替代件任一满足即可。

性能结构：基础供给流（现存、在途、所有可能改港路径的到达抽样）只依赖
随机种子与情景迟延，在一次选优中预计算一次；每套候选动作只做增量
扣减/追加，避免重复抽样与重复排序。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataio import DataPack
from .forecast import ForecastResult, forecast
from .inventory import on_hand_at
from .supply import (DEFAULT_BORROW_DAYS, effective_shipment_window,
                     purchase_lead_time)

N_SIM_DEFAULT = 2000
SCREEN_SIMS = 400


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    """去掉时区，保留 +08:00 墙钟值；仿真内全部用朴素墙钟时间比较。"""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


@dataclass
class Action:
    kind: str                  # normal | expedite | redirect | borrow
    station_id: str
    part_id: str
    quantity: int
    unit_cost: float
    fee: float                 # 加急费/改港费（总额）
    eta_earliest: pd.Timestamp
    eta_latest: pd.Timestamp
    ref: str = ""
    detail: str = ""

    @property
    def cash(self) -> float:
        return self.unit_cost * self.quantity + self.fee


@dataclass
class PlanEvaluation:
    actions: list[Action]
    cutoff: pd.Timestamp
    horizon_days: int
    scenario_id: str | None
    on_time_weighted: float
    on_time_unweighted: float
    expected_backlog_units: float
    sla_breach_weighted: float
    cash_committed: float
    budget: float
    within_budget: bool
    station_metrics: pd.DataFrame
    assumptions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 动作枚举
# ---------------------------------------------------------------------------

def candidate_actions(pack: DataPack, fc: ForecastResult, cutoff: pd.Timestamp,
                      route_delay: int, borrow_days: int = DEFAULT_BORROW_DAYS
                      ) -> list[Action]:
    """枚举可行动作：采购/加急补 P95 缺口、有效改港路径、可借存量互借。

    全局互斥约束（由 optimize 的可行性校验强制）：
    - 同一 shipment 最多改港一次（一个批次物理上只能到一个目的站）；
    - 同一出借站×物料累计外借不超过其可保障存量。
    """
    actions: list[Action] = []
    parts = pack.parts.set_index("part_id")
    demand_keys = set(zip(fc.part_demand.station_id, fc.part_demand.part_id))

    for _, d in fc.part_demand.iterrows():
        station, part = d.station_id, d.part_id
        in_transit = int(pack.shipments[
            (pack.shipments.station_id == station)
            & (pack.shipments.part_id == part)].quantity.sum())
        gap95 = max(0, int(np.ceil(d.demand_p95))
                    - on_hand_at(pack, station, part, cutoff) - in_transit)
        if gap95 <= 0:
            continue
        p = parts.loc[part]
        for mode in ("normal", "expedite"):
            lead = purchase_lead_time(p, mode, route_delay)
            eta = cutoff + pd.Timedelta(days=lead)
            fee = 0.0 if mode == "normal" else float(p.expedite_cost) * gap95
            actions.append(Action(
                mode, station, part, gap95, float(p.unit_cost), fee,
                eta, eta, ref=mode,
                detail=f"{mode} 交期 {lead} 天，补 P95 缺口 {gap95} 件"))

    # 改港：每个批次只保留“该物料预测缺口最大的一个目的站”作为候选，
    # 从源头避免同批次多目的站的重复承诺
    demand = fc.part_demand.set_index(["station_id", "part_id"])
    for _, shp in pack.shipments.iterrows():
        e0, l0 = effective_shipment_window(shp, route_delay)
        width = l0 - e0
        options = [t for t in pack.stations.station_id
                   if t != shp.station_id and (t, shp.part_id) in demand_keys]
        if not options:
            continue

        def gap_of(t):
            row = demand.loc[(t, shp.part_id)]
            in_transit = int(pack.shipments[
                (pack.shipments.station_id == t)
                & (pack.shipments.part_id == shp.part_id)].quantity.sum())
            return (float(row.demand_p95)
                    - on_hand_at(pack, t, shp.part_id, cutoff) - in_transit)

        target = max(options, key=gap_of)
        if gap_of(target) <= 0:
            continue
        actions.append(Action(
            "redirect", target, shp.part_id, int(shp.quantity),
            0.0, float(shp.redirect_cost), e0 + width, l0 + width,
            ref=shp.shipment_id,
            detail=f"批次 {shp.shipment_id} 改港 {shp.station_id}->{target}"))

    # 互借候选：他站现存扣除“自身 P95 需求与在途”后仍有真实剩余时可借。
    # 只对需求表中出现的物料开放；P3/P5 无故障需求，不进入互借。
    demand_idx = fc.part_demand.set_index(["station_id", "part_id"])
    for _, d in fc.part_demand.iterrows():
        station, part = d.station_id, d.part_id
        for other in pack.stations.station_id:
            if other == station:
                continue
            if (other, part) not in demand_idx.index:
                spare = on_hand_at(pack, other, part, cutoff)
            else:
                other_row = demand_idx.loc[(other, part)]
                other_in_transit = int(pack.shipments[
                    (pack.shipments.station_id == other)
                    & (pack.shipments.part_id == part)].quantity.sum())
                spare = (on_hand_at(pack, other, part, cutoff)
                         + other_in_transit
                         - int(np.ceil(other_row.demand_p95)))
            if spare <= 0:
                continue
            eta = cutoff + pd.Timedelta(days=borrow_days)
            actions.append(Action(
                "borrow", station, part,
                min(spare, int(np.ceil(d.demand_p95))),
                0.0, 0.0, eta, eta, ref=other,
                detail=f"互借 {other}->{station}，承诺即扣 {other} 保障库存"))
    return actions


def _feasible_with(chosen: list[Action], candidate: Action,
                   pack: DataPack, cutoff: pd.Timestamp) -> bool:
    """全局可行性：批次唯一改港、互借不超出借站可保障存量。"""
    if candidate.kind == "redirect":
        if any(a.kind == "redirect" and a.ref == candidate.ref for a in chosen):
            return False
    if candidate.kind == "borrow":
        already = sum(a.quantity for a in chosen
                      if a.kind == "borrow" and a.ref == candidate.ref
                      and a.part_id == candidate.part_id)
        available = on_hand_at(pack, candidate.ref, candidate.part_id, cutoff)
        if already + candidate.quantity > available:
            return False
    return True


# ---------------------------------------------------------------------------
# 基础供给流预计算（只依赖种子与情景迟延，与动作无关）
# ---------------------------------------------------------------------------

def _base_supply_draws(pack: DataPack, cutoff: pd.Timestamp,
                       route_delay: int, n_sims: int, seed: int) -> dict:
    """每个 (站, 物料) 的基础供给行：(来源id, 数量, 到达时刻数组 datetime64)。

    同时预生成每个批次改往每个其他站的到达流，动作评估时直接选取，
    保证同一改港动作在不同候选组合中的抽样完全一致（公共随机数）。
    """
    rng = np.random.default_rng(seed)
    cutoff_n = np.datetime64(_naive(cutoff).value, "ns")
    base: dict[tuple[str, str], list[tuple[str, int, np.ndarray]]] = {}
    for s in pack.stations.station_id:
        for p in pack.parts.part_id:
            rows = []
            oh = on_hand_at(pack, s, p, cutoff)
            if oh > 0:
                rows.append(("on_hand", oh, np.full(n_sims, cutoff_n)))
            base[(s, p)] = rows

    for _, shp in pack.shipments.iterrows():
        e, l = effective_shipment_window(shp, route_delay)
        e64, l64 = np.datetime64(_naive(e).value, "ns"), np.datetime64(
            _naive(l).value, "ns")
        u = rng.uniform(size=n_sims)
        arrivals = e64 + (l64 - e64) * u
        base[(shp.station_id, shp.part_id)].append(
            (shp.shipment_id, int(shp.quantity), arrivals))
        # 改港流：改港额外迟延取原区间宽度（supply 模块的显式假设）
        width = l64 - e64
        for target in pack.stations.station_id:
            if target == shp.station_id:
                continue
            base[(target, shp.part_id)].append(
                (f"redirect:{shp.shipment_id}", int(shp.quantity),
                 arrivals + width))
    return base


def _build_matrix(base_rows: list[tuple[str, int, np.ndarray]],
                  actions: list[Action], station: str, part: str,
                  cutoff: pd.Timestamp, n_sims: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """在基础供给流上应用动作：改港移除原批次、互借扣出借站现存、追加新供给。"""
    cutoff64 = np.datetime64(_naive(cutoff).value, "ns")
    removed = {a.ref for a in actions if a.kind == "redirect"
               and a.station_id == station and a.part_id == part}
    lender_cut = sum(a.quantity for a in actions
                     if a.kind == "borrow" and a.ref == station
                     and a.part_id == part)
    rows: list[tuple[int, np.ndarray]] = []
    for source, qty, arrivals in base_rows:
        if source in removed:
            continue
        if source == "on_hand" and lender_cut > 0:
            take = min(lender_cut, qty)
            qty -= take
            lender_cut -= take
            if qty <= 0:
                continue
        rows.append((qty, arrivals))
    for a in actions:
        if a.station_id != station or a.part_id != part:
            continue
        if a.kind == "redirect":
            stream = next(arr for src, _, arr in base_rows
                          if src == f"redirect:{a.ref}")
            rows.append((a.quantity, stream))
        elif a.kind in ("borrow", "normal", "expedite"):
            rows.append((a.quantity, np.full(
                n_sims, np.datetime64(_naive(a.eta_latest).value, "ns"))))
    if not rows:
        return (np.empty((0, n_sims), dtype="datetime64[ns]"),
                np.empty((0, n_sims), dtype=int))
    mat = np.vstack([np.asarray(d, dtype="datetime64[ns]")
                     for _, d in rows])
    qty_vec = np.array([q for q, _ in rows], dtype=int)
    order = np.argsort(mat, axis=0)
    sorted_mat = np.take_along_axis(mat, order, axis=0)
    sorted_qty = np.take_along_axis(
        np.broadcast_to(qty_vec.reshape(-1, 1), mat.shape), order, axis=0)
    return sorted_mat, np.cumsum(sorted_qty, axis=0)


def _nth_arrival(mat: np.ndarray, cum_qty: np.ndarray, nth: np.ndarray,
                 n_sims: int) -> np.ndarray:
    """第 nth 件物料在每次模拟中的到达时刻；供给不足返回 NaT。"""
    out = np.full(n_sims, np.datetime64("NaT"), dtype="datetime64[ns]")
    if cum_qty.size == 0:
        return out
    has_cap = cum_qty.max(axis=0) >= nth
    rows_with = np.where(has_cap, (cum_qty >= nth).argmax(axis=0), 0)
    out[has_cap] = mat[rows_with[has_cap], np.arange(n_sims)[has_cap]]
    return out


# ---------------------------------------------------------------------------
# 履约蒙特卡洛（工单级配套判定）
# ---------------------------------------------------------------------------

def _simulate(pack: DataPack, fc: ForecastResult, actions: list[Action],
              cutoff: pd.Timestamp, route_delay: int, n_sims: int,
              seed: int, borrow_days: int,
              base_draws: dict | None = None) -> dict:
    rng = np.random.default_rng(seed + 1)
    st = pack.stations.set_index("station_id")
    cutoff_n = _naive(cutoff)
    if base_draws is None:
        base_draws = _base_supply_draws(pack, cutoff, route_delay, n_sims, seed)

    # 后果权重分布（每站一次）
    wo_with_station = pack.work_orders.merge(
        pack.installed_base[["unit_id", "station_id"]], on="unit_id")
    cons_lookup = {
        s: sub.consequence.value_counts(normalize=True).sort_index()
        for s, sub in wo_with_station.groupby("station_id")}

    backlog_events = []
    for order_id, grp in fc.backlog.groupby("order_id"):
        first = grp.iloc[0]
        backlog_events.append({
            "station_id": first.station_id,
            "needs": dict(zip(grp.part_id, grp.need_qty.astype(int))),
            "failed_at": _naive(first.failed_at),
            "consequence": int(first.consequence)})

    bom = (pack.compatibility.sort_values(["generation", "firmware",
                                           "required_group", "part_id"])
           .groupby("generation")["part_id"]
           .apply(lambda s: sorted(set(s))).to_dict())
    new_batches = []
    for _, row in fc.failure_forecast.iterrows():
        counts = rng.poisson(max(row.failures_mean, 0.0), size=n_sims)
        max_n = int(counts.max())
        if max_n == 0:
            continue
        offsets = rng.uniform(0, fc.horizon_days, size=(n_sims, max_n))
        dist = cons_lookup[row.station_id]
        cons = rng.choice(dist.index.astype(int), size=(n_sims, max_n),
                          p=dist.values)
        new_batches.append((row.station_id,
                            {p: 1 for p in bom[row.generation]},
                            offsets, cons, counts))

    matrix_cache: dict[tuple[str, str], tuple] = {}

    def get_matrix(station, part):
        key = (station, part)
        if key not in matrix_cache:
            matrix_cache[key] = _build_matrix(
                base_draws[key], actions, station, part, cutoff, n_sims)
        return matrix_cache[key]

    allocated = {(s, p): np.zeros(n_sims, dtype=int)
                 for s in pack.stations.station_id
                 for p in pack.parts.part_id}
    met_w = np.zeros(n_sims)
    met_u = np.zeros(n_sims)
    total_w = np.zeros(n_sims)
    total_u = np.zeros(n_sims)
    unfilled = np.zeros(n_sims)
    per_station = {s: {"met_w": np.zeros(n_sims), "total_w": np.zeros(n_sims),
                       "met_u": np.zeros(n_sims), "total_u": np.zeros(n_sims),
                       "backlog": np.zeros(n_sims)}
                   for s in pack.stations.station_id}

    def judge(station, needs, failed_times, cons_arr, valid):
        service_h = int(st.loc[station, "service_hours"])
        deadlines = np.asarray(
            failed_times + pd.Timedelta(hours=service_h),
            dtype="datetime64[ns]")
        on_time = valid.copy()
        ever_supplied = valid.copy()
        for part, qty in needs.items():
            mat, cum = get_matrix(station, part)
            rank = np.maximum(allocated[(station, part)] + qty, 1)
            arrivals = _nth_arrival(mat, cum, rank, n_sims)
            nat = np.isnat(arrivals)
            on_time &= (~nat) & (arrivals <= deadlines)
            ever_supplied &= ~nat
            allocated[(station, part)] += np.where(valid, qty, 0)
        w = np.where(valid, cons_arr, 0)
        nonlocal met_w, met_u, total_w, total_u, unfilled
        total_w += w
        total_u += valid.astype(int)
        met_w += np.where(on_time, w, 0)
        met_u += on_time.astype(int)
        unfilled += np.where(valid & ~ever_supplied, sum(needs.values()), 0)
        d = per_station[station]
        d["total_w"] += w
        d["total_u"] += valid.astype(int)
        d["met_w"] += np.where(on_time, w, 0)
        d["met_u"] += on_time.astype(int)
        d["backlog"] += np.where(valid & ~ever_supplied,
                                 sum(needs.values()), 0)

    for ev in backlog_events:
        judge(ev["station_id"], ev["needs"],
              np.full(n_sims, np.datetime64(ev["failed_at"].value, "ns")),
              np.full(n_sims, ev["consequence"]),
              np.ones(n_sims, dtype=bool))

    for station, needs, offsets, cons, counts in sorted(
            new_batches, key=lambda b: b[2].mean()):
        for k in range(offsets.shape[1]):
            valid = counts > k
            if not valid.any():
                continue
            failed_times = np.datetime64(cutoff_n.value, "ns") + (
                offsets[:, k] * 86400 * 1e9).astype("timedelta64[ns]")
            judge(station, needs, failed_times, cons[:, k], valid)

    station_metrics = pd.DataFrame([
        {"station_id": s,
         "on_time_weighted": float(
             (d["met_w"] / np.maximum(d["total_w"], 1)).mean()),
         "on_time_unweighted": float(
             (d["met_u"] / np.maximum(d["total_u"], 1)).mean()),
         "expected_backlog_units": float(d["backlog"].mean())}
        for s, d in per_station.items()])
    return {
        "on_time_weighted": float((met_w / np.maximum(total_w, 1)).mean()),
        "on_time_unweighted": float((met_u / np.maximum(total_u, 1)).mean()),
        "sla_breach_weighted": float(
            ((total_w - met_w) / np.maximum(total_w, 1)).mean()),
        "expected_backlog_units": float(unfilled.mean()),
        "station_metrics": station_metrics,
    }


def evaluate(pack: DataPack, fc: ForecastResult, actions: list[Action],
             cutoff: pd.Timestamp, budget: float, scenario_id: str | None,
             route_delay: int, n_sims: int = N_SIM_DEFAULT, seed: int = 7,
             borrow_days: int = DEFAULT_BORROW_DAYS,
             base_draws: dict | None = None) -> PlanEvaluation:
    """评估一套动作的履约表现、资金占用与预算合规性。"""
    sim = _simulate(pack, fc, actions, cutoff, route_delay, n_sims, seed,
                    borrow_days, base_draws)
    cash = sum(a.cash for a in actions)
    assumptions = [
        "新故障在预测窗内均匀发生；资料只覆盖34天且日计数近似常数，"
        "无法识别窗内季节性。",
        "后果权重沿用历史工单的站点分布抽样。",
        "在途与改港到达在区间内均匀抽样；互借运输时长为显式参数"
        f"（默认 {borrow_days} 天），资料未提供真实运输数据。",
        "同路线批次共享 route_delay_days 情景迟延；不同路线是否共同受扰，"
        "资料不支持独立假设，情景内按同一区域事件处理，情景间不做相关性推断。",
        "采购在承诺日按交期确定到达（单点估计），交期本身的不确定性未建模；"
        "加急对情景迟延减半敏感是显式假设。",
        "先坏先修（FCFS）占用站点供给；资料未提供派工优先级规则，"
        "优化器按后果权重选动作，但仿真内不做人工插队。",
    ]
    return PlanEvaluation(
        actions=actions, cutoff=cutoff, horizon_days=fc.horizon_days,
        scenario_id=scenario_id,
        on_time_weighted=sim["on_time_weighted"],
        on_time_unweighted=sim["on_time_unweighted"],
        expected_backlog_units=sim["expected_backlog_units"],
        sla_breach_weighted=sim["sla_breach_weighted"],
        cash_committed=cash, budget=budget, within_budget=cash <= budget,
        station_metrics=sim["station_metrics"], assumptions=assumptions)


# ---------------------------------------------------------------------------
# 预算内贪心选优
# ---------------------------------------------------------------------------

def optimize(pack: DataPack, cutoff: pd.Timestamp, budget: float,
             scenario_id: str | None = None, n_sims: int = N_SIM_DEFAULT,
             seed: int = 7, borrow_days: int = DEFAULT_BORROW_DAYS
             ) -> tuple[list[Action], PlanEvaluation, list[dict]]:
    """预算内贪心选优：以单位资金带来的加权 SLA 改善排序。

    筛选阶段用较少模拟次数与公共随机数（同一基础供给流）比较候选，
    选中后才用完整 n_sims 复算最终指标，兼顾速度与稳定性。
    """
    route_delay = 0
    if scenario_id is not None:
        hit = pack.plans[pack.plans.scenario_id == scenario_id]
        route_delay = int(hit.route_delay_days.iloc[0])
    fc = forecast(pack, cutoff, scenario_id=scenario_id, seed=seed)
    screen_base = _base_supply_draws(pack, cutoff, route_delay, SCREEN_SIMS,
                                     seed)
    base = _simulate(pack, fc, [], cutoff, route_delay, SCREEN_SIMS, seed,
                     borrow_days, screen_base)
    cands = candidate_actions(pack, fc, cutoff, route_delay, borrow_days)
    dedup = {}
    for a in cands:
        key = (a.kind, a.station_id, a.part_id, a.ref)
        if key not in dedup or a.quantity > dedup[key].quantity:
            dedup[key] = a
    cands = list(dedup.values())

    chosen: list[Action] = []
    spent = 0.0
    current_metrics = base
    trace = [{"step": 0, "action": "(基线无动作)", "cash": 0.0,
              "on_time_weighted": round(base["on_time_weighted"], 4),
              "expected_backlog_units": round(
                  base["expected_backlog_units"], 2)}]
    while True:
        best, best_score, best_metrics = None, -1.0, None
        for a in cands:
            if a in chosen or spent + a.cash > budget + 1e-9:
                continue
            if not _feasible_with(chosen, a, pack, cutoff):
                continue
            m = _simulate(pack, fc, chosen + [a], cutoff, route_delay,
                          SCREEN_SIMS, seed, borrow_days, screen_base)
            gain = m["on_time_weighted"] - current_metrics["on_time_weighted"]
            score = gain / max(a.cash, 1.0)
            if gain > 1e-5 and score > best_score:
                best, best_score, best_metrics = a, score, m
        if best is None:
            break
        chosen.append(best)
        spent += best.cash
        current_metrics = best_metrics
        trace.append({
            "step": len(trace),
            "action": f"{best.kind}:{best.part_id}x{best.quantity}@{best.station_id}"
                      f"({best.ref})",
            "cash": round(best.cash, 2),
            "on_time_weighted": round(best_metrics["on_time_weighted"], 4),
            "expected_backlog_units": round(
                best_metrics["expected_backlog_units"], 2),
        })
    final = evaluate(pack, fc, chosen, cutoff, budget, scenario_id,
                     route_delay, n_sims, seed, borrow_days)
    return chosen, final, trace
