"""故障与备件需求预测。

设计原则（对应业务要求）：
1. 四个事件层严格分开：故障发生（failed_at）、客户报修（opened_at）、
   实际领件（part_uses.issue）、履约结果（state/repair_at）。
   备件消耗不是故障需求：缺货时段没有领用，不能解释为没有故障；
   替代领用会抬高另一种物料的消耗；返修退回不取消原工单需求。
2. 故障率随机龄变化，并允许代际/环境差异；小样本单元向总体收缩，
   新站用装机计划承接暴露量，而不是按历史领件外推。
3. 只使用截点可知资料：估计与预测只读取 point_in_time 切片，
   waiting 工单的 repair_at 为空，绝不参与任何估计。
4. 输出带区间与假设标记，不用单一确定数。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .compat import repair_requirements
from .dataio import DataPack

AGE_BINS = (0, 30, 45, 60, 75, np.inf)
AGE_LABELS = ("0-30", "31-45", "46-60", "61-75", "76+")
# Gamma-Poisson 共轭先验：弱信息先验，相当于 0.5 次故障 / 100 台日
PRIOR_ALPHA = 0.5
PRIOR_BETA = 100.0
# 代际/环境效应的收缩强度（观测台日相对该尺度越多，越相信单元自身）
SHRINK_SCALE_DAYS = 2000.0


def _age_bin(age_days: float) -> str:
    idx = np.digitize(age_days, AGE_BINS[1:-1], right=False)
    return AGE_LABELS[idx]


def _bin_edges(label: str) -> tuple[float, float]:
    i = AGE_LABELS.index(label)
    return AGE_BINS[i], AGE_BINS[i + 1]


# ---------------------------------------------------------------------------
# 历史需求重建
# ---------------------------------------------------------------------------

def order_demand_lines(pack: DataPack) -> pd.DataFrame:
    """把每个工单展开为“按兼容关系应有”的需求行，并标注实际履约。

    每行是 (工单, 配套组) 的真实故障需求，候选物料列出该站实际可选用料；
    issued 记录该工单实际领用（毛额，退回不冲销需求）。
    """
    wo = pack.work_orders.merge(
        pack.installed_base[["unit_id", "station_id", "generation",
                             "firmware", "environment", "commissioned"]],
        on="unit_id")
    wo = wo.merge(pack.stations[["station_id", "qualification",
                                 "service_hours", "region"]], on="station_id")
    issues = pack.part_uses[pack.part_uses.kind == "issue"]
    rows = []
    for _, o in wo.iterrows():
        reqs = repair_requirements(
            pack, o.generation, o.firmware, o.qualification)
        used = issues[issues.order_id == o.order_id]
        for group, options in reqs.items():
            candidate_parts = [x.part_id for x in options]
            need_qty = options[0].parts_per_repair
            group_issues = used[used.part_id.isin(candidate_parts)]
            issued_parts = sorted(group_issues.part_id.unique().tolist())
            issued_qty = int(group_issues.quantity.sum())
            rows.append({
                "order_id": o.order_id,
                "station_id": o.station_id,
                "region": o.region,
                "unit_id": o.unit_id,
                "generation": o.generation,
                "firmware": o.firmware,
                "environment": o.environment,
                "required_group": group,
                "candidate_parts": ",".join(candidate_parts),
                "need_qty": need_qty,
                "issued_parts": ",".join(issued_parts),
                "issued_qty": issued_qty,
                "demand_occluded": issued_qty == 0,
                "state": o.state,
                "consequence": int(o.consequence),
                "failed_at": o.failed_at,
                "opened_at": o.opened_at,
            })
    return pd.DataFrame(rows)


def occlusion_summary(pack: DataPack) -> pd.DataFrame:
    """缺货遮挡与替代抬高的可核对汇总。

    - stockout_occluded：工单故障时点落在登记缺货区间且未领用；
    - zero_stock_occluded：库存事件显示为零（无独立登记）且未领用；
    - backlog_no_stock_cause：未领用但故障时点库存充足（原因资料不支持，
      如实标为未知，不编造）；
    - substitution_observed：同组内实际领用了非首选物料。
    """
    lines = order_demand_lines(pack)
    wo = pack.work_orders.merge(
        pack.installed_base[["unit_id", "station_id"]], on="unit_id")
    rows = []
    for _, ln in lines[lines.demand_occluded].iterrows():
        o = wo[wo.order_id == ln.order_id].iloc[0]
        fdate = o.failed_at.date()
        registered = pack.stockouts[
            (pack.stockouts.station_id == ln.station_id)
            & (pack.stockouts.part_id.isin(ln.candidate_parts.split(",")))
            & (pack.stockouts.start <= fdate)
            & (pack.stockouts.end >= fdate)
        ]
        if len(registered):
            cause = "stockout_registered"
            detail = ";".join(
                f"{r.part_id}:{r.reason}" for _, r in registered.iterrows())
        else:
            from .inventory import on_hand_at
            avail = {p: on_hand_at(pack, ln.station_id, p, o.failed_at)
                     for p in ln.candidate_parts.split(",")}
            if all(v == 0 for v in avail.values()):
                cause = "zero_stock_unregistered"
                detail = str(avail)
            else:
                cause = "backlog_reason_unknown"
                detail = str(avail)
        rows.append({
            "order_id": ln.order_id, "station_id": ln.station_id,
            "required_group": ln.required_group,
            "candidate_parts": ln.candidate_parts,
            "failed_at": o.failed_at, "state": ln.state,
            "consequence": ln.consequence, "occlusion_cause": cause,
            "evidence": detail,
        })
    return pd.DataFrame(rows)


def consumption_vs_demand(pack: DataPack) -> pd.DataFrame:
    """按物料对比“账面消耗”与“真实故障需求”，暴露替代与遮挡偏差。"""
    lines = order_demand_lines(pack)
    issues = pack.part_uses[pack.part_uses.kind == "issue"]
    gross = issues.groupby("part_id").quantity.sum().rename("gross_issued")
    returned = pack.part_uses[pack.part_uses.kind == "return_accepted"]
    ret = returned.groupby("part_id").quantity.sum().rename("returned")
    # 真实需求：每个工单配套组按候选物料首选（compatibility 中同组第一行）
    pref = pack.compatibility.sort_values(["generation", "firmware",
                                           "required_group", "part_id"])
    pref = pref.groupby(["generation", "firmware", "required_group"]).first().reset_index()
    need_rows = []
    for _, ln in lines.iterrows():
        p = pref[(pref.generation == ln.generation)
                 & (pref.firmware == ln.firmware)
                 & (pref.required_group == ln.required_group)].iloc[0]
        need_rows.append((p.part_id, ln.need_qty, ln.demand_occluded))
    need_df = pd.DataFrame(need_rows, columns=["part_id", "qty", "occluded"])
    true_need = need_df.groupby("part_id").qty.sum().rename("true_failure_demand")
    occluded = need_df.groupby("part_id").occluded.sum().rename("occluded_demand_units")
    out = pd.concat([gross, ret, true_need, occluded], axis=1).fillna(0)
    out["net_consumed"] = out.gross_issued - out.returned
    return out.reset_index().sort_values("part_id")


# ---------------------------------------------------------------------------
# 故障率模型：龄期分层 + 代际/环境收缩的 Gamma-Poisson
# ---------------------------------------------------------------------------

@dataclass
class HazardModel:
    """分层泊松过程的后验故障率（次/台日）。"""
    global_bin: dict[str, tuple[float, float]]          # label -> (alpha, beta)
    cell: dict[tuple, tuple[float, float, float, float]]  # (gen,env,bin)-> posterior a,b, shrink, exposure
    assumptions: list[str] = field(default_factory=list)

    def rate_draws(self, rng: np.random.Generator, n: int) -> dict:
        """每个 (gen,env,bin) 抽 n 个后验故障率样本。"""
        draws = {}
        for key, (a, b, _, _) in self.cell.items():
            draws[key] = rng.gamma(a, 1.0 / b, size=n)
        return draws


def fit_hazard_model(pack: DataPack) -> HazardModel:
    """只用截点切片内的故障与暴露拟合故障率。

    暴露按每台设备从投运到切片内最晚故障日（或资料最晚可知日）逐日归入龄期桶；
    单元（代际×环境×龄期桶）故障率向全网同桶总体做经验贝叶斯收缩，
    样本越少收缩越强。
    """
    ib = pack.installed_base.copy()
    ib["commissioned"] = pd.to_datetime(ib["commissioned"])
    wo = pack.work_orders.merge(ib[["unit_id", "generation",
                                    "environment", "commissioned"]], on="unit_id")
    horizon_end = pack.work_orders.failed_at.max().tz_localize(None)
    ib["exposure_end_age"] = (horizon_end - ib.commissioned).dt.days + 1

    # 全网龄期桶暴露与故障
    bin_exposure = {b: 0.0 for b in AGE_LABELS}
    bin_fails = {b: 0 for b in AGE_LABELS}
    for _, u in ib.iterrows():
        end_age = int(u.exposure_end_age)
        for label in AGE_LABELS:
            lo, hi = _bin_edges(label)
            hi = min(hi, end_age + 1)
            if hi > lo:
                bin_exposure[label] += hi - lo
    wo2 = wo.copy()
    wo2["age"] = (wo.failed_at.dt.tz_localize(None)
                  - wo2.commissioned).dt.days
    wo2["bin"] = wo2.age.map(_age_bin)
    for label, cnt in wo2.groupby("bin").size().items():
        bin_fails[label] = int(cnt)

    global_bin = {}
    for label in AGE_LABELS:
        a = PRIOR_ALPHA + bin_fails[label]
        b = PRIOR_BETA + bin_exposure[label]
        global_bin[label] = (a, b)

    # 单元暴露与故障（代际×环境×龄期桶）
    cells = {}
    groups = ib.groupby(["generation", "environment"])
    for (gen, env), units in groups:
        cell_exposure = {b: 0.0 for b in AGE_LABELS}
        for _, u in units.iterrows():
            end_age = int(u.exposure_end_age)
            for label in AGE_LABELS:
                lo, hi = _bin_edges(label)
                hi = min(hi, end_age + 1)
                if hi > lo:
                    cell_exposure[label] += hi - lo
        sub = wo2[(wo2.generation == gen) & (wo2.environment == env)]
        for label in AGE_LABELS:
            exposure = cell_exposure[label]
            fails = int((sub["bin"] == label).sum())
            ga, gb = global_bin[label]
            shrink = SHRINK_SCALE_DAYS / (SHRINK_SCALE_DAYS + exposure)
            # 收缩后验：先验强度按 shrink 折入全网同桶均值
            prior_rate = ga / gb
            a = fails + (1.0 - shrink) * PRIOR_ALPHA + shrink * prior_rate * PRIOR_BETA
            b = exposure + PRIOR_BETA
            cells[(gen, env, label)] = (a, b, shrink, exposure)

    assumptions = [
        "故障在固定协变量下服从齐次泊松过程（按龄期桶分段常数），"
        "这是资料未直接证实的建模假设；日频计数近似常数但样本仅34天。",
        "代际/环境效应采用经验贝叶斯收缩，小样本单元向全网同龄期桶靠拢；"
        "资料中代际×环境故障率差异很小，收缩后差异更弱。",
        "未观测到重复故障（每台至多1单），预测按伯努利过程处理："
        "每台设备在窗内至多故障一次，概率由龄期分层强度 1-exp(-λ) 给出；"
        "窗内已故障设备是否再发无资料支持，按不再发处理属保守下限。",
        "情景参数 failure_multiplier 作为外部假设线性缩放故障率，"
        "其与航线延迟的关联只在 plans 情景内成对出现，不假设站点间独立或相关。",
    ]
    return HazardModel(global_bin=global_bin, cell=cells, assumptions=assumptions)


# ---------------------------------------------------------------------------
# 前向预测
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    cutoff: pd.Timestamp
    horizon_days: int
    failure_forecast: pd.DataFrame
    part_demand: pd.DataFrame
    backlog: pd.DataFrame
    model: HazardModel
    assumptions: list[str]


def _unit_exposure_by_bin(commissioned: pd.Timestamp, cutoff: pd.Timestamp,
                          horizon_days: int) -> dict[str, float]:
    """预测窗内每台设备落在各龄期桶的台日数。"""
    start_age = (cutoff.tz_localize(None) - commissioned).days
    out = {}
    remaining = horizon_days
    for label in AGE_LABELS:
        lo, hi = _bin_edges(label)
        in_bin = max(0.0, min(hi, start_age + remaining) - max(lo, start_age))
        out[label] = in_bin
        remaining -= in_bin
    return out


def forecast(pack: DataPack, cutoff: pd.Timestamp, horizon_days: int = 90,
             scenario_id: str | None = None, n_sims: int = 4000,
             seed: int = 42) -> ForecastResult:
    """预测截点后 horizon_days 天的故障与备件需求（含区间）。

    返回：
    - failure_forecast：站×代际×环境 的故障数后验均值与 50%/90% 区间；
    - part_demand：站×物料 的毛需求区间（按兼容首选落料，含现有积压）；
    - backlog：截点未结工单的待满足需求。
    """
    cutoff = pd.Timestamp(cutoff)
    rng = np.random.default_rng(seed)
    model = fit_hazard_model(pack)
    rate_draws = model.rate_draws(rng, n_sims)

    scen = None
    if scenario_id is not None:
        hit = pack.plans[pack.plans.scenario_id == scenario_id]
        if hit.empty:
            raise ValueError(f"未知情景：{scenario_id}")
        multiplier = float(hit.failure_multiplier.iloc[0])
        delay = int(hit.route_delay_days.iloc[0])
    else:
        multiplier, delay = 1.0, 0

    ib = pack.installed_base.copy()
    ib["commissioned"] = pd.to_datetime(ib["commissioned"])
    # 情景新增设备：窗内均匀投运，代际构成同本站当前在保设备
    new_unit_rows = []
    if scen is None and scenario_id is not None:
        scen = hit
    if scenario_id is not None:
        gen_share = (ib.groupby(["station_id", "generation"]).size()
                     / ib.groupby("station_id").size())
        uid = 0
        for _, p in hit.iterrows():
            station = p.station_id
            shares = gen_share[station]
            for gen, share in shares.items():
                k = int(round(share * p.new_units))
                firmware = ib[ib.generation == gen].firmware.iloc[0]
                env = ib[ib.station_id == station].environment.iloc[0]
                for j in range(k):
                    day = int((j + 0.5) / max(1, k) * horizon_days)
                    comm = (cutoff.tz_localize(None) + pd.Timedelta(days=day)).date()
                    uid += 1
                    new_unit_rows.append({
                        "unit_id": f"NEW{uid}", "station_id": station,
                        "generation": gen, "firmware": firmware,
                        "environment": env, "commissioned": pd.Timestamp(comm),
                        "warranty_end": pd.Timestamp("2027-06-20"),
                    })
        if new_unit_rows:
            ib = pd.concat([ib, pd.DataFrame(new_unit_rows)], ignore_index=True)

    # 每台设备在窗内的累计故障强度 lam；资料显示每台至多 1 次故障，
    # 故按伯努利（至少故障一次的概率 1-exp(-lam)）抽样，不允许重复计数
    fail_sims = np.zeros((n_sims, len(ib)), dtype=int)
    keys = []
    for i, (_, u) in enumerate(ib.iterrows()):
        expo = _unit_exposure_by_bin(u.commissioned, cutoff, horizon_days)
        lam = np.zeros(n_sims)
        for label, days in expo.items():
            if days <= 0:
                continue
            key = (u.generation, u.environment, label)
            lam += rate_draws[key] * days * multiplier
        prob = 1.0 - np.exp(-lam)
        fail_sims[:, i] = (rng.uniform(size=n_sims) < prob).astype(int)
        keys.append((u.station_id, u.generation, u.environment))
    key_df = pd.DataFrame(keys, columns=["station_id", "generation", "environment"])
    key_df["sim_unit"] = np.arange(len(ib))

    ff_rows = []
    for (station, gen, env), idx in key_df.groupby(
            ["station_id", "generation", "environment"]).groups.items():
        sims = fail_sims[:, list(idx)].sum(axis=1)
        ff_rows.append({
            "station_id": station, "generation": gen, "environment": env,
            "units": len(idx),
            "failures_mean": sims.mean(),
            "failures_p25": np.percentile(sims, 25),
            "failures_p75": np.percentile(sims, 75),
            "failures_p05": np.percentile(sims, 5),
            "failures_p95": np.percentile(sims, 95),
        })
    failure_forecast = pd.DataFrame(ff_rows).sort_values(
        ["station_id", "generation", "environment"]).reset_index(drop=True)

    # 物料需求：按每站代际故障数 × BOM 首选映射；G1 在 high 站首选仍为 P2，
    # P3 是受约束时的替代选项，在规划模块中单独评估
    pref = (pack.compatibility.sort_values(["generation", "firmware",
                                            "required_group", "part_id"])
            .groupby(["generation", "required_group"]).first().reset_index())
    station_qual = pack.stations.set_index("station_id").qualification
    # 模拟矩阵：站×物料
    parts = pack.parts.part_id.tolist()
    demand_sims = {(s, p): np.zeros(n_sims) for s in pack.stations.station_id
                   for p in parts}
    for (station, gen, env), idx in key_df.groupby(
            ["station_id", "generation", "environment"]).groups.items():
        sims = fail_sims[:, list(idx)].sum(axis=1)
        bom = pref[pref.generation == gen]
        for _, b in bom.iterrows():
            demand_sims[(station, b.part_id)] += sims * int(b.parts_per_repair)

    # 现有积压（截点 waiting 工单）按同样 BOM 加入，属确定的待满足需求
    backlog_rows = _backlog(pack)
    for _, bl in backlog_rows.iterrows():
        demand_sims[(bl.station_id, bl.part_id)] += bl.need_qty

    pd_rows = []
    for (station, part), sims in demand_sims.items():
        if sims.max() == 0:
            continue
        pd_rows.append({
            "station_id": station, "part_id": part,
            "demand_mean": sims.mean(),
            "demand_p25": np.percentile(sims, 25),
            "demand_p75": np.percentile(sims, 75),
            "demand_p05": np.percentile(sims, 5),
            "demand_p95": np.percentile(sims, 95),
            "includes_backlog": bool(
                ((backlog_rows.station_id == station)
                 & (backlog_rows.part_id == part)).any()),
        })
    part_demand = pd.DataFrame(pd_rows).sort_values(
        ["station_id", "part_id"]).reset_index(drop=True)

    assumptions = list(model.assumptions)
    if scenario_id is not None:
        assumptions.append(
            f"情景 {scenario_id}：failure_multiplier={multiplier}，"
            f"route_delay_days={delay}（用户输入假设，非观测概率真值）；"
            "新增设备均匀投运、代际构成同本站在保设备，均为 plans 声明的情景规则。")
    return ForecastResult(
        cutoff=cutoff, horizon_days=horizon_days,
        failure_forecast=failure_forecast, part_demand=part_demand,
        backlog=backlog_rows, model=model, assumptions=assumptions)


def _backlog(pack: DataPack) -> pd.DataFrame:
    """截点未结工单的待满足物料需求（按兼容首选落料）。"""
    waiting = pack.work_orders[pack.work_orders.state == "waiting"].merge(
        pack.installed_base[["unit_id", "station_id", "generation",
                             "firmware"]], on="unit_id")
    pref = (pack.compatibility.sort_values(["generation", "firmware",
                                            "required_group", "part_id"])
            .groupby(["generation", "required_group"]).first().reset_index())
    rows = []
    for _, o in waiting.iterrows():
        for _, b in pref[pref.generation == o.generation].iterrows():
            rows.append({
                "order_id": o.order_id, "station_id": o.station_id,
                "part_id": b.part_id, "need_qty": int(b.parts_per_repair),
                "consequence": int(o.consequence), "failed_at": o.failed_at,
            })
    return pd.DataFrame(rows)
