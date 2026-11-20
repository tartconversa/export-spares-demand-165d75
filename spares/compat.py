"""兼容与配套关系解析。

口径：
- 同一 required_group 内的物料互为替代，任选其一；
- 不同 required_group 的物料必须同时满足（数量按 parts_per_repair）才有修复价值；
- 替代受固件（firmware）与服务站资质（min_qualification）共同限制；
- 合成数据中资质有序：basic < high。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dataio import DataPack

QUAL_ORDER = {"basic": 1, "high": 2}


@dataclass(frozen=True)
class GroupOption:
    required_group: str
    part_id: str
    parts_per_repair: int
    min_qualification: str


def _qualified(station_qual: str, required: str) -> bool:
    return QUAL_ORDER[station_qual] >= QUAL_ORDER[required]


def repair_requirements(pack: DataPack, generation: str, firmware: str,
                        station_qualification: str) -> dict[str, list[GroupOption]]:
    """返回某代际/固件设备在某资质站可执行的修复配套。

    返回 {required_group: [可选用料...]}；每个组都必须被满足。
    资质不足的用料选项直接剔除（不是降权，而是不可用）。
    """
    rows = pack.compatibility[
        (pack.compatibility.generation == generation)
        & (pack.compatibility.firmware == firmware)
    ]
    result: dict[str, list[GroupOption]] = {}
    for _, r in rows.iterrows():
        if not _qualified(station_qualification, r.min_qualification):
            continue
        result.setdefault(r.required_group, []).append(
            GroupOption(r.required_group, r.part_id,
                        int(r.parts_per_repair), r.min_qualification))
    if not result:
        raise ValueError(
            f"无兼容配套：{generation}/{firmware} @ {station_qualification}")
    return result


def demand_groups(pack: DataPack, generation: str, firmware: str) -> list[str]:
    """该设备修复所需的配套组（与资质无关的结构层）。"""
    return sorted(pack.compatibility[
        (pack.compatibility.generation == generation)
        & (pack.compatibility.firmware == firmware)
    ].required_group.unique().tolist())


def eligible_parts(pack: DataPack, generation: str, firmware: str,
                   station_qualification: str) -> pd.DataFrame:
    """展平的可用物料清单（含配套组与数量），用于需求-物料映射。"""
    reqs = repair_requirements(pack, generation, firmware, station_qualification)
    rows = []
    for group, options in reqs.items():
        for opt in options:
            rows.append({
                "required_group": group, "part_id": opt.part_id,
                "parts_per_repair": opt.parts_per_repair,
                "min_qualification": opt.min_qualification,
            })
    return pd.DataFrame(rows)


def can_fulfill(stock_by_part: dict[str, int],
                requirements: dict[str, list[GroupOption]]) -> tuple[bool, dict]:
    """给定各物料可用数量，判断一次修复的全部配套组能否同时满足。

    同组内只要任一可选物料数量足够即可；所有组同时满足才可修复。
    返回 (是否可修复, 每组的判定明细)。
    """
    detail = {}
    ok_all = True
    for group, options in requirements.items():
        choices = [
            {"part_id": o.part_id, "need": o.parts_per_repair,
             "available": stock_by_part.get(o.part_id, 0),
             "ok": stock_by_part.get(o.part_id, 0) >= o.parts_per_repair}
            for o in options
        ]
        ok = any(c["ok"] for c in choices)
        detail[group] = {"ok": ok, "choices": choices}
        ok_all = ok_all and ok
    return ok_all, detail
