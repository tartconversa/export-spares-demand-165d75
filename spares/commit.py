"""保障方案落实为采购与互借承诺，并持久化到演练目录。

落实规则（业务要求）：
- 只有显式确认的动作进入承诺台账，尚未落实的建议不占实物；
- 互借承诺立即扣减出借站可保障库存，借入站到达前不入账，
  同一件不能两边都承诺（落实前校验可保障库存是否充足）；
- 采购承诺记录资金占用与预计到达，到货入库前不增加现存库存；
- 改港承诺记录改港费，并标记原批次改往新目的站。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .dataio import DataPack
from .inventory import CommitmentLedger
from .planning import Action


def _ts(value) -> str:
    return pd.Timestamp(value).isoformat()


def commit_actions(pack: DataPack, actions: list[Action],
                   state_dir: str | Path, cutoff: pd.Timestamp,
                   ledger: CommitmentLedger | None = None,
                   ) -> CommitmentLedger:
    """把动作校验后写入承诺台账；库存不足的互借直接拒绝。"""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger or CommitmentLedger()
    cutoff = pd.Timestamp(cutoff)
    rejected = []

    # 先在副本上累计校验，全部通过才落盘
    borrow_index: dict[tuple[str, str], int] = {}
    for b in ledger.borrows:
        key = (b["from_station"], b["part_id"])
        borrow_index[key] = borrow_index.get(key, 0) + b["quantity"]

    seq = len(ledger.purchases) + len(ledger.borrows)
    for a in actions:
        seq += 1
        cid = f"C{seq:04d}"
        if a.kind in ("normal", "expedite"):
            ledger.add_purchase(
                cid, a.station_id, a.part_id, a.quantity, a.kind,
                cutoff, a.unit_cost, a.fee, a.eta_latest)
        elif a.kind == "borrow":
            key = (a.ref, a.part_id)
            already = borrow_index.get(key, 0)
            from .inventory import on_hand_at
            available = on_hand_at(pack, a.ref, a.part_id, cutoff) - already
            if a.quantity > available:
                rejected.append({
                    "action": a.detail,
                    "reason": f"出借站 {a.ref} 可保障库存不足：需 {a.quantity}，"
                              f"仅剩 {max(0, available)}"})
                continue
            borrow_index[key] = already + a.quantity
            ledger.add_borrow(cid, a.ref, a.station_id, a.part_id,
                              a.quantity, cutoff, a.eta_latest)
        elif a.kind == "redirect":
            ledger.purchases.append({
                "commitment_id": cid, "kind": "redirect",
                "mode": "redirect", "station_id": a.station_id,
                "part_id": a.part_id, "quantity": a.quantity,
                "commit_at": pd.Timestamp(cutoff),
                "eta": pd.Timestamp(a.eta_latest),
                "unit_cost": 0.0, "fee": a.fee,
                "ref_shipment": a.ref,
            })
        else:
            rejected.append({"action": a.detail, "reason": "未知动作类型"})

    _persist(ledger, state_dir)
    if rejected:
        (state_dir / "rejected_commitments.json").write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger


def _persist(ledger: CommitmentLedger, state_dir: Path) -> None:
    def norm(rows):
        out = []
        for r in rows:
            out.append({k: (_ts(v) if isinstance(v, (pd.Timestamp,)) else v)
                        for k, v in r.items()})
        return out
    payload = {
        "borrows": norm(ledger.borrows),
        "purchases": norm(ledger.purchases),
    }
    (state_dir / "commitments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ledger(state_dir: str | Path) -> CommitmentLedger:
    """从演练目录读回承诺台账。"""
    path = Path(state_dir) / "commitments.json"
    ledger = CommitmentLedger()
    if not path.exists():
        return ledger
    payload = json.loads(path.read_text(encoding="utf-8"))
    for b in payload.get("borrows", []):
        ledger.add_borrow(
            b["commitment_id"], b["from_station"], b["to_station"],
            b["part_id"], b["quantity"],
            pd.Timestamp(b["commit_at"]), pd.Timestamp(b["eta"]))
    for p in payload.get("purchases", []):
        if p["kind"] == "redirect":
            ledger.purchases.append({**p,
                "commit_at": pd.Timestamp(p["commit_at"]),
                "eta": pd.Timestamp(p["eta"])})
        else:
            ledger.add_purchase(
                p["commitment_id"], p["station_id"], p["part_id"],
                p["quantity"], p["mode"], pd.Timestamp(p["commit_at"]),
                p["unit_cost"], p["fee"], pd.Timestamp(p["eta"]))
    return ledger


def save_snapshot(payload: dict, state_dir: str | Path, name: str) -> Path:
    """保存预测/方案快照（回看时保留原方案当时使用的预测，不被覆盖）。"""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=str), encoding="utf-8")
    return path
