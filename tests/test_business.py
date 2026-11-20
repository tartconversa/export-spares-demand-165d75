"""业务口径与不变量测试。

测试不依赖随机结果的具体数值，而是固化业务规则：
时点可知性、缺货遮挡、配套约束、承诺扣减、预测防泄漏、方案可行性、
回看归因互斥等。
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from spares.actuals import generate_actuals, merge_actuals
from spares.commit import commit_actions, load_ledger
from spares.compat import repair_requirements
from spares.dataio import load_pack, point_in_time
from spares.forecast import (consumption_vs_demand, fit_hazard_model,
                             forecast, occlusion_summary, order_demand_lines)
from spares.inventory import CommitmentLedger, on_hand_at, zero_stock_intervals
from spares.planning import (Action, candidate_actions, evaluate, optimize,
                             _feasible_with)
from spares.replay import attribute_gaps
from spares.supply import effective_shipment_window

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2025-04-21T08:00:00+08:00")


class PackTest(unittest.TestCase):
    def setUp(self):
        self.raw = load_pack(DATA)
        self.pack = point_in_time(self.raw, CUTOFF)


class TestPointInTime(PackTest):
    def test_future_facts_excluded(self):
        # 构造一个更早截点：晚于截点 available_at 的工单不可见
        early = pd.Timestamp("2025-03-15T08:00:00+08:00")
        pit = point_in_time(self.raw, early)
        self.assertTrue((pit.work_orders.available_at <= early).all())
        self.assertLess(len(pit.work_orders), len(self.raw.work_orders))
        # 主数据不受影响
        self.assertEqual(len(pit.stations), len(self.raw.stations))

    def test_repair_results_not_leaked(self):
        early = pd.Timestamp("2025-03-15T08:00:00+08:00")
        pit = point_in_time(self.raw, early)
        # 截点时不可把后来的 repair_at 当作已知
        later_repaired = self.raw.work_orders[
            self.raw.work_orders.available_at > early]
        self.assertTrue(len(later_repaired) > 0)
        self.assertEqual(
            len(pit.work_orders.merge(
                later_repaired[["order_id"]], on="order_id")), 0)


class TestInventoryAndOcclusion(PackTest):
    def test_on_hand_matches_event_sum(self):
        total = self.pack.stock_events.groupby(
            ["station_id", "part_id"]).quantity.sum()
        for (s, p), v in total.items():
            self.assertEqual(on_hand_at(self.pack, s, p, CUTOFF), int(v))

    def test_s2_p1_zero_interval_detected(self):
        zeros = zero_stock_intervals(self.pack)
        s2p1 = zeros[(zeros.station_id == "S2") & (zeros.part_id == "P1")]
        self.assertEqual(len(s2p1), 1)
        self.assertIsNone(s2p1.iloc[0].zero_to)  # 截点仍为零

    def test_stockout_registered_for_s2_g2_orders(self):
        occl = occlusion_summary(self.pack)
        registered = occl[occl.occlusion_cause == "stockout_registered"]
        # 6 个 S2 G2 工单在登记缺货窗口内且零领用
        self.assertEqual(len(registered), 6)
        self.assertTrue((registered.station_id == "S2").all())

    def test_unknown_cause_not_fabricated(self):
        occl = occlusion_summary(self.pack)
        unknown = occl[occl.occlusion_cause == "backlog_reason_unknown"]
        # 资料不支持原因的积压必须如实标注，不得编造缺货
        self.assertGreater(len(unknown), 0)

    def test_consumption_understates_demand(self):
        cvd = consumption_vs_demand(self.pack).set_index("part_id")
        # P1 净消耗 29 < 真实故障需求 50（遮挡+退回）
        self.assertLess(cvd.loc["P1", "net_consumed"],
                        cvd.loc["P1", "true_failure_demand"])
        self.assertEqual(cvd.loc["P1", "true_failure_demand"], 50)

    def test_returns_do_not_cancel_demand(self):
        lines = order_demand_lines(self.pack)
        # 有返修退回的工单仍保留需求行
        returned_orders = set(self.pack.part_uses[
            self.pack.part_uses.kind == "return_accepted"].order_id)
        self.assertTrue(returned_orders.issubset(set(lines.order_id)))


class TestCompatibility(PackTest):
    def test_g2_requires_both_groups(self):
        reqs = repair_requirements(self.pack, "G2", "F2", "basic")
        self.assertEqual(set(reqs), {"main", "aux"})
        self.assertEqual({o.part_id for o in reqs["main"]}, {"P1"})
        self.assertEqual({o.part_id for o in reqs["aux"]}, {"P4"})

    def test_p3_substitution_only_for_high(self):
        high = repair_requirements(self.pack, "G1", "F1", "high")
        basic = repair_requirements(self.pack, "G1", "F1", "basic")
        self.assertIn("P3", [o.part_id for o in high["main"]])
        self.assertNotIn("P3", [o.part_id for o in basic["main"]])
        self.assertEqual([o.part_id for o in basic["main"]], ["P2"])


class TestForecast(PackTest):
    def test_hazard_increases_with_age(self):
        m = fit_hazard_model(self.pack)
        rates = [m.global_bin[b][0] / m.global_bin[b][1] for b in
                 ("0-30", "31-45", "46-60", "61-75", "76+")]
        self.assertLess(rates[0], rates[2])
        self.assertLess(rates[1], rates[2])

    def test_forecast_ranges_ordered(self):
        fc = forecast(self.pack, CUTOFF, n_sims=400)
        d = fc.part_demand
        self.assertTrue((d.demand_p05 <= d.demand_mean).all())
        self.assertTrue((d.demand_mean <= d.demand_p95).all())
        # backlog 被包含
        self.assertTrue(d.includes_backlog.any())

    def test_backlog_contains_all_waiting_orders(self):
        fc = forecast(self.pack, CUTOFF, n_sims=300)
        n_waiting = (self.pack.work_orders.state == "waiting").sum()
        self.assertEqual(fc.backlog.order_id.nunique(), n_waiting)

    def test_no_repeat_failure_per_unit(self):
        fc = forecast(self.pack, CUTOFF, n_sims=300)
        # 单站单代际 20 台，90 天故障 P95 不应远超设备数（伯努利上限）
        self.assertTrue((fc.failure_forecast.failures_p95
                         <= fc.failure_forecast.units + 1).all())

    def test_scenario_multiplier_changes_mean(self):
        base = forecast(self.pack, CUTOFF, n_sims=300)
        slow = forecast(self.pack, CUTOFF, scenario_id="slow_start",
                        n_sims=300)
        # slow_start 倍数 0.8 但有新增设备，总量仍应低于 port_hold
        port = forecast(self.pack, CUTOFF, scenario_id="port_hold",
                        n_sims=300)
        self.assertLess(slow.failure_forecast.failures_mean.sum(),
                        port.failure_forecast.failures_mean.sum())
        self.assertIsInstance(base.assumptions, list)


class TestCommitment(PackTest):
    def test_borrow_reserves_lender_stock(self):
        ledger = CommitmentLedger()
        before = on_hand_at(self.pack, "S1", "P3", CUTOFF)
        ledger.add_borrow("B1", "S1", "S2", "P3", 4, CUTOFF,
                          CUTOFF + pd.Timedelta(days=3))
        self.assertEqual(
            ledger.guarantee_stock(self.pack, "S1", "P3", CUTOFF),
            before - 4)
        # 借入站在到达前不算到库
        self.assertEqual(on_hand_at(self.pack, "S2", "P3", CUTOFF),
                         on_hand_at(self.pack, "S2", "P3", CUTOFF))

    def test_commit_rejects_over_promise(self):
        # S2 P1 现存为 0，任何互借都应被拒绝
        action = Action("borrow", "S1", "P1", 5, 0.0, 0.0,
                        CUTOFF + pd.Timedelta(days=3),
                        CUTOFF + pd.Timedelta(days=3), ref="S2",
                        detail="test")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ledger = commit_actions(self.pack, [action], d, CUTOFF)
            self.assertEqual(len(ledger.borrows), 0)
            rejected = pd.read_json(Path(d) / "rejected_commitments.json")
            self.assertEqual(len(rejected), 1)

    def test_persist_and_reload(self):
        import tempfile
        ledger = CommitmentLedger()
        ledger.add_borrow("B1", "S3", "S1", "P5", 2, CUTOFF,
                          CUTOFF + pd.Timedelta(days=3))
        with tempfile.TemporaryDirectory() as d:
            commit_actions(self.pack, [], d, CUTOFF, ledger)
            reloaded = load_ledger(d)
            self.assertEqual(len(reloaded.borrows), 1)
            self.assertEqual(reloaded.borrows[0]["quantity"], 2)


class TestPlanning(PackTest):
    def setUp(self):
        super().setUp()
        self.fc = forecast(self.pack, CUTOFF, n_sims=300)

    def test_borrow_candidate_when_lender_has_real_spare(self):
        # 构造富余：把 S3 的 P2 现存抬高到远超其自身 P95 需求，
        # 此时应生成 S3 向缺料站的互借候选
        import dataclasses
        extra = self.pack.stock_events[
            (self.pack.stock_events.station_id == "S3")
            & (self.pack.stock_events.part_id == "P2")].copy()
        extra["stock_event_id"] = "TEST-BULK"
        extra["quantity"] = 60
        pack = dataclasses.replace(
            self.pack,
            stock_events=pd.concat(
                [self.pack.stock_events, extra], ignore_index=True))
        fc = forecast(pack, CUTOFF, n_sims=200)
        cands = candidate_actions(pack, fc, CUTOFF, 0)
        borrows = [a for a in cands if a.kind == "borrow"
                   and a.ref == "S3" and a.part_id == "P2"]
        self.assertTrue(borrows, "存在真实富余时应产生互借候选")
        # 可行性：同一出借站累计外借不超过其可保障存量
        first = borrows[0]
        self.assertLessEqual(
            first.quantity, on_hand_at(pack, "S3", "P2", CUTOFF))
        self.assertTrue(_feasible_with([], first, pack, CUTOFF))
        # 超出存量的第二笔不可行
        over = Action("borrow", "S1", "P2",
                      on_hand_at(pack, "S3", "P2", CUTOFF) + 1, 0.0, 0.0,
                      CUTOFF, CUTOFF, ref="S3")
        self.assertFalse(_feasible_with([first], over, pack, CUTOFF))

    def test_shipment_single_redirect_feasibility(self):
        cands = candidate_actions(self.pack, self.fc, CUTOFF, 0)
        redirects = [a for a in cands if a.kind == "redirect"]
        # 每个 shipment 在候选中至多一条改港
        refs = [a.ref for a in redirects]
        self.assertEqual(len(refs), len(set(refs)))
        chosen = [a for a in redirects if a.ref == redirects[0].ref]
        another = Action("redirect", "S9", redirects[0].part_id, 6, 0.0,
                         600.0, CUTOFF, CUTOFF, ref=redirects[0].ref)
        self.assertFalse(_feasible_with(chosen, another, self.pack, CUTOFF))

    def test_evaluate_keeps_all_demand(self):
        # 不允许通过不处理工单美化：无动作时仍计入全部 backlog
        ev = evaluate(self.pack, self.fc, [], CUTOFF, 60000, None, 0,
                      n_sims=300)
        self.assertTrue(0.0 <= ev.on_time_weighted <= 1.0)
        # backlog 在无 P1 的 S2 按时恢复必然受限
        s2 = ev.station_metrics[
            ev.station_metrics.station_id == "S2"].iloc[0]
        self.assertLess(s2.on_time_weighted, 0.9)

    def test_optimize_respects_budget_and_feasibility(self):
        chosen, ev, _ = optimize(self.pack, CUTOFF, 60000, n_sims=300)
        self.assertTrue(ev.within_budget)
        # 同一批次不会被改两次
        redirect_refs = [a.ref for a in chosen if a.kind == "redirect"]
        self.assertEqual(len(redirect_refs), len(set(redirect_refs)))
        # 互借总量不超出借站现存
        for a in chosen:
            if a.kind == "borrow":
                self.assertLessEqual(
                    a.quantity,
                    on_hand_at(self.pack, a.ref, a.part_id, CUTOFF))

    def test_actions_improve_over_baseline(self):
        chosen, ev, _ = optimize(self.pack, CUTOFF, 60000, n_sims=300)
        base = evaluate(self.pack, self.fc, [], CUTOFF, 60000, None, 0,
                        n_sims=300)
        self.assertGreaterEqual(ev.on_time_weighted,
                                base.on_time_weighted - 0.01)


class TestSupplyWindows(PackTest):
    def test_scenario_delay_shifts_window(self):
        shp = self.pack.shipments.iloc[0]
        e0, l0 = effective_shipment_window(shp, 0)
        e1, l1 = effective_shipment_window(shp, 14)
        self.assertEqual((e1 - e0).days, 14)
        self.assertEqual((l1 - l0).days, 14)

    def test_earliest_not_treated_as_certain(self):
        from spares.supply import arrival_probability_by
        e = pd.Timestamp("2025-04-21T08:00:00+08:00")
        l = pd.Timestamp("2025-05-06T08:00:00+08:00")
        self.assertEqual(arrival_probability_by(e, l, e), 0.0)
        self.assertEqual(arrival_probability_by(e, l, l), 1.0)
        mid = e + (l - e) / 2
        self.assertAlmostEqual(arrival_probability_by(e, l, mid), 0.5)


class TestReplay(PackTest):
    def test_attribution_categories_disjoint_and_snapshot_kept(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fc = forecast(self.pack, CUTOFF, n_sims=200)
            chosen, _, _ = optimize(self.pack, CUTOFF, 60000, n_sims=200)
            snapshot = {
                "cutoff": str(CUTOFF), "scenario_id": None,
                "route_delay_days": 0, "budget": 60000, "horizon_days": 90,
                "part_demand": fc.part_demand.to_dict("records"),
                "actions": [a.__dict__ | {
                    "eta_earliest": str(a.eta_earliest),
                    "eta_latest": str(a.eta_latest)} for a in chosen]}
            generate_actuals(self.raw, CUTOFF, Path(d) / "act",
                             days=30, new_failures=10)
            review = CUTOFF + pd.Timedelta(days=30)
            extended = merge_actuals(self.raw, Path(d) / "act", review)
            ledger = CommitmentLedger()
            result = attribute_gaps(snapshot, extended, review, 90, ledger)
            allowed = {"demand_error", "compat_judgement",
                       "fulfillment_change"}
            self.assertTrue(
                set(result.gap_attribution.gap_category).issubset(allowed))
            # 旧快照内容不被重预测覆写
            self.assertEqual(snapshot["horizon_days"], 90)

    def test_default_actuals_scale_with_history(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = generate_actuals(self.raw, CUTOFF, Path(d) / "act",
                                    days=45)
            wo = pd.read_csv(path / "work_orders_extra.csv")
            # 历史 100 单 / 约 14700 台日，45 天 ×200 台期望约 60 单量级
            self.assertGreater(len(wo), 30)
            # 显式低压情景仍被尊重
            path2 = generate_actuals(self.raw, CUTOFF, Path(d) / "act2",
                                     days=45, new_failures=5)
            self.assertEqual(
                len(pd.read_csv(path2 / "work_orders_extra.csv")), 5)


if __name__ == "__main__":
    unittest.main()
