import pandas as pd, numpy as np
D = "/workspace/data"
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned","warranty_end"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
se = pd.read_csv(f"{D}/stock_events.csv", parse_dates=["occurred_at"])
pu = pd.read_csv(f"{D}/part_uses.csv", parse_dates=["occurred_at"])
w = wo.merge(ib[["unit_id","station_id","generation","environment","commissioned"]], on="unit_id")
w["age_days"] = (w.failed_at.dt.tz_localize(None) - w.commissioned).dt.days

print("== failures per day ==")
print(w.groupby(w.failed_at.dt.date).size().to_dict())

print("\n== age at failure by gen/env ==")
print(w.groupby(["generation","environment"]).age_days.describe()[["mean","std","min","max"]])
print("\n== failure counts by gen/env (each cell has 50 units? check base) ==")
base = ib.groupby(["generation","environment"]).unit_id.count()
cnt = w.groupby(["generation","environment"]).size()
print(pd.DataFrame({"failures":cnt,"base":base,"rate_per_unit":cnt/base}))

print("\n== station x gen failure rates (base=20 each) ==")
b2 = ib.groupby(["station_id","generation"]).unit_id.count()
c2 = w.groupby(["station_id","generation"]).size()
print(pd.DataFrame({"fail":c2,"base":b2}).fillna(0))

print("\n== consequence distribution by station ==")
print(w.groupby("station_id").consequence.agg(["count","sum"]))

print("\n== stock event kinds ==")
print(se.kind.value_counts())
print("\n== stock events: receipts by station/part ==")
r = se[se.kind=="receipt"].pivot_table(index="station_id",columns="part_id",values="quantity",aggfunc="sum",fill_value=0)
print(r)
print("\n== stock events: issues by station/part ==")
i = se[se.kind=="issue"].pivot_table(index="station_id",columns="part_id",values="quantity",aggfunc="sum",fill_value=0)
print(i)
print("\n== returns by station/part ==")
rt = se[se.kind=="return"].pivot_table(index="station_id",columns="part_id",values="quantity",aggfunc="sum",fill_value=0)
print(rt)
print("\n== ending on-hand ==")
oh = se.groupby(["station_id","part_id"]).quantity.sum().unstack(fill_value=0)
print(oh)
