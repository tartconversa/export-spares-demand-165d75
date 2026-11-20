import pandas as pd, numpy as np
D = "/workspace/data"
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
w = wo.merge(ib, on="unit_id", suffixes=("","_ib"))

# exposure: units commissioned from 2025-01-01..02-19, failures from 03-07..04-09
# check repeated failures per unit
print("failures per unit distribution:", wo.unit_id.value_counts().value_counts().to_dict())
print("unique failed units:", wo.unit_id.nunique(), "of", len(ib))

# Any unit fail more than once?
vc = wo.unit_id.value_counts()
print("units with >1 failure:", vc[vc>1].to_dict())

# exposure window per unit: commissioned -> end of observation (last failed_at date 2025-04-09)
obs_end = pd.Timestamp("2025-04-09 08:00:00+08:00")
ib2 = ib.copy()
ib2["exposure_days"] = (obs_end - ib2.commissioned.dt.tz_localize("Asia/Shanghai")).dt.total_seconds()/86400
fail_cnt = wo.groupby("unit_id").size().rename("fails")
ib2 = ib2.merge(fail_cnt, on="unit_id", how="left").fillna({"fails":0})

# rate per gen/env
agg = ib2.groupby(["generation","environment"]).agg(units=("unit_id","count"), expo=("exposure_days","sum"), fails=("fails","sum"))
agg["rate_per_unit_day"] = agg.fails/agg.expo
print("\n== failure rate per unit-day by gen/env ==")
print(agg)

# by age buckets: age at failure
w["age"] = (w.failed_at.dt.tz_localize(None)-w.commissioned).dt.days
ib2["age_at_obs_end"] = (obs_end.dt.tz_localize(None)-ib2.commissioned).dt.days
# piecewise exposure by age bins
bins=[0,30,45,60,75,200]
ib2["bin"]=pd.cut(ib2.age_at_obs_end,bins)
print("\nage at obs end by unit:", ib2.age_at_obs_end.describe()[["min","max","mean"]].to_dict())
print("age at failure:", w.age.describe()[["min","25%","50%","75%","max"]].to_dict())

# failures per calendar week, check stationarity
w["week"] = w.failed_at.dt.isocalendar().week
print("\nfailures per week:", w.groupby("week").size().to_dict())

# consequence 3 orders - any pattern?
print("\nconsequence-3 orders:")
print(w[w.consequence==3][["order_id","station_id","generation","environment","failed_at","state"]].to_string())

# shipments: routes vs station regions
st = pd.read_csv(f"{D}/stations.csv")
ship = pd.read_csv(f"{D}/shipments.csv")
print("\nshipments with station region:")
print(ship.merge(st[["station_id","region"]],on="station_id")[["shipment_id","part_id","station_id","region","route","quantity","earliest","latest"]].to_string())

# plans: budget total vs cost of e.g. covering demand
parts = pd.read_csv(f"{D}/parts.csv")
print("\nplans budget per scenario:", pd.read_csv(f"{D}/plans.csv").groupby("scenario_id").budget.unique())
print("new units total:", pd.read_csv(f"{D}/plans.csv").groupby("scenario_id").new_units.sum())

# firmware mapping: G1->F1, G2->F2 unique?
print("\ngen-firmware combos:", ib.groupby(["generation","firmware"]).size().to_dict())
