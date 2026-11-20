import pandas as pd, numpy as np
D = "/workspace/data"
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
w = wo.merge(ib, on="unit_id", suffixes=("","_ib"))
obs_end_naive = pd.Timestamp("2025-04-09 08:00:00")
ib["age_at_obs_end"] = (obs_end_naive-ib.commissioned).dt.days
w["age"] = (w.failed_at.dt.tz_localize(None)-w.commissioned).dt.days
print("age at obs end:", ib.age_at_obs_end.describe()[["min","max","mean"]].round(1).to_dict())
print("age at failure:", w.age.describe()[["min","25%","50%","75%","max"]].round(1).to_dict())

# age-bucket hazard
bins=[0,30,45,60,75,200]; labels=["0-30","31-45","46-60","61-75","76+"]
w["bin"]=pd.cut(w.age,bins,labels=labels)
# exposure per bin per unit (units enter at commissioned, observed till obs_end)
ib_s = ib.sort_values("commissioned")
# total unit-days in each bin
expo = np.zeros(len(labels)); fails = np.zeros(len(labels))
for _, u in ib.iterrows():
    age0, age1 = 0, u.age_at_obs_end
    for i,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
        lo2=max(lo,age0); hi2=min(hi,age1+1)
        if hi2>lo2: expo[i]+=hi2-lo2
for b in w.bin.cat.categories:
    pass
f = w.bin.value_counts().reindex(labels).fillna(0)
print("\nhazard by age bin:")
for i,l in enumerate(labels):
    print(f"  {l}: fails={int(f[i])}, unit-days={int(expo[i])}, hazard/1000d={1000*f[i]/expo[i]:.3f}")

# weekly stationarity
w["week"] = w.failed_at.dt.isocalendar().week
print("\nfailures per ISO week:", w.groupby("week").size().to_dict())

# consequence 3
print("\nconsequence-3:")
print(w[w.consequence==3][["order_id","station_id","generation","environment","failed_at","state"]].to_string(index=False))

# shipments vs regions
st = pd.read_csv(f"{D}/stations.csv"); ship = pd.read_csv(f"{D}/shipments.csv")
print("\nshipments:")
print(ship.merge(st[["station_id","region"]],on="station_id")[["shipment_id","part_id","station_id","region","route"]].to_string(index=False))
pl = pd.read_csv(f"{D}/plans.csv")
print("\nbudget per scenario:", pl.groupby("scenario_id").budget.unique().to_dict())
print("new units total per scenario:", pl.groupby("scenario_id").new_units.sum().to_dict())
print("\ngen-firmware:", ib.groupby(["generation","firmware"]).size().to_dict())
print("env by region:", ib.merge(st,on="station_id").groupby(["region","environment"]).size().to_dict())

# service_hours vs observed repair: all closed = 48h; basic stations have 72h commitment
# check waiting order ages at obs end
wait = w[w.state=="waiting"].copy()
wait["wait_days_at_obs"] = (obs_end_naive - wait.failed_at.dt.tz_localize(None)).dt.days
print("\nwaiting order age at obs end (days):")
print(wait.groupby("station_id").wait_days_at_obs.describe()[["min","mean","max"]])

# Poisson check: 3/day network-wide = 0.6/day/station; over 34 days each station 20 failures
per_station_day = w.groupby(["station_id", w.failed_at.dt.date]).size()
print("\nmean failures/station/day:", per_station_day.mean().round(3), "var:", per_station_day.var().round(3))
