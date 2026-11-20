import pandas as pd
D = "/workspace/data"
stations = pd.read_csv(f"{D}/stations.csv")
parts = pd.read_csv(f"{D}/parts.csv")
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned","warranty_end"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
se = pd.read_csv(f"{D}/stock_events.csv", parse_dates=["occurred_at"])
pu = pd.read_csv(f"{D}/part_uses.csv", parse_dates=["occurred_at"])
comp = pd.read_csv(f"{D}/compatibility.csv")
ship = pd.read_csv(f"{D}/shipments.csv", parse_dates=["earliest","latest"])
so = pd.read_csv(f"{D}/stockouts.csv", parse_dates=["start","end"])
plans = pd.read_csv(f"{D}/plans.csv")

print("== time ranges ==")
for name, df, col in [("WO failed",wo,"failed_at"),("WO opened",wo,"opened_at"),("WO repair",wo,"repair_at"),
                      ("WO available",wo,"available_at"),("SE",se,"occurred_at"),("PU",pu,"occurred_at")]:
    s = df[col].dropna()
    print(f"{name}: {s.min()} -> {s.max()}  n={len(s)}")

print("\n== installed base by station/gen/env ==")
print(ib.groupby(["station_id","generation","environment"]).size().unstack(fill_value=0))
print("\ncommissioned range:", ib.commissioned.min(), ib.commissioned.max())
print("warranty_end unique:", ib.warranty_end.unique())
print("firmware:", ib.firmware.unique(), "gen:", ib.generation.unique(), "env:", ib.environment.unique())

print("\n== work orders ==")
print(wo.state.value_counts())
print("consequence:", wo.consequence.value_counts().to_dict())
w = wo.merge(ib[["unit_id","station_id","generation","firmware","environment","commissioned"]], on="unit_id")
print("\nWO by station/gen/env:")
print(w.groupby(["station_id","generation","environment"]).size())
print("\nwaiting orders by station:")
print(w[w.state=="waiting"].groupby("station_id").size())
print("\nlead failed->opened (days):", ((w.opened_at-w.failed_at).dt.total_seconds()/86400).describe()[["min","mean","max"]].to_dict())
print("repair duration closed (hours):", ((w[w.state=="closed"].repair_at - w[w.state=="closed"].failed_at).dt.total_seconds()/3600).describe()[["min","mean","max"]].to_dict())
