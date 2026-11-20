import pandas as pd, numpy as np
D = "/workspace/data"
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
se = pd.read_csv(f"{D}/stock_events.csv", parse_dates=["occurred_at"])
pu = pd.read_csv(f"{D}/part_uses.csv", parse_dates=["occurred_at"])
st = pd.read_csv(f"{D}/stations.csv")
w = wo.merge(ib[["unit_id","station_id","generation"]],on="unit_id").merge(st,on="station_id")

# SLA check for closed orders
w["dur_h"] = (w.repair_at-w.failed_at).dt.total_seconds()/3600
closed = w[w.state=="closed"]
print("closed: repair hours vs service_hours:", closed.groupby(["service_hours"]).dur_h.unique().to_dict())
breached = closed[closed.dur_h > closed.service_hours]
print("breached closed:", len(breached))
# waiting => already breached?
wait = w[w.state=="waiting"].copy()
cutoff = pd.Timestamp("2025-04-21 08:00:00+08:00")
wait["age_h_at_cut"] = (cutoff-wait.failed_at).dt.total_seconds()/3600
print("waiting orders already beyond SLA at cutoff:", (wait.age_h_at_cut>wait.service_hours).sum(), "of", len(wait))
print(wait[["order_id","station_id","service_hours","age_h_at_cut","consequence"]].assign(breach=wait.age_h_at_cut>wait.service_hours).to_string(index=False))

# point-in-time: what is knowable at 2025-04-21? max available_at
print("\nmax available_at:", wo.available_at.max())
print("max stock event:", se.occurred_at.max(), "max part_use:", pu.occurred_at.max())
# so at cutoff 4/21 all observation data (through 4/13) is knowable.

# stock_events kinds beyond receipt/issue/return?
print("\nSE kinds:", se.kind.unique())
# check any negative stock ever (data integrity)
for (s,p),g in se.sort_values("occurred_at").groupby(["station_id","part_id"]):
    if g.quantity.cumsum().min() < 0: print("NEGATIVE:", s, p)
# stock events vs part_uses consistency per order? SE has no order link; check totals per station/part/day
a = se[se.kind.isin(["issue","return"])].groupby(["station_id","part_id",se.occurred_at.dt.date]).quantity.sum()
b = pu.assign(sgn=np.where(pu.kind=="issue",-pu.quantity,pu.quantity)).groupby(["station_id","part_id",pu.occurred_at.dt.date]).sgn.sum()
cmp = pd.DataFrame({"se":a,"pu":b}).fillna(0)
print("\nSE vs PU daily mismatches:", (cmp.se!=cmp.pu).sum())

# P3/P5: any work order demand potential — G1 orders at basic stations cannot use P3
g1_basic = w[(w.generation=="G1") & (w.qualification=="basic")]
print("\nG1 orders at basic stations (P2 only):", len(g1_basic), "at high:", len(w[(w.generation=='G1')&(w.qualification=='high')]))

# order issue lag for closed: always 48h. Waiting orders never issued.
# Summary of net part demand observed (issues minus returns = consumed)
cons = pu.assign(sgn=np.where(pu.kind=="issue",pu.quantity,-pu.quantity)).groupby("part_id").sgn.sum()
print("\nnet consumed parts:", cons.to_dict())
print("gross issues:", pu[pu.kind=='issue'].groupby('part_id').quantity.sum().to_dict())

# G2 closed count and G1 closed count -> implied true part needs
print("\nclosed by gen:", w[w.state=='closed'].groupby('generation').size().to_dict())
print("waiting by gen:", w[w.state=='waiting'].groupby('generation').size().to_dict())
