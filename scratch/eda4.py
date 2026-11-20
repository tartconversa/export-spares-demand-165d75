import pandas as pd, numpy as np
D = "/workspace/data"
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
se = pd.read_csv(f"{D}/stock_events.csv", parse_dates=["occurred_at"])
pu = pd.read_csv(f"{D}/part_uses.csv", parse_dates=["occurred_at"])
w = wo.merge(ib[["unit_id","station_id","generation","firmware","environment"]], on="unit_id")

# running on-hand timeline at each failure date for waiting orders
wait = w[w.state=="waiting"].sort_values("failed_at")
def onhand(station, part, t):
    s = se[(se.station_id==station)&(se.part_id==part)&(se.occurred_at<=t)]
    return int(s.quantity.sum())
def need_parts(gen, station):
    if gen=="G2": return [("main","P1"),("aux","P4")]
    return [("main","P2")]
print("== waiting orders: on-hand at failure ==")
for _, o in wait.iterrows():
    t = o.failed_at
    rows=[]
    for grp,p in need_parts(o.generation,o.station_id):
        rows.append(f"{p}={onhand(o.station_id,p,t)}")
    print(o.order_id, o.station_id, o.generation, str(t.date()), "cons=%d"%o.consequence, " ".join(rows))

# check: did stock hit zero at any point per station/part? build daily min
print("\n== min running on-hand over whole window (per station/part) ==")
for (s,p), g in se.sort_values("occurred_at").groupby(["station_id","part_id"]):
    cum = g.quantity.cumsum()
    print(s,p,"min=",cum.min(),"end=",cum.iloc[-1])

# G1 orders at high stations (S1,S2,S3): any P3 used? verify substitution never exercised
print("\n== G1 closed orders and parts ==")
g1 = w[(w.generation=="G1")]
g1u = pu[pu.order_id.isin(g1.order_id) & (pu.kind=="issue")]
print(g1u.groupby(["station_id","part_id"]).size())
# high stations qualification
print("\nstations:", pd.read_csv(f"{D}/stations.csv").set_index("station_id").qualification.to_dict())

# returns: which orders, closed? what happened after return (re-issue?)
ret = pu[pu.kind=="return_accepted"]
print("\n== return orders state ==")
print(ret.merge(wo[["order_id","state","repair_at"]],on="order_id")[["order_id","station_id","part_id","occurred_at","state","repair_at"]].to_string())

# timing: issue vs failed_at for closed orders
closed = w[w.state=="closed"]
ci = pu[(pu.order_id.isin(closed.order_id))&(pu.kind=="issue")].merge(wo[["order_id","failed_at"]],on="order_id")
ci["lag_h"] = (ci.occurred_at-ci.failed_at).dt.total_seconds()/3600
print("\nissue lag hours vs failure:", ci.lag_h.value_counts().to_dict())

# available_at lag
print("available_at - failed_at days:", ((wo.available_at-wo.failed_at).dt.total_seconds()/86400).value_counts().to_dict())

# failures per day per station - check regional correlation (same region events)
w2 = w.copy(); w2["d"]=w2.failed_at.dt.date
tab = w2.groupby(["d","station_id"]).size().unstack(fill_value=0)
print("\ndaily failures head:\n", tab.head(8))
# region of each station
st = pd.read_csv(f"{D}/stations.csv")
print("\ncorrelation of daily counts within region:")
print(tab.corr())
