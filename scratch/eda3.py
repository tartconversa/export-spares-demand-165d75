import pandas as pd, numpy as np
D = "/workspace/data"
ib = pd.read_csv(f"{D}/installed_base.csv", parse_dates=["commissioned"])
wo = pd.read_csv(f"{D}/work_orders.csv", parse_dates=["failed_at","opened_at","repair_at","available_at"])
se = pd.read_csv(f"{D}/stock_events.csv", parse_dates=["occurred_at"])
pu = pd.read_csv(f"{D}/part_uses.csv", parse_dates=["occurred_at"])
so = pd.read_csv(f"{D}/stockouts.csv", parse_dates=["start","end"])
w = wo.merge(ib[["unit_id","station_id","generation","firmware","environment"]], on="unit_id")

print("== part_uses by kind/part ==")
print(pu.groupby(["kind","part_id"]).size())
print("\npart_uses net qty by station/part:")
print(pu.assign(sgn=np.where(pu.kind=="issue",pu.quantity,-pu.quantity)).groupby(["station_id","part_id"]).sgn.sum().unstack(fill_value=0))

print("\n== stock_events vs part_uses totals ==")
print("SE issues qty:", -se[se.kind=="issue"].quantity.sum(), " PU issue qty:", pu[pu.kind=="issue"].quantity.sum())
print("SE returns qty:", se[se.kind=="return"].quantity.sum(), " PU returns:", pu[pu.kind=="return_accepted"].quantity.sum())

# orders with multiple uses
g = pu.groupby(["order_id","kind"]).size().unstack(fill_value=0)
print("\norders with >1 issue:", (g.get("issue",0)>1).sum())
multi = g[g.get("issue",0)>1].index
print(pu[pu.order_id.isin(multi)].sort_values(["order_id","occurred_at"]).to_string())

# which parts did each generation order use
pw = pu.merge(w[["order_id","station_id","generation","state"]], on="order_id", suffixes=("","_wo"))
print("\n== parts used by gen (issue only) ==")
print(pw[pw.kind=="issue"].groupby(["generation","part_id"]).size())

# waiting orders: what parts did they issue, and stockout overlap
wait = w[w.state=="waiting"][["order_id","station_id","generation","failed_at","consequence"]]
print("\n== waiting orders ==")
wu = pu[pu.order_id.isin(wait.order_id)].sort_values(["order_id","occurred_at"])
print(wu.to_string())
print("\nwaiting with no part_uses at all:", set(wait.order_id)-set(pu.order_id))
print("\nwaiting order detail:")
print(wait.sort_values(["station_id","failed_at"]).to_string())

# stockout windows vs failed dates of waiting orders
print("\n== stockout windows ==")
print(so)
for _, r in so.iterrows():
    affected = wait[(wait.station_id==r.station_id)]
    for _, o in affected.iterrows():
        d = o.failed_at.date()
        if r.start.date() <= d <= r.end.date():
            print("stockout overlap:", r.station_id, r.part_id, r.start.date(), r.end.date(), "->", o.order_id, d)
