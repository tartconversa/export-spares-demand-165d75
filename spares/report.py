"""单文件离线 HTML 演练报告（内联 SVG，无外部依赖、无脚本）。

图表只用三类分类色（蓝/橙/青，经调色板校验的前三个槽位），
状态色仅用于 good/warning/critical 语义并始终配文字标签，
颜色从不单独承载含义；每张图都配同数据的表格视图。
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .dataio import DataPack
from .forecast import ForecastResult, consumption_vs_demand, occlusion_summary
from .planning import Action, PlanEvaluation

CSS = """
.viz-root { color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --on-warn:#0b0b0b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b; } }
body.viz-root { margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }
main { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; }
h3 { font-size:15px; margin:20px 0 6px; color:var(--ink-2); }
.sub { color:var(--ink-2); font-size:13px; margin-bottom:20px; }
.card { background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:16px 18px; margin:12px 0; }
.tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.tile { background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:14px 16px; }
.tile .v { font-size:26px; font-weight:650; }
.tile .k { font-size:12px; color:var(--ink-2); }
table { border-collapse:collapse; width:100%; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); }
th { color:var(--ink-2); font-weight:600; }
.tag { display:inline-block; padding:1px 8px; border-radius:999px;
  font-size:11.5px; font-weight:650; color:#fff; }
.tag.good{background:var(--good);} .tag.warn{background:var(--warn);color:var(--on-warn);}
.tag.serious{background:var(--serious);} .tag.crit{background:var(--crit);}
ul.assume { font-size:12.5px; color:var(--ink-2); padding-left:18px; }
.legend { font-size:12px; color:var(--ink-2); margin:6px 0; }
.sw { display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px; }
svg text { fill:var(--ink-2); font-size:11px; }
.axis { stroke:var(--axis); stroke-width:1; } .gridl{stroke:var(--grid);stroke-width:1;}
.note { font-size:12px; color:var(--muted); }
"""


def _table(df: pd.DataFrame, float_digits: int = 2) -> str:
    if df is None or len(df) == 0:
        return '<p class="note">（无记录）</p>'
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:.{float_digits}f}")
    head = "".join(f"<th>{escape(str(c))}</th>" for c in d.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
        for row in d.itertuples(index=False))
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _range_bars(df: pd.DataFrame, label_col: str, mean_col: str,
                lo_col: str, hi_col: str, color: str) -> str:
    """水平区间条：90% 区间线 + 均值点。color 为实际十六进制色值。"""
    rows = df.reset_index(drop=True)
    n = len(rows)
    rh, top, left = 26, 10, 150
    width, height = 760, top * 2 + n * rh
    vmax = float(rows[hi_col].max()) * 1.1 + 0.1
    def x(v): return left + (v / vmax) * (width - left - 20)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="区间预测条形图">']
    for g in range(5):
        gv = vmax * g / 4
        parts.append(f'<line class="gridl" x1="{x(gv):.1f}" y1="{top}" '
                     f'x2="{x(gv):.1f}" y2="{height-8}"/>')
        parts.append(f'<text x="{x(gv):.1f}" y="{height-0}">{gv:.0f}</text>')
    for i, r in rows.iterrows():
        y = top + i * rh + 12
        parts.append(f'<text x="8" y="{y+4}">{escape(str(r[label_col]))}</text>')
        parts.append(f'<line x1="{x(r[lo_col]):.1f}" y1="{y}" '
                     f'x2="{x(r[hi_col]):.1f}" y2="{y}" stroke="{color}" '
                     f'stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x(r[mean_col]):.1f}" cy="{y}" r="4.5" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{x(r[hi_col])+6:.1f}" y="{y+4}">'
                     f'{r[lo_col]:.1f}–{r[hi_col]:.1f}（均{r[mean_col]:.1f}）</text>')
    parts.append("</svg>")
    return "".join(parts)


def _grouped_bars(metrics: pd.DataFrame) -> str:
    """分站按时恢复概率双系列（加权/单量）分组柱。"""
    rows = metrics.reset_index(drop=True)
    n = len(rows)
    top, base_left, bw, gap = 28, 40, 26, 18
    width, height = 760, 240
    plot_h = height - top - 34
    def y(v): return top + (1 - v) * plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="分站按时恢复概率">']
    for g in range(5):
        v = g / 4
        parts.append(f'<line class="gridl" x1="{base_left}" y1="{y(v):.1f}" '
                     f'x2="{width-20}" y2="{y(v):.1f}"/>')
        parts.append(f'<text x="6" y="{y(v)+4:.1f}">{v:.2f}</text>')
    for i, r in rows.iterrows():
        x0 = base_left + 30 + i * (2 * bw + gap + 30)
        for j, (col, c) in enumerate(
                [("on_time_weighted", "#2a78d6"),
                 ("on_time_unweighted", "#eb6834")]):
            h = plot_h * r[col]
            parts.append(f'<rect x="{x0+j*(bw+4)}" y="{y(r[col]):.1f}" '
                         f'width="{bw}" height="{h:.1f}" rx="4" fill="{c}"/>')
            parts.append(f'<text x="{x0+j*(bw+4)+2}" y="{y(r[col])-4:.1f}">'
                         f'{r[col]:.2f}</text>')
        parts.append(f'<text x="{x0+8}" y="{height-12}">{r.station_id}</text>')
    parts.append(f'<line class="axis" x1="{base_left}" y1="{top+plot_h}" '
                 f'x2="{width-20}" y2="{top+plot_h}"/>')
    parts.append("</svg>")
    parts.append('<div class="legend"><span class="sw" style="background:#2a78d6"></span>'
                 '加权按时恢复（按后果权重）'
                 '<span class="sw" style="background:#eb6834"></span>单量按时恢复</div>')
    return "".join(parts)


def write_report(path: str | Path, pack: DataPack, fc: ForecastResult,
                 chosen: list[Action], ev: PlanEvaluation, trace: list[dict],
                 cutoff: pd.Timestamp, scenario_id: str | None) -> Path:
    cutoff = pd.Timestamp(cutoff)
    occl = occlusion_summary(pack)
    cons = consumption_vs_demand(pack)
    ff = fc.failure_forecast.assign(
        label=lambda d: d.station_id + " " + d.generation + "/" + d.environment)
    pd_demand = fc.part_demand.assign(
        label=lambda d: d.station_id + "·" + d.part_id)

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="v">{ev.on_time_weighted:.1%}</div>
        <div class="k">加权按时恢复概率（方案后）</div></div>
      <div class="tile"><div class="v">{ev.expected_backlog_units:.1f}</div>
        <div class="k">窗末期望积压（件）</div></div>
      <div class="tile"><div class="v">{ev.cash_committed:,.0f}</div>
        <div class="k">资金占用（元，预算 {ev.budget:,.0f}）</div></div>
      <div class="tile"><div class="v">{len(occl)}</div>
        <div class="k">被遮挡/积压需求行（无领用≠无故障）</div></div>
    </div>"""

    action_rows = pd.DataFrame([{
        "动作": a.kind, "站": a.station_id, "物料": a.part_id,
        "数量": a.quantity, "资金": f"{a.cash:,.0f}",
        "预计到达": pd.Timestamp(a.eta_latest).date(),
        "依据": a.detail} for a in chosen])

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>备件保障季度演练报告</title><style>{CSS}</style></head>
<body class="viz-root"><main>
<h1>海外储能备件保障 · 季度演练报告</h1>
<div class="sub">截点 {cutoff.date()}（资料第110日） · 展望 {fc.horizon_days} 天 ·
情景 {escape(str(scenario_id or "基准"))} · 全部数据为合成资料 ·
评估仅使用截点可知信息，waiting 工单 repair_at 为空不参与估计</div>
{tiles}

<h2>1. 故障预测（站×代际×环境，含 90% 区间）</h2>
<div class="card">{_range_bars(ff, "label", "failures_mean",
    "failures_p05", "failures_p95", "#2a78d6")}
<h3>数据表</h3>{_table(fc.failure_forecast.round(2))}</div>

<h2>2. 备件需求区间（含现有积压，按兼容首选落料）</h2>
<div class="card">{_range_bars(pd_demand, "label", "demand_mean",
    "demand_p05", "demand_p95", "#1baf7a")}
<h3>数据表</h3>{_table(fc.part_demand.round(2))}</div>

<h2>3. 账面消耗与真实故障需求的偏差</h2>
<div class="card"><p class="note">毛领用受替代与缺货双重扭曲；返修退回不取消
原工单需求（net_consumed 仍低于 true_failure_demand 的部分即被遮挡需求）。</p>
{_table(cons)}</div>

<h2>4. 缺货遮挡与未结单清单</h2>
<div class="card">{_table(occl)}</div>

<h2>5. 保障方案：四种动作的选优结果</h2>
<div class="card"><h3>选定动作（可落实为采购/互借承诺）</h3>
{_table(action_rows)}
<h3>贪心选优轨迹</h3>{_table(pd.DataFrame(trace))}</div>

<h2>6. 履约评估（服务承诺视角，非库存排行榜）</h2>
<div class="card">{_grouped_bars(ev.station_metrics)}
<h3>分站指标表</h3>{_table(ev.station_metrics.round(3), 3)}</div>

<h2>7. 假设登记（资料不支持的独立假设均在此标出）</h2>
<div class="card"><ul class="assume">
{''.join(f'<li>{escape(a)}</li>' for a in fc.assumptions + ev.assumptions)}
</ul></div>
<p class="note">本报告为离线单文件，无脚本、无外部资源；颜色不单独承载含义，
全部图形均附数据表。</p>
</main></body></html>"""
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
