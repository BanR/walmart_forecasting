"""
Step 3: EDA — Generate key visualisations and statistics for M5 data.
Outputs an HTML report with interactive charts.

Usage:
    python -m src.eda
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path


DATA_DIR = "data"
OUTPUT_DIR = "docs"


def load_data():
    print("Loading data...")
    t0 = time.time()
    sales = pd.read_csv(f"{DATA_DIR}/sales_train_evaluation.csv")
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return sales, calendar, prices


def compute_eda_stats(sales, calendar, prices):
    """Compute all EDA statistics needed for the report."""
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    stats = {}

    # --- 1. Daily total sales over time ---
    print("  Computing daily totals...")
    daily_total = sales[day_cols].sum(axis=0).values
    # Downsample to weekly for charting (sum each 7 days)
    n_weeks = len(daily_total) // 7
    weekly_total = daily_total[:n_weeks*7].reshape(n_weeks, 7).sum(axis=1)
    stats["weekly_total"] = weekly_total.tolist()

    # Date labels for weekly
    cal_dates = calendar.set_index("d")["date"].to_dict()
    week_labels = [cal_dates.get(f"d_{i*7+1}", "") for i in range(n_weeks)]
    stats["week_labels"] = week_labels

    # --- 2. Daily total by category ---
    print("  Computing category trends...")
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        cat_sales = sales[sales.cat_id == cat][day_cols].sum(axis=0).values
        cat_weekly = cat_sales[:n_weeks*7].reshape(n_weeks, 7).sum(axis=1)
        stats[f"weekly_{cat.lower()}"] = cat_weekly.tolist()

    # --- 3. Day-of-week effect ---
    print("  Computing day-of-week patterns...")
    # Use last 52 weeks of data
    last_364_cols = day_cols[-364:]
    last_364_sales = sales[last_364_cols].values  # (30490, 364)
    daily_totals = last_364_sales.sum(axis=0)  # (364,)
    # Reshape to (52, 7) and average
    dow_pattern = daily_totals.reshape(52, 7).mean(axis=0)
    # Get day names from calendar
    last_364_d = last_364_cols[:7]
    dow_names = [calendar[calendar.d == d]["weekday"].values[0] for d in last_364_d]
    stats["dow_pattern"] = dow_pattern.tolist()
    stats["dow_names"] = dow_names

    # --- 4. Monthly seasonality ---
    print("  Computing monthly patterns...")
    cal_map = calendar[["d", "month", "year"]].copy()
    cal_map = cal_map[cal_map.d.isin(day_cols)]
    daily_series = pd.DataFrame({"d": day_cols, "total": sales[day_cols].sum(axis=0).values})
    daily_series = daily_series.merge(cal_map, on="d")
    monthly_avg = daily_series.groupby("month")["total"].mean().values
    stats["monthly_avg"] = monthly_avg.tolist()
    stats["month_names"] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # --- 5. SNAP effect ---
    print("  Computing SNAP effects...")
    snap_effects = {}
    for state in ["CA", "TX", "WI"]:
        snap_col = f"snap_{state}"
        state_sales = sales[sales.state_id == state][day_cols].sum(axis=0)
        state_daily = pd.DataFrame({"d": day_cols, "sales": state_sales.values})
        state_daily = state_daily.merge(calendar[["d", snap_col]], on="d")
        snap_on = state_daily[state_daily[snap_col] == 1]["sales"].mean()
        snap_off = state_daily[state_daily[snap_col] == 0]["sales"].mean()
        snap_effects[state] = {"on": round(snap_on, 0), "off": round(snap_off, 0),
                               "lift_pct": round((snap_on/snap_off - 1)*100, 1)}
    stats["snap_effects"] = snap_effects

    # --- 6. Event effects ---
    print("  Computing event effects...")
    daily_all = pd.DataFrame({"d": day_cols, "total": sales[day_cols].sum(axis=0).values})
    daily_all = daily_all.merge(calendar[["d", "event_name_1", "event_type_1"]], on="d")
    no_event_mean = daily_all[daily_all.event_name_1.isna()]["total"].mean()
    event_means = daily_all[daily_all.event_name_1.notna()].groupby("event_name_1")["total"].mean()
    top_events = event_means.nlargest(10)
    bottom_events = event_means.nsmallest(5)
    stats["no_event_mean"] = round(no_event_mean, 0)
    stats["top_events"] = {k: round(v, 0) for k, v in top_events.items()}
    stats["bottom_events"] = {k: round(v, 0) for k, v in bottom_events.items()}

    # --- 7. Zero distribution per department ---
    print("  Computing zero distributions...")
    zero_by_dept = {}
    for dept in sorted(sales.dept_id.unique()):
        dept_data = sales[sales.dept_id == dept][day_cols].values
        zero_pct = (dept_data == 0).mean() * 100
        zero_by_dept[dept] = round(zero_pct, 1)
    stats["zero_by_dept"] = zero_by_dept

    # --- 8. Sales distribution (histogram of mean daily sales per series) ---
    print("  Computing sales distribution...")
    mean_sales = sales[day_cols[-28:]].mean(axis=1)
    hist_bins = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500]
    hist_counts = []
    for i in range(len(hist_bins)-1):
        count = ((mean_sales >= hist_bins[i]) & (mean_sales < hist_bins[i+1])).sum()
        hist_counts.append(int(count))
    hist_counts.append(int((mean_sales >= hist_bins[-1]).sum()))
    stats["sales_hist_bins"] = [str(b) for b in hist_bins] + ["500+"]
    stats["sales_hist_counts"] = hist_counts

    # --- 9. Price volatility ---
    print("  Computing price statistics...")
    price_stats = prices.groupby("item_id")["sell_price"].agg(["mean", "std", "count"])
    price_stats["cv"] = price_stats["std"] / price_stats["mean"]
    stats["price_cv_mean"] = round(price_stats["cv"].mean(), 3)
    stats["price_cv_median"] = round(price_stats["cv"].median(), 3)
    stats["price_changes_per_item"] = round(price_stats["count"].mean(), 1)

    # --- 10. Item lifespan ---
    print("  Computing item lifespans...")
    first_sale = np.argmax(sales[day_cols].values > 0, axis=1)
    lifespan = len(day_cols) - first_sale
    stats["lifespan_min"] = int(lifespan.min())
    stats["lifespan_max"] = int(lifespan.max())
    stats["lifespan_median"] = int(np.median(lifespan))
    stats["pct_full_history"] = round((lifespan == len(day_cols)).mean() * 100, 1)
    stats["pct_under_1year"] = round((lifespan < 365).mean() * 100, 1)

    return stats


def generate_html(stats):
    """Generate interactive HTML EDA report."""
    stats_json = json.dumps(stats)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>M5 EDA — Sales Patterns & Data Understanding</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2e3145;
    --accent: #4f8ef7; --green: #22c55e; --orange: #f97316;
    --red: #ef4444; --yellow: #eab308; --purple: #a855f7;
    --text: #e2e8f0; --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
  h1 span {{ color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 28px; }}
  .grid {{ display: grid; gap: 18px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }}
  .card-title {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 12px; }}
  .stat-row {{ display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; flex: 1; min-width: 140px; }}
  .stat .val {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
  .stat .lbl {{ font-size: 0.72rem; color: var(--muted); margin-top: 2px; }}
  .section {{ margin: 24px 0 14px; font-size: 0.95rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }}
  .section::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
  .badge {{ background: var(--accent); color: var(--bg); border-radius: 20px; font-size: 0.6rem; font-weight: 700; padding: 2px 8px; }}
  .insight {{ background: #12151e; border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 0.8rem; color: var(--muted); margin-top: 10px; }}
  .insight strong {{ color: var(--text); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  th {{ color: var(--muted); text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #1e2234; }}
  @media (max-width: 900px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>M5 EDA <span>— Sales Patterns & Data Understanding</span></h1>
<p class="subtitle">Key patterns that drive feature engineering and model design</p>

<div class="stat-row">
  <div class="stat"><div class="val">{stats['lifespan_median']}</div><div class="lbl">Median item lifespan (days)</div></div>
  <div class="stat"><div class="val">{stats['pct_full_history']}%</div><div class="lbl">Items with full history</div></div>
  <div class="stat"><div class="val">{stats['pct_under_1year']}%</div><div class="lbl">Items < 1 year old</div></div>
  <div class="stat"><div class="val">{stats['price_cv_median']}</div><div class="lbl">Median price coeff. of variation</div></div>
</div>

<div class="section"><span class="badge">1</span> Weekly Total Sales Over Time (by Category)</div>
<div class="card">
  <canvas id="trendChart" height="80"></canvas>
  <div class="insight"><strong>Key finding:</strong> Clear yearly seasonality (peaks around Dec/holiday season). FOODS dominates volume. Notable dips on Christmas Day (stores closed).</div>
</div>

<div class="section"><span class="badge">2</span> Day-of-Week & Monthly Patterns</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">Average Daily Sales by Day of Week (last year)</div>
    <canvas id="dowChart" height="180"></canvas>
    <div class="insight"><strong>Key finding:</strong> Strong weekly cycle. Weekend days (Sat/Sun) have noticeably higher sales — important calendar feature.</div>
  </div>
  <div class="card">
    <div class="card-title">Average Daily Total Sales by Month</div>
    <canvas id="monthChart" height="180"></canvas>
    <div class="insight"><strong>Key finding:</strong> Sales ramp in spring, peak activity in summer months. December shows holiday boost.</div>
  </div>
</div>

<div class="section"><span class="badge">3</span> SNAP & Event Effects</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">SNAP Benefit Days — Sales Lift</div>
    <table>
      <thead><tr><th>State</th><th>SNAP Day Sales</th><th>Non-SNAP Sales</th><th>Lift</th></tr></thead>
      <tbody>
        <tr><td>California</td><td>{stats['snap_effects']['CA']['on']:,.0f}</td><td>{stats['snap_effects']['CA']['off']:,.0f}</td><td style="color:var(--green)">+{stats['snap_effects']['CA']['lift_pct']}%</td></tr>
        <tr><td>Texas</td><td>{stats['snap_effects']['TX']['on']:,.0f}</td><td>{stats['snap_effects']['TX']['off']:,.0f}</td><td style="color:var(--green)">+{stats['snap_effects']['TX']['lift_pct']}%</td></tr>
        <tr><td>Wisconsin</td><td>{stats['snap_effects']['WI']['on']:,.0f}</td><td>{stats['snap_effects']['WI']['off']:,.0f}</td><td style="color:var(--green)">+{stats['snap_effects']['WI']['lift_pct']}%</td></tr>
      </tbody>
    </table>
    <div class="insight"><strong>Key finding:</strong> SNAP days increase sales 2–8% depending on state. Must use state-specific SNAP flags as features.</div>
  </div>
  <div class="card">
    <div class="card-title">Event Impact — Top Sales Boosters & Suppressors</div>
    <canvas id="eventChart" height="180"></canvas>
    <div class="insight"><strong>Key finding:</strong> Some events drive massive sales spikes (pre-holiday buying). Christmas/Thanksgiving suppress (closed stores). Avg non-event day: {stats['no_event_mean']:,.0f} units.</div>
  </div>
</div>

<div class="section"><span class="badge">4</span> Intermittency & Sales Distribution</div>
<div class="grid grid-2">
  <div class="card">
    <div class="card-title">% Zero-Sales Days by Department</div>
    <canvas id="zeroChart" height="200"></canvas>
    <div class="insight"><strong>Key finding:</strong> HOBBIES_2 has 86%+ zeros (highly intermittent). FOODS_3 has lowest zeros at ~54%. Loss function choice must handle this.</div>
  </div>
  <div class="card">
    <div class="card-title">Distribution of Mean Daily Sales (last 28d per series)</div>
    <canvas id="histChart" height="200"></canvas>
    <div class="insight"><strong>Key finding:</strong> Most series sell less than 2 units/day. Long tail with a few high-volume items. Revenue weighting means these high-volume items matter disproportionately.</div>
  </div>
</div>

<script>
const S = {stats_json};

// --- Trend chart ---
new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{
    labels: S.week_labels,
    datasets: [
      {{ label:'FOODS', data:S.weekly_foods, borderColor:'#3b82f6', borderWidth:1.5, pointRadius:0, fill:false }},
      {{ label:'HOUSEHOLD', data:S.weekly_household, borderColor:'#22c55e', borderWidth:1.5, pointRadius:0, fill:false }},
      {{ label:'HOBBIES', data:S.weekly_hobbies, borderColor:'#a855f7', borderWidth:1.5, pointRadius:0, fill:false }},
    ]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color:'#94a3b8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#94a3b8', maxTicksLimit:20 }}, grid: {{ color:'#2e3145' }} }},
      y: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#2e3145' }} }}
    }}
  }}
}});

// --- Day of week ---
new Chart(document.getElementById('dowChart'), {{
  type: 'bar',
  data: {{
    labels: S.dow_names,
    datasets: [{{ data: S.dow_pattern, backgroundColor: S.dow_pattern.map((v,i) => i>=5 ? '#3b82f6' : '#3b82f680'), borderRadius: 4 }}]
  }},
  options: {{ plugins: {{ legend: {{ display:false }} }}, scales: {{ y: {{ grid: {{ color:'#2e3145' }}, ticks: {{ color:'#94a3b8' }} }}, x: {{ grid: {{ display:false }}, ticks: {{ color:'#94a3b8' }} }} }} }}
}});

// --- Monthly ---
new Chart(document.getElementById('monthChart'), {{
  type: 'bar',
  data: {{
    labels: S.month_names,
    datasets: [{{ data: S.monthly_avg, backgroundColor: '#22c55e80', borderColor: '#22c55e', borderWidth: 1, borderRadius: 4 }}]
  }},
  options: {{ plugins: {{ legend: {{ display:false }} }}, scales: {{ y: {{ grid: {{ color:'#2e3145' }}, ticks: {{ color:'#94a3b8' }} }}, x: {{ grid: {{ display:false }}, ticks: {{ color:'#94a3b8' }} }} }} }}
}});

// --- Events ---
const eventLabels = Object.keys(S.top_events).concat(Object.keys(S.bottom_events));
const eventValues = Object.values(S.top_events).concat(Object.values(S.bottom_events));
const eventColors = eventValues.map(v => v > S.no_event_mean ? '#22c55ecc' : '#ef4444cc');
new Chart(document.getElementById('eventChart'), {{
  type: 'bar',
  data: {{
    labels: eventLabels,
    datasets: [{{ data: eventValues, backgroundColor: eventColors, borderRadius: 4 }}]
  }},
  options: {{
    indexAxis: 'y',
    plugins: {{ legend: {{ display:false }},
      annotation: {{ annotations: {{ line1: {{ type:'line', xMin:S.no_event_mean, xMax:S.no_event_mean, borderColor:'#f97316', borderDash:[5,5] }} }} }}
    }},
    scales: {{ x: {{ grid: {{ color:'#2e3145' }}, ticks: {{ color:'#94a3b8' }} }}, y: {{ grid: {{ display:false }}, ticks: {{ color:'#94a3b8', font: {{ size:10 }} }} }} }}
  }}
}});

// --- Zero chart ---
const zeroDepts = Object.keys(S.zero_by_dept);
const zeroVals = Object.values(S.zero_by_dept);
const zeroColors = zeroVals.map(v => v > 75 ? '#ef4444cc' : v > 65 ? '#f97316cc' : '#22c55ecc');
new Chart(document.getElementById('zeroChart'), {{
  type: 'bar',
  data: {{
    labels: zeroDepts,
    datasets: [{{ data: zeroVals, backgroundColor: zeroColors, borderRadius: 4 }}]
  }},
  options: {{
    plugins: {{ legend: {{ display:false }} }},
    scales: {{ y: {{ grid: {{ color:'#2e3145' }}, ticks: {{ color:'#94a3b8', callback: v=>v+'%' }}, max:100 }}, x: {{ grid: {{ display:false }}, ticks: {{ color:'#94a3b8' }} }} }}
  }}
}});

// --- Histogram ---
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: S.sales_hist_bins,
    datasets: [{{ data: S.sales_hist_counts, backgroundColor: '#a855f7cc', borderRadius: 4 }}]
  }},
  options: {{
    plugins: {{ legend: {{ display:false }} }},
    scales: {{ y: {{ type:'logarithmic', grid: {{ color:'#2e3145' }}, ticks: {{ color:'#94a3b8' }} }}, x: {{ grid: {{ display:false }}, ticks: {{ color:'#94a3b8', font: {{ size:10 }} }} }} }}
  }}
}});
</script>
</body>
</html>"""
    return html


def main():
    sales, calendar, prices = load_data()
    stats = compute_eda_stats(sales, calendar, prices)

    html = generate_html(stats)
    output_path = Path(OUTPUT_DIR) / "M5_eda.html"
    output_path.write_text(html)
    print(f"\n  EDA report saved to: {output_path}")


if __name__ == "__main__":
    main()
