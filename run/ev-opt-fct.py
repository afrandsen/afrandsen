import os
import json
import math
import numpy as np
import pandas as pd
import requests
from pytz import timezone
from pandas.api.types import is_datetime64_any_dtype
from datetime import datetime, timedelta

INITIAL_SOC_PCT = float(os.getenv("INITIAL_SOC_PCT"))
SOC_MIN_PCT = float(os.getenv("SOC_MIN_PCT"))
SOC_MAX_PCT = float(os.getenv("SOC_MAX_PCT"))
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))
trips_json = os.getenv("TRIPS")
trips = pd.DataFrame(json.loads(trips_json))

BATTERY_KWH=75
CHARGER_KW=11
CHARGER_MIN_A=6
CHARGER_VOLT=400
PHASES=3
EFF_KWH_PER_KM=0.128
SOLAR_EFF=0.95
PANEL_AREA=11.5
PANEL_EFF=0.2046
SYSTEMTARIF=0.09250
NETTARIF_TSO=0.07625
ELAFGIFT=0.40000
LOOAD_TILLAEG=0.08000
REFUSION=0.5
TILT=25
AZIMUTH=0
tz = "Europe/Copenhagen"

# Read prices.json with fields: date (ISO8601 UTC), price (øre/kWh)
#prices = pd.read_json("prices.json")
#assert {"date", "price"}.issubset(prices.columns)

def fetch_dk1_prices_dkk():
    today = datetime.now().date()
    start = today
    end = today + timedelta(days=2)  # today + tomorrow

    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        "filter": '{"PriceArea":"DK1"}',
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sort": "HourUTC asc"
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["records"])
    df["date"] = pd.to_datetime(df["HourUTC"], utc=True)
    df["price"] = (df["SpotPriceDKK"]/10)*1.25 

    return df[["date", "price"]].sort_values("date").reset_index(drop=True)

prices_actual = fetch_dk1_prices_dkk()
prices_actual["source"] = "Nordpool"

def fetch_forecast_prices(url="https://raw.githubusercontent.com/solmoller/Spotprisprognose/refs/heads/main/DK1.json"):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()  # dict {timestamp: price}

    # Convert dict → DataFrame
    df = pd.DataFrame(list(j.items()), columns=["date", "price"])

    # Parse datetime
    df["date"] = pd.to_datetime(df["date"], utc=True)

    return df.sort_values("date").reset_index(drop=True)

prices_forecast = fetch_forecast_prices()
prices_forecast["source"] = "Forecast"

def combine_actuals_and_forecast(prices_actual, prices_forecast, tz="Europe/Copenhagen"):
    last_actual = prices_actual["date"].max()
    future = prices_forecast[prices_forecast["date"] > last_actual]

    df = pd.concat([prices_actual, future], ignore_index=True).sort_values("date").reset_index(drop=True)

    # Alternativ now = pd.Timestamp.now(tz=tz).floor("h")
    now = pd.Timestamp.now(tz="UTC").floor("15min")

    # filter from current hour and forward
    df = df[df["date"] >= now]

    return df.reset_index(drop=True)

prices = combine_actuals_and_forecast(prices_actual=prices_actual, prices_forecast=prices_forecast, tz="Europe/Copenhagen")

prices = prices.sort_values("date").reset_index(drop=True)

def optimize_ev_charging(
    trips,                 # DataFrame: columns ["day", "away_start", "away_end", "distance_km", "trip_kwh"]
    prices,                # DataFrame: columns ["date", "price"] (price in øre/kWh)
    battery_kwh=75,
    soc_min_pct=0.30,
    soc_max_pct=0.80,
    charger_kw=11,
    charger_min_a=6,
    charger_volt=400,
    phases=3,
    eff_kwh_per_km=0.128,
    initial_soc_pct=0.78,
    solar_eff=0.95,
    panel_area=11.5,
    panel_eff=0.2046,
    systemtarif=0.09250,
    nettarif_tso=0.07625,
    elafgift=0.40000,
    looad_tillaeg=0.08000,
    lat=0,
    lon=0,
    tilt=25,
    azimuth=0,
    tz="Europe/Copenhagen"
):
    import numpy as np, pandas as pd, math, pulp, requests
    from pytz import timezone
    from pandas.api.types import is_datetime64_any_dtype

    # --- Prices preprocessing ---
    assert {"date", "price"}.issubset(prices.columns)
    if not is_datetime64_any_dtype(prices["date"]):
        prices["date"] = pd.to_datetime(prices["date"], utc=True)
    prices = prices.sort_values("date").reset_index(drop=True)

    # --- Battery & charger parameters ---
    CHARGER_MIN_KW = (charger_min_a * charger_volt * math.sqrt(phases)) / 1000.0
    SOC_MIN = battery_kwh * soc_min_pct
    SOC_MAX = battery_kwh * soc_max_pct
    SOC0    = battery_kwh * initial_soc_pct
    FLAT_ADDERS = systemtarif + nettarif_tso + elafgift + looad_tillaeg

    # --- Build timeline & prices ---
    df = pd.DataFrame({"datetime_utc": prices["date"]})
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(tz)
    df["wday_label"] = df["datetime_local"].dt.day_name().str.lower()
    df["hour_local"] = df["datetime_local"].dt.hour
    df["minute_local"] = df["datetime_local"].dt.minute
    df["spot_kr_kwh"] = prices["price"] / 100.0

    h = df["hour_local"].values
    dso = np.full(len(df), 0.12763)
    dso[(h >= 0) & (h < 6)] = 0.08512
    dso[(h >= 6) & (h < 17)] = 0.12763
    dso[(h >= 17) & (h < 21)] = 0.33200
    df["dso_tariff"] = dso
    df["total_price_kr_kwh"] = df["spot_kr_kwh"] + FLAT_ADDERS + df["dso_tariff"]

    # --- Expand to 15-min resolution ---
    N = len(df)
    df_q = df.loc[df.index.repeat(4)].copy().reset_index(drop=True)
    df_q["datetime_local"] = df_q["datetime_local"] + pd.to_timedelta(np.tile([0,15,30,45], N), unit="m")
    df_q["hour_local"] = df_q["datetime_local"].dt.hour
    df_q["minute_local"] = df_q["datetime_local"].dt.minute
    df_q["wday_label"] = df_q["datetime_local"].dt.day_name().str.lower()

    df = df_q

    # Alternativ now = pd.Timestamp.now(tz=tz)
    now = pd.Timestamp.now(tz=tz).floor("15min")

    # filter from current hour and forward
    df = df.loc[df["datetime_local"] >= now].copy()

    H = len(df)

    # --- Parse trip times (accept HH:MM) ---
    trips = trips.copy()
    for col in ["away_start", "away_end"]:
        if trips[col].dtype == object:
            trips[col] = pd.to_datetime(trips[col], format="%H:%M").dt.time

    # --- Availability ---
    available = np.ones(H, dtype=int)
    for _, t in trips.iterrows():
        idx_day = np.where(df["wday_label"].values == t["day"].lower())[0]
        start_minutes = t["away_start"].hour * 60 + t["away_start"].minute
        end_minutes   = t["away_end"].hour * 60 + t["away_end"].minute
        minutes_of_day = df["hour_local"].values[idx_day] * 60 + df["minute_local"].values[idx_day]
        mask = (minutes_of_day >= start_minutes) & (minutes_of_day < end_minutes)
        available[idx_day[mask]] = 0
    df["available"] = available

    # --- Solar irradiance ---
    start_date = df["datetime_local"].min().strftime("%Y-%m-%d")
    end_date   = df["datetime_local"].max().strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=global_tilted_irradiance_instant"
        f"&tilt={tilt}&azimuth={azimuth}"
        f"&start={start_date}&end={end_date}"
        "&timezone=Europe/Copenhagen"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    irradiance = np.array(j["hourly"]["global_tilted_irradiance_instant"], dtype=float)
    irr_times = pd.to_datetime(j["hourly"]["time"]).tz_localize(tz)
    irr_q_times = irr_times.repeat(4) + pd.to_timedelta(np.tile([0,15,30,45], len(irr_times)), unit="m")
    irr_q_vals  = irradiance.repeat(4)
    match_idx = pd.Series(irr_q_vals, index=irr_q_times)
    irradiance_aligned = match_idx.reindex(df["datetime_local"]).values
    if np.isnan(irradiance_aligned).any():
        raise RuntimeError("Irradiance alignment error at quarter level.")
    solar_energy = (irradiance_aligned / 1000.0) * panel_area * panel_eff * solar_eff * 0.25
    df["irradiance_syn"] = irradiance_aligned
    df["solar_energy_syn"] = solar_energy


    ##
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&minutely_15=global_tilted_irradiance_instant"
        f"&tilt={tilt}&azimuth={azimuth}"
        f"&start={start_date}&end={end_date}"
        "&timezone=Europe/Copenhagen"
    )
    resp_real = requests.get(url, timeout=30)
    resp_real.raise_for_status()
    j_real = resp_real.json()
    irradiance_real = np.array(j_real["minutely_15"]["global_tilted_irradiance_instant"], dtype=float)
    irr_times_real = pd.to_datetime(j_real["minutely_15"]["time"]).tz_localize(tz)
    match_idx_real = pd.Series(irradiance_real, index=irr_times_real)
    irradiance_aligned_real = match_idx_real.reindex(df["datetime_local"]).values
    solar_energy_real = (irradiance_aligned_real / 1000.0) * panel_area * panel_eff * solar_eff * 0.25
    df["irradiance_real"] = irradiance_aligned_real
    df["solar_energy_real"] = solar_energy_real

    df["solar_energy"] = df["solar_energy_real"].fillna(df["solar_energy_syn"])
    df["irradiance"] = df["irradiance_real"].fillna(df["irradiance_syn"])
    
    #####

    # --- Trip energy vector ---
    trip_energy_vec = np.zeros(H)
    for _, t in trips.iterrows():
        need_kwh = float(t["trip_kwh"]) if pd.notna(t["trip_kwh"]) else float(t["distance_km"]) * eff_kwh_per_km
        dep_minutes = t["away_start"].hour * 60 + t["away_start"].minute
        idx_dep = df.index[
            (df["wday_label"].values == t["day"].lower()) &
            ((df["hour_local"].values * 60 + df["minute_local"].values) == dep_minutes)
        ]
        if len(idx_dep) >= 1:
            trip_energy_vec[idx_dep[0]] += need_kwh
        if SOC_MIN + need_kwh > SOC_MAX:
            raise RuntimeError(f"Trip on {t['day']} {t['away_start']} infeasible (need {need_kwh:.1f} kWh + reserve)")

    # --- Build MILP ---
    cap_per_quarter = charger_kw * 0.25
    min_per_quarter = CHARGER_MIN_KW * 0.25
    prob = pulp.LpProblem("ev_charging_opt", pulp.LpMinimize)
    grid  = pulp.LpVariable.dicts("grid",  range(H), lowBound=0, cat=pulp.LpContinuous)
    solar = pulp.LpVariable.dicts("solar", range(H), lowBound=0, cat=pulp.LpContinuous)
    soc   = pulp.LpVariable.dicts("soc",   range(H), lowBound=SOC_MIN, upBound=SOC_MAX, cat=pulp.LpContinuous)
    z     = pulp.LpVariable.dicts("z",     range(H), cat=pulp.LpBinary)
    prices_k = df["total_price_kr_kwh"].values
    prob += pulp.lpSum(grid[h] * float(prices_k[h]) for h in range(H))

    for h in range(H):
        if h == 0:
            prob += soc[h] - grid[h] - solar[h] == (SOC0 - float(trip_energy_vec[h]))
        else:
            prob += soc[h] - soc[h-1] - grid[h] - solar[h] == (-float(trip_energy_vec[h]))
    avail = df["available"].values.astype(float)
    solar_cap = df["solar_energy"].values
    for h in range(H):
        prob += grid[h] + solar[h] <= (cap_per_quarter * avail[h]) * z[h]
        prob += solar[h] <= solar_cap[h] * z[h]
        prob += grid[h] + solar[h] >= min_per_quarter * z[h]
    trip_rows = np.where(trip_energy_vec > 0)[0]
    for h in trip_rows:
        if h > 0:
            prob += soc[h-1] >= SOC_MIN + float(trip_energy_vec[h])

    solver = pulp.PULP_CBC_CMD(msg=False)
    res_status = prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP not optimal. Status: {pulp.LpStatus[prob.status]}")

    # --- Results ---
    grid_opt  = np.array([pulp.value(grid[h])  for h in range(H)])
    solar_opt = np.array([pulp.value(solar[h]) for h in range(H)])
    soc_opt   = np.array([pulp.value(soc[h])   for h in range(H)])
    
    trip_energy_df = pd.DataFrame({
    "trip_kwh_at_departure": pd.Series(trip_energy_vec)
    })

    trip_energy_df["datetime_local"] = df["datetime_local"]
    
    df_out = pd.DataFrame({
        "datetime_local": df["datetime_local"],
        "weekday": df["wday_label"].values,
        "hour": df["hour_local"].values,
        "minute": df["minute_local"].values,
        "price_kr_per_kwh": np.round(df["total_price_kr_kwh"].values, 5),
        "available": df["available"].values,
        "trip_kwh_at_departure": np.round(trip_energy_vec, 3),
        "grid_charge_kwh":  np.round(grid_opt, 4),
        "solar_charge_kwh": np.round(solar_opt, 4),
        "total_charge_kwh": np.round(grid_opt + solar_opt, 4),
        "amp": np.round((((grid_opt + solar_opt) / 0.25) * 1000) / (math.sqrt(phases) * charger_volt), 0),
        "irradiance": df["irradiance"].values,
        "soc_kwh": np.round(soc_opt, 3),
        "cost_kr": np.round(grid_opt * df["total_price_kr_kwh"].values, 4),
    })

    df_out = df_out.merge(
        trip_energy_df[["datetime_local", "trip_kwh_at_departure"]],
        on="datetime_local",
        how="left"
    )

    return df_out


df_out = optimize_ev_charging(
        trips,
        prices,
        BATTERY_KWH, SOC_MIN_PCT, SOC_MAX_PCT,
        CHARGER_KW, CHARGER_MIN_A, CHARGER_VOLT, PHASES,
        EFF_KWH_PER_KM, INITIAL_SOC_PCT,
        SOLAR_EFF, PANEL_AREA, PANEL_EFF,
        SYSTEMTARIF, NETTARIF_TSO, ELAFGIFT, LOOAD_TILLAEG, LAT, LON, TILT, AZIMUTH
    )

# soc before/after (to mirror the R output logic)
soc_kwh_before = np.where(
    (df_out["grid_charge_kwh"].values + df_out["solar_charge_kwh"].values) > 0,
    df_out["soc_kwh"].values - df_out["total_charge_kwh"].values,
    df_out["soc_kwh"].values + df_out["trip_kwh_at_departure"].values,
)

df_out["soc_kwh_before"] = soc_kwh_before

df_out["soc_pct_before"] = np.round((df_out["soc_kwh_before"].values / BATTERY_KWH) * 100.0, 1)
df_out["soc_pct_after"]  = np.round((df_out["soc_kwh"].values / BATTERY_KWH) * 100.0, 1)

mask_events = (
    (df_out["trip_kwh_at_departure"].values > 0) |
    (df_out["grid_charge_kwh"].values > 0) |
    (df_out["solar_charge_kwh"].values > 0)
)

print("\n=== Optimal Charging & Trip Events (15-min) ===")
header = (
    f"{'datetime_local':<16} | {'weekday':<9} | {'hour':<2} | {'minute':<2} | {'irradiance':<10} | "
    f"{'price_kr/kWh':>12} | {'grid_kWh':>8} | {'solar_kWh':>9} | {'total_kwh':>9} | "
    f"{'amp':>3} | {'trip_kWh':>8} | {'soc_kWh':>7} | {'soc_%_before':>12} | {'soc_%_after':>11}"
)
print(header)
print("-" * len(header))

for _, row in df_out.loc[mask_events].iterrows():
    print(
        f"{row['datetime_local']:%Y-%m-%d %H:%M} | "
        f"{row['weekday']:<9} | "
        f"{int(row['hour']):<4d} | "
        f"{int(row['minute']):<6d} | "
        f"{row['irradiance']:>10.0f} | "
        f"{row['price_kr_per_kwh']:>12.2f} | "
        f"{row['grid_charge_kwh']:>8.2f} | "
        f"{row['solar_charge_kwh']:>9.2f} | "
        f"{row['total_charge_kwh']:>9.2f} | "
        f"{int(row['amp']):>3d} | "
        f"{row['trip_kwh_at_departure']:>8.2f} | "
        f"{row['soc_kwh']:>7.2f} | "
        f"{row['soc_pct_before']:>12.1f} | "
        f"{row['soc_pct_after']:>11.1f}"
    )

# Totals
total_cost = float((df_out["cost_kr"]).sum())
effective_cost = total_cost - float(df_out["solar_charge_kwh"].sum()) * REFUSION

total_charge = float(df_out["total_charge_kwh"].sum())
from_grid = float(df_out["grid_charge_kwh"].sum())
from_solar = float(df_out["solar_charge_kwh"].sum())

avg_cost = (total_cost / total_charge) if total_charge > 0 else float("nan")
avg_cost_eff = (effective_cost / total_charge) if total_charge > 0 else float("nan")

# --- Daily summary ---
df_out["date"] = df_out["datetime_local"].dt.date
df_out["weekday"] = df_out["datetime_local"].dt.day_name()

# base aggregations
daily_summary = df_out.groupby(["date", "weekday"]).agg(
    grid_kWh=("grid_charge_kwh", "sum"),
    solar_kWh=("solar_charge_kwh", "sum"),
    total_kWh=("total_charge_kwh", "sum"),
    trip_kWh=("trip_kwh_at_departure", "sum"),
    cost=("cost_kr", "sum"),
).reset_index()

# get start/end SoC properly
soc_start = (
    df_out.sort_values("datetime_local")
    .groupby("date")["soc_pct_before"]
    .first()
)
soc_end = (
    df_out.sort_values("datetime_local")
    .groupby("date")["soc_pct_after"]
    .last()
)

daily_summary["soc_start"] = daily_summary["date"].map(soc_start)
daily_summary["soc_end"] = daily_summary["date"].map(soc_end)

# effective cost
daily_summary["effective_cost"] = daily_summary["cost"] - daily_summary["solar_kWh"] * REFUSION

# average costs
daily_summary["cost_per_kWh"] = daily_summary["cost"] / daily_summary["total_kWh"].replace(0, np.nan)
daily_summary["eff_cost_per_kWh"] = daily_summary["effective_cost"] / daily_summary["total_kWh"].replace(0, np.nan)

# print
print("\n=== Daily Summary ===")
header = (
    f"{'date':<10} | {'weekday':<9} | {'grid_kWh':>8} | {'solar_kWh':>8} | "
    f"{'total_kWh':>8} | {'trip_kWh':>8} | {'soc_start%':>9} | {'soc_end%':>7} | "
    f"{'cost':>8} | {'eff_cost':>10} | {'avg_cost':>9} | {'avg_eff':>9}"
)
print(header)
print("-" * len(header))

for _, row in daily_summary.iterrows():
    print(
        f"{row['date']} | "
        f"{row['weekday']:<9} | "
        f"{row['grid_kWh']:8.2f} | "
        f"{row['solar_kWh']:9.2f} | "
        f"{row['total_kWh']:9.2f} | "
        f"{row['trip_kWh']:8.2f} | "
        f"{row['soc_start']:10.1f} | "
        f"{row['soc_end']:8.1f} | "
        f"{row['cost']:8.2f} | "
        f"{row['effective_cost']:10.2f} | "
        f"{row['cost_per_kWh']:9.2f} | "
        f"{row['eff_cost_per_kWh']:9.2f}"
    )

print(
    f"Total cost: {total_cost:.2f} kr. "
    f"Total effective cost: {effective_cost:.2f} kr. "
    f"Total charging: {total_charge:.2f} kWh ({from_grid:.2f} grid, {from_solar:.2f} solar). "
    f"Cost per kWh: {avg_cost:.2f} kr/kWh. Eff. cost per kWh: {avg_cost_eff:.2f} kr/kWh."
)

# import matplotlib.pyplot as plt

# # Assuming prices has columns: 'datetime_local', 'price', 'source'
# plt.figure(figsize=(15,5))

# # Set colors: blue for Nordpool, semi-transparent blue for Forecast
# colors = ['tab:blue' if s == 'Nordpool' else 'tab:green' for s in prices['source']]
# alphas = [1.0 if s == 'Nordpool' else 0.5 for s in prices['source']]

# prices["datetime_local"] = prices["date"].dt.tz_convert(tz)

# # Plot bars
# plt.bar(prices['datetime_local'], prices['price'], width=0.02, color=colors, alpha=1.0)

# # Apply transparency individually
# for i, (d, p, a) in enumerate(zip(prices['datetime_local'], prices['price'], alphas)):
#     plt.bar(d, p, width=0.02, color='tab:blue', alpha=a)

# plt.xlabel('Datetime')
# plt.ylabel('Price [kr/kWh]')
# plt.title('DK1 Spot Prices')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# --- Export function ---

def export_interactive_table(
    df, 
    filename="table.html", 
    table_id="table", 
    drop_cols=None, 
    mask=None, 
    scale=0.85, 
    footer_text=None   # <-- new argument
):
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if mask is not None:
        df = df.loc[mask].copy()

    folder_path = "static/bare/ev"
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("""
        <html>
        <head>
          <link rel="stylesheet" 
                href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
          <script src="https://code.jquery.com/jquery-3.7.1.js"></script>
          <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
          <style>
            table.dataTable tbody td {
                font-size: 12px;
                padding: 2px 6px;
            }
            table.dataTable thead th {
                font-size: 13px;
            }
            .summary-text {
                margin-top: 20px;
                font-size: 14px;
                font-weight: bold;
            }
          </style>
        </head>
        <body>
        """)
        f.write(f"""
        <div style="transform: scale({scale}); transform-origin: top left; width: 110%;">
        {df.to_html(index=False, table_id=table_id, border=0, classes="display compact")}
        </div>
        """)
        # Add footer text if provided
        if footer_text:
            f.write(f'<div class="summary-text">{footer_text}</div>')
        f.write(f"""
        <script>
          $(document).ready(function() {{
              $('#{table_id}').DataTable({{
                  "pageLength": 25,
                  "order": [[0, "asc"]]
              }});
          }});
        </script>
        </body>
        </html>
        """)

# --- Construct cost summary string ---
summary_text = (
    f"Total cost: {total_cost:.2f} kr. "
    f"Total effective cost: {effective_cost:.2f} kr. "
    f"Total charging: {total_charge:.2f} kWh ({from_grid:.2f} grid, {from_solar:.2f} solar). "
    f"Cost per kWh: {avg_cost:.2f} kr/kWh. Eff. cost per kWh: {avg_cost_eff:.2f} kr/kWh."
)

mask_events = (
    (df_out["trip_kwh_at_departure"].values > 0) |
    (df_out["grid_charge_kwh"].values > 0) |
    (df_out["solar_charge_kwh"].values > 0)
)

# --- Export tables with summary text ---
export_interactive_table(
    df_out,
    "ev_schedule.html",
    "schedule",
    drop_cols=["available", "date"],
    mask=mask_events,
    footer_text=summary_text   # 👈 add summary
)

export_interactive_table(
    daily_summary,
    "ev_summary.html",
    "summary",
    footer_text=summary_text   # 👈 add summary
)

