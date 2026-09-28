"""OmniTwin - Urban Environmental Digital Twin (hackathon prototype).

Run:  pip install -r requirements.txt && streamlit run app.py
"""
import datetime as dt

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import engine as E

BLUE, ORANGE, GREY, GREEN, RED = "#1f77b4", "#ff7f0e", "#444444", "#2e9e4f", "#d62728"
SRC_COLORS = {"Vehicular": "#1f77b4", "Industrial": "#7f7f7f",
              "Dust/Weather": "#e0b040", "Background/regional": "#c9d6df"}
ASSUMPTION = "Assumption: Industrial baseline is constant; traffic scales with rush hour; dust correlates with wind speed."

st.set_page_config(page_title="OmniTwin - Urban Environmental Digital Twin",
                   page_icon="🌫️", layout="wide")


def badge(kind: str):
    label, col = {"obs": ("OBSERVED DATA", BLUE),
                  "mod": ("MODELED SCENARIO", ORANGE),
                  "val": ("ACTUAL HISTORICAL (VALIDATION)", GREY)}[kind]
    st.markdown(f"<div style='background:{col};color:#fff;padding:10px 16px;border-radius:10px;"
                f"font-size:1.5rem;font-weight:800;letter-spacing:.05em;text-align:center;margin:6px 0'>"
                f"{label}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached wrappers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(file_bytes):
    if file_bytes is None:
        return E.make_synthetic()
    import io
    return pd.read_csv(io.BytesIO(file_bytes), parse_dates=["date"])


@st.cache_data(show_spinner=False)
def series_for(df, station):
    return E.get_series(df, station)


@st.cache_resource(show_spinner="Training forecast model...")
def model_for(_s, key, kind):
    return E.fit_forecaster(_s, kind)


@st.cache_data(show_spinner="Running historical back-test...")
def backtest_for(_s, _model, key, kind):
    return E.backtest(_model, _s)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🌫️ OmniTwin")
st.sidebar.caption("Urban Environmental Digital Twin · Pune demo area")

up = st.sidebar.file_uploader("Optional: your own CSV (CPCB / Kaggle)", type="csv")
file_bytes = up.getvalue() if up else None
with st.sidebar.expander("If your CSV has no lat/lon or station"):
    area_name = st.text_input("Area name", "Pune")
    area_lat = st.number_input("Latitude", value=18.5204, format="%.4f")
    area_lon = st.number_input("Longitude", value=73.8567, format="%.4f")
raw = load_data(file_bytes)
try:
    df, load_notes = E.prepare_df(raw, area_lat, area_lon, area_name)
except ValueError as err:
    st.error(str(err))
    st.stop()
if load_notes:
    st.sidebar.info("Data notes:\n\n" + "\n\n".join("• " + n for n in load_notes))
data_id = "synthetic" if file_bytes is None else str(hash(file_bytes))
is_demo = file_bytes is None

with st.sidebar.expander("CSV format / demo data"):
    st.write("Minimum: **Date, PM2.5, Traffic_Index, Wind_Speed** (daily). Optional: station, lat, lon, "
             "temp, humidity, rain, industrial_index. Missing optional columns are filled with defaults.")
    st.download_button("Download demo CSV", E.make_synthetic().to_csv(index=False),
                       "omnitwin_demo.csv", "text/csv")

stations = list(df.groupby("station").size().index)
station = st.sidebar.selectbox("Forecast area", (["City average"] if len(stations) > 1 else []) + stations)
kind = st.sidebar.selectbox("Forecast model", ["Gradient boosting", "Ridge regression"])

s_sel = series_for(df, station)
if len(s_sel) < E.TEST_DAYS + 120:
    st.error(f"Need at least {E.TEST_DAYS + 120} days of daily data (found {len(s_sel)}).")
    st.stop()
test_start = E.test_start_date(s_sel)
lo = (test_start + pd.Timedelta(days=7)).date()
hi = (s_sel.index.max() - pd.Timedelta(days=E.HORIZON)).date()
origin_d = st.sidebar.slider("'Today' (forecast origin, inside held-out test period)",
                             min_value=lo, max_value=hi, value=lo + dt.timedelta(days=30),
                             format="DD MMM YYYY")
origin = pd.Timestamp(origin_d)

st.sidebar.markdown("---")
ind_share = st.sidebar.slider("Assumed industrial baseline share (%) - used only if no industrial data",
                              5, 40, 15) / 100
st.sidebar.subheader("🎛️ Scenario simulator")
st.sidebar.caption("Toggle interventions - map, forecast and comparison update.")
active = {}
eff_inputs = {}
with st.sidebar.expander("Edit effectiveness assumptions"):
    for name, cat, default in E.ACTIONS:
        eff_inputs[cat] = st.slider(f"{name} - cut in {cat} share (%)", 0, 100, int(default * 100)) / 100
for name, cat, default in E.ACTIONS:
    if st.sidebar.toggle(name, value=False):
        active[cat] = eff_inputs[cat]

# ----------------------------------------------------------------------------
# Compute everything once per run
# ----------------------------------------------------------------------------
lr = E.fit_attribution(df, origin, ind_share)


def bundle(name):
    s = series_for(df, name)
    m = model_for(s, f"{data_id}|{name}", kind)
    fc = E.forecast(m, s, origin)
    return s, m, fc, E.apply_actions(fc, lr, s, active)


s, model, fc, fc_scn = bundle(station)
bt = backtest_for(s, model, f"{data_id}|{station}", kind)
sc = E.score(bt)
obs7 = s.loc[origin - pd.Timedelta(days=6):origin, "pm25"].mean()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("OmniTwin · Urban Environmental Digital Twin")
st.markdown(
    f"<div style='padding:8px 12px;border-radius:8px;background:#f4f6f8'>"
    f"<b>Legend:</b> <span style='color:{BLUE}'>■ Observed data (solid blue)</span> &nbsp; "
    f"<span style='color:{ORANGE}'>■ Modeled scenario (dashed orange)</span> &nbsp; "
    f"<span style='color:{GREY}'>■ Actual historical, for validation (dotted grey)</span></div>",
    unsafe_allow_html=True)
if is_demo:
    st.warning("Demo mode: using a **synthetic** Pune-like dataset (seasonal patterns, 5 neighbourhoods). "
               "Upload a real CPCB/Kaggle CSV in the sidebar to run on real data.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("OBSERVED · PM2.5, 7d to 'Today'", f"{obs7:.0f} µg/m³")
k2.metric("MODELED SCENARIO · baseline, next 7d", f"{fc.mean():.0f} µg/m³")
k3.metric("MODELED SCENARIO · with your actions", f"{fc_scn.mean():.0f} µg/m³",
          f"{fc_scn.mean() - fc.mean():+.1f} µg/m³" if active else None, delta_color="inverse")
k4.metric("Back-test MAE (held-out 90d)", f"{sc.loc['OmniTwin model', 'MAE (ug/m3)']:.1f} µg/m³",
          f"{(1 - sc.loc['OmniTwin model', 'MAE (ug/m3)'] / sc.loc['Baseline: persistence', 'MAE (ug/m3)']) * 100:.0f}% better than persistence",
          delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Hotspot map", "📈 Forecast & validation", "🥧 Source attribution",
     "🎛️ Scenario comparison", "📋 Method & assumptions"])

# ----------------------------------------------------------------------------
# Tab 1: map
# ----------------------------------------------------------------------------
with tab1:
    mode = st.radio("Map layer", ["Observed data (7 days up to 'Today')",
                                  "Modeled scenario (next 7 days, selected actions applied)"],
                    horizontal=True)
    observed_mode = mode.startswith("Observed")
    coords = df.groupby("station")[["lat", "lon"]].first()
    rows = []
    for name in stations:
        ss, _, f_i, scn_i = bundle(name)
        rows.append({"Neighbourhood": name, "lat": coords.loc[name, "lat"], "lon": coords.loc[name, "lon"],
                     "Observed 7d (µg/m³)": ss.loc[origin - pd.Timedelta(days=6):origin, "pm25"].mean(),
                     "Modeled baseline (µg/m³)": f_i.mean(), "Modeled + actions (µg/m³)": scn_i.mean()})
    tbl = pd.DataFrame(rows)
    col = "Observed 7d (µg/m³)" if observed_mode else "Modeled + actions (µg/m³)"

    m = folium.Map(location=[tbl.lat.mean(), tbl.lon.mean()], zoom_start=11, tiles="cartodbpositron")
    HeatMap([[r.lat, r.lon, r[col] / 150] for _, r in tbl.iterrows()], radius=55, blur=40,
            min_opacity=0.25).add_to(m)
    for _, r in tbl.iterrows():
        c, cat = E.pm_category(r[col])
        folium.CircleMarker([r.lat, r.lon], radius=9 + r[col] / 8, color="#222", weight=1,
                            fill=True, fill_color=c, fill_opacity=0.85,
                            tooltip=f"{r['Neighbourhood']}: {r[col]:.0f} µg/m³ ({cat})").add_to(m)
    tag, tcol = ("OBSERVED DATA", BLUE) if observed_mode else ("MODELED SCENARIO", ORANGE)
    legend = f"""
    <div style="position:fixed;top:12px;left:60px;z-index:9999;background:{tcol};color:#fff;
         padding:10px 20px;border-radius:8px;font:800 22px sans-serif;letter-spacing:1px">{tag}</div>
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:8px 12px;
         border:1px solid #bbb;border-radius:6px;font:12px sans-serif;line-height:1.6">
      <b>PM2.5 (CPCB)</b><br>
      <span style="color:#2e9e4f">●</span> 0-30 Good<br><span style="color:#8bc34a">●</span> 31-60 Satisfactory<br>
      <span style="color:#f2c500">●</span> 61-90 Moderate<br><span style="color:#ff8c00">●</span> 91-120 Poor<br>
      <span style="color:#e53935">●</span> 121-250 Very poor</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    badge("obs" if observed_mode else "mod")
    st_folium(m, height=520, use_container_width=True, returned_objects=[])
    show = tbl.drop(columns=["lat", "lon"]).set_index("Neighbourhood").round(1)
    st.dataframe(show)
    st.caption("Hotspot = highest-PM2.5 neighbourhood in the selected layer. "
               "Toggle actions in the sidebar to see the modeled hotspots shift.")

# ----------------------------------------------------------------------------
# Tab 2: forecast + validation
# ----------------------------------------------------------------------------
with tab2:
    st.subheader(f"7-day PM2.5 forecast · {station}")
    lb1, lb2, lb3 = st.columns(3)
    with lb1:
        badge("obs")
    with lb2:
        badge("mod")
    with lb3:
        badge("val")
    hist = s.loc[origin - pd.Timedelta(days=30):origin, "pm25"]
    act = s.loc[fc.index, "pm25"]
    last = hist.iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Observed (up to 'Today')",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=[origin] + list(fc.index), y=[last] + list(fc.values),
                             name="Modeled scenario: 7-day forecast (baseline)",
                             line=dict(color=ORANGE, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=[origin] + list(act.index), y=[last] + list(act.values),
                             name="Actual historical (validation)",
                             line=dict(color=GREY, width=3, dash="dot")))
    if active:
        fig.add_trace(go.Scatter(x=[origin] + list(fc_scn.index), y=[last] + list(fc_scn.values),
                                 name="Modeled scenario + your selected actions",
                                 line=dict(color=GREEN, width=4, dash="dash")))
    fig.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper",
                  line=dict(color="#999", width=1))
    fig.add_annotation(x=origin, y=1, yref="paper", text="'Today'", showarrow=False, yanchor="bottom")
    fig.add_hline(y=E.NAAQS_PM25, line_dash="dash", line_color=RED, line_width=1,
                  annotation_text="India 24h standard: 60 µg/m³")
    fig.update_layout(height=430, xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    st.plotly_chart(fig)
    st.caption("The model was trained only on data **before** the held-out test period. The 7 days "
               "after 'Today' were never seen in training; the dotted line is what actually happened. "
               "Weather/traffic drivers for those days are taken from the historical record "
               "(i.e. assumes a good weather forecast).")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Back-test over the whole held-out period (rolling 7-day forecasts)**")
        st.dataframe(sc.round(2))
        h = bt.assign(err=(bt.pred - bt.actual).abs()).groupby("horizon")["err"].mean()
        fig_h = go.Figure(go.Bar(x=[f"+{i}d" for i in h.index], y=h.values, marker_color=ORANGE))
        fig_h.update_layout(height=250, yaxis_title="MAE (µg/m³)", margin=dict(t=10),
                            title=dict(text="Error by forecast horizon", font=dict(size=13)))
        st.plotly_chart(fig_h)
    with c2:
        bts = bt.sort_values("date")
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=bts.date, y=bts.actual, name="Actual historical",
                                   line=dict(color=GREY, width=2, dash="dot")))
        fig_b.add_trace(go.Scatter(x=bts.date, y=bts.pred, name="Modeled forecast",
                                   line=dict(color=ORANGE, width=2, dash="dash")))
        fig_b.update_layout(height=430, yaxis_title="PM2.5 (µg/m³)", margin=dict(t=30),
                            legend=dict(orientation="h", y=-0.2),
                            title=dict(text="Held-out test period: forecast vs actual", font=dict(size=13)))
        st.plotly_chart(fig_b)

# ----------------------------------------------------------------------------
# Tab 3: attribution
# ----------------------------------------------------------------------------
with tab3:
    st.subheader(f"Source attribution · {station}")
    st.info(ASSUMPTION)
    badge("mod")
    if lr.industrial_assumed:
        st.warning(f"No usable industrial variation in the data, so following the stated assumption: "
                   f"industrial = constant baseline of {ind_share:.0%} of mean PM2.5 (adjust in sidebar).")
    st.caption("Attribution is a **modeled estimate** derived from observed data via regression - "
               "it is not a measured emissions inventory.")
    recent = s.loc[origin - pd.Timedelta(days=29):origin]
    c_recent = E.contributions(lr, recent).mean()
    a, b = st.columns(2)
    with a:
        fig_p = go.Figure(go.Pie(labels=list(c_recent.index), values=c_recent.values, hole=0.45,
                                 marker=dict(colors=[SRC_COLORS[k] for k in c_recent.index]),
                                 textinfo="label+percent"))
        fig_p.update_layout(height=380, showlegend=False, margin=dict(t=40),
                            title=dict(text="Estimated contribution, last 30 days to 'Today'", font=dict(size=13)))
        st.plotly_chart(fig_p)
    with b:
        c_fc = E.contributions(lr, s.loc[fc.index])
        scale = fc / c_fc.sum(axis=1)
        c_fc = c_fc.mul(scale, axis=0)
        fig_s = go.Figure()
        for k in c_fc.columns:
            fig_s.add_trace(go.Bar(x=[d.strftime("%d %b") for d in c_fc.index], y=c_fc[k], name=k,
                                   marker_color=SRC_COLORS[k]))
        fig_s.update_layout(barmode="stack", height=380, yaxis_title="PM2.5 (µg/m³)", margin=dict(t=40),
                            legend=dict(orientation="h", y=-0.2),
                            title=dict(text="Modeled forecast split by source, next 7 days", font=dict(size=13)))
        st.plotly_chart(fig_s)

    per = []
    for name in stations:
        ss = series_for(df, name).loc[origin - pd.Timedelta(days=29):origin]
        cc = E.contributions(lr, ss).mean()
        per.append((cc / cc.sum() * 100).rename(name))
    per = pd.DataFrame(per)
    fig_n = go.Figure()
    for k in per.columns:
        fig_n.add_trace(go.Bar(y=per.index, x=per[k], name=k, orientation="h", marker_color=SRC_COLORS[k]))
    fig_n.update_layout(barmode="stack", height=300, xaxis_title="Share of PM2.5 (%)",
                        legend=dict(orientation="h", y=-0.35), margin=dict(t=40),
                        title=dict(text="Why each neighbourhood differs (modeled)", font=dict(size=13)))
    st.plotly_chart(fig_n)
    st.caption("Fitted coefficients (µg/m³ per unit): traffic "
               f"{lr.coef_[0]:.2f}, industrial {lr.coef_[1]:.2f}, dust {lr.coef_[2]:.2f}. "
               "'Background/regional' = PM2.5 not explained by local sources.")

# ----------------------------------------------------------------------------
# Tab 4: scenarios
# ----------------------------------------------------------------------------
with tab4:
    st.subheader(f"Compare interventions · next 7 days · {station}")
    badge("mod")
    st.info(ASSUMPTION)
    cost = {"Vehicular": "Medium", "Industrial": "High", "Dust/Weather": "Low"}
    scen = {"Baseline (no action)": fc}
    for name, cat, _ in E.ACTIONS:
        scen[name] = E.apply_actions(fc, lr, s, {cat: eff_inputs[cat]})
    scen["All three combined"] = E.apply_actions(fc, lr, s, {c: eff_inputs[c] for _, c, _ in E.ACTIONS})
    if active:
        scen["▶ Your selection (sidebar)"] = fc_scn
    rows = []
    for name, ser in scen.items():
        cat = next((c for n, c, _ in E.ACTIONS if n == name), None)
        rows.append({"Scenario": name, "Mean PM2.5 (µg/m³)": ser.mean(),
                     "Reduction (µg/m³)": fc.mean() - ser.mean(),
                     "Reduction (%)": (1 - ser.mean() / fc.mean()) * 100,
                     "Days above 60 µg/m³ (of 7)": int((ser > E.NAAQS_PM25).sum()),
                     "Economic/social cost (qualitative)": cost.get(cat, "-")})
    res = pd.DataFrame(rows).set_index("Scenario")

    if not active:
        st.caption("Tip: switch on actions in the sidebar to add a '▶ Your selection' scenario.")
    cA, cB = st.columns([1, 1])
    with cA:
        colors = [ORANGE if i == 0 else (GREEN if "combined" in n or "▶" in n else BLUE)
                  for i, n in enumerate(res.index)]
        fig_c = go.Figure(go.Bar(x=res["Mean PM2.5 (µg/m³)"], y=res.index, orientation="h",
                                 marker_color=colors, text=res["Mean PM2.5 (µg/m³)"].round(1)))
        fig_c.add_vline(x=E.NAAQS_PM25, line_dash="dash", line_color=RED)
        fig_c.update_layout(height=340, xaxis_title="Mean modeled PM2.5 (µg/m³)",
                            yaxis=dict(autorange="reversed"), margin=dict(t=20))
        st.plotly_chart(fig_c)
    with cB:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Baseline",
                                   line=dict(color=ORANGE, width=3, dash="dash")))
        for name, ser in scen.items():
            if name.startswith("Action") or "combined" in name or "▶" in name:
                fig_d.add_trace(go.Scatter(x=ser.index, y=ser.values, name=name.replace("Action ", "A"),
                                           line=dict(width=2, dash="dash")))
        fig_d.add_hline(y=E.NAAQS_PM25, line_dash="dot", line_color=RED)
        fig_d.update_layout(height=340, yaxis_title="PM2.5 (µg/m³)", margin=dict(t=20),
                            legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_d)
    st.dataframe(res.round(1))
    best = res.drop(index=["Baseline (no action)", "All three combined"] +
                    (["▶ Your selection (sidebar)"] if active else []))["Reduction (µg/m³)"].idxmax()
    st.success(f"Largest single-action effect in this window: **{best}**. "
               "Effects depend on the local source mix - see the attribution tab.")
    st.caption("Reductions are modeled: each day's forecast is split by source share, and the targeted "
               "share is cut by the assumed % (editable in the sidebar). Costs are qualitative judgements, "
               "not computed.")

# ----------------------------------------------------------------------------
# Tab 5: method
# ----------------------------------------------------------------------------
with tab5:
    st.markdown(f"""
### How it works
| Stage | Label in UI | Method |
|---|---|---|
| Hotspot map | **Observed** / **Modeled** toggle | 7-day mean PM2.5 per neighbourhood; modeled layer = 7-day forecast with actions applied |
| Forecast | Observed → Modeled → Actual | Lagged PM2.5 + weather + traffic + industrial index + seasonality; trained on data **before** the last {E.TEST_DAYS} days; recursive 7-day forecast |
| Validation | Validation | Rolling 7-day back-test on the held-out {E.TEST_DAYS} days vs a persistence baseline |
| Attribution | Modeled | Pooled non-negative regression: PM2.5 ~ traffic + industrial + dust (+ background intercept) |
| Scenarios | Modeled | Daily forecast split by attribution share; targeted share cut by action effectiveness |

### Stated assumptions
- {ASSUMPTION.replace('Assumption: ', '')}
- Action effectiveness defaults: odd-even −30% vehicular, industry halt −80% industrial, sprinklers −40% dust (editable).
- Actions are applied independently to each source; no rebound or displacement effects are modeled.
- Future weather/traffic for the 7 forecast days come from the historical record (assumes a perfect weather forecast). In production this would be an IMD/weather-API forecast.

### Limitations (honest)
- Attribution is statistical (regression), not a chemical-transport or emissions-inventory model; correlation ≠ causation.
- The 'Dust/Weather' share also absorbs some meteorology effects such as season and dilution.
- {'This run uses SYNTHETIC demo data. Numbers illustrate the method, not real Pune air quality.' if is_demo else 'Running on your uploaded dataset.'}

### Roadmap
Live CPCB/OpenAQ + traffic API feeds · sensor-level interpolation across the city grid · chemical-transport or satellite (AOD) priors for attribution · cost-benefit optimizer over combinations of actions.
""")
