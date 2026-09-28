"""
OmniEarthers — Urban Environmental Digital Twin
HackMatrix 24-hour MVP.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import datetime as dt
import io

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import engine as E


# ---------------------------------------------------------------------
# Theme / constants
# ---------------------------------------------------------------------

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREY = "#555555"
GREEN = "#2e9e4f"
RED = "#d62728"
DARK = "#15202b"

SRC_COLORS = {
    "Vehicular": "#1f77b4",
    "Industrial": "#7f7f7f",
    "Dust/Weather": "#e0b040",
    "Background/regional": "#c9d6df",
}

ASSUMPTION = (
    "Traffic scales with activity/rush-hour patterns; industrial baseline "
    "is treated as approximately constant where historical variation is "
    "insufficient; dust correlates with wind on dry days."
)

st.set_page_config(
    page_title="OmniEarthers — Urban Environmental Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------

def badge(kind: str):
    labels = {
        "obs": ("OBSERVED DATA", BLUE),
        "mod": ("MODELED SCENARIO", ORANGE),
        "val": ("HISTORICAL VALIDATION", GREY),
        "assumed": ("ASSUMED / DERIVED", "#8a6d1d"),
    }

    label, color = labels[kind]

    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:8px 14px;
            border-radius:8px;
            font-size:0.88rem;
            font-weight:800;
            letter-spacing:.06em;
            text-align:center;
            margin:5px 0 10px 0;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def metric_card(title, value, help_text=""):
    st.metric(title, value, help=help_text or None)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Loading environmental data...")
def load_data(file_bytes):
    if file_bytes is None:
        return E.make_synthetic()

    return pd.read_csv(
        io.BytesIO(file_bytes),
        parse_dates=["date"],
    )


@st.cache_data(show_spinner=False)
def series_for(df, station):
    return E.get_series(df, station)


@st.cache_resource(show_spinner="Training forecast model...")
def model_for(_s, key, kind):
    return E.fit_forecaster(_s, kind)


@st.cache_data(show_spinner="Running historical back-test...")
def backtest_for(_s, _model, key):
    return E.backtest(_model, _s)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.markdown(
    """
    <h1 style="margin-bottom:0;">🌍 OmniEarthers</h1>
    <p style="margin-top:0;color:#777;">
    Urban Environmental Digital Twin
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption(
    "HackMatrix MVP · Pune Urban Region"
)

st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader(
    "Use historical CSV",
    type=["csv"],
    help=(
        "Upload a historical dataset. The prototype requires Date, PM2.5, "
        "Traffic_Index and Wind_Speed. Other fields are optional."
    ),
)

file_bytes = uploaded.getvalue() if uploaded else None
is_demo = file_bytes is None

with st.sidebar.expander("Study-area settings"):
    area_name = st.text_input(
        "Area name",
        "Pune Urban Region",
    )
    area_lat = st.number_input(
        "Latitude",
        value=18.5204,
        format="%.4f",
    )
    area_lon = st.number_input(
        "Longitude",
        value=73.8567,
        format="%.4f",
    )

raw = load_data(file_bytes)

try:
    df, load_notes = E.prepare_df(
        raw,
        area_lat,
        area_lon,
        area_name,
    )
except ValueError as err:
    st.error(str(err))
    st.stop()

if load_notes:
    st.sidebar.info(
        "Data preparation notes:\n\n"
        + "\n\n".join("• " + n for n in load_notes)
    )

data_id = (
    "synthetic-demo"
    if file_bytes is None
    else str(hash(file_bytes))
)

with st.sidebar.expander("CSV schema"):
    st.markdown(
        """
        **Required**
        - `date`
        - `PM2.5`
        - `Traffic_Index`
        - `Wind_Speed`

        **Recommended**
        - `station`
        - `lat`, `lon`
        - `temp`
        - `humidity`
        - `rain`
        - `industrial_index`
        """
    )

    st.download_button(
        "Download demo dataset",
        E.make_synthetic().to_csv(index=False),
        "omniearthers_demo.csv",
        "text/csv",
    )

stations = list(df.groupby("station").size().index)

station_options = (
    ["City average"] + stations
    if len(stations) > 1
    else stations
)

station = st.sidebar.selectbox(
    "Twin view",
    station_options,
)

model_kind = st.sidebar.selectbox(
    "Forecast model",
    ["Gradient boosting", "Ridge regression"],
)

s_selected = series_for(df, station)

minimum_required = E.TEST_DAYS + 30

if len(s_selected) < minimum_required:
    st.error(
        f"Need at least {minimum_required} daily observations for the "
        f"historical replay and validation. Found {len(s_selected)}."
    )
    st.stop()

test_start = E.test_start_date(s_selected)

# Give enough room for a seven-day forecast.
min_replay = test_start + pd.Timedelta(days=7)
max_replay = s_selected.index.max() - pd.Timedelta(days=E.HORIZON)

if min_replay > max_replay:
    st.error("Not enough data to create a valid historical replay window.")
    st.stop()

default_replay = min(
    min_replay + pd.Timedelta(days=30),
    max_replay,
)

replay_date = st.sidebar.date_input(
    "Historical replay date",
    value=default_replay.date(),
    min_value=min_replay.date(),
    max_value=max_replay.date(),
)

origin = pd.Timestamp(replay_date)

st.sidebar.markdown("---")

st.sidebar.subheader("🎛️ What-if simulator")

industrial_share = (
    st.sidebar.slider(
        "Industrial baseline assumption (%)",
        5,
        40,
        15,
        help=(
            "Used only when industrial activity does not vary enough "
            "to estimate its coefficient statistically."
        ),
    )
    / 100
)

effectiveness = {}

with st.sidebar.expander("Edit intervention assumptions"):
    for action_name, category, default in E.ACTIONS:
        effectiveness[category] = (
            st.slider(
                f"{action_name} — reduction (%)",
                0,
                100,
                int(default * 100),
            )
            / 100
        )

active = {}

for action_name, category, default in E.ACTIONS:
    if st.sidebar.checkbox(action_name, value=False):
        active[category] = effectiveness[category]


# ---------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------

attribution = E.fit_attribution(
    df,
    origin,
    industrial_share,
)


def bundle(name):
    series = series_for(df, name)
    model = model_for(
        series,
        f"{data_id}|{name}",
        model_kind,
    )

    forecast = E.forecast(
        model,
        series,
        origin,
    )

    scenario = E.apply_actions(
        forecast,
        attribution,
        series,
        active,
    )

    return series, model, forecast, scenario


series, model, forecast, scenario = bundle(station)

backtest = backtest_for(
    series,
    model,
    f"{data_id}|{station}|{model_kind}",
)

scores = E.score(backtest)

observed_7d = series.loc[
    origin - pd.Timedelta(days=6):origin,
    "pm25",
].mean()

baseline_mean = forecast.mean()
scenario_mean = scenario.mean()

model_mae = scores.loc[
    "OmniEarthers model",
    "MAE (µg/m³)",
]

baseline_mae = scores.loc[
    "Persistence baseline",
    "MAE (µg/m³)",
]

improvement = (
    (1 - model_mae / baseline_mae) * 100
    if baseline_mae > 0
    else 0
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    # 🌍 OmniEarthers
    ### Urban Environmental Digital Twin — Pune

    **Observe → Understand → Forecast → Simulate → Validate**
    """
)

st.info(
    "OmniEarthers turns historical urban-environment data into an "
    "interactive decision-support model. The MVP forecasts PM2.5, "
    "estimates likely source contributions, maps hotspots and simulates "
    "potential interventions."
)

if is_demo:
    st.warning(
        "DEMO DATA: This run uses a synthetic Pune-like dataset. "
        "Synthetic values are used to demonstrate the workflow and are "
        "not presented as real Pune measurements. Upload a real historical "
        "dataset before the final competition demo."
    )
else:
    st.success(
        "DATA MODE: Running on the uploaded historical dataset."
    )


# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card(
        "Observed PM2.5",
        f"{observed_7d:.1f} µg/m³",
        "Mean of the seven observed days ending on the replay date.",
    )

with k2:
    metric_card(
        "7-day modeled baseline",
        f"{baseline_mean:.1f} µg/m³",
        "Model forecast with no intervention.",
    )

with k3:
    metric_card(
        "7-day modeled scenario",
        f"{scenario_mean:.1f} µg/m³",
        "Forecast after selected intervention assumptions.",
    )

with k4:
    metric_card(
        "Historical test MAE",
        f"{model_mae:.1f} µg/m³",
        "Rolling 7-day error on the held-out historical test period.",
    )

st.markdown("---")

st.caption(
    f"Historical replay date: **{origin.strftime('%d %b %Y')}** · "
    f"Forecast horizon: **7 days** · "
    f"Validation period: **90 days**"
)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tabs = st.tabs([
    "🗺️ City Twin",
    "📈 Forecast & Validation",
    "🧪 Pollution Sources",
    "🎛️ What-If Simulator",
    "🔬 Methodology",
])


# ---------------------------------------------------------------------
# TAB 1 — CITY TWIN
# ---------------------------------------------------------------------

with tabs[0]:

    section_title(
        "Pollution Hotspots",
        "Toggle between measured historical conditions and modeled intervention outcomes.",
    )

    map_mode = st.radio(
        "Map layer",
        [
            "Observed data — historical",
            "Modeled scenario — next 7 days",
        ],
        horizontal=True,
    )

    observed_mode = map_mode.startswith("Observed")

    coords = (
        df.groupby("station")[["lat", "lon"]]
        .first()
    )

    rows = []

    for name in stations:
        ss, _, fc_i, scn_i = bundle(name)

        rows.append({
            "Neighbourhood": name,
            "lat": coords.loc[name, "lat"],
            "lon": coords.loc[name, "lon"],
            "Observed PM2.5": ss.loc[
                origin - pd.Timedelta(days=6):origin,
                "pm25",
            ].mean(),
            "Modeled baseline": fc_i.mean(),
            "Modeled + actions": scn_i.mean(),
        })

    hotspot_table = pd.DataFrame(rows)

    map_column = (
        "Observed PM2.5"
        if observed_mode
        else "Modeled + actions"
    )

    m = folium.Map(
        location=[
            hotspot_table.lat.mean(),
            hotspot_table.lon.mean(),
        ],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    HeatMap(
        [
            [
                row.lat,
                row.lon,
                min(float(row[map_column]) / 150, 1.0),
            ]
            for _, row in hotspot_table.iterrows()
        ],
        radius=55,
        blur=40,
        min_opacity=0.25,
    ).add_to(m)

    for _, row in hotspot_table.iterrows():
        category_color, category = E.pm_category(
            float(row[map_column])
        )

        folium.CircleMarker(
            [
                row.lat,
                row.lon,
            ],
            radius=max(8, 9 + row[map_column] / 8),
            color="#222222",
            weight=1,
            fill=True,
            fill_color=category_color,
            fill_opacity=0.85,
            tooltip=(
                f"{row['Neighbourhood']}: "
                f"{row[map_column]:.1f} µg/m³ — {category}"
            ),
        ).add_to(m)

    label = (
        "OBSERVED DATA"
        if observed_mode
        else "MODELED SCENARIO"
    )

    label_color = BLUE if observed_mode else ORANGE

    legend = f"""
    <div style="
        position:fixed;
        top:12px;
        left:60px;
        z-index:9999;
        background:{label_color};
        color:white;
        padding:9px 18px;
        border-radius:8px;
        font:800 18px sans-serif;">
        {label}
    </div>

    <div style="
        position:fixed;
        bottom:24px;
        left:24px;
        z-index:9999;
        background:white;
        padding:9px 12px;
        border:1px solid #bbb;
        border-radius:6px;
        font:12px sans-serif;
        line-height:1.6;">
        <b>PM2.5 category</b><br>
        <span style="color:#2e9e4f">●</span> 0–30 Good<br>
        <span style="color:#8bc34a">●</span> 31–60 Satisfactory<br>
        <span style="color:#f2c500">●</span> 61–90 Moderate<br>
        <span style="color:#ff8c00">●</span> 91–120 Poor<br>
        <span style="color:#e53935">●</span> 121–250 Very Poor
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend)
    )

    badge("obs" if observed_mode else "mod")

    st_folium(
        m,
        height=540,
        use_container_width=True,
        returned_objects=[],
    )

    display_table = (
        hotspot_table
        .drop(columns=["lat", "lon"])
        .set_index("Neighbourhood")
        .round(1)
    )

    st.dataframe(
        display_table,
        use_container_width=True,
    )

    st.caption(
        "Observed layer = historical measurements. "
        "Modeled layer = seven-day intervention scenario. "
        "Modeled values are not observations."
    )


# ---------------------------------------------------------------------
# TAB 2 — FORECAST & VALIDATION
# ---------------------------------------------------------------------

with tabs[1]:

    section_title(
        "Seven-Day PM2.5 Forecast",
        f"Historical replay from {origin.strftime('%d %B %Y')} for {station}.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        badge("obs")

    with c2:
        badge("mod")

    with c3:
        badge("val")

    history = series.loc[
        origin - pd.Timedelta(days=30):origin,
        "pm25",
    ]

    actual = series.loc[
        forecast.index,
        "pm25",
    ]

    last_observed = history.iloc[-1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            name="Observed historical data",
            line=dict(color=BLUE, width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[origin] + list(forecast.index),
            y=[last_observed] + list(forecast.values),
            name="Modeled baseline forecast",
            line=dict(
                color=ORANGE,
                width=3,
                dash="dash",
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[origin] + list(actual.index),
            y=[last_observed] + list(actual.values),
            name="Actual historical validation",
            line=dict(
                color=GREY,
                width=3,
                dash="dot",
            ),
        )
    )

    if active:
        fig.add_trace(
            go.Scatter(
                x=[origin] + list(scenario.index),
                y=[last_observed] + list(scenario.values),
                name="Modeled scenario + selected actions",
                line=dict(
                    color=GREEN,
                    width=4,
                    dash="dash",
                ),
            )
        )

    fig.add_shape(
        type="line",
        x0=origin,
        x1=origin,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#999", width=1),
    )

    fig.add_annotation(
        x=origin,
        y=1,
        yref="paper",
        text="Historical replay",
        showarrow=False,
        yanchor="bottom",
    )

    fig.add_hline(
        y=E.NAAQS_PM25,
        line_dash="dash",
        line_color=RED,
        annotation_text="India 24-h PM2.5 standard: 60 µg/m³",
    )

    fig.update_layout(
        height=460,
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        legend=dict(
            orientation="h",
            y=-0.22,
        ),
        margin=dict(t=30),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "Validation design: the forecasting model is trained only on "
        "data before the held-out test period. The seven historical days "
        "shown in grey were not used for model training."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Historical back-test")

        score_display = scores.copy().round(2)

        st.dataframe(
            score_display,
            use_container_width=True,
        )

        st.metric(
            "Model MAE vs persistence",
            f"{improvement:.1f}%",
            help="Positive value means lower MAE than the persistence baseline.",
        )

    with right:
        bt_sorted = backtest.sort_values("date")

        fig_bt = go.Figure()

        fig_bt.add_trace(
            go.Scatter(
                x=bt_sorted.date,
                y=bt_sorted.actual,
                name="Actual historical",
                line=dict(
                    color=GREY,
                    width=2,
                    dash="dot",
                ),
            )
        )

        fig_bt.add_trace(
            go.Scatter(
                x=bt_sorted.date,
                y=bt_sorted.pred,
                name="Modeled forecast",
                line=dict(
                    color=ORANGE,
                    width=2,
                    dash="dash",
                ),
            )
        )

        fig_bt.update_layout(
            height=400,
            title="Held-out test period: forecast vs actual",
            yaxis_title="PM2.5 (µg/m³)",
            legend=dict(
                orientation="h",
                y=-0.2,
            ),
        )

        st.plotly_chart(
            fig_bt,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# TAB 3 — POLLUTION SOURCES
# ---------------------------------------------------------------------

with tabs[2]:

    section_title(
        "Likely Pollution Sources",
        "Interpretable statistical attribution based on historical relationships.",
    )

    badge("mod")

    st.info(
        ASSUMPTION
    )

    if attribution.industrial_assumed:
        st.warning(
            f"Industrial activity did not contain enough usable variation "
            f"for a stable coefficient. The model therefore uses an assumed "
            f"industrial baseline share of {industrial_share:.0%} of mean PM2.5."
        )

    st.caption(
        "Important: attribution is a modeled estimate, not a measured "
        "emissions inventory and not proof of causation."
    )

    recent = series.loc[
        origin - pd.Timedelta(days=29):origin
    ]

    recent_contributions = E.contributions(
        attribution,
        recent,
    ).mean()

    a, b = st.columns(2)

    with a:

        fig_pie = go.Figure(
            go.Pie(
                labels=list(recent_contributions.index),
                values=recent_contributions.values,
                hole=0.45,
                marker=dict(
                    colors=[
                        SRC_COLORS[k]
                        for k in recent_contributions.index
                    ]
                ),
                textinfo="label+percent",
            )
        )

        fig_pie.update_layout(
            height=390,
            title="Estimated contribution — recent 30 days",
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True,
        )

    with b:

        forecast_contributions = E.contributions(
            attribution,
            series.loc[forecast.index],
        )

        scale = (
            forecast
            / forecast_contributions.sum(axis=1).replace(0, np.nan)
        )

        forecast_contributions = (
            forecast_contributions
            .mul(scale, axis=0)
            .fillna(0)
        )

        fig_stack = go.Figure()

        for source in forecast_contributions.columns:
            fig_stack.add_trace(
                go.Bar(
                    x=[
                        d.strftime("%d %b")
                        for d in forecast_contributions.index
                    ],
                    y=forecast_contributions[source],
                    name=source,
                    marker_color=SRC_COLORS[source],
                )
            )

        fig_stack.update_layout(
            barmode="stack",
            height=390,
            title="Modeled source split — next 7 days",
            yaxis_title="PM2.5 (µg/m³)",
            legend=dict(
                orientation="h",
                y=-0.22,
            ),
        )

        st.plotly_chart(
            fig_stack,
            use_container_width=True,
        )

    st.subheader("Source differences across neighbourhoods")

    neighbourhood_rows = []

    for name in stations:
        ss = series_for(df, name).loc[
            origin - pd.Timedelta(days=29):origin
        ]

        contributions = E.contributions(
            attribution,
            ss,
        ).mean()

        shares = (
            contributions
            / max(contributions.sum(), 1e-9)
            * 100
        )

        neighbourhood_rows.append(
            shares.rename(name)
        )

    neighbourhood_df = pd.DataFrame(
        neighbourhood_rows
    )

    fig_neighbourhood = go.Figure()

    for source in neighbourhood_df.columns:
        fig_neighbourhood.add_trace(
            go.Bar(
                y=neighbourhood_df.index,
                x=neighbourhood_df[source],
                name=source,
                orientation="h",
                marker_color=SRC_COLORS[source],
            )
        )

    fig_neighbourhood.update_layout(
        barmode="stack",
        height=330,
        title="Modeled source shares by neighbourhood",
        xaxis_title="Estimated share of PM2.5 (%)",
        legend=dict(
            orientation="h",
            y=-0.3,
        ),
    )

    st.plotly_chart(
        fig_neighbourhood,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# TAB 4 — WHAT-IF SIMULATOR
# ---------------------------------------------------------------------

with tabs[3]:

    section_title(
        "What-If Simulator",
        "Test interventions on the virtual city before considering real-world action.",
    )

    badge("mod")

    st.info(
        "Scenario results are modeled outputs. They represent the effect of "
        "the stated intervention assumptions on the forecast and are not "
        "observed measurements."
    )

    scenario_sets = {
        "Baseline — no intervention": {},
    }

    for action_name, category, _ in E.ACTIONS:
        scenario_sets[action_name] = {
            category: effectiveness[category]
        }

    scenario_sets["Combined — all three"] = {
        category: effectiveness[category]
        for _, category, _ in E.ACTIONS
    }

    if active:
        scenario_sets["Current sidebar selection"] = active

    scenario_rows = []

    for name, cuts in scenario_sets.items():

        values = E.apply_actions(
            forecast,
            attribution,
            series,
            cuts,
        )

        scenario_rows.append({
            "Scenario": name,
            "Mean PM2.5 (µg/m³)": values.mean(),
            "Reduction (µg/m³)": baseline_mean - values.mean(),
            "Reduction (%)": (
                (1 - values.mean() / baseline_mean) * 100
                if baseline_mean > 0
                else 0
            ),
            "Days > 60 µg/m³": int(
                (values > E.NAAQS_PM25).sum()
            ),
        })

    scenario_df = pd.DataFrame(
        scenario_rows
    ).set_index("Scenario")

    left, right = st.columns(2)

    with left:

        fig_scenario = go.Figure()

        fig_scenario.add_trace(
            go.Bar(
                x=scenario_df["Mean PM2.5 (µg/m³)"],
                y=scenario_df.index,
                orientation="h",
                text=scenario_df["Mean PM2.5 (µg/m³)"].round(1),
                textposition="auto",
            )
        )

        fig_scenario.add_vline(
            x=E.NAAQS_PM25,
            line_dash="dash",
            line_color=RED,
            annotation_text="60 µg/m³",
        )

        fig_scenario.update_layout(
            height=400,
            title="Scenario comparison",
            xaxis_title="Mean modeled PM2.5 (µg/m³)",
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(
            fig_scenario,
            use_container_width=True,
        )

    with right:

        fig_lines = go.Figure()

        for name, cuts in scenario_sets.items():

            values = E.apply_actions(
                forecast,
                attribution,
                series,
                cuts,
            )

            line_style = (
                "dash"
                if name.startswith("Baseline")
                else "solid"
            )

            fig_lines.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values,
                    name=name,
                    line=dict(
                        width=2.5,
                        dash=line_style,
                    ),
                )
            )

        fig_lines.add_hline(
            y=E.NAAQS_PM25,
            line_dash="dot",
            line_color=RED,
            annotation_text="60 µg/m³",
        )

        fig_lines.update_layout(
            height=400,
            title="Projected PM2.5 trajectories",
            yaxis_title="PM2.5 (µg/m³)",
            legend=dict(
                orientation="h",
                y=-0.28,
            ),
        )

        st.plotly_chart(
            fig_lines,
            use_container_width=True,
        )

    st.dataframe(
        scenario_df.round(1),
        use_container_width=True,
    )

    st.markdown("### Intervention assumptions")

    assumption_table = pd.DataFrame([
        {
            "Intervention": name,
            "Target source": category,
            "Default modeled reduction": f"{default:.0%}",
        }
        for name, category, default in E.ACTIONS
    ])

    st.dataframe(
        assumption_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Scenario effects are applied to estimated source shares. "
        "No rebound, displacement or secondary atmospheric chemistry "
        "effects are modeled in this MVP."
    )


# ---------------------------------------------------------------------
# TAB 5 — METHODOLOGY
# ---------------------------------------------------------------------

with tabs[4]:

    section_title(
        "How OmniEarthers Works",
        "A transparent MVP designed around the competition requirements.",
    )

    st.markdown(
        """
        ### 1. Historical data

        Historical PM2.5 observations are combined with weather and activity
        indicators for a defined urban area.

        ### 2. Forecasting

        The model uses recent PM2.5 history, weather, traffic, industrial
        activity and seasonal features to produce a seven-day PM2.5 forecast.

        ### 3. Historical validation

        The last 90 days are held out from training. Rolling seven-day
        predictions are compared with what actually happened.

        ### 4. Source attribution

        A non-negative regression estimates the likely contribution of:

        - 🚗 Vehicular activity
        - 🏭 Industrial activity
        - 🌬️ Dust / weather

        ### 5. Digital twin scenarios

        The attribution shares are used to simulate interventions and
        estimate the resulting PM2.5 under each scenario.
        """
    )

    st.markdown("### Observed vs modeled")

    provenance = pd.DataFrame([
        {
            "Component": "Historical PM2.5",
            "Status": "OBSERVED",
            "Meaning": "Measurement in the supplied historical dataset",
        },
        {
            "Component": "Weather / activity inputs",
            "Status": "OBSERVED / DERIVED",
            "Meaning": "Historical drivers supplied or derived from the dataset",
        },
        {
            "Component": "7-day PM2.5 forecast",
            "Status": "MODELED",
            "Meaning": "Model output",
        },
        {
            "Component": "Source contribution",
            "Status": "MODELED",
            "Meaning": "Statistical estimate, not an emissions inventory",
        },
        {
            "Component": "Intervention result",
            "Status": "MODELED SCENARIO",
            "Meaning": "Counterfactual estimate under stated assumptions",
        },
        {
            "Component": "Historical validation line",
            "Status": "OBSERVED",
            "Meaning": "Actual values used only for evaluation",
        },
    ])

    st.dataframe(
        provenance,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Stated assumptions")

    st.markdown(
        f"""
        - {ASSUMPTION}
        - Traffic intervention default: 30% reduction in modeled vehicular contribution.
        - Industrial intervention default: 80% reduction in modeled industrial contribution.
        - Dust intervention default: 40% reduction in modeled dust contribution.
        - Future driver values in the historical replay come from the historical
          record, equivalent to assuming those driver forecasts are available.
        - Intervention effects are independent in this MVP.
        """
    )

    st.markdown("### Limitations")

    st.markdown(
        """
        - Source attribution is statistical and should not be interpreted as
          proof of causality.
        - The model is not a chemical-transport model.
        - Dust/weather is represented through a proxy based on wind, humidity
          and dry conditions.
        - Synthetic demo data, when enabled, is illustrative and must not be
          presented as real Pune observations.
        - Production deployment should replace historical driver replay with
          live weather and traffic forecasts.
        """
    )

    st.markdown("### Production roadmap")

    st.markdown(
        """
        **MVP → Production**

        Historical CSV  
        ↓  
        Live CPCB / OpenAQ data  
        ↓  
        Live weather + traffic feeds  
        ↓  
        Higher-resolution city grid  
        ↓  
        Satellite/AOD and emissions priors  
        ↓  
        More advanced atmospheric modeling  
        ↓  
        Cost-aware intervention optimization
        """
    )

    st.success(
        "Core MVP loop: Observe → Understand → Forecast → Simulate → Validate"
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown("---")

st.caption(
    "OmniEarthers · Urban Environmental Digital Twin · HackMatrix MVP"
)
