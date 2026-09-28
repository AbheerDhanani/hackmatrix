"""
OmniEarthers engine
Urban Environmental Digital Twin — forecasting, back-testing,
source attribution and intervention scenarios.

No UI code lives here.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 7
TEST_DAYS = 90
NAAQS_PM25 = 60

NUM = [
    "pm25", "temp", "humidity", "wind", "rain",
    "traffic_index", "industrial_index"
]
REQUIRED = ["date", "station", "lat", "lon"] + NUM
SOURCES = ["Vehicular", "Industrial", "Dust/Weather"]

ACTIONS = [
    ("Traffic restriction", "Vehicular", 0.30),
    ("Heavy-industry control", "Industrial", 0.80),
    ("Construction dust suppression", "Dust/Weather", 0.40),
]

STATIONS = {
    "Shivajinagar": (18.5308, 73.8475, 1.15, 0.5, 0.8),
    "Hadapsar": (18.5089, 73.9260, 1.10, 1.3, 1.0),
    "Pimpri-Chinchwad": (18.6298, 73.7997, 1.00, 1.9, 0.9),
    "Katraj": (18.4575, 73.8677, 0.90, 0.4, 1.5),
    "Kothrud": (18.5074, 73.8077, 0.85, 0.3, 0.7),
}

ALIASES = {
    "pm2.5": "pm25", "pm_2.5": "pm25", "pm_2_5": "pm25",
    "pm25": "pm25", "wind_speed": "wind", "windspeed": "wind",
    "wind_spd": "wind", "traffic": "traffic_index",
    "traffic_index": "traffic_index", "industrial": "industrial_index",
    "temperature": "temp", "rainfall": "rain",
    "precipitation": "rain", "humid": "humidity",
    "latitude": "lat", "longitude": "lon", "city": "station",
    "location": "station", "neighbourhood": "station",
    "neighborhood": "station", "area": "station"
}

MINIMUM = ["date", "pm25", "traffic_index", "wind"]
DEFAULTS = {
    "temp": 25.0,
    "humidity": 50.0,
    "rain": 0.0,
    "industrial_index": 50.0,
}


def prepare_df(df: pd.DataFrame, lat=18.5204, lon=73.8567,
               area="Pune Urban Region"):
    """Normalize an uploaded CSV into the OmniEarthers schema."""
    df = df.copy()
    norm = lambda c: str(c).strip().lower().replace(" ", "_")
    df.columns = [ALIASES.get(norm(c), norm(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in MINIMUM if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV needs at least: Date, PM2.5, Traffic_Index, Wind_Speed. "
            f"Missing: {missing}. Found: {list(df.columns)}"
        )

    notes = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "pm25"])

    if "station" not in df.columns:
        df["station"] = area
        notes.append(
            "No station/neighbourhood column: treating the dataset as one area."
        )

    if "lat" not in df.columns or "lon" not in df.columns:
        df["lat"], df["lon"] = lat, lon
        if df["station"].nunique() > 1:
            notes.append(
                "No latitude/longitude columns: all stations use the supplied "
                "study-area coordinates."
            )

    for key, value in DEFAULTS.items():
        if key not in df.columns:
            df[key] = value
            notes.append(f"No '{key}' column: filled with default {value}.")

    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["pm25", "traffic_index", "wind"])
    agg = {c: "mean" for c in NUM}
    agg.update(lat="first", lon="first")

    df = (
        df.groupby(["station", "date"], as_index=False)
        .agg(agg)
        .sort_values(["station", "date"])
    )
    return df, notes


def make_synthetic(seed: int = 7) -> pd.DataFrame:
    """Pune-like synthetic demonstration data.
    Clearly label this as synthetic in the UI.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values

    winter = np.clip(np.cos(2 * np.pi * (doy - 10) / 365), 0, None)
    monsoon = ((doy >= 160) & (doy <= 272)).astype(float)

    rain = (
        (rng.random(n) < 0.12 + 0.6 * monsoon)
        * rng.gamma(2, 4, n)
    )
    temp = (
        26
        + 6 * np.cos(2 * np.pi * (doy - 120) / 365)
        + rng.normal(0, 1.2, n)
    )
    humidity = np.clip(
        45 + 35 * monsoon + 15 * (rain > 0)
        + rng.normal(0, 6, n), 15, 98
    )
    wind = np.clip(
        7 + 3 * monsoon - 2 * winter + rng.normal(0, 2.5, n),
        0.5, None
    )
    weekday = (dates.dayofweek.values < 5).astype(float)
    dust_raw = wind * (1 - humidity / 100) * (rain < 0.5)
    dilution = np.clip(
        1 + 0.6 * winter - 0.4 * monsoon
        - 0.03 * (wind - 7), 0.3, None
    )
    wash = np.exp(-0.06 * rain)

    frames = []
    for name, (lat, lon, traffic_mult, industrial_mult, dust_mult) in STATIONS.items():
        traffic = np.clip(
            (55 + 18 * weekday) * traffic_mult
            + rng.normal(0, 6, n) - 0.6 * rain, 5, None
        )
        industrial = np.clip(
            50 * industrial_mult + rng.normal(0, 2, n), 1, None
        )

        residual = np.zeros(n)
        for t in range(1, n):
            residual[t] = 0.6 * residual[t - 1] + rng.normal(0, 6)

        pm = (
            0.30 * traffic
            + 0.22 * industrial
            + 4.0 * dust_mult * dust_raw
            + 18
        ) * dilution * wash + residual

        frames.append(pd.DataFrame({
            "date": dates,
            "station": name,
            "lat": lat,
            "lon": lon,
            "pm25": np.clip(pm, 5, None).round(1),
            "temp": temp.round(1),
            "humidity": humidity.round(0),
            "wind": wind.round(1),
            "rain": rain.round(1),
            "traffic_index": traffic.round(1),
            "industrial_index": industrial.round(1),
        }))

    return pd.concat(frames, ignore_index=True)


def get_series(df: pd.DataFrame, station: str) -> pd.DataFrame:
    if station == "City average":
        s = df.groupby("date")[NUM].mean()
    else:
        s = (
            df[df.station == station]
            .set_index("date")[NUM]
            .sort_index()
        )

    s = s.asfreq("D").interpolate(limit_direction="both")
    s["dust"] = (
        s["wind"]
        * (1 - s["humidity"] / 100)
        * (s["rain"] < 0.5)
    )
    return s


def test_start_date(s: pd.DataFrame) -> pd.Timestamp:
    return s.index.max() - pd.Timedelta(days=TEST_DAYS - 1)


def row_features(window, ex, date):
    doy = date.dayofyear
    return [
        window[-1], window[-2], window[-3], window[-7],
        float(np.mean(window)),
        ex["temp"], ex["humidity"], ex["wind"], ex["rain"],
        ex["traffic_index"], ex["industrial_index"],
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ]


def make_xy(s: pd.DataFrame):
    pm = s["pm25"].values
    X, y = [], []

    for t in range(7, len(s)):
        X.append(row_features(pm[t - 7:t], s.iloc[t], s.index[t]))
        y.append(pm[t])

    return np.array(X), np.array(y)


def new_model(kind: str):
    if kind == "Ridge regression":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=5.0)
        )

    return GradientBoostingRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    )


def fit_forecaster(s: pd.DataFrame, kind: str):
    """Train only before the held-out historical test period."""
    train = s[s.index < test_start_date(s)]

    if len(train) < 30:
        raise ValueError("At least 30 daily training observations are required.")

    X, y = make_xy(train)
    return new_model(kind).fit(X, y)


def forecast(model, s: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Recursive 7-day PM2.5 forecast from a historical replay date."""
    if origin not in s.index:
        raise ValueError("Replay date is not available in the selected time series.")

    window = list(s.loc[:origin, "pm25"].values[-7:])
    if len(window) < 7:
        raise ValueError("At least seven PM2.5 observations are required before replay date.")

    days = pd.date_range(
        origin + pd.Timedelta(days=1),
        periods=HORIZON
    )

    out = []
    for d in days:
        if d not in s.index:
            raise ValueError(
                f"Historical driver data is missing for forecast date {d.date()}."
            )

        x = row_features(window[-7:], s.loc[d], d)
        prediction = float(max(model.predict([x])[0], 1.0))
        out.append(prediction)
        window.append(prediction)

    return pd.Series(out, index=days, name="forecast")


def backtest(model, s: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin 7-day back-test across the held-out period."""
    ts = test_start_date(s)
    origin = ts - pd.Timedelta(days=1)
    rows = []

    while origin + pd.Timedelta(days=HORIZON) <= s.index.max():
        fc = forecast(model, s, origin)
        persistence = s.loc[origin, "pm25"]

        for horizon, (date, pred) in enumerate(fc.items(), start=1):
            rows.append((
                origin, date, horizon, pred,
                s.loc[date, "pm25"], persistence
            ))

        origin += pd.Timedelta(days=HORIZON)

    return pd.DataFrame(
        rows,
        columns=[
            "origin", "date", "horizon",
            "pred", "actual", "persistence"
        ],
    )


def score(bt: pd.DataFrame) -> pd.DataFrame:
    def metrics(column):
        return {
            "MAE (µg/m³)": mean_absolute_error(bt.actual, bt[column]),
            "RMSE (µg/m³)": float(
                np.sqrt(mean_squared_error(bt.actual, bt[column]))
            ),
            "R²": r2_score(bt.actual, bt[column]),
        }

    return pd.DataFrame({
        "OmniEarthers model": metrics("pred"),
        "Persistence baseline": metrics("persistence"),
    }).T


class Attribution:
    """Interpretable statistical source-attribution model."""

    def __init__(self, coef, intercept, industrial_assumed):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.industrial_assumed = industrial_assumed


def fit_attribution(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    industrial_share: float = 0.15,
) -> Attribution:
    """
    Estimate contributions from observed historical relationships.

    This is statistical attribution, not a chemical-transport model.
    """
    window = df[
        (df.date <= origin)
        & (df.date > origin - pd.Timedelta(days=365))
    ].copy()

    window["dust"] = (
        window["wind"]
        * (1 - window["humidity"] / 100)
        * (window["rain"] < 0.5)
    )

    industrial = window["industrial_index"]

    # If industrial activity is effectively constant, its coefficient
    # cannot be identified from temporal variation.
    if (
        industrial.std() < 0.05 * max(industrial.mean(), 1e-9)
        or window["station"].nunique() == 1
    ):
        k = (
            industrial_share
            * window["pm25"].mean()
            / max(industrial.mean(), 1e-9)
        )
        residual = window["pm25"] - k * industrial

        lr = LinearRegression(positive=True).fit(
            window[["traffic_index", "dust"]],
            residual,
        )

        return Attribution(
            [lr.coef_[0], k, lr.coef_[1]],
            lr.intercept_,
            True,
        )

    lr = LinearRegression(positive=True).fit(
        window[[
            "traffic_index",
            "industrial_index",
            "dust",
        ]],
        window["pm25"],
    )

    return Attribution(
        lr.coef_,
        lr.intercept_,
        False,
    )


def contributions(model, rows: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({
        "Vehicular": model.coef_[0] * rows["traffic_index"],
        "Industrial": model.coef_[1] * rows["industrial_index"],
        "Dust/Weather": model.coef_[2] * rows["dust"],
    })

    c["Background/regional"] = max(
        float(model.intercept_), 0.0
    )

    return c


def apply_actions(
    forecast_values: pd.Series,
    attribution_model,
    series: pd.DataFrame,
    cuts: dict,
) -> pd.Series:
    """Apply source-specific intervention assumptions to a forecast."""
    c = contributions(
        attribution_model,
        series.loc[forecast_values.index]
    )

    source_total = c[SOURCES].sum(axis=1).replace(0, np.nan)
    shares = c[SOURCES].div(source_total, axis=0).fillna(0)

    reduction = sum(
        cuts.get(source, 0.0) * shares[source]
        for source in SOURCES
    )

    return forecast_values * (1 - reduction.clip(0, 0.95))


def pm_category(value: float):
    """CPCB India PM2.5 category thresholds used for visualization."""
    if value <= 30:
        return "#2e9e4f", "Good"
    if value <= 60:
        return "#8bc34a", "Satisfactory"
    if value <= 90:
        return "#f2c500", "Moderate"
    if value <= 120:
        return "#ff8c00", "Poor"
    if value <= 250:
        return "#e53935", "Very Poor"
    return "#8b0000", "Severe"
