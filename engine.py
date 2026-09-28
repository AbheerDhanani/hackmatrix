"""OmniTwin engine: data, forecasting, back-testing, source attribution, scenarios.

Pure pandas / scikit-learn. No UI code here, so it can be tested on its own.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 7          # forecast days
TEST_DAYS = 90       # last N days are the "historical test period"
NAAQS_PM25 = 60      # India 24-h PM2.5 standard, ug/m3

NUM = ["pm25", "temp", "humidity", "wind", "rain", "traffic_index", "industrial_index"]
REQUIRED = ["date", "station", "lat", "lon"] + NUM
SOURCES = ["Vehicular", "Industrial", "Dust/Weather"]

# (name, category it acts on, default reduction of that category's contribution)
ACTIONS = [
    ("Action 1: Odd-Even Traffic Rule", "Vehicular", 0.30),
    ("Action 2: Halt Heavy Industry", "Industrial", 0.80),
    ("Action 3: Construction Sprinklers", "Dust/Weather", 0.40),
]

# name: (lat, lon, traffic mult, industrial mult, dust mult)  -- Pune demo area
STATIONS = {
    "Shivajinagar": (18.5308, 73.8475, 1.15, 0.5, 0.8),
    "Hadapsar": (18.5089, 73.9260, 1.10, 1.3, 1.0),
    "Pimpri-Chinchwad": (18.6298, 73.7997, 1.00, 1.9, 0.9),
    "Katraj": (18.4575, 73.8677, 0.90, 0.4, 1.5),
    "Kothrud": (18.5074, 73.8077, 0.85, 0.3, 0.7),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

ALIASES = {"pm2.5": "pm25", "pm_2.5": "pm25", "pm_2_5": "pm25", "pm25": "pm25",
           "wind_speed": "wind", "windspeed": "wind", "wind_spd": "wind",
           "traffic": "traffic_index", "traffic_index": "traffic_index",
           "industrial": "industrial_index", "temperature": "temp",
           "rainfall": "rain", "precipitation": "rain", "humid": "humidity",
           "latitude": "lat", "longitude": "lon", "city": "station", "location": "station",
           "neighbourhood": "station", "neighborhood": "station", "area": "station"}
MINIMUM = ["date", "pm25", "traffic_index", "wind"]
DEFAULTS = {"temp": 25.0, "humidity": 50.0, "rain": 0.0, "industrial_index": 50.0}


def prepare_df(df: pd.DataFrame, lat=18.5204, lon=73.8567, area="Study area"):
    """Accepts a minimal CSV (Date, PM2.5, Traffic_Index, Wind_Speed) or a richer one.
    Returns (clean_df, notes) where notes lists anything that was filled with a default."""
    df = df.copy()
    norm = lambda c: str(c).strip().lower().replace(" ", "_")
    df.columns = [ALIASES.get(norm(c), norm(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in MINIMUM if c not in df.columns]
    if missing:
        raise ValueError(f"CSV needs at least these columns: Date, PM2.5, Traffic_Index, Wind_Speed "
                         f"(missing: {missing}; found: {list(df.columns)})")
    notes = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "pm25"])
    if "station" not in df:
        df["station"] = area
        notes.append("No station/neighbourhood column: treating data as ONE area (single map marker).")
    if "lat" not in df or "lon" not in df:
        df["lat"], df["lon"] = lat, lon
        if "station" in df and df["station"].nunique() > 1:
            notes.append("No lat/lon columns: all stations placed at the same point. Add lat/lon for a real map.")
    for k, v in DEFAULTS.items():
        if k not in df:
            df[k] = v
            notes.append(f"No '{k}' column: filled with constant {v}.")
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = {c: "mean" for c in NUM}
    agg.update(lat="first", lon="first")
    df = df.groupby(["station", "date"], as_index=False).agg(agg)
    return df, notes

def make_synthetic(seed: int = 7) -> pd.DataFrame:
    """Demo dataset with Pune-like seasonality (winter smog, monsoon wash-out).

    SYNTHETIC. Replace with a real CPCB / Kaggle CSV of the same schema.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    winter = np.clip(np.cos(2 * np.pi * (doy - 10) / 365), 0, None)
    monsoon = ((doy >= 160) & (doy <= 272)).astype(float)
    rain = (rng.random(n) < 0.12 + 0.6 * monsoon) * rng.gamma(2, 4, n)
    temp = 26 + 6 * np.cos(2 * np.pi * (doy - 120) / 365) + rng.normal(0, 1.2, n)
    humidity = np.clip(45 + 35 * monsoon + 15 * (rain > 0) + rng.normal(0, 6, n), 15, 98)
    wind = np.clip(7 + 3 * monsoon - 2 * winter + rng.normal(0, 2.5, n), 0.5, None)
    weekday = (dates.dayofweek.values < 5).astype(float)
    dust_raw = wind * (1 - humidity / 100) * (rain < 0.5)
    dilution = np.clip(1 + 0.6 * winter - 0.4 * monsoon - 0.03 * (wind - 7), 0.3, None)
    wash = np.exp(-0.06 * rain)

    frames = []
    for name, (lat, lon, tm, im, dm) in STATIONS.items():
        traffic = np.clip((55 + 18 * weekday) * tm + rng.normal(0, 6, n) - 0.6 * rain, 5, None)
        industrial = np.clip(50 * im + rng.normal(0, 2, n), 1, None)
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.6 * e[t - 1] + rng.normal(0, 6)
        pm = (0.30 * traffic + 0.22 * industrial + 4.0 * dm * dust_raw + 18) * dilution * wash + e
        frames.append(pd.DataFrame({
            "date": dates, "station": name, "lat": lat, "lon": lon,
            "pm25": np.clip(pm, 5, None).round(1), "temp": temp.round(1),
            "humidity": humidity.round(0), "wind": wind.round(1), "rain": rain.round(1),
            "traffic_index": traffic.round(1), "industrial_index": industrial.round(1),
        }))
    return pd.concat(frames, ignore_index=True)


def get_series(df: pd.DataFrame, station: str) -> pd.DataFrame:
    if station == "City average":
        s = df.groupby("date")[NUM].mean()
    else:
        s = df[df.station == station].set_index("date")[NUM].sort_index()
    s = s.asfreq("D").interpolate(limit_direction="both")
    # dust proxy: wind speed on dry days (assumption: dust correlates with wind speed)
    s["dust"] = s["wind"] * (1 - s["humidity"] / 100) * (s["rain"] < 0.5)
    return s


def test_start_date(s: pd.DataFrame) -> pd.Timestamp:
    return s.index.max() - pd.Timedelta(days=TEST_DAYS - 1)


# ----------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------
def row_features(window, ex, date):
    """window = last 7 PM2.5 values (oldest -> newest); ex = that day's drivers."""
    doy = date.dayofyear
    return [window[-1], window[-2], window[-3], window[-7], float(np.mean(window)),
            ex["temp"], ex["humidity"], ex["wind"], ex["rain"],
            ex["traffic_index"], ex["industrial_index"],
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]


def make_xy(s: pd.DataFrame):
    pm = s["pm25"].values
    X, y = [], []
    for t in range(7, len(s)):
        X.append(row_features(pm[t - 7:t], s.iloc[t], s.index[t]))
        y.append(pm[t])
    return np.array(X), np.array(y)


def new_model(kind: str):
    if kind == "Ridge regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    return GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=0)


def fit_forecaster(s: pd.DataFrame, kind: str):
    """Train ONLY on data before the historical test period (no leakage)."""
    train = s[s.index < test_start_date(s)]
    X, y = make_xy(train)
    return new_model(kind).fit(X, y)


def forecast(model, s: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Recursive 7-day forecast from `origin` ('Today'). Uses actual weather/traffic
    for the future days (i.e. assumes a good weather forecast is available)."""
    window = list(s.loc[:origin, "pm25"].values[-7:])
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON)
    out = []
    for d in days:
        x = row_features(window[-7:], s.loc[d], d)
        p = float(max(model.predict([x])[0], 1.0))
        out.append(p)
        window.append(p)
    return pd.Series(out, index=days, name="forecast")


def backtest(model, s: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin back-test over the held-out test period, every 7 days."""
    ts = test_start_date(s)
    origin = ts - pd.Timedelta(days=1)
    rows = []
    while origin + pd.Timedelta(days=HORIZON) <= s.index.max():
        fc = forecast(model, s, origin)
        persist = s.loc[origin, "pm25"]
        for h, (d, p) in enumerate(fc.items(), start=1):
            rows.append((origin, d, h, p, s.loc[d, "pm25"], persist))
        origin += pd.Timedelta(days=HORIZON)
    return pd.DataFrame(rows, columns=["origin", "date", "horizon", "pred", "actual", "persistence"])


def score(bt: pd.DataFrame) -> pd.DataFrame:
    def m(col):
        return {"MAE (ug/m3)": mean_absolute_error(bt.actual, bt[col]),
                "RMSE (ug/m3)": float(np.sqrt(mean_squared_error(bt.actual, bt[col]))),
                "R2": r2_score(bt.actual, bt[col])}
    return pd.DataFrame({"OmniTwin model": m("pred"), "Baseline: persistence": m("persistence")}).T


# ----------------------------------------------------------------------------
# Source attribution + scenarios
# ----------------------------------------------------------------------------
class Attribution:
    """coef_ = [traffic, industrial, dust]; intercept_ = background/regional."""
    def __init__(self, coef, intercept, industrial_assumed):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.industrial_assumed = industrial_assumed


def fit_attribution(df: pd.DataFrame, origin: pd.Timestamp, industrial_share: float = 0.15) -> Attribution:
    """Regression on the trailing year of OBSERVED data: PM2.5 ~ traffic + industrial + dust
    (non-negative). Pooled across stations because industrial activity is ~constant in time at
    one station and is only identifiable BETWEEN stations. If there is no usable variation
    (single area / constant industrial data) we follow the stated assumption instead:
    industrial is a constant baseline equal to `industrial_share` of mean PM2.5."""
    w = df[(df.date <= origin) & (df.date > origin - pd.Timedelta(days=365))].copy()
    w["dust"] = w["wind"] * (1 - w["humidity"] / 100) * (w["rain"] < 0.5)
    ind = w["industrial_index"]
    if ind.std() < 0.05 * ind.mean():
        k = industrial_share * w["pm25"].mean() / ind.mean()
        res = w["pm25"] - k * ind
        lr = LinearRegression(positive=True).fit(w[["traffic_index", "dust"]], res)
        return Attribution([lr.coef_[0], k, lr.coef_[1]], lr.intercept_, True)
    lr = LinearRegression(positive=True).fit(w[["traffic_index", "industrial_index", "dust"]], w["pm25"])
    return Attribution(lr.coef_, lr.intercept_, False)


def contributions(lr, rows: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({
        "Vehicular": lr.coef_[0] * rows["traffic_index"],
        "Industrial": lr.coef_[1] * rows["industrial_index"],
        "Dust/Weather": lr.coef_[2] * rows["dust"],
    })
    c["Background/regional"] = max(float(lr.intercept_), 0.0)
    return c


def apply_actions(fc: pd.Series, lr, s: pd.DataFrame, cuts: dict) -> pd.Series:
    """cuts = {"Vehicular": 0.3, ...}. Each day's forecast is split by the attribution
    shares for that day, then each targeted share is reduced by its cut."""
    c = contributions(lr, s.loc[fc.index])
    share = c[SOURCES].div(c.sum(axis=1), axis=0)
    reduction = sum(cuts.get(k, 0.0) * share[k] for k in SOURCES)
    return fc * (1 - reduction)


def pm_category(v: float):
    """CPCB India PM2.5 categories -> (colour, label)."""
    if v <= 30: return "#2e9e4f", "Good"
    if v <= 60: return "#8bc34a", "Satisfactory"
    if v <= 90: return "#f2c500", "Moderate"
    if v <= 120: return "#ff8c00", "Poor"
    if v <= 250: return "#e53935", "Very Poor"
    return "#8b0000", "Severe"
