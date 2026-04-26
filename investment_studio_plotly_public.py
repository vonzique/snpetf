from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None


# ============================================================
# Streamlit setup
# ============================================================

st.set_page_config(
    page_title="CAPE Real/Nominal Investment Studio",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# Theory note used in the app
# ============================================================

THEORY_NOTE = """
### CAPE and inflation: important interpretation

Professor Robert Shiller's CAPE ratio is normally calculated using inflation-adjusted prices and inflation-adjusted 10-year average earnings.
That means CAPE is primarily a **valuation metric** and is best calibrated against **future real returns**.

So if the CAPE model says the expected return is **4% per year**, this should usually be interpreted as approximately **4% real annual return**, meaning after inflation.

To estimate the absolute account value that you may see in the future, convert the real return into nominal return:

$$
1+r_{nominal} = (1+r_{real})(1+\pi)
$$

where:

- $r_{real}$ = CAPE-implied real return
- $\pi$ = inflation assumption
- $r_{nominal}$ = expected absolute account growth rate

Example: 4% real return and 3% inflation gives:

$$
(1.04)(1.03)-1 = 7.12\% \text{ nominal return}
$$

Therefore, the app keeps two separate layers:

1. **CAPE calibration layer** → estimates real returns.
2. **Projection layer** → converts real returns into nominal values using inflation.

This avoids mixing real valuation signals with nominal wealth projections.
"""


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ColumnMap:
    date: Optional[str] = None
    price: Optional[str] = None
    earnings: Optional[str] = None
    dividend: Optional[str] = None
    cpi: Optional[str] = None
    cape: Optional[str] = None
    total_return_index: Optional[str] = None
    real_total_return_index: Optional[str] = None


@dataclass
class CalibrationResult:
    model_name: str
    horizon_years: int
    slope: float
    intercept: float
    r2: float
    n_obs: int
    x_name: str
    y_name: str
    fitted: pd.DataFrame


# ============================================================
# Utility functions
# ============================================================


def clean_col_name(col: object) -> str:
    return str(col).strip().replace("\n", " ").replace("\r", " ")


def normalise_col_key(col: object) -> str:
    return (
        clean_col_name(col)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
    )


def first_matching_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    keys = {normalise_col_key(c): c for c in columns}
    candidate_keys = [normalise_col_key(c) for c in candidates]

    # Exact normalised match first
    for ck in candidate_keys:
        if ck in keys:
            return keys[ck]

    # Then contains match
    for col in columns:
        k = normalise_col_key(col)
        for ck in candidate_keys:
            if ck and ck in k:
                return col

    return None


def parse_shiller_decimal_date(value: object) -> pd.Timestamp:
    """
    Handles Shiller style dates such as 1871.01, 1871.02, ..., 2024.12.
    This is not a true decimal year; the part after the dot usually represents month.
    """
    if pd.isna(value):
        return pd.NaT

    try:
        text = str(value).strip()
        if not text:
            return pd.NaT

        # Already date-like
        if "-" in text or "/" in text:
            return pd.to_datetime(text, errors="coerce")

        val = float(text)
        year = int(math.floor(val))
        raw_month = int(round((val - year) * 100))

        if raw_month < 1 or raw_month > 12:
            # fallback for genuine decimal-year values
            raw_month = int(round((val - year) * 12)) + 1
            raw_month = min(max(raw_month, 1), 12)

        return pd.Timestamp(year=year, month=raw_month, day=1)
    except Exception:
        return pd.NaT


def infer_and_parse_date(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()

    if date_col is None:
        # Try index if it looks date-like
        try:
            parsed_index = pd.to_datetime(out.index, errors="coerce")
            if parsed_index.notna().mean() > 0.8:
                out["Date"] = parsed_index
                return out
        except Exception:
            pass
        return out

    raw = out[date_col]

    # If date column is numeric, likely Shiller yyyy.mm format
    if pd.api.types.is_numeric_dtype(raw):
        parsed = raw.apply(parse_shiller_decimal_date)
    else:
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().mean() < 0.7:
            parsed = raw.apply(parse_shiller_decimal_date)

    out["Date"] = parsed
    out = out.dropna(subset=["Date"])
    out = out.sort_values("Date")
    return out


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))

    if name.endswith(".mat"):
        if loadmat is None:
            raise RuntimeError("scipy is required to read .mat files. Install scipy or upload CSV/XLSX.")
        mat = loadmat(io.BytesIO(content), squeeze_me=True, struct_as_record=False)
        return mat_to_dataframe(mat)

    raise ValueError("Unsupported file type. Please upload CSV, XLSX, XLS, or MAT.")


def mat_to_dataframe(mat: Dict[str, object]) -> pd.DataFrame:
    """
    Best-effort MAT reader.
    Works when the MAT file contains either:
    - one 2D numeric array, or
    - multiple equal-length vectors.

    If your MAT file is complex, export it to CSV from MATLAB for best reliability.
    """
    usable = {
        k: v
        for k, v in mat.items()
        if not k.startswith("__") and not callable(v)
    }

    # Prefer a 2D numeric array
    for key, value in usable.items():
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            if arr.shape[0] < arr.shape[1]:
                # Usually rows are observations; do not transpose if clearly already tall
                pass
            return pd.DataFrame(arr, columns=[f"col_{i+1}" for i in range(arr.shape[1])])

    # Equal-length vectors
    vectors = {}
    lengths = []
    for key, value in usable.items():
        arr = np.asarray(value).squeeze()
        if arr.ndim == 1 and arr.size > 1 and np.issubdtype(arr.dtype, np.number):
            vectors[key] = arr
            lengths.append(arr.size)

    if vectors:
        common_len = pd.Series(lengths).mode().iloc[0]
        selected = {k: v for k, v in vectors.items() if len(v) == common_len}
        if selected:
            return pd.DataFrame(selected)

    raise ValueError("Could not convert MAT file into a table. Please export the dataset as CSV/XLSX.")


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    cols = [clean_col_name(c) for c in df.columns]
    df.columns = cols

    date = first_matching_column(cols, ["Date", "Month", "Time", "YearMonth", "Year_Month"])
    price = first_matching_column(cols, ["Price", "P", "S&P", "S&P 500", "SP500", "Index", "Close"])
    earnings = first_matching_column(cols, ["Earnings", "E", "EPS", "Trailing earnings"])
    dividend = first_matching_column(cols, ["Dividend", "Dividends", "D", "Dividend Yield"])
    cpi = first_matching_column(cols, ["CPI", "Consumer Price Index", "Inflation Index"])
    cape = first_matching_column(cols, ["CAPE", "CAPE Ratio", "Shiller PE", "Cyclically Adjusted PE", "P/E10", "PE10"])
    tri = first_matching_column(cols, ["Total Return", "Total Return Index", "TR", "TRI", "Nominal Total Return Index"])
    real_tri = first_matching_column(cols, ["Real Total Return", "Real Total Return Index", "Real TR", "Real TRI"])

    # Avoid conflict where column "P/E10" is detected as price due contains P.
    if price == cape:
        price = first_matching_column(cols, ["Price", "S&P 500 Price", "SP500 Price", "Close"])

    return ColumnMap(
        date=date,
        price=price,
        earnings=earnings,
        dividend=dividend,
        cpi=cpi,
        cape=cape,
        total_return_index=tri,
        real_total_return_index=real_tri,
    )


def to_numeric_series(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def annualised_return(start: pd.Series, end: pd.Series, years: float) -> pd.Series:
    ratio = end / start
    ratio = ratio.where((ratio > 0) & np.isfinite(ratio))
    return ratio.pow(1.0 / years) - 1.0


def real_to_nominal_return(real_return: pd.Series | float, inflation: pd.Series | float) -> pd.Series | float:
    return (1.0 + real_return) * (1.0 + inflation) - 1.0


def nominal_to_real_return(nominal_return: pd.Series | float, inflation: pd.Series | float) -> pd.Series | float:
    return (1.0 + nominal_return) / (1.0 + inflation) - 1.0


def compute_cape_from_price_earnings_cpi(
    df: pd.DataFrame,
    price_col: str,
    earnings_col: str,
    cpi_col: str,
    window_months: int = 120,
) -> pd.Series:
    price = pd.to_numeric(df[price_col], errors="coerce")
    earnings = pd.to_numeric(df[earnings_col], errors="coerce")
    cpi = pd.to_numeric(df[cpi_col], errors="coerce")

    cpi_ref = cpi.iloc[-1]
    real_price = price / cpi * cpi_ref
    real_earnings = earnings / cpi * cpi_ref
    avg_real_earnings = real_earnings.rolling(window_months, min_periods=window_months).mean()

    cape = real_price / avg_real_earnings
    cape = cape.replace([np.inf, -np.inf], np.nan)
    return cape


def build_real_total_return_index(
    df: pd.DataFrame,
    cmap: ColumnMap,
    assume_shiller_dividend_is_annual: bool = True,
) -> pd.Series:
    """
    Returns a real total return index.

    Priority:
    1. Existing real total return index.
    2. Existing nominal total return index deflated by CPI.
    3. Price + dividend approximation deflated by CPI.
    4. Real price index only, if dividends are unavailable.

    Shiller's historical monthly dataset usually has P, D, E, CPI, CAPE.
    The D column is commonly interpreted as annualised dividend amount.
    A monthly total-return approximation therefore uses D / 12.
    """
    cpi = to_numeric_series(df, cmap.cpi)

    real_tri = to_numeric_series(df, cmap.real_total_return_index)
    if real_tri is not None:
        idx = real_tri / real_tri.dropna().iloc[0]
        return idx.rename("real_total_return_index")

    tri = to_numeric_series(df, cmap.total_return_index)
    if tri is not None and cpi is not None:
        real = tri / cpi * cpi.iloc[-1]
        idx = real / real.dropna().iloc[0]
        return idx.rename("real_total_return_index")

    price = to_numeric_series(df, cmap.price)
    dividend = to_numeric_series(df, cmap.dividend)

    if price is None:
        raise ValueError("Need either a total return index or a price column to build return series.")

    if cpi is None:
        raise ValueError("Need CPI to convert returns to real terms for CAPE calibration.")

    if dividend is not None:
        monthly_div = dividend / 12.0 if assume_shiller_dividend_is_annual else dividend
        prev_price = price.shift(1)
        nominal_monthly_return = (price + monthly_div) / prev_price - 1.0
    else:
        nominal_monthly_return = price.pct_change()

    inflation_monthly = cpi.pct_change()
    real_monthly_return = nominal_to_real_return(nominal_monthly_return, inflation_monthly)
    real_monthly_return = real_monthly_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    idx = (1.0 + real_monthly_return).cumprod()
    return idx.rename("real_total_return_index")


def prepare_dataset(
    raw_df: pd.DataFrame,
    user_col_map: Optional[ColumnMap] = None,
    cape_window_months: int = 120,
    assume_shiller_dividend_is_annual: bool = True,
) -> Tuple[pd.DataFrame, ColumnMap, List[str]]:
    df = raw_df.copy()
    df.columns = [clean_col_name(c) for c in df.columns]

    cmap = infer_columns(df) if user_col_map is None else user_col_map
    notes: List[str] = []

    df = infer_and_parse_date(df, cmap.date)
    if "Date" not in df.columns:
        raise ValueError("Could not infer a date column. Please select one manually or rename it to Date.")

    df = df.sort_values("Date").reset_index(drop=True)

    # Numeric conversion for selected columns
    for col in [cmap.price, cmap.earnings, cmap.dividend, cmap.cpi, cmap.cape, cmap.total_return_index, cmap.real_total_return_index]:
        if col is not None and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # CAPE: use existing if present, otherwise calculate
    if cmap.cape is not None and cmap.cape in df.columns:
        df["CAPE"] = pd.to_numeric(df[cmap.cape], errors="coerce")
        notes.append("Using existing CAPE column from the uploaded data.")
    else:
        if cmap.price and cmap.earnings and cmap.cpi:
            df["CAPE"] = compute_cape_from_price_earnings_cpi(
                df,
                price_col=cmap.price,
                earnings_col=cmap.earnings,
                cpi_col=cmap.cpi,
                window_months=cape_window_months,
            )
            notes.append("Calculated CAPE from real price divided by 10-year average real earnings.")
        else:
            raise ValueError(
                "No CAPE column found and not enough columns to calculate it. Need Price, Earnings, and CPI."
            )

    # Real total return index for calibration
    df["real_total_return_index"] = build_real_total_return_index(
        df,
        cmap,
        assume_shiller_dividend_is_annual=assume_shiller_dividend_is_annual,
    )

    # Also build nominal price/index for display if possible
    if cmap.total_return_index:
        df["nominal_reference_index"] = to_numeric_series(df, cmap.total_return_index)
        notes.append("Nominal reference index uses uploaded total return index.")
    elif cmap.price:
        df["nominal_reference_index"] = to_numeric_series(df, cmap.price)
        notes.append("Nominal reference index uses price column only; dividends may not be included in nominal display.")
    else:
        df["nominal_reference_index"] = np.nan

    # Inflation series
    if cmap.cpi:
        cpi = to_numeric_series(df, cmap.cpi)
        df["inflation_monthly"] = cpi.pct_change()
        df["inflation_annualised_12m"] = cpi.pct_change(12)
    else:
        df["inflation_monthly"] = np.nan
        df["inflation_annualised_12m"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    return df, cmap, notes


# ============================================================
# Calibration functions
# ============================================================


def add_forward_real_returns(df: pd.DataFrame, horizon_years: int) -> pd.DataFrame:
    out = df.copy()
    months = horizon_years * 12
    start = out["real_total_return_index"]
    end = out["real_total_return_index"].shift(-months)
    out[f"forward_{horizon_years}y_real_return_ann"] = annualised_return(start, end, horizon_years)
    return out


def fit_linear_regression(x: pd.Series, y: pd.Series) -> Tuple[float, float, float, int, pd.DataFrame]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data["x"].abs() < 1e6) & (data["y"].abs() < 10)]

    if len(data) < 20:
        raise ValueError("Not enough valid observations for calibration. Try a shorter horizon or check the data.")

    X = np.vstack([data["x"].values, np.ones(len(data))]).T
    slope, intercept = np.linalg.lstsq(X, data["y"].values, rcond=None)[0]
    y_hat = slope * data["x"].values + intercept

    ss_res = np.sum((data["y"].values - y_hat) ** 2)
    ss_tot = np.sum((data["y"].values - np.mean(data["y"].values)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fitted = data.copy()
    fitted["y_hat"] = y_hat
    return float(slope), float(intercept), float(r2), len(data), fitted


def calibrate_cape_model(
    df: pd.DataFrame,
    horizon_years: int = 10,
    predictor: str = "cape_yield",
) -> CalibrationResult:
    work = add_forward_real_returns(df, horizon_years)
    y_col = f"forward_{horizon_years}y_real_return_ann"

    if predictor == "cape":
        x = work["CAPE"]
        x_name = "CAPE"
    elif predictor == "cape_yield":
        x = 1.0 / work["CAPE"]
        x_name = "1 / CAPE"
    elif predictor == "log_cape":
        x = np.log(work["CAPE"])
        x_name = "log(CAPE)"
    else:
        raise ValueError("Unsupported predictor.")

    y = work[y_col]
    slope, intercept, r2, n_obs, fitted = fit_linear_regression(x, y)

    return CalibrationResult(
        model_name=f"Forward {horizon_years}Y real return ~ {x_name}",
        horizon_years=horizon_years,
        slope=slope,
        intercept=intercept,
        r2=r2,
        n_obs=n_obs,
        x_name=x_name,
        y_name=f"Forward {horizon_years}Y annualised real return",
        fitted=fitted,
    )


def predict_real_return_from_cape(cape: float, calibration: CalibrationResult) -> float:
    if calibration.x_name == "CAPE":
        x = cape
    elif calibration.x_name == "1 / CAPE":
        x = 1.0 / cape
    elif calibration.x_name == "log(CAPE)":
        x = math.log(cape)
    else:
        raise ValueError("Unknown calibration predictor.")
    return calibration.slope * x + calibration.intercept


def empirical_real_return_distribution(
    df: pd.DataFrame,
    horizon_years: int,
    cape_now: float,
    matching_strength: float = 0.35,
    min_obs: int = 50,
) -> pd.Series:
    """
    Pulls historical forward real returns from observations with CAPE close to today's CAPE.
    matching_strength is a fractional band around CAPE.
    Example: 0.35 means use CAPE within ±35% of current CAPE.
    """
    work = add_forward_real_returns(df, horizon_years)
    y_col = f"forward_{horizon_years}y_real_return_ann"

    lower = cape_now * (1.0 - matching_strength)
    upper = cape_now * (1.0 + matching_strength)

    matched = work.loc[work["CAPE"].between(lower, upper), y_col].dropna()
    all_returns = work[y_col].dropna()

    if len(matched) < min_obs:
        return all_returns

    return matched


# ============================================================
# Projection and simulation functions
# ============================================================


def project_lump_sum(
    initial_amount: float,
    annual_real_return: float,
    annual_inflation: float,
    years: int,
) -> pd.DataFrame:
    rows = []
    nominal_return = real_to_nominal_return(annual_real_return, annual_inflation)

    for y in range(years + 1):
        real_value = initial_amount * ((1.0 + annual_real_return) ** y)
        nominal_value = initial_amount * ((1.0 + nominal_return) ** y)
        inflation_index = (1.0 + annual_inflation) ** y
        rows.append(
            {
                "Year": y,
                "Real value_today_money": real_value,
                "Nominal account_value": nominal_value,
                "Inflation index": inflation_index,
                "Nominal return used": nominal_return,
                "Real return used": annual_real_return,
            }
        )
    return pd.DataFrame(rows)


def project_monthly_contributions(
    initial_amount: float,
    monthly_contribution: float,
    annual_real_return: float,
    annual_inflation: float,
    years: int,
    contribution_growth_with_inflation: bool = True,
) -> pd.DataFrame:
    months = years * 12
    monthly_real_return = (1.0 + annual_real_return) ** (1.0 / 12.0) - 1.0
    annual_nominal_return = real_to_nominal_return(annual_real_return, annual_inflation)
    monthly_nominal_return = (1.0 + annual_nominal_return) ** (1.0 / 12.0) - 1.0
    monthly_inflation = (1.0 + annual_inflation) ** (1.0 / 12.0) - 1.0

    real_value = initial_amount
    nominal_value = initial_amount

    rows = []
    for m in range(months + 1):
        year = m / 12.0
        rows.append(
            {
                "Month": m,
                "Year": year,
                "Real value_today_money": real_value,
                "Nominal account_value": nominal_value,
            }
        )

        if m == months:
            break

        contribution_nominal = monthly_contribution
        if contribution_growth_with_inflation:
            contribution_nominal = monthly_contribution * ((1.0 + monthly_inflation) ** m)

        contribution_real = contribution_nominal / ((1.0 + monthly_inflation) ** m)

        real_value = (real_value + contribution_real) * (1.0 + monthly_real_return)
        nominal_value = (nominal_value + contribution_nominal) * (1.0 + monthly_nominal_return)

    return pd.DataFrame(rows)


def run_monte_carlo_projection(
    initial_amount: float,
    monthly_contribution: float,
    years: int,
    real_return_samples: np.ndarray,
    annual_inflation: float,
    n_sims: int,
    contribution_growth_with_inflation: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    sampled_real_returns = rng.choice(real_return_samples, size=n_sims, replace=True)

    results = []
    for i, rr in enumerate(sampled_real_returns):
        path = project_monthly_contributions(
            initial_amount=initial_amount,
            monthly_contribution=monthly_contribution,
            annual_real_return=float(rr),
            annual_inflation=annual_inflation,
            years=years,
            contribution_growth_with_inflation=contribution_growth_with_inflation,
        )
        last = path.iloc[-1]
        results.append(
            {
                "Simulation": i + 1,
                "Sampled annual real return": float(rr),
                "Sampled annual nominal return": float(real_to_nominal_return(float(rr), annual_inflation)),
                "Final real value_today_money": float(last["Real value_today_money"]),
                "Final nominal account_value": float(last["Nominal account_value"]),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# Chart functions
# ============================================================


def plot_cape_history(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["CAPE"], mode="lines", name="CAPE"))
    fig.update_layout(
        title="Historical CAPE",
        xaxis_title="Date",
        yaxis_title="CAPE",
        height=420,
        hovermode="x unified",
    )
    return fig


def plot_forward_returns(df: pd.DataFrame, horizon_years: int) -> go.Figure:
    work = add_forward_real_returns(df, horizon_years)
    y_col = f"forward_{horizon_years}y_real_return_ann"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["Date"],
            y=work[y_col] * 100,
            mode="lines",
            name=f"Forward {horizon_years}Y real return",
        )
    )
    fig.update_layout(
        title=f"Historical Forward {horizon_years}-Year Annualised Real Returns",
        xaxis_title="Start date",
        yaxis_title="Annualised real return (%)",
        height=420,
        hovermode="x unified",
    )
    return fig


def plot_calibration(cal: CalibrationResult) -> go.Figure:
    fitted = cal.fitted.copy()
    fitted["y_pct"] = fitted["y"] * 100
    fitted["y_hat_pct"] = fitted["y_hat"] * 100

    fig = px.scatter(
        fitted,
        x="x",
        y="y_pct",
        labels={"x": cal.x_name, "y_pct": cal.y_name + " (%)"},
        title=f"CAPE Calibration: {cal.model_name}",
    )

    line_df = fitted.sort_values("x")
    fig.add_trace(
        go.Scatter(
            x=line_df["x"],
            y=line_df["y_hat_pct"],
            mode="lines",
            name="Fitted line",
        )
    )

    fig.update_layout(height=480)
    return fig


def plot_projection(path: pd.DataFrame) -> go.Figure:
    x = path["Year"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=path["Nominal account_value"],
            mode="lines",
            name="Nominal account value",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=path["Real value_today_money"],
            mode="lines",
            name="Real value in today's money",
        )
    )
    fig.update_layout(
        title="Projection: nominal vs real wealth",
        xaxis_title="Year",
        yaxis_title="Portfolio value",
        height=480,
        hovermode="x unified",
    )
    return fig


def plot_monte_carlo(mc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=mc["Final nominal account_value"],
            name="Nominal final value",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=mc["Final real value_today_money"],
            name="Real final value",
            opacity=0.75,
        )
    )
    fig.update_layout(
        title="Monte Carlo final wealth distribution",
        xaxis_title="Final value",
        yaxis_title="Count",
        barmode="overlay",
        height=480,
    )
    return fig


# ============================================================
# Formatting helpers
# ============================================================


def pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def money(x: float, currency: str = "£") -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{currency}{x:,.0f}"


def metric_card(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label, value, help=help_text)


# ============================================================
# Demo data
# ============================================================


def make_demo_data() -> pd.DataFrame:
    """
    Synthetic demonstration dataset only.
    The app is designed for uploaded Shiller/CAPE historical data.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("1950-01-01", "2025-12-01", freq="MS")
    n = len(dates)

    inflation_m = rng.normal(0.0025, 0.002, n).clip(-0.01, 0.02)
    cpi = 100 * np.cumprod(1 + inflation_m)

    real_return_m = rng.normal(0.005, 0.04, n)
    real_price = 100 * np.cumprod(1 + real_return_m)
    price = real_price * cpi / cpi[-1]

    # Synthetic earnings and CAPE-like cycle
    real_earnings = real_price / (18 + 8 * np.sin(np.linspace(0, 10 * np.pi, n)) + rng.normal(0, 2, n))
    earnings = real_earnings * cpi / cpi[-1]
    dividend = price * 0.018

    df = pd.DataFrame(
        {
            "Date": dates,
            "Price": price,
            "Earnings": earnings,
            "Dividend": dividend,
            "CPI": cpi,
        }
    )
    return df


# ============================================================
# Main app
# ============================================================


def main():
    st.title("📈 CAPE Real/Nominal Investment Studio")
    st.caption("CAPE calibration in real returns, then conversion to nominal wealth projections.")

    with st.expander("How this version treats CAPE, inflation, real returns, and nominal account value", expanded=True):
        st.markdown(THEORY_NOTE)

    st.sidebar.header("1) Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Shiller/CAPE historical data",
        type=["csv", "xlsx", "xls", "mat"],
        help="CSV/XLSX is recommended. The app also tries to read simple MATLAB .mat files.",
    )

    use_demo = st.sidebar.checkbox("Use synthetic demo data", value=uploaded_file is None)

    if uploaded_file is not None:
        try:
            raw_df = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Could not load file: {exc}")
            return
    elif use_demo:
        raw_df = make_demo_data()
        st.info("Using synthetic demo data. Upload Shiller/CAPE historical data for real analysis.")
    else:
        st.warning("Upload a data file or enable synthetic demo data.")
        return

    raw_df.columns = [clean_col_name(c) for c in raw_df.columns]
    inferred = infer_columns(raw_df.copy())

    st.sidebar.header("2) Column mapping")
    cols = [None] + list(raw_df.columns)

    def select_col(label: str, inferred_col: Optional[str]) -> Optional[str]:
        index = cols.index(inferred_col) if inferred_col in cols else 0
        selected = st.sidebar.selectbox(label, cols, index=index)
        return selected if selected is not None else None

    with st.sidebar.expander("Manual column mapping", expanded=False):
        date_col = select_col("Date column", inferred.date)
        price_col = select_col("Price column", inferred.price)
        earnings_col = select_col("Earnings column", inferred.earnings)
        dividend_col = select_col("Dividend column", inferred.dividend)
        cpi_col = select_col("CPI column", inferred.cpi)
        cape_col = select_col("CAPE column", inferred.cape)
        tri_col = select_col("Nominal total return index", inferred.total_return_index)
        real_tri_col = select_col("Real total return index", inferred.real_total_return_index)

    user_map = ColumnMap(
        date=date_col,
        price=price_col,
        earnings=earnings_col,
        dividend=dividend_col,
        cpi=cpi_col,
        cape=cape_col,
        total_return_index=tri_col,
        real_total_return_index=real_tri_col,
    )

    st.sidebar.header("3) CAPE settings")
    cape_window_months = st.sidebar.slider("CAPE earnings smoothing window, months", 60, 180, 120, step=12)
    assume_annual_dividend = st.sidebar.checkbox(
        "Dividend column is annualised", value=True,
        help="Shiller-style data usually reports dividend amount annualised. Monthly total-return approximation uses D/12."
    )

    try:
        df, cmap, notes = prepare_dataset(
            raw_df,
            user_col_map=user_map,
            cape_window_months=cape_window_months,
            assume_shiller_dividend_is_annual=assume_annual_dividend,
        )
    except Exception as exc:
        st.error(f"Could not prepare dataset: {exc}")
        st.stop()

    valid_cape = df["CAPE"].dropna()
    if valid_cape.empty:
        st.error("No valid CAPE observations were found.")
        st.stop()

    latest_row = df.dropna(subset=["CAPE"]).iloc[-1]
    latest_date = latest_row["Date"]
    latest_cape = float(latest_row["CAPE"])

    st.subheader("Dataset check")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Start date", df["Date"].min().strftime("%Y-%m"))
    with c2:
        metric_card("End date", df["Date"].max().strftime("%Y-%m"))
    with c3:
        metric_card("Latest CAPE", f"{latest_cape:.2f}")
    with c4:
        metric_card("Valid rows", f"{len(df.dropna(subset=['CAPE'])):,}")

    if notes:
        with st.expander("Data preparation notes", expanded=False):
            for note in notes:
                st.write(f"- {note}")
            st.write("Detected/selected columns:")
            st.json(cmap.__dict__)

    with st.expander("Preview prepared data", expanded=False):
        st.dataframe(df.tail(20), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_cape_history(df), use_container_width=True)
    with right:
        st.line_chart(
            df.set_index("Date")[["real_total_return_index"]].dropna(),
            use_container_width=True,
        )

    st.sidebar.header("4) Calibration")
    horizon_years = st.sidebar.slider("Forward return horizon, years", 5, 30, 10, step=1)
    predictor = st.sidebar.selectbox(
        "CAPE predictor",
        ["cape_yield", "cape", "log_cape"],
        index=0,
        format_func=lambda x: {"cape_yield": "1 / CAPE", "cape": "CAPE", "log_cape": "log(CAPE)"}[x],
    )

    try:
        cal = calibrate_cape_model(df, horizon_years=horizon_years, predictor=predictor)
        expected_real_return = predict_real_return_from_cape(latest_cape, cal)
    except Exception as exc:
        st.error(f"Calibration failed: {exc}")
        st.stop()

    st.subheader("CAPE calibration result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Model", cal.model_name)
    with c2:
        metric_card("Observations", f"{cal.n_obs:,}")
    with c3:
        metric_card("R²", f"{cal.r2:.3f}")
    with c4:
        metric_card("CAPE-implied real return", pct(expected_real_return), "This is an inflation-adjusted expected annual return.")

    st.plotly_chart(plot_calibration(cal), use_container_width=True)
    st.plotly_chart(plot_forward_returns(df, horizon_years), use_container_width=True)

    st.sidebar.header("5) Projection")
    currency_symbol = st.sidebar.selectbox("Currency symbol", ["£", "$", "€"], index=0)
    initial_amount = st.sidebar.number_input("Initial investment", min_value=0.0, value=10000.0, step=500.0)
    monthly_contribution = st.sidebar.number_input("Monthly contribution", min_value=0.0, value=500.0, step=50.0)
    projection_years = st.sidebar.slider("Projection horizon, years", 1, 50, 25)
    annual_inflation = st.sidebar.slider("Inflation assumption", -2.0, 10.0, 3.0, step=0.1) / 100.0
    contribution_growth = st.sidebar.checkbox("Increase monthly contribution with inflation", value=True)

    manual_real_return = st.sidebar.checkbox("Override CAPE-implied real return", value=False)
    if manual_real_return:
        expected_real_return = st.sidebar.slider("Manual real return", -10.0, 15.0, float(expected_real_return * 100), step=0.1) / 100.0

    expected_nominal_return = real_to_nominal_return(expected_real_return, annual_inflation)

    projection = project_monthly_contributions(
        initial_amount=initial_amount,
        monthly_contribution=monthly_contribution,
        annual_real_return=expected_real_return,
        annual_inflation=annual_inflation,
        years=projection_years,
        contribution_growth_with_inflation=contribution_growth,
    )

    final = projection.iloc[-1]

    st.subheader("Projection: real vs nominal")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Expected real return", pct(expected_real_return), "CAPE-calibrated return after inflation.")
    with c2:
        metric_card("Inflation assumption", pct(annual_inflation))
    with c3:
        metric_card("Expected nominal return", pct(expected_nominal_return), "Absolute expected account growth rate.")
    with c4:
        metric_card("Latest CAPE date", latest_date.strftime("%Y-%m"))

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Final nominal account value", money(final["Nominal account_value"], currency_symbol))
    with c2:
        metric_card("Final real value today-money", money(final["Real value_today_money"], currency_symbol))
    with c3:
        gap = final["Nominal account_value"] - final["Real value_today_money"]
        metric_card("Inflation component", money(gap, currency_symbol), "Difference between nominal account value and today's-money value.")

    st.plotly_chart(plot_projection(projection), use_container_width=True)

    with st.expander("Projection table", expanded=False):
        display_projection = projection.copy()
        display_projection["Nominal account_value"] = display_projection["Nominal account_value"].round(2)
        display_projection["Real value_today_money"] = display_projection["Real value_today_money"].round(2)
        st.dataframe(display_projection, use_container_width=True)

    st.subheader("Monte Carlo using historical CAPE-conditioned real returns")
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        matching_strength = st.slider(
            "CAPE matching band",
            0.05,
            1.00,
            0.35,
            step=0.05,
            help="0.35 means use historical observations where CAPE was within ±35% of latest CAPE. If too few observations exist, the app uses all observations.",
        )
    with mc_col2:
        n_sims = st.slider("Number of simulations", 100, 10000, 2000, step=100)
    with mc_col3:
        random_seed = st.number_input("Random seed", value=42, step=1)

    empirical_returns = empirical_real_return_distribution(
        df,
        horizon_years=horizon_years,
        cape_now=latest_cape,
        matching_strength=matching_strength,
        min_obs=50,
    )

    if empirical_returns.empty:
        st.warning("No empirical returns available for Monte Carlo.")
    else:
        mc = run_monte_carlo_projection(
            initial_amount=initial_amount,
            monthly_contribution=monthly_contribution,
            years=projection_years,
            real_return_samples=empirical_returns.values,
            annual_inflation=annual_inflation,
            n_sims=n_sims,
            contribution_growth_with_inflation=contribution_growth,
            random_seed=int(random_seed),
        )

        q = mc[["Final nominal account_value", "Final real value_today_money"]].quantile([0.1, 0.5, 0.9])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Historical return samples", f"{len(empirical_returns):,}")
        with c2:
            metric_card("Median nominal", money(q.loc[0.5, "Final nominal account_value"], currency_symbol))
        with c3:
            metric_card("Median real", money(q.loc[0.5, "Final real value_today_money"], currency_symbol))
        with c4:
            metric_card("Median sampled real return", pct(mc["Sampled annual real return"].median()))

        st.plotly_chart(plot_monte_carlo(mc), use_container_width=True)

        summary = pd.DataFrame(
            {
                "Metric": ["10th percentile", "Median", "90th percentile"],
                "Nominal final account value": [
                    q.loc[0.1, "Final nominal account_value"],
                    q.loc[0.5, "Final nominal account_value"],
                    q.loc[0.9, "Final nominal account_value"],
                ],
                "Real final value today-money": [
                    q.loc[0.1, "Final real value_today_money"],
                    q.loc[0.5, "Final real value_today_money"],
                    q.loc[0.9, "Final real value_today_money"],
                ],
            }
        )
        st.dataframe(summary, use_container_width=True)

        with st.expander("Monte Carlo raw output", expanded=False):
            st.dataframe(mc, use_container_width=True)

    st.subheader("Key interpretation")
    st.markdown(
        f"""
- The CAPE model currently estimates approximately **{pct(expected_real_return)} real annual return**.
- With your inflation assumption of **{pct(annual_inflation)}**, that becomes approximately **{pct(expected_nominal_return)} nominal annual return**.
- Therefore, the **absolute account value** can grow faster than the CAPE real-return number, but the difference is mainly inflation.
- For long-term planning, the **real value** is the better measure of actual purchasing-power improvement.
- For account balance targets, tax wrappers, pension pots, or nominal targets such as £200k/£400k, the **nominal value** is the number you will see on the account.
"""
    )

    st.caption(
        "Important: This is a historical valuation-based model, not financial advice. CAPE has long-horizon statistical usefulness but poor short-term timing ability."
    )


if __name__ == "__main__":
    main()
