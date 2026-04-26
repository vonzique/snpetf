from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Streamlit page setup
# ============================================================
st.set_page_config(
    page_title="S&P 500 Investment analysis by Dimi",
    page_icon="📈",
    layout="wide",
)

st.title("S&P 500 Investment analysis by Dimi")
st.caption(
    "Long-term S&P 500 projection tool using historical returns and optional CAPE-based real-return conditioning."
)


# ============================================================
# Core utility functions
# ============================================================

def annual_to_monthly_return(annual_return: float) -> float:
    """Convert annual return to equivalent monthly compounded return."""
    return (1.0 + annual_return) ** (1.0 / 12.0) - 1.0


def monthly_to_annual_return(monthly_return: float) -> float:
    """Convert monthly return to equivalent annual compounded return."""
    return (1.0 + monthly_return) ** 12.0 - 1.0


def real_to_nominal_return(real_return: float | np.ndarray, inflation_rate: float | np.ndarray) -> float | np.ndarray:
    """
    Convert real return to nominal return.

    Formula:
        (1 + nominal) = (1 + real) * (1 + inflation)
    """
    return (1.0 + real_return) * (1.0 + inflation_rate) - 1.0


def nominal_to_real_return(nominal_return: float | np.ndarray, inflation_rate: float | np.ndarray) -> float | np.ndarray:
    """
    Convert nominal return to real return.

    Formula:
        (1 + real) = (1 + nominal) / (1 + inflation)
    """
    return (1.0 + nominal_return) / (1.0 + inflation_rate) - 1.0


def compute_cape(
    price: pd.Series,
    earnings: pd.Series,
    cpi: pd.Series,
    window_months: int = 120,
) -> pd.Series:
    """
    Compute Shiller-style CAPE using real price divided by 10-year average real earnings.

    price: nominal price index
    earnings: nominal earnings
    cpi: CPI index
    window_months: default 120 months = 10 years

    CAPE is a valuation metric, so price and earnings are made real before comparison.
    """
    price = pd.to_numeric(price, errors="coerce")
    earnings = pd.to_numeric(earnings, errors="coerce")
    cpi = pd.to_numeric(cpi, errors="coerce")

    cpi_base = cpi.dropna().iloc[-1]
    real_price = price / cpi * cpi_base
    real_earnings = earnings / cpi * cpi_base
    avg_real_earnings = real_earnings.rolling(window_months, min_periods=window_months).mean()

    cape = real_price / avg_real_earnings
    return cape.replace([np.inf, -np.inf], np.nan)


def estimate_real_return_from_cape(
    cape: float,
    alpha: float = 0.015,
    beta: float = 1.10,
    floor_return: float = -0.03,
    cap_return: float = 0.12,
) -> float:
    """
    Simple CAPE-to-real-return mapping.

    This uses CAPE yield, 1/CAPE, as the predictor:
        expected_real_return = alpha + beta * (1 / CAPE)

    Default values are deliberately conservative placeholders.
    For serious use, calibrate alpha/beta using historical out-of-sample data.
    """
    if cape <= 0 or np.isnan(cape):
        return np.nan

    expected = alpha + beta * (1.0 / cape)
    return float(np.clip(expected, floor_return, cap_return))


def safe_percent(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100 * x:.2f}%"


# ============================================================
# Data loading
# ============================================================

def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV and make best-effort date parsing."""
    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]

    date_candidates = [
        c for c in df.columns
        if c.lower() in ["date", "time", "month", "year", "datetime"]
        or "date" in c.lower()
        or "time" in c.lower()
    ]

    if date_candidates:
        date_col = date_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).set_index(date_col)

    return df


def infer_return_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Infer a monthly return series from common column names."""
    lower_map = {c.lower(): c for c in df.columns}

    for name in ["return", "returns", "monthly_return", "monthly returns", "real_return", "real return"]:
        if name in lower_map:
            s = pd.to_numeric(df[lower_map[name]], errors="coerce").dropna()
            # If returns look like percentages, convert to decimals.
            if s.abs().median() > 1:
                s = s / 100.0
            return s

    for name in ["price", "close", "sp500", "s&p 500", "snp", "index"]:
        if name in lower_map:
            price = pd.to_numeric(df[lower_map[name]], errors="coerce").dropna()
            return price.pct_change().dropna()

    return None


def infer_cape_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Infer CAPE if directly available, otherwise compute if Price/Earnings/CPI exist."""
    lower_map = {c.lower(): c for c in df.columns}

    for name in ["cape", "shiller cape", "cape ratio", "cyclically adjusted pe"]:
        if name in lower_map:
            return pd.to_numeric(df[lower_map[name]], errors="coerce")

    price_col = None
    earnings_col = None
    cpi_col = None

    for candidate in ["price", "real price", "p", "sp500", "s&p 500", "index"]:
        if candidate in lower_map:
            price_col = lower_map[candidate]
            break

    for candidate in ["earnings", "e", "eps", "trailing earnings"]:
        if candidate in lower_map:
            earnings_col = lower_map[candidate]
            break

    for candidate in ["cpi", "consumer price index"]:
        if candidate in lower_map:
            cpi_col = lower_map[candidate]
            break

    if price_col and earnings_col and cpi_col:
        return compute_cape(df[price_col], df[earnings_col], df[cpi_col])

    return None


# ============================================================
# Simulation engine
# ============================================================

@dataclass
class SimulationConfig:
    starting_wealth: float
    monthly_contribution: float
    years: int
    n_sims: int
    expected_real_return_annual: float
    volatility_annual: float
    expected_inflation_annual: float
    projection_mode: str
    seed: int


def simulate_monthly_wealth(config: SimulationConfig) -> pd.DataFrame:
    """
    Monte Carlo wealth simulation.

    expected_real_return_annual is treated as a REAL return.
    If projection_mode is 'Nominal values', inflation is added back before wealth projection.
    """
    rng = np.random.default_rng(config.seed)
    n_months = config.years * 12

    real_mu_monthly = annual_to_monthly_return(config.expected_real_return_annual)
    real_sigma_monthly = config.volatility_annual / np.sqrt(12.0)
    monthly_inflation = annual_to_monthly_return(config.expected_inflation_annual)

    simulated_real_returns = rng.normal(
        loc=real_mu_monthly,
        scale=real_sigma_monthly,
        size=(config.n_sims, n_months),
    )

    # Avoid impossible returns below -100%.
    simulated_real_returns = np.clip(simulated_real_returns, -0.95, None)

    if config.projection_mode == "Nominal values":
        simulated_returns = real_to_nominal_return(simulated_real_returns, monthly_inflation)
    else:
        simulated_returns = simulated_real_returns

    wealth = np.zeros((config.n_sims, n_months + 1), dtype=float)
    wealth[:, 0] = config.starting_wealth

    for m in range(1, n_months + 1):
        wealth[:, m] = wealth[:, m - 1] * (1.0 + simulated_returns[:, m - 1]) + config.monthly_contribution

    months = np.arange(n_months + 1)
    summary = pd.DataFrame({
        "Month": months,
        "Year": months / 12.0,
        "P10": np.percentile(wealth, 10, axis=0),
        "P25": np.percentile(wealth, 25, axis=0),
        "Median": np.percentile(wealth, 50, axis=0),
        "Mean": np.mean(wealth, axis=0),
        "P75": np.percentile(wealth, 75, axis=0),
        "P90": np.percentile(wealth, 90, axis=0),
    })

    return summary


def plot_wealth(summary: pd.DataFrame, projection_mode: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=summary["Year"],
        y=summary["P90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=summary["Year"],
        y=summary["P10"],
        mode="lines",
        fill="tonexty",
        name="10th-90th percentile range",
        line=dict(width=0),
    ))

    fig.add_trace(go.Scatter(
        x=summary["Year"],
        y=summary["Median"],
        mode="lines",
        name="Median",
        line=dict(width=3),
    ))

    fig.add_trace(go.Scatter(
        x=summary["Year"],
        y=summary["Mean"],
        mode="lines",
        name="Mean",
        line=dict(width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"Projected wealth ({projection_mode})",
        xaxis_title="Years",
        yaxis_title="Portfolio value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Inputs")

starting_wealth = st.sidebar.number_input(
    "Starting wealth",
    min_value=0.0,
    value=10_000.0,
    step=500.0,
)

monthly_contribution = st.sidebar.number_input(
    "Monthly contribution",
    min_value=0.0,
    value=500.0,
    step=50.0,
)

years = st.sidebar.slider("Projection horizon (years)", 1, 50, 25)
n_sims = st.sidebar.slider("Monte Carlo simulations", 100, 20_000, 5_000, step=100)

st.sidebar.header("Return model")

model_mode = st.sidebar.radio(
    "Expected return source",
    ["Manual real return", "CAPE-based real return"],
    index=0,
)

manual_real_return_annual = st.sidebar.number_input(
    "Manual expected annual real return (%)",
    min_value=-10.0,
    max_value=20.0,
    value=4.0,
    step=0.1,
) / 100.0

volatility_annual = st.sidebar.number_input(
    "Annual volatility (%)",
    min_value=1.0,
    max_value=80.0,
    value=15.0,
    step=0.5,
) / 100.0

st.sidebar.header("Inflation / nominal projection")

projection_mode = st.sidebar.radio(
    "Projection mode",
    ["Real values", "Nominal values"],
    index=0,
    help="Real values show purchasing-power-adjusted results. Nominal values show estimated future account balances before inflation adjustment.",
)

expected_inflation_annual = st.sidebar.number_input(
    "Expected annual inflation (%)",
    min_value=-5.0,
    max_value=20.0,
    value=2.5,
    step=0.1,
) / 100.0

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=999_999, value=42, step=1)


# ============================================================
# Optional data upload
# ============================================================
st.subheader("Historical data / CAPE input")

uploaded_file = st.file_uploader(
    "Optional: upload CSV with returns, CAPE, or Price/Earnings/CPI columns",
    type=["csv"],
)

cape_value = np.nan
historical_monthly_returns = None
uploaded_df = None

if uploaded_file is not None:
    uploaded_df = load_uploaded_csv(uploaded_file)
    st.success("CSV loaded successfully.")

    with st.expander("Preview uploaded data"):
        st.dataframe(uploaded_df.head(20), use_container_width=True)

    historical_monthly_returns = infer_return_series(uploaded_df)
    cape_series = infer_cape_series(uploaded_df)

    if cape_series is not None and cape_series.dropna().shape[0] > 0:
        cape_value = float(cape_series.dropna().iloc[-1])
        st.metric("Latest inferred CAPE", f"{cape_value:.2f}")
    else:
        st.warning("No CAPE column found and CAPE could not be computed from Price/Earnings/CPI columns.")

    if historical_monthly_returns is not None and historical_monthly_returns.dropna().shape[0] > 12:
        hist_ann_return = monthly_to_annual_return(historical_monthly_returns.mean())
        hist_ann_vol = historical_monthly_returns.std() * np.sqrt(12.0)
        col_a, col_b = st.columns(2)
        col_a.metric("Historical annualised return", safe_percent(hist_ann_return))
        col_b.metric("Historical annualised volatility", safe_percent(hist_ann_vol))
    else:
        st.info("No usable return or price series was inferred from the uploaded file.")
else:
    st.info(
        "You can run the app without data using manual assumptions, or upload a CSV containing either returns, CAPE, or Price/Earnings/CPI."
    )


# ============================================================
# CAPE calibration controls
# ============================================================
if model_mode == "CAPE-based real return":
    st.subheader("CAPE-based real return model")

    col1, col2, col3 = st.columns(3)
    with col1:
        cape_manual = st.number_input(
            "Current CAPE if not uploaded",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=0.5,
        )
    with col2:
        cape_alpha = st.number_input(
            "CAPE model alpha",
            min_value=-0.10,
            max_value=0.10,
            value=0.015,
            step=0.001,
            format="%.3f",
        )
    with col3:
        cape_beta = st.number_input(
            "CAPE model beta",
            min_value=0.0,
            max_value=5.0,
            value=1.10,
            step=0.05,
        )

    effective_cape = cape_value if not pd.isna(cape_value) else cape_manual
    expected_real_return_annual = estimate_real_return_from_cape(
        effective_cape,
        alpha=cape_alpha,
        beta=cape_beta,
    )

    st.info(
        "CAPE-based expected returns are treated as real returns. "
        "If you select nominal values, the app adds expected inflation back to estimate the future account balance."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("CAPE used", f"{effective_cape:.2f}")
    col2.metric("Expected real annual return", safe_percent(expected_real_return_annual))
    col3.metric(
        "Expected nominal annual return",
        safe_percent(real_to_nominal_return(expected_real_return_annual, expected_inflation_annual)),
    )
else:
    expected_real_return_annual = manual_real_return_annual


# ============================================================
# Run simulation
# ============================================================
st.subheader("Projection")

if projection_mode == "Real values":
    st.caption("Results are shown in today's purchasing-power terms.")
else:
    st.caption("Results are shown as nominal future account balances before inflation adjustment.")

config = SimulationConfig(
    starting_wealth=starting_wealth,
    monthly_contribution=monthly_contribution,
    years=years,
    n_sims=n_sims,
    expected_real_return_annual=expected_real_return_annual,
    volatility_annual=volatility_annual,
    expected_inflation_annual=expected_inflation_annual,
    projection_mode=projection_mode,
    seed=int(seed),
)

summary = simulate_monthly_wealth(config)
final_row = summary.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final median", f"{final_row['Median']:,.0f}")
col2.metric("Final mean", f"{final_row['Mean']:,.0f}")
col3.metric("Final P10", f"{final_row['P10']:,.0f}")
col4.metric("Final P90", f"{final_row['P90']:,.0f}")

fig = plot_wealth(summary, projection_mode)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Projection summary table"):
    display_summary = summary.copy()
    display_summary["Year"] = display_summary["Year"].round(2)
    st.dataframe(display_summary, use_container_width=True)


# ============================================================
# Download outputs
# ============================================================
csv_buffer = io.StringIO()
summary.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download projection summary CSV",
    data=csv_buffer.getvalue(),
    file_name="sp500_projection_summary.csv",
    mime="text/csv",
)


# ============================================================
# Methodology notes
# ============================================================
with st.expander("Methodology notes"):
    st.markdown(
        """
### CAPE logic

CAPE is a valuation ratio. It compares real price against the 10-year average of real earnings.
That means CAPE is normally used to estimate **real returns**, not nominal account balances.

### Real vs nominal projection

If the CAPE model gives a 4% annual return, that should usually be interpreted as approximately
4% after inflation.

To estimate nominal account growth, this app uses:

```text
(1 + nominal return) = (1 + real return) × (1 + inflation)
```

Example:

```text
Real return = 4%
Inflation = 2.5%
Nominal return = (1.04 × 1.025) - 1 = 6.60%
```

### Important limitation

The default CAPE model here is a transparent placeholder:

```text
Expected real return = alpha + beta × (1 / CAPE)
```

For high-confidence use, calibrate alpha and beta using historical out-of-sample testing.
        """
    )
