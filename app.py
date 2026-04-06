
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pricing import BlackScholes, Contract, Market


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Option Portfolio Visualizer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚙️ Black–Scholes Portfolio Visualizer")
st.caption("Build arbitrary stock / option portfolios, inspect Greeks, and visualize payoff + sensitivities.")


# =========================================================
# Engine + registry
# =========================================================
bs = BlackScholes()
greeks = list(bs.greeks())
greek_keys = [g.key for g in greeks]
greek_by_key = {g.key: g for g in greeks}
METRICS_ALL = ["price"] + greek_keys


# =========================================================
# Session state helpers
# =========================================================
def ensure_state() -> None:
    if "portfolio_legs" not in st.session_state:
        st.session_state.portfolio_legs = [
            {
                "name": "Underlying",
                "type": "stock",
                "qty": 100.0,
            },
            {
                "name": "Short Call",
                "type": "call",
                "qty": -1.0,
                "K": 105.0,
                "T": 0.25,
            },
        ]


def default_leg(index: int) -> dict[str, Any]:
    return {
        "name": f"Leg {index + 1}",
        "type": "call",
        "qty": 1.0,
        "K": 100.0,
        "T": 0.25,
    }


def add_leg(leg: dict[str, Any] | None = None) -> None:
    legs = st.session_state.portfolio_legs
    legs.append(copy.deepcopy(leg) if leg is not None else default_leg(len(legs)))


def delete_leg(idx: int) -> None:
    legs = st.session_state.portfolio_legs
    if len(legs) > 1:
        del legs[idx]


def set_portfolio(legs: list[dict[str, Any]]) -> None:
    st.session_state.portfolio_legs = copy.deepcopy(legs)


ensure_state()


# =========================================================
# Strategy presets
# =========================================================
PRESET_STRATEGIES: dict[str, list[dict[str, Any]]] = {
    "Covered Call": [
        {"name": "Long Stock", "type": "stock", "qty": 100.0},
        {"name": "Short Call", "type": "call", "qty": -1.0, "K": 105.0, "T": 0.25},
    ],
    "Bull Call Spread": [
        {"name": "Long Lower Call", "type": "call", "qty": 1.0, "K": 95.0, "T": 0.25},
        {"name": "Short Upper Call", "type": "call", "qty": -1.0, "K": 105.0, "T": 0.25},
    ],
    "Long Straddle": [
        {"name": "Long Call", "type": "call", "qty": 1.0, "K": 100.0, "T": 0.25},
        {"name": "Long Put", "type": "put", "qty": 1.0, "K": 100.0, "T": 0.25},
    ],
    "Long Strangle": [
        {"name": "Long Put", "type": "put", "qty": 1.0, "K": 95.0, "T": 0.25},
        {"name": "Long Call", "type": "call", "qty": 1.0, "K": 105.0, "T": 0.25},
    ],
    "Iron Condor": [
        {"name": "Long Put Wing", "type": "put", "qty": 1.0, "K": 90.0, "T": 0.25},
        {"name": "Short Put", "type": "put", "qty": -1.0, "K": 95.0, "T": 0.25},
        {"name": "Short Call", "type": "call", "qty": -1.0, "K": 105.0, "T": 0.25},
        {"name": "Long Call Wing", "type": "call", "qty": 1.0, "K": 110.0, "T": 0.25},
    ],
}


# =========================================================
# Pricing helpers
# =========================================================
def stock_metric(greek_key: str, market: Market) -> float:
    """
    Return the chosen metric for one share of stock.
    """
    if greek_key == "price":
        return float(market.S)
    if greek_key == "delta":
        return 1.0
    return 0.0


def stock_payoff_at_expiry(spot_at_expiry: float) -> float:
    return float(spot_at_expiry)


def option_payoff_at_expiry(option_type: str, K: float, spot_at_expiry: float) -> float:
    if option_type == "call":
        return max(float(spot_at_expiry) - float(K), 0.0)
    if option_type == "put":
        return max(float(K) - float(spot_at_expiry), 0.0)
    raise KeyError(option_type)


def leg_metric(greek_key: str, leg: dict[str, Any], market: Market) -> float:
    """
    Return quantity-adjusted Black-Scholes metric for a single leg.
    """
    leg_type = leg["type"]
    qty = float(leg["qty"])

    if leg_type == "stock":
        return qty * stock_metric(greek_key, market)

    c = Contract(
        K=float(leg["K"]),
        T=max(float(leg["T"]), 1e-12),
        option_type=leg_type,
    )
    spec = bs.price if greek_key == "price" else greek_by_key[greek_key]
    return qty * float(bs.metric(spec, c, market))


def portfolio_metric(greek_key: str, portfolio_legs: list[dict[str, Any]], market: Market) -> float:
    return float(sum(leg_metric(greek_key, leg, market) for leg in portfolio_legs))


def leg_payoff_at_expiry(leg: dict[str, Any], spot_at_expiry: float) -> float:
    qty = float(leg["qty"])
    leg_type = leg["type"]

    if leg_type == "stock":
        return qty * stock_payoff_at_expiry(spot_at_expiry)

    return qty * option_payoff_at_expiry(leg_type, float(leg["K"]), spot_at_expiry)


def portfolio_payoff_at_expiry(portfolio_legs: list[dict[str, Any]], spot_at_expiry: float) -> float:
    return float(sum(leg_payoff_at_expiry(leg, spot_at_expiry) for leg in portfolio_legs))


def unique_maturities(legs: list[dict[str, Any]]) -> list[float]:
    ts = sorted({float(leg["T"]) for leg in legs if leg["type"] != "stock"})
    return ts if ts else [0.25]


def build_plot_targets(legs: list[dict[str, Any]]) -> list[str]:
    """
    Global market variables + per-leg K/T selectors.
    """
    targets = ["S", "sigma", "r", "q"]
    for i, leg in enumerate(legs):
        name = leg.get("name", f"Leg {i + 1}") or f"Leg {i + 1}"
        if leg["type"] != "stock":
            targets.append(f"leg_{i + 1}_K ({name})")
            targets.append(f"leg_{i + 1}_T ({name})")
    return targets


def apply_plot_variable(
    market: Market,
    portfolio_legs: list[dict[str, Any]],
    var: str,
    value: float,
) -> tuple[Market, list[dict[str, Any]]]:
    """
    Return a new (market, legs) pair after applying one plot variable change.
    """
    new_market = market
    new_legs = [dict(leg) for leg in portfolio_legs]

    if var == "S":
        new_market = market.with_(S=float(value))
    elif var == "sigma":
        new_market = market.with_(sigma=max(float(value), 0.0))
    elif var == "r":
        new_market = market.with_(r=float(value))
    elif var == "q":
        new_market = market.with_(q=float(value))
    elif var.startswith("leg_"):
        prefix, field_part = var.split("_", 2)
        # field_part looks like "1_K (Name)" or "2_T (Name)"
        leg_idx_str, rest = field_part.split("_", 1)
        leg_idx = int(leg_idx_str) - 1
        field = rest.split(" ", 1)[0]  # "K" or "T"
        if field not in {"K", "T"}:
            raise KeyError(var)
        if not (0 <= leg_idx < len(new_legs)):
            raise IndexError(var)
        if new_legs[leg_idx]["type"] == "stock":
            return new_market, new_legs
        if field == "K":
            new_legs[leg_idx]["K"] = max(float(value), 1e-12)
        else:
            new_legs[leg_idx]["T"] = max(float(value), 0.0)
    else:
        raise KeyError(var)

    return new_market, new_legs


@st.cache_data(show_spinner=False)
def compute_metric_grid(
    greek_key: str,
    portfolio_legs: list[dict[str, Any]],
    S: float,
    sigma: float,
    r: float,
    q: float,
    mode: str,
    x_var: str,
    x_min: float,
    x_max: float,
    n: int,
    y_var: str | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    n2: int | None = None,
):
    base_market = Market(S=float(S), r=float(r), sigma=float(sigma), q=float(q))
    xs = np.linspace(float(x_min), float(x_max), int(n))

    if mode == "1D line":
        ys = np.empty_like(xs, dtype=float)
        for i, xv in enumerate(xs):
            m_i, legs_i = apply_plot_variable(base_market, portfolio_legs, x_var, float(xv))
            ys[i] = portfolio_metric(greek_key, legs_i, m_i)
        return xs, ys, None

    assert y_var is not None and y_min is not None and y_max is not None and n2 is not None
    ys = np.linspace(float(y_min), float(y_max), int(n2))
    Z = np.zeros((len(ys), len(xs)), dtype=float)

    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            m1, legs1 = apply_plot_variable(base_market, portfolio_legs, x_var, float(xv))
            m2, legs2 = apply_plot_variable(m1, legs1, y_var, float(yv))
            Z[j, i] = portfolio_metric(greek_key, legs2, m2)

    return xs, ys, Z


@st.cache_data(show_spinner=False)
def compute_payoff_curve(
    portfolio_legs: list[dict[str, Any]],
    x_min: float,
    x_max: float,
    n: int,
    premium: float,
):
    xs = np.linspace(float(x_min), float(x_max), int(n))
    gross = np.array([portfolio_payoff_at_expiry(portfolio_legs, x) for x in xs], dtype=float)
    pnl = gross - float(premium)
    return xs, gross, pnl


# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.subheader("Portfolio Builder")

    with st.expander("Strategy presets", expanded=True):
        preset_names = ["Custom"] + list(PRESET_STRATEGIES.keys())
        selected_preset = st.selectbox("Load preset", preset_names, index=0)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            if st.button("Apply preset", use_container_width=True, disabled=(selected_preset == "Custom")):
                set_portfolio(PRESET_STRATEGIES[selected_preset])
                st.rerun()
        with c_p2:
            if st.button("Add new leg", use_container_width=True):
                add_leg()
                st.rerun()

    with st.expander("Portfolio legs", expanded=True):
        legs = st.session_state.portfolio_legs

        for i, leg in enumerate(legs):
            st.markdown(f"**Leg {i + 1}**")
            c_top1, c_top2 = st.columns([3.0, 1.1])

            with c_top1:
                leg["name"] = st.text_input(
                    f"Name #{i + 1}",
                    value=leg.get("name", f"Leg {i + 1}"),
                    key=f"name_{i}",
                )

            with c_top2:
                delete_disabled = len(legs) == 1
                if st.button("Delete", key=f"delete_{i}", use_container_width=True, disabled=delete_disabled):
                    delete_leg(i)
                    st.rerun()

            c1, c2 = st.columns(2)
            with c1:
                current_type = ["stock", "call", "put"].index(leg["type"])
                leg["type"] = st.selectbox(
                    f"Type #{i + 1}",
                    ["stock", "call", "put"],
                    index=current_type,
                    key=f"type_{i}",
                )
            with c2:
                leg["qty"] = float(
                    st.number_input(
                        f"Quantity #{i + 1}",
                        value=float(leg.get("qty", 1.0)),
                        step=1.0,
                        key=f"qty_{i}",
                        help="Use negative quantity for short positions.",
                    )
                )

            if leg["type"] != "stock":
                leg.setdefault("K", 100.0)
                leg.setdefault("T", 0.25)
                c3, c4 = st.columns(2)
                with c3:
                    leg["K"] = float(
                        st.number_input(
                            f"Strike K #{i + 1}",
                            min_value=0.0001,
                            value=float(leg.get("K", 100.0)),
                            step=1.0,
                            key=f"K_{i}",
                        )
                    )
                with c4:
                    leg["T"] = float(
                        st.number_input(
                            f"Expiry T #{i + 1}",
                            min_value=0.0,
                            value=float(leg.get("T", 0.25)),
                            step=0.01,
                            key=f"T_{i}",
                        )
                    )
            else:
                leg.pop("K", None)
                leg.pop("T", None)

            st.markdown("---")

    with st.expander("Market", expanded=True):
        S = st.number_input("Spot (S)", value=100.0, min_value=0.0001, step=1.0)
        sigma = st.number_input("Vol (σ)", value=0.20, min_value=0.0, step=0.01, format="%.4f")
        r = st.number_input("Rate (r)", value=0.03, step=0.005, format="%.4f")
        q = st.number_input("Dividend (q)", value=0.00, step=0.005, format="%.4f")

    plot_targets = build_plot_targets(st.session_state.portfolio_legs)

    with st.expander("Sensitivity plot", expanded=True):
        greek_key = st.selectbox(
            "Metric",
            METRICS_ALL,
            index=METRICS_ALL.index("delta") if "delta" in METRICS_ALL else 0,
        )
        mode = st.segmented_control("Mode", options=["1D line", "3D surface"], default="1D line")

        c5, c6 = st.columns(2)
        with c5:
            x_var = st.selectbox("X variable", plot_targets, index=0)
        with c6:
            n_metric = st.slider("X points", min_value=25, max_value=400, value=160)

        c7, c8 = st.columns(2)
        with c7:
            x_min_metric = st.number_input("X min", value=50.0, step=1.0)
        with c8:
            x_max_metric = st.number_input("X max", value=150.0, step=1.0)

        if mode == "3D surface":
            remaining_targets = [v for v in plot_targets if v != x_var]
            c9, c10 = st.columns(2)
            with c9:
                y_var = st.selectbox("Y variable", remaining_targets, index=0)
            with c10:
                n2 = st.slider("Y points", min_value=20, max_value=250, value=90)

            c11, c12 = st.columns(2)
            with c11:
                y_min = st.number_input("Y min", value=0.05 if "T" in y_var or y_var == "sigma" else 50.0, step=0.01, format="%.4f")
            with c12:
                y_max = st.number_input("Y max", value=0.60 if "T" in y_var or y_var == "sigma" else 150.0, step=0.01, format="%.4f")
        else:
            y_var = None
            y_min = None
            y_max = None
            n2 = None

    with st.expander("Payoff plot", expanded=True):
        payoff_x_min = st.number_input("Payoff spot min", value=50.0, step=1.0)
        payoff_x_max = st.number_input("Payoff spot max", value=150.0, step=1.0)
        payoff_n = st.slider("Payoff points", min_value=25, max_value=500, value=220)
        show_gross = st.checkbox("Show gross payoff", value=True)
        show_pnl = st.checkbox("Show net P&L (subtract premium)", value=True)

    st.markdown("---")
    st.caption("Tip: choose leg-specific variables like `leg_2_K (...)` to vary only one option leg.")


# =========================================================
# Base valuation + exposures
# =========================================================
portfolio_legs = copy.deepcopy(st.session_state.portfolio_legs)
base_market = Market(S=float(S), r=float(r), sigma=float(sigma), q=float(q))
portfolio_price = portfolio_metric("price", portfolio_legs, base_market)

exposure_rows: list[dict[str, Any]] = []
for metric in METRICS_ALL:
    exposure_rows.append(
        {
            "Metric": metric,
            "Value": portfolio_metric(metric, portfolio_legs, base_market),
        }
    )

exposure_df = pd.DataFrame(exposure_rows)


# =========================================================
# Top KPIs
# =========================================================
primary_metric_val = portfolio_metric(greek_key, portfolio_legs, base_market)

k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1, 1, 1])
k1.metric("Portfolio Price", f"{portfolio_price:.6f}")
k2.metric(greek_key, f"{primary_metric_val:.6f}")
k3.metric("S", f"{S:.4f}")
k4.metric("σ", f"{sigma:.4f}")
k5.metric("r", f"{r:.4f}")

st.divider()


# =========================================================
# Main layout
# =========================================================
left_col, right_col = st.columns([3.3, 1.35], gap="large")

with left_col:
    tab_metric, tab_payoff = st.tabs(["Sensitivity", "Payoff at Expiry"])

    with tab_metric:
        with st.spinner("Computing sensitivity grid..."):
            xs, ys, Z = compute_metric_grid(
                greek_key=greek_key,
                portfolio_legs=portfolio_legs,
                S=float(S),
                sigma=float(sigma),
                r=float(r),
                q=float(q),
                mode=mode,
                x_var=x_var,
                x_min=float(x_min_metric),
                x_max=float(x_max_metric),
                n=int(n_metric),
                y_var=y_var,
                y_min=None if y_min is None else float(y_min),
                y_max=None if y_max is None else float(y_max),
                n2=None if n2 is None else int(n2),
            )

        if mode == "1D line":
            fig_metric = go.Figure()
            fig_metric.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=greek_key,
                    hovertemplate=f"{x_var}=%{{x:.6f}}<br>{greek_key}=%{{y:.6f}}<extra></extra>",
                )
            )
            fig_metric.update_layout(
                title=f"Portfolio {greek_key} vs {x_var}",
                xaxis_title=x_var,
                yaxis_title=greek_key,
                hovermode="x",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            fig_metric.update_xaxes(showgrid=True)
            fig_metric.update_yaxes(showgrid=True)
            st.plotly_chart(fig_metric, use_container_width=True, config={"scrollZoom": True})
        else:
            fig_metric = go.Figure(
                data=go.Surface(
                    x=xs,
                    y=ys,
                    z=Z,
                    colorbar=dict(title=greek_key),
                    hovertemplate=f"{x_var}=%{{x:.6f}}<br>{y_var}=%{{y:.6f}}<br>{greek_key}=%{{z:.6f}}<extra></extra>",
                )
            )
            fig_metric.update_layout(
                title=f"Portfolio {greek_key} — 3D surface",
                scene=dict(
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    zaxis_title=greek_key,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    zaxis=dict(showgrid=True),
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_metric, use_container_width=True, config={"scrollZoom": True})

    with tab_payoff:
        with st.spinner("Computing payoff curve..."):
            pxs, gross_payoff, net_pnl = compute_payoff_curve(
                portfolio_legs=portfolio_legs,
                x_min=float(payoff_x_min),
                x_max=float(payoff_x_max),
                n=int(payoff_n),
                premium=float(portfolio_price),
            )

        fig_payoff = go.Figure()
        if show_gross:
            fig_payoff.add_trace(
                go.Scatter(
                    x=pxs,
                    y=gross_payoff,
                    mode="lines",
                    name="Gross payoff",
                    hovertemplate="Spot=%{x:.6f}<br>Gross payoff=%{y:.6f}<extra></extra>",
                )
            )
        if show_pnl:
            fig_payoff.add_trace(
                go.Scatter(
                    x=pxs,
                    y=net_pnl,
                    mode="lines",
                    name="Net P&L",
                    hovertemplate="Spot=%{x:.6f}<br>Net P&L=%{y:.6f}<extra></extra>",
                )
            )

        fig_payoff.add_hline(y=0.0)
        fig_payoff.update_layout(
            title="Portfolio payoff at expiry",
            xaxis_title="Spot at expiry",
            yaxis_title="Value",
            hovermode="x",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig_payoff.update_xaxes(showgrid=True)
        fig_payoff.update_yaxes(showgrid=True)
        st.plotly_chart(fig_payoff, use_container_width=True, config={"scrollZoom": True})

with right_col:
    st.subheader("Snapshot")

    st.write("**Portfolio legs**")
    portfolio_lines = []
    for i, leg in enumerate(portfolio_legs, start=1):
        name = leg.get("name", f"Leg {i}")
        if leg["type"] == "stock":
            portfolio_lines.append(f"{i}. {name}: {leg['qty']:+.2f} x stock")
        else:
            portfolio_lines.append(
                f"{i}. {name}: {leg['qty']:+.2f} x {leg['type']} | "
                f"K={float(leg['K']):.4f} | T={float(leg['T']):.6f}"
            )
    st.code("\n".join(portfolio_lines), language="text")

    st.write("**Market**")
    st.code(
        f"S={float(S):.4f}\n"
        f"σ={float(sigma):.6f}\n"
        f"r={float(r):.6f}\n"
        f"q={float(q):.6f}",
        language="text",
    )

    st.write("**Exposure table**")
    st.dataframe(
        exposure_df.style.format({"Value": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Quick notes**")
    maturity_list = ", ".join(f"{t:.4f}" for t in unique_maturities(portfolio_legs))
    st.code(
        f"Active maturities: {maturity_list}\n"
        f"Metric plot: {greek_key}\n"
        f"X variable: {x_var}\n"
        f"Mode: {mode}",
        language="text",
    )


with st.expander("Notes", expanded=False):
    st.markdown(
        """
        - Portfolio value and Greeks are computed as the **sum across all legs**.
        - `stock` supports `price` and `delta = 1`; other stock Greeks are set to `0` in this app.
        - You can vary **one specific option leg** by choosing plot variables like `leg_2_K (...)` or `leg_3_T (...)`.
        - The payoff tab shows:
          - **Gross payoff at expiry**
          - **Net P&L**, computed as payoff minus the current Black–Scholes premium of the portfolio
        - Negative quantities represent **short positions**.
        """
    )
