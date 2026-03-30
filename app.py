from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pricing import BlackScholes, Market, Contract


# Page + theme
st.set_page_config(
    page_title="Option Visualizer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚙️ Black–Scholes Portfolio Visualizer")
st.caption("Finite-difference Greeks on top of a Black–Scholes engine.")

# Engine + registry
bs = BlackScholes()
greeks = list(bs.greeks())
greek_keys = [g.key for g in greeks]
greek_by_key = {g.key: g for g in greeks}

VARS_ALL = ["S", "sigma", "T", "r", "q", "K"]


def stock_metric(greek_key: str, market: Market) -> float:
    """
    Greeks for a stock position under this app's conventions.
    Price = S
    Delta = 1
    All other higher derivatives default to 0 unless you want
    to explicitly define rate/dividend carry sensitivities differently.
    """
    if greek_key == "price":
        return market.S
    if greek_key == "delta":
        return 1.0
    return 0.0


def leg_metric(
    greek_key: str,
    leg_type: str,
    qty: float,
    K: float | None,
    T: float | None,
    market: Market,
) -> float:
    """
    Value one portfolio leg.
    leg_type: 'stock', 'call', or 'put'
    """
    if leg_type == "stock":
        base_val = stock_metric(greek_key, market)
        return qty * base_val

    assert K is not None and T is not None
    c = Contract(K=K, T=T, option_type=leg_type)
    g = bs.price if greek_key == "price" else greek_by_key[greek_key]
    base_val = bs.metric(g, c, market)
    return qty * base_val


def portfolio_metric(
    greek_key: str,
    portfolio_legs: list[dict],
    market: Market,
) -> float:
    total = 0.0
    for leg in portfolio_legs:
        total += leg_metric(
            greek_key=greek_key,
            leg_type=leg["type"],
            qty=leg["qty"],
            K=leg.get("K"),
            T=leg.get("T"),
            market=market,
        )
    return total


@st.cache_data(show_spinner=False)
def compute_grid(
    greek_key: str,
    portfolio_legs: list[dict],
    # base market params
    S: float,
    sigma: float,
    r: float,
    q: float,
    # plot params
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
    """
    Compute either 1D line (xs, ys) or 2D heatmap (xs, ys, Z)
    for the WHOLE portfolio.
    """
    base_market = Market(S=S, r=r, sigma=sigma, q=q)

    def with_var(market: Market, legs: list[dict], var: str, value: float):
        new_market = market
        new_legs = [dict(leg) for leg in legs]

        if var == "S":
            new_market = market.with_(S=value)
        elif var == "sigma":
            new_market = market.with_(sigma=value)
        elif var == "r":
            new_market = market.with_(r=value)
        elif var == "q":
            new_market = market.with_(q=value)
        elif var == "T":
            for leg in new_legs:
                if leg["type"] != "stock":
                    leg["T"] = value
        elif var == "K":
            for leg in new_legs:
                if leg["type"] != "stock":
                    leg["K"] = value
        else:
            raise KeyError(var)

        return new_market, new_legs

    xs = np.linspace(float(x_min), float(x_max), int(n))

    if mode == "1D line":
        ys = np.empty_like(xs, dtype=float)
        for i, xv in enumerate(xs):
            m_i, legs_i = with_var(base_market, portfolio_legs, x_var, float(xv))
            ys[i] = portfolio_metric(greek_key, legs_i, m_i)
        return xs, ys, None

    assert y_var is not None and y_min is not None and y_max is not None and n2 is not None
    ys = np.linspace(float(y_min), float(y_max), int(n2))
    Z = np.zeros((len(ys), len(xs)), dtype=float)

    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            m1, legs1 = with_var(base_market, portfolio_legs, x_var, float(xv))
            m2, legs2 = with_var(m1, legs1, y_var, float(yv))
            Z[j, i] = portfolio_metric(greek_key, legs2, m2)

    return xs, ys, Z


# Sidebar controls
with st.sidebar:
    st.subheader("Inputs")

    with st.expander("Portfolio", expanded=True):
        n_legs = st.number_input("Number of legs", min_value=1, max_value=12, value=2, step=1)

        portfolio_legs = []
        for i in range(int(n_legs)):
            st.markdown(f"**Leg {i + 1}**")
            c1, c2, c3 = st.columns([1.2, 1, 1])

            with c1:
                leg_type = st.selectbox(
                    f"Type #{i+1}",
                    ["stock", "call", "put"],
                    index=0 if i == 0 else 1,
                    key=f"type_{i}",
                )
            with c2:
                qty = st.number_input(
                    f"Qty #{i+1}",
                    value=1.0,
                    step=1.0,
                    key=f"qty_{i}",
                )
            with c3:
                st.write("")
                st.write("")

            leg = {"type": leg_type, "qty": float(qty)}

            if leg_type != "stock":
                c4, c5 = st.columns(2)
                with c4:
                    K = st.number_input(
                        f"Strike K #{i+1}",
                        value=100.0,
                        min_value=0.0001,
                        step=1.0,
                        key=f"K_{i}",
                    )
                with c5:
                    T = st.number_input(
                        f"Expiry T #{i+1}",
                        value=0.25,
                        min_value=0.0,
                        step=0.01,
                        key=f"T_{i}",
                    )
                leg["K"] = float(K)
                leg["T"] = float(T)

            portfolio_legs.append(leg)
            st.markdown("---")

    with st.expander("Market", expanded=True):
        S = st.number_input("Spot (S)", value=100.0, min_value=0.0001, step=1.0)
        sigma = st.number_input("Vol (σ)", value=0.20, min_value=0.0, step=0.01, format="%.4f")
        r = st.number_input("Rate (r)", value=0.03, step=0.005, format="%.4f")
        q = st.number_input("Dividend (q)", value=0.00, step=0.005, format="%.4f")

    with st.expander("Plot", expanded=True):
        greek_key = st.selectbox(
            "Metric",
            ["price"] + greek_keys,
            index=(["price"] + greek_keys).index("delta") if "delta" in greek_keys else 0,
        )
        mode = st.segmented_control("Mode", options=["1D line", "2D heatmap"], default="1D line")

        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("X variable", VARS_ALL, index=VARS_ALL.index("S"))
        with c2:
            n = st.slider("X points", min_value=25, max_value=400, value=150)

        c3, c4 = st.columns(2)
        with c3:
            x_min = st.number_input("X min", value=50.0, step=1.0)
        with c4:
            x_max = st.number_input("X max", value=150.0, step=1.0)

        if mode == "2D heatmap":
            st.markdown("---")
            c5, c6 = st.columns(2)
            with c5:
                y_var = st.selectbox("Y variable", [v for v in VARS_ALL if v != x_var], index=0)
            with c6:
                n2 = st.slider("Y points", min_value=20, max_value=250, value=80)

            c7, c8 = st.columns(2)
            with c7:
                y_min_default = 0.05 if y_var in {"sigma", "T"} else 50.0
                y_min = st.number_input("Y min", value=y_min_default, step=0.01, format="%.4f")
            with c8:
                y_max_default = 0.60 if y_var in {"sigma", "T"} else 150.0
                y_max = st.number_input("Y max", value=y_max_default, step=0.01, format="%.4f")
        else:
            y_var = None
            y_min = None
            y_max = None
            n2 = None

    st.markdown("---")
    st.caption("Tip: portfolio Greeks add linearly across legs.")


# Base market + KPIs
base_market = Market(S=float(S), r=float(r), sigma=float(sigma), q=float(q))
price = portfolio_metric("price", portfolio_legs, base_market)
gval = portfolio_metric(greek_key, portfolio_legs, base_market) if greek_key != "price" else price

k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1, 1, 1])
k1.metric("Portfolio Price", f"{price:.6f}")
k2.metric(greek_key, f"{gval:.6f}")
k3.metric("S", f"{S:.4f}")
k4.metric("σ", f"{sigma:.4f}")
k5.metric("r", f"{r:.4f}")

st.divider()

# Plot area
plot_col, info_col = st.columns([3.2, 1.2], gap="large")

with plot_col:
    with st.spinner("Computing grid..."):
        xs, ys, Z = compute_grid(
            greek_key=greek_key,
            portfolio_legs=portfolio_legs,
            S=float(S),
            sigma=float(sigma),
            r=float(r),
            q=float(q),
            mode=mode,
            x_var=x_var,
            x_min=float(x_min),
            x_max=float(x_max),
            n=int(n),
            y_var=y_var,
            y_min=None if y_min is None else float(y_min),
            y_max=None if y_max is None else float(y_max),
            n2=None if n2 is None else int(n2),
        )

    if mode == "1D line":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=greek_key,
                hovertemplate=f"{x_var}=%{{x:.6f}}<br>{greek_key}=%{{y:.6f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Portfolio {greek_key} vs {x_var}",
            xaxis_title=x_var,
            yaxis_title=greek_key,
            hovermode="x",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    else:
        fig = go.Figure(
            data=go.Heatmap(
                x=xs,
                y=ys,
                z=Z,
                colorbar=dict(title=greek_key),
                hovertemplate=f"{x_var}=%{{x:.6f}}<br>{y_var}=%{{y:.6f}}<br>{greek_key}=%{{z:.6f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Portfolio {greek_key} heatmap",
            xaxis_title=x_var,
            yaxis_title=y_var,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


with info_col:
    st.subheader("Snapshot")

    st.write("**Portfolio**")
    portfolio_lines = []
    for i, leg in enumerate(portfolio_legs, start=1):
        if leg["type"] == "stock":
            portfolio_lines.append(f"{i}. {leg['qty']:+.2f} x stock")
        else:
            portfolio_lines.append(
                f"{i}. {leg['qty']:+.2f} x {leg['type']}  K={leg['K']:.4f}  T={leg['T']:.6f}"
            )
    st.code("\n".join(portfolio_lines), language="text")

    st.write("**Market**")
    st.code(f"S={S:.4f}\nσ={sigma:.6f}\nr={r:.6f}\nq={q:.6f}", language="text")

    st.write("**Plot**")
    if mode == "1D line":
        st.code(f"{greek_key} vs {x_var}\n[{x_min:.4f}, {x_max:.4f}]  ({n} pts)", language="text")
    else:
        st.code(
            f"{greek_key} vs ({x_var}, {y_var})\n"
            f"x:[{x_min:.4f}, {x_max:.4f}] ({n} pts)\n"
            f"y:[{y_min:.4f}, {y_max:.4f}] ({n2} pts)",
            language="text",
        )


with st.expander("Notes / gotchas", expanded=False):
    st.markdown(
        """
        - Portfolio value and Greeks are computed as the **sum across all legs**.
        - `stock` supports `price` and `delta=1`; other stock Greeks are set to 0 here.
        - When you plot **K** or **T**, the app currently applies that same K/T to **all option legs**.
        - **Theta convention:** `theta = -dPrice/dT` (market convention).
        - All option metrics here use **finite differences**.
        """
    )