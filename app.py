# app.py

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from pricing import BlackScholes, Market, Contract


# Page + theme
st.set_page_config(
    page_title="Option Visualizer",
    # page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚙️ Black–Scholes Visualizer")
st.caption("Finite-difference Greeks on top of a Black–Scholes engine.")

# Engine + registry
bs = BlackScholes()
greeks = list(bs.greeks())
greek_keys = [g.key for g in greeks]
greek_by_key = {g.key: g for g in greeks}

VARS_ALL = ["S", "sigma", "T", "r", "q", "K"]

@st.cache_data(show_spinner=False)
def compute_grid(
    greek_key: str,
    option_type: str,
    # base params
    S: float,
    sigma: float,
    r: float,
    q: float,
    K: float,
    T: float,
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
    """Compute either 1D line (xs, ys) or 2D heatmap (xs, ys, Z)."""
    base_contract = Contract(K=K, T=T, option_type=option_type)
    base_market = Market(S=S, r=r, sigma=sigma, q=q)

    def with_var(contract: Contract, market: Market, var: str, value: float):
        if var == "S":
            return contract, market.with_(S=value)
        if var == "sigma":
            return contract, market.with_(sigma=value)
        if var == "r":
            return contract, market.with_(r=value)
        if var == "q":
            return contract, market.with_(q=value)
        if var == "T":
            return contract.with_(T=value), market
        if var == "K":
            return contract.with_(K=value), market
        raise KeyError(var)

    def metric(c: Contract, m: Market) -> float:
        return bs.metric(greek_by_key[greek_key], c, m)

    xs = np.linspace(float(x_min), float(x_max), int(n))

    if mode == "1D line":
        ys = np.empty_like(xs, dtype=float)
        for i, xv in enumerate(xs):
            c_i, m_i = with_var(base_contract, base_market, x_var, float(xv))
            ys[i] = metric(c_i, m_i)
        return xs, ys, None

    # 2D
    assert y_var is not None and y_min is not None and y_max is not None and n2 is not None
    ys = np.linspace(float(y_min), float(y_max), int(n2))
    Z = np.zeros((len(ys), len(xs)), dtype=float)

    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            c1, m1 = with_var(base_contract, base_market, x_var, float(xv))
            c2, m2 = with_var(c1, m1, y_var, float(yv))
            Z[j, i] = metric(c2, m2)

    return xs, ys, Z


def metric_now(greek_key: str, c: Contract, m: Market) -> float:
    return bs.metric(greek_by_key[greek_key], c, m)


# Sidebar controls
with st.sidebar:
    st.subheader("Inputs")

    with st.expander("Contract", expanded=True):
        option_type = st.segmented_control("Type", options=["call", "put"], default="call")
        K = st.number_input("Strike (K)", value=100.0, min_value=0.0001, step=1.0)
        T = st.number_input("Time to expiry (T, years)", value=0.25, min_value=0.0, step=0.01)

    with st.expander("Market", expanded=True):
        S = st.number_input("Spot (S)", value=100.0, min_value=0.0001, step=1.0)
        sigma = st.number_input("Vol (σ)", value=0.20, min_value=0.0, step=0.01, format="%.4f")
        r = st.number_input("Rate (r)", value=0.03, step=0.005, format="%.4f")
        q = st.number_input("Dividend (q)", value=0.00, step=0.005, format="%.4f")

    with st.expander("Plot", expanded=True):
        greek_key = st.selectbox(
            "Greek",
            greek_keys,
            index=greek_keys.index("delta") if "delta" in greek_keys else 0,
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
                y_min = st.number_input("Y min", value=0.05, step=0.01, format="%.4f")
            with c8:
                y_max = st.number_input("Y max", value=0.60, step=0.01, format="%.4f")
        else:
            y_var = None
            y_min = None
            y_max = None
            n2 = None

    st.markdown("---")
    st.caption("Tip: for noisy higher-order greeks, increase your FD step sizes.")


# Derived objects + KPIs
base_contract = Contract(K=K, T=T, option_type=option_type)
base_market = Market(S=S, r=r, sigma=sigma, q=q)

price = bs.metric(bs.price, base_contract, base_market)
gval = metric_now(greek_key, base_contract, base_market)

k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1, 1, 1])
k1.metric("Price", f"{price:.6f}")
k2.metric(greek_key, f"{gval:.6f}")
k3.metric("S", f"{S:.4f}")
k4.metric("σ", f"{sigma:.4f}")
k5.metric("T", f"{T:.4f}")

st.divider()

# Plot area (with spinner)
plot_col, info_col = st.columns([3.2, 1.2], gap="large")

with plot_col:
    with st.spinner("Computing grid..."):
        xs, ys, Z = compute_grid(
            greek_key=greek_key,
            option_type=option_type,
            S=float(S),
            sigma=float(sigma),
            r=float(r),
            q=float(q),
            K=float(K),
            T=float(T),
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
            title=f"{greek_key} vs {x_var}  ({option_type}, K={K:g}, T={T:g})",
            xaxis_title=x_var,
            yaxis_title=greek_key,
            hovermode="x",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        # scrollZoom = True enables mouse-wheel zoom
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    else:
        # Heatmap: Z[j, i] corresponds to (ys[j], xs[i])
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
            title=f"{greek_key} heatmap  ({option_type}, base K={K:g}, base T={T:g})",
            xaxis_title=x_var,
            yaxis_title=y_var,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


with info_col:
    st.subheader("Snapshot")

    st.write("**Contract**")
    st.code(f"type={option_type}\nK={K:.4f}\nT={T:.6f} yrs", language="text")

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
        - **Theta convention:** `theta = -dPrice/dT` (market convention).
        - All metrics here use **finite differences**.
        - If high-order greeks look noisy (speed/ultima/etc.), try widening your FD step sizes.
        """
    )