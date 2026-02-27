# app.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from pricing.black_scholes import BlackScholes, Market, Contract


st.set_page_config(page_title="Option Greeks Plotter", layout="wide")

st.title("Black–Scholes Greeks Plotter (Finite Differences)")

bs = BlackScholes()

greeks = list(bs.greeks())
greek_keys = [g.key for g in greeks]
greek_by_key = {g.key: g for g in greeks}

vars_all = ["S", "sigma", "T", "r", "q", "K"]


def compute(bs: BlackScholes, greek_key: str, c: Contract, m: Market) -> float:
    return bs.metric(greek_by_key[greek_key], c, m)


with st.sidebar:
    st.header("Contract")
    option_type = st.selectbox("Option type", ["call", "put"], index=0)
    K = st.number_input("Strike K", value=100.0, min_value=0.0001, step=1.0)
    T = st.number_input("Time to expiry T (years)", value=0.25, min_value=0.0, step=0.01)

    st.header("Market")
    S = st.number_input("Spot S", value=100.0, min_value=0.0001, step=1.0)
    sigma = st.number_input("Vol sigma", value=0.20, min_value=0.0, step=0.01, format="%.4f")
    r = st.number_input("Rate r", value=0.03, step=0.005, format="%.4f")
    q = st.number_input("Dividend q", value=0.00, step=0.005, format="%.4f")

    st.header("Plot")
    greek_key = st.selectbox("Greek", greek_keys, index=greek_keys.index("delta") if "delta" in greek_keys else 0)

    mode = st.radio("Mode", ["1D line", "2D heatmap"], index=0)

    x_var = st.selectbox("X variable", vars_all, index=vars_all.index("S"))
    x_min = st.number_input("X min", value=50.0, step=1.0)
    x_max = st.number_input("X max", value=150.0, step=1.0)
    n = st.slider("Points", min_value=25, max_value=400, value=150)

    if mode == "2D heatmap":
        y_var = st.selectbox("Y variable", [v for v in vars_all if v != x_var], index=0)
        y_min = st.number_input("Y min", value=0.05, step=0.01, format="%.4f")
        y_max = st.number_input("Y max", value=0.60, step=0.01, format="%.4f")
        n2 = st.slider("Y points", min_value=20, max_value=250, value=80)


base_contract = Contract(K=K, T=T, option_type=option_type)
base_market = Market(S=S, r=r, sigma=sigma, q=q)

# quick KPI row
colA, colB, colC, colD = st.columns(4)
price = bs.metric(bs.price, base_contract, base_market)
val = compute(bs, greek_key, base_contract, base_market)
colA.metric("Price", f"{price:.6f}")
colB.metric(greek_key, f"{val:.6f}")
colC.metric("S", f"{S:.4f}")
colD.metric("sigma", f"{sigma:.4f}")

st.divider()


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


if mode == "1D line":
    xs = np.linspace(float(x_min), float(x_max), int(n))
    ys = np.empty_like(xs, dtype=float)

    for i, xv in enumerate(xs):
        c_i, m_i = with_var(base_contract, base_market, x_var, float(xv))
        ys[i] = compute(bs, greek_key, c_i, m_i)

    fig = plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(x_var)
    plt.ylabel(greek_key)
    plt.title(f"{greek_key} vs {x_var}  ({option_type}, K={K}, T={T})")
    plt.grid(True)
    st.pyplot(fig, clear_figure=True)

else:
    xs = np.linspace(float(x_min), float(x_max), int(n))
    ys = np.linspace(float(y_min), float(y_max), int(n2))

    Z = np.zeros((len(ys), len(xs)), dtype=float)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            c1, m1 = with_var(base_contract, base_market, x_var, float(xv))
            c2, m2 = with_var(c1, m1, y_var, float(yv))
            Z[j, i] = compute(bs, greek_key, c2, m2)

    fig = plt.figure()
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f"{greek_key} heatmap  ({option_type}, base K={K}, base T={T})")
    plt.colorbar(im)
    st.pyplot(fig, clear_figure=True)


with st.expander("Notes / gotchas", expanded=False):
    st.markdown(
        """
- Theta here uses **market convention**: `theta = -dPrice/dT`.
- Everything is computed via **finite differences** (your `pricing/diff.py`).
- If you see noise for very high-order greeks (speed/ultima/etc), widen `rel_step` a bit.
"""
    )