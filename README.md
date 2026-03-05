# Option Greeks Plotter

A **Black–Scholes visualization tool** that allows users to visualize **arbitrary-order Greeks** and option sensitivities through 2D plots and 3D surfaces.

The application is built with **Streamlit** and uses a **finite-difference engine** to compute derivatives of the Black–Scholes price with respect to any parameter. Unlike standard textbook plots, plots of **arbitrary higher-order derivatives** are supported. It is great both for educational and for training purposes.

---

## Features

- Adjustable market parameters
- Support for plotting any **arbitrary-order Greek** against any variable (e.g. Delta vs Spot, Gamma vs Volatility)
- Interactive plotting with Streamlit UI

---

## Example Greeks Supported

Because Greeks are defined as derivatives of the price function

\[
P = P(S, \sigma, T, r, q, K)
\]

this engine can compute derivatives such as:

| Greek | Definition |
|------|-------------|
| Delta | ∂P / ∂S |
| Gamma | ∂²P / ∂S² |
| Vega | ∂P / ∂σ |
| Theta | ∂P / ∂T |
| Vanna | ∂²P / (∂S ∂σ) |
| Volga | ∂²P / ∂σ² |
| Higher-order Greeks | Any arbitrary derivative combination |
