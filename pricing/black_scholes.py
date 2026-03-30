# pricing/black_scholes.py

import math
from typing import Dict, Callable, Iterable
from .diff import DiffConfig, DerivativeSpec, step, finite_diff
from .types import Greek, Contract, Market


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class BlackScholes:
    """
    Design:
      - Greeks are attributes: bs.delta, bs.gamma, ...
      - They are Greek objects (key + derivative spec)
      - metric(greek, contract, market) computes it numerically via mixed partials
    """

    def __init__(self, diff: DiffConfig = DiffConfig()):
        self.diff = diff

        # Zeroth order
        self.price = Greek("price", [])

        # First order
        self.delta = Greek("delta", [("S", 1)])
        self.gamma = Greek("gamma", [("S", 2)])
        self.vega = Greek("vega", [("sigma", 1)])
        self.theta = Greek("theta", [("T", 1)], theta_market=True)  # market: -dP/dT
        self.rho = Greek("rho", [("r", 1)])

        # Second order
        self.vanna = Greek("vanna", [("S", 1), ("sigma", 1)])
        self.volga = Greek("volga", [("sigma", 2)])  # aka vomma
        self.charm = Greek("charm", [("S", 1), ("T", 1)], theta_market=True)  # delta decay
        self.veta = Greek("veta", [("sigma", 1), ("T", 1)], theta_market=True)  # vega_decay

        # Third order
        self.speed = Greek("speed", [("S", 3)])
        self.color = Greek("color", [("S", 2), ("T", 1)], theta_market=True)  # gamma decay
        self.ultima = Greek("ultima", [("sigma", 3)])
        self.zomma = Greek("zomma", [("S", 2), ("sigma", 1)])
        
        # Acronyms
        self.delta_decay = self.charm
        self.vega_decay = self.veta
        self.gamma_decay = self.color

    def greeks(self) -> Iterable[Greek]:
        return [
            self.price, self.delta, self.gamma, self.vega, self.theta, self.rho,
            self.vanna, self.volga, self.charm, self.veta,
            self.speed, self.color, self.ultima, self.zomma
        ]

    # Core pricing function
    def _price_value(self, c: Contract, m: Market) -> float:
        """
        Closed-form Black-Scholes price for a European call/put with dividend yield.

        Implements the standard formula with continuous dividend yield q:
          C = e^{-qT} S Φ(d1) - e^{-rT} K Φ(d2)
          P = e^{-rT} K Φ(-d2) - e^{-qT} S Φ(-d1)

        Where:
          d1 = [ln(S/K) + (r - q + 0.5 σ^2) T] / (σ sqrt(T))
          d2 = d1 - σ sqrt(T)

        Edge cases handled
        ------------------
        - Invalid option_type -> ValueError
        - T <= 0:
            Treat as expired: intrinsic value max(S-K,0) or max(K-S,0)
        - sigma <= 0:
            Degenerate (deterministic) case:
              * Under risk-neutral dynamics with drift (r-q), forward is:
                    F = S * exp((r - q) T)
              * Price becomes discounted intrinsic on the forward:
                    disc * max(F-K, 0)  (call)
                    disc * max(K-F, 0)  (put)

        Parameters
        ----------
        c: contract (K, T, call/put).
        m: market (S, r, sigma, q).

        Returns
        -------
        Option present value (float).
        """
        S, K, T, r, sigma, q = m.S, c.K, c.T, m.r, m.sigma, m.q
        opt = c.option_type.lower().strip()

        if opt not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got: {c.option_type}")

        # Expired or effectively expired
        if T <= 0.0:
            if opt == "call":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        # Deterministic forward under risk-neutral drift (r - q)
        if sigma <= 0.0:
            F = S * math.exp((r - q) * T)
            disc = math.exp(-r * T)
            if opt == "call":
                return disc * max(F - K, 0.0)
            else:
                return disc * max(K - F, 0.0)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if opt == "call":
            return disc_q * S * _norm_cdf(d1) - disc_r * K * _norm_cdf(d2)
        else:
            return disc_r * K * _norm_cdf(-d2) - disc_q * S * _norm_cdf(-d1)

    # Mixed partial engine
    def _base_state(self, c: Contract, m: Market) -> Dict[str, float]:
        return {
            "S": float(m.S),
            "sigma": float(m.sigma),
            "r": float(m.r),
            "q": float(m.q),
            "T": float(c.T),
            "K": float(c.K),
        }

    def _price_from_state(self, state: Dict[str, float], opt: str) -> float:
        m = Market(S=state["S"], r=state["r"], sigma=state["sigma"], q=state["q"])
        c = Contract(K=state["K"], T=state["T"], option_type=opt)
        return self._price_value(c, m)

    def _mixed_diff(
        self,
        base_state: Dict[str, float],
        spec: DerivativeSpec,
        price_from_state: Callable[[Dict[str, float]], float],
    ) -> float:
        """
        Computes mixed partial derivatives of the pricing function using finite differences.

        This function applies a sequence of derivatives specified by `spec` by
        iteratively wrapping the pricing function. Each step transforms the function
        into its partial derivative with respect to a given variable.

        Parameters
        ----------
        base_state: dict of input variables representing the point at which the derivative is evaluated.
        spec (DerivativeSpec): list of (variable, order) pairs specifying the derivative.
            For example:
                [("S", 1)] -> first derivative w.r.t. S (Delta)
                [("S", 2)] -> second derivative w.r.t. S (Gamma)
                [("S", 1), ("sigma", 1)]  -> mixed derivative (Vanna)
        price_from_state: function that maps a state dictionary to a scalar price value.

        Returns
        -------
        Floating point numerical approximation of the requested mixed partial derivative.

        Notes
        -----
        - Derivatives are applied sequentially in the order given by `spec`.
        - Functions are repeatedly wrapped so that derivatives are computed without explicit formulas.
        """

        f = price_from_state

        for var_name, order in spec:
            if var_name not in base_state:
                raise KeyError(f"Unknown state variable '{var_name}'. Known: {sorted(base_state.keys())}")

            def wrap(prev_f: Callable[[Dict[str, float]], float], vn: str, ord_: int) -> Callable[[Dict[str, float]], float]:
                def new_f(st: Dict[str, float]) -> float:
                    x0 = float(st[vn])
                    h = step(x0, self.diff)

                    def one_d(z: float) -> float:
                        st2 = dict(st)
                        st2[vn] = float(z)
                        return prev_f(st2)

                    return float(finite_diff(one_d, x0, ord_, config=self.diff, h=h))

                return new_f

            f = wrap(f, var_name, order)

        return float(f(base_state))

    # Public API
    def metric(self, greek: Greek, c: Contract, m: Market) -> float:
        """
        Computes the value of a specified Greek for a given contract and market.
        It supports arbitrary-order and mixed partial derivatives.

        Parameters
        ----------
        greek: object defining the Greek to compute, including its derivative specification
        c: contract specification (strike, time to expiry, option type).
        m: market data (spot, volatility, interest rate, dividend yield).

        Returns
        -------
        Numerical value of the requested Greek.

        Notes
        -----
        - If `greek.theta_market` is True, it returns -dP/dT instead of dP/dT).
        - For mixed derivatives involving time (T), the sign is flipped if the
            total order of differentiation with respect to T is odd.
        """
        base = self._base_state(c, m)
        opt = c.option_type.lower().strip()

        val = self._mixed_diff(
            base_state=base,
            spec=greek.spec,
            price_from_state=lambda st: self._price_from_state(st, opt),
        )

        # Market theta convention: -d/dT (and for mixed partials, flip sign if odd total T-order)
        if greek.theta_market:
            t_order = sum(ord_ for (v, ord_) in greek.spec if v == "T")
            if t_order % 2 == 1:
                val = -val

        return float(val)

    def metric_by_key(self, key: str, c: Contract, m: Market) -> float:
        """
        Computes a Greek by its string identifier - a convenience wrapper 
        around `metric` that allows users to request a Greek using its name
        
        Parameters
        ----------
        key: string name of the Greek (case-insensitive), e.g. "delta", "vega", "theta".
        c: contract specification.
        m: market data.

        Returns
        -------
        Numerical value of the requested Greek.

        Raises
        ------
        KeyError if the provided key does not correspond to a known Greek.
        """
        key = key.strip().lower()
        gmap = {g.key.lower(): g for g in self.greeks()}
        if key not in gmap:
            raise KeyError(f"Unknown greek key '{key}'. Known: {sorted(gmap.keys())}")
        return self.metric(gmap[key], c, m)