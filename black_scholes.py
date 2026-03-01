# pricing/black_scholes.py
import math
from typing import Dict, List, Tuple, Callable, Optional, Iterable
from dataclasses import dataclass, replace

from diff import DiffConfig, finite_diff, step
from metrics import Greek, DerivativeSpec


@dataclass(frozen=True)
class Market:
    S: float  # spot
    r: float  # risk-free rate (cc)
    sigma: float  # vol
    q: float = 0.0  # dividend yield (cc)

    def with_(self, **kwargs) -> "Market":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Contract:
    K: float
    T: float  # time to expiry in years
    option_type: str  # "call" or "put"

    def with_(self, **kwargs) -> "Contract":
        return replace(self, **kwargs)


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
        self.charm = Greek("charm", [("S", 1), ("T", 1)], theta_market=True)
        self.vega_decay = Greek("vega_decay", [("sigma", 1), ("T", 1)], theta_market=True)

        # Third order
        self.speed = Greek("speed", [("S", 3)])
        self.color = Greek("color", [("S", 2), ("T", 1)], theta_market=True)
        self.ultima = Greek("ultima", [("sigma", 3)])
        self.zomma = Greek("zomma", [("S", 2), ("sigma", 1)])

    def greeks(self) -> Iterable[Greek]:
        return [
            self.price, self.delta, self.gamma, self.vega, self.theta, self.rho,
            self.vanna, self.volga, self.charm, self.vega_decay,
            self.speed, self.color, self.ultima, self.zomma
        ]

    # -----------------------------
    # Core pricing function
    # -----------------------------
    def _price_value(self, c: Contract, m: Market) -> float:
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

        if sigma <= 0.0:
            # Deterministic forward under risk-neutral drift (r - q)
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

    # -----------------------------
    # Mixed partial engine
    # -----------------------------
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
        Applies derivatives in sequence by wrapping the function.
        Each step creates a new function that, when called on a state,
        computes the requested partial derivative w.r.t. one variable.
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

    # -----------------------------
    # Public API
    # -----------------------------
    def metric(self, greek: Greek, c: Contract, m: Market) -> float:
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
        key = key.strip().lower()
        gmap = {g.key.lower(): g for g in self.greeks()}
        if key not in gmap:
            raise KeyError(f"Unknown greek key '{key}'. Known: {sorted(gmap.keys())}")
        return self.metric(gmap[key], c, m)