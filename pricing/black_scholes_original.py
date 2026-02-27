import math
from typing import Dict, List, Tuple, Callable, Optional, Iterable
from dataclasses import dataclass, replace

from pricing.diff import DiffConfig, finite_diff, step
from pricing.metrics import Greek, DerivativeSpec


@dataclass(frozen=True)
class Market:
    S: float  # stock price
    r: float  # interest rate
    sigma: float  # volatility
    q: float = 0.0  # dividend yield

    def with_(self, **kwargs) -> "Market":
        return replace(self, **kwargs)
    

@dataclass(frozen=True)
class Contract:
    K: float
    T: float
    option_type: str  # "call" or "put"

    def with_(self, **kwargs) -> "Contract":
        return replace(self, **kwargs)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class BlackScholes:
    """
    Class design
        - Greeks are attributes: bs.delta, bs.gamma, bs.zomma etc.
        - These return Greek objects (not strings)
        - metric(greek, contract, market) computes it
    """

    def __init__(self, diff: DiffConfig = DiffConfig()):
        self.diff = diff 

        # Zeroth order
        self.price = Greek("price", [])
        
        # First order greeks
        self.delta = Greek("delta", [("S", 1)])
        self.gamma = Greek("gamma", [("S", 2)])
        self.vega = Greek("vega", [("sigma", 1)])
        self.theta = Greek("theta", [("T", 1)], theta_market=True)
        self.rho = Greek("rho", [("r", 1)])

        # Second order greeks
        self.vanna = Greek("vanna", [("S", 1), ("sigma", 1)])
        self.volga = Greek("volga", [("sigma", 2)])
        self.charm = Greek("charm", [("S", 1), ("T", 1)], theta_market=True)
        self.vega_decay = Greek("charm", [("sigma", 1), ("T", 1)], theta_market=True)

        # Third order greeks
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
    

    def _price_value(self, c: Contract, m: Market) -> float:
        S, K, T, r, sigma, q = m.S, m.K, m.T, m.r, m.sigma, m.q 
        opt = c.option_type.lower()

        if T <= 0.0:
            F = S * math.exp((r - 1) * T)
            disc = math.exp(-r * T)
            return disc * max(F - K, 0.0) if opt == "call" else disc * max(K - F, 0.0)
        
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2 ) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if opt == "call":
            return disc_q * S * _norm_cdf(d1) - disc_r * K * _norm_cdf(d2)
        else:
            return disc_r * K * _norm_cdf(-d2) - disc_q * S * _norm_cdf(-d1)
    

    def _mixed_diff(self, base_state: Dict[str, float], spec: DerivativeSpec, price_from_state: Callable[[Dict[str, float]], float]) -> float:
        """ Mixed partial calculation engine """
        f = price_from_state

        for var, ord in spec:
            x0 = base_state[var]
            h = step(x0, self.diff)

            def make_new_f(prev_f, var_name, order, h_local):
                def new_f(state):
                    x = state[var_name]
                    def one_d(z):
                        st2 = dict(state)
                        st2[var_name] = z
                        return prev_f(st2)
                    return finite_diff(one_d, x, order, config=self.diff, h=h_local)
                return new_f
            
            f = make_new_f(f, var, ord, h)
        
        return f(base_state)
    

    