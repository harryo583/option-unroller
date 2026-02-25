import math
from typing import Dict, List, Tuple, Callable, Optional
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
        self._price = Greek("price", [])
        
        # First order greeks
        self._delta = Greek("delta", [("S", 1)])
        self._gamma = Greek("gamma", [("S", 2)])
        self._vega = Greek("vega", [("sigma", 1)])
        self._theta = Greek("theta", [("T", 1)], theta_market=True)
        self._rho = Greek("rho", [("r", 1)])

        # Second order greeks
        self._vanna = Greek()
        self._volga = Greek()
        self._charm = Greek()
        self._vega_decay = Greek() 

        # Third order greeks
        self._speed = None
        self._color = None
        self._ultima = None
        self._zomma = None