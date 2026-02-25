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
        self._price = Greek("price", [])
        
        # First order greeks
        self._delta = Greek("delta", [("S", 1)])
        self._gamma = Greek("gamma", [("S", 2)])
        self._vega = Greek("vega", [("sigma", 1)])
        self._theta = Greek("theta", [("T", 1)], theta_market=True)
        self._rho = Greek("rho", [("r", 1)])

        # Second order greeks
        self._vanna = Greek("vanna", [("S", 1), ("sigma", 1)])
        self._volga = Greek("volga", [("sigma", 2)])
        self._charm = Greek("charm", [("S", 1), ("T", 1)], theta_market=True)
        self._vega_decay = Greek("charm", [("sigma", 1), ("T", 1)], theta_market=True)

        # Third order greeks
        self._speed = Greek("speed", [("S", 3)])
        self._color = Greek("color", [("S", 2), ("T", 1)], theta_market=True)
        self._ultima = Greek("ultima", [("sigma", 3)])
        self._zomma = Greek("zomma", [("S", 2), ("sigma", 1)])


    @property
    def price(self) -> Greek: return self._price

    @property
    def delta(self) -> Greek: return self._delta
    
    @property
    def gamma(self) -> Greek: return self._gamma

    @property
    def vega(self) -> Greek: return self._vega

    @property
    def theta(self) -> Greek: return self._theta

    @property
    def rho(self) -> Greek: return self._rho

    @property
    def vanna(self) -> Greek: return self._vanna

    @property
    def volga(self) -> Greek: return self._volga

    @property
    def charm(self) -> Greek: return self._charm

    @property
    def vega_decay(self) -> Greek: return self._vega_decay

    @property
    def speed(self) -> Greek: return self._speed

    @property
    def color(self) -> Greek: return self._color

    @property
    def ultima(self) -> Greek: return self._ultima

    @property
    def zomma(self) -> Greek: return self._zomma


    def greeks(self) -> Iterable[Greek]:
        return [
            self.price, self.delta, self.gamma, self.vega, self.theta, self.rho,
            self.vanna, self.volga, self.charm, self.vega_decay,
            self.speed, self.color, self.ultima, self.zomma
        ]
    

    def _price_value(self, c: Contract, m: Market) -> float:
        pass