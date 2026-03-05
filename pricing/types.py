# pricing/types.py

from dataclasses import dataclass, replace
from .diff import DerivativeSpec


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


@dataclass(frozen=True)
class Greek:
    key: str  # greek id
    spec: DerivativeSpec  # [] for price (no derivatives)
    theta_market: bool = False  # if True return -dP/dT instead of dP/dT
