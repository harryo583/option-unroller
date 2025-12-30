import math
from typing import Dict, List, Tuple, Callable, Optional
from dataclass import dataclass


@dataclass(frozen=True)
class BSParams:
    S: float # stock price
    K: float # strike price
    T: float # time in years
    r: float # continuously compounded risk-free rate
    sigma: float # volatility
    q: float # divident yield



class BlackScholes:
    def __init__(self):
        
        pass

    def _N(x: float) -> float:
        """Cumulative distribution function for standard normal distribution"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    def _n(x: float) -> float:
        """Probability density function for standard normal distribution"""
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


    def price(self, p: BSParams, option_type: str = "call") -> float:
        S, K, T, r, sigma, q = p.S, p.K, p.T, p.r, p.sigma, p.q
        option_type = option_type.lower()

        if not option_type in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        if T <= 0.0:
            if option_type == "call":
                return max(S - K, 0.0)
            return max(K - S, 0.0)
            
        if sigma <= 0.0:
            F = S * math.exp((r - q) * T)
            discount_rate = math.exp(-r * T)
            if option_type == "call":
                return discount_rate * max(F - K, 0.0)
            return discount_rate * max(K - F, 0.0)
        
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if option_type == "call":
            return S * disc_q * self._N(d1) - K * disc_r * self._N(d2)
        else:
            return K * disc_r * self._N(-d2) - S * disc_q * self._N(-d1)
        

