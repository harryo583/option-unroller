from dataclasses import dataclass
from typing import Callable, Optional 


@dataclass(frozen=True)
class DiffConfig:
    rel_step: float = 1e-4
    abs_step: float = 1e-6


def step(x: float, config: DiffConfig) -> float:
    return max(config.abs_step, config.rel_step * (abs(x) + 1.0))


def finite_diff(f: Callable[[float], float], x0: float, order: int, config: DiffConfig, h: Optional[float] = None) -> float:
    """ Returns f^(n)(x0), the nth derivative of f evaluated at x0 """
    
    h = step(x0, config) if h == None else h

    if order == 1:
        return (f(x0 + h) - f(x0 - h)) / (2 * h)
    elif order == 2:
        return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h)
    
    return finite_diff(lambda x: finite_diff(f, x, order - 1, config, h = h), x0, 1, config, h = h)

    # def g(x: float) -> float:
    #     return finite_diff(f, x, order - 1, config, h = h)
    
    # return finite_diff(g, x0, 1, config, h = h)

