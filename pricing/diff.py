# pricing/diff.py

from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass


DerivativeSpec = List[Tuple[str, int]]


@dataclass(frozen=True)
class DiffConfig:
    rel_step: float = 1e-4
    abs_step: float = 1e-6


def step(x: float, config: DiffConfig) -> float:
    """
    Parameters
    ----------
    x: the current value of the variable being perturbed (e.g. spot S, vol sigma).
    config: controls the relative and absolute step sizes.

    Returns
    -------
    Floating-point step size h for use in finite-difference formulas.
    """
    return max(config.abs_step, config.rel_step * (abs(x) + 1.0))


def finite_diff(f: Callable[[float], float], x0: float, order: int, config: DiffConfig, h: Optional[float] = None) -> float:
    """
    Numerically approximates the nth derivative of f at x0 with central differences f^(n)(x0).

    Parameters
    ----------
    f: scalar function f(x) to differentiate.
    x0: point where the derivative is evaluated.
    order: derivative order n.
    config: step-size configuration.
    h: optional explicit step size.

    Returns
    -------
    Floating-point approximation to f^(order)(x0).

    If h is provided, it is used directly. Otherwise, we compute an adaptive step.
    Recurses with f^(n)(x0) = d/dx [ f^(n-1)(x) ] at x0 until it reduces to base case with order 1 or 2.
        - Inner call: builds f^(n-1) at a point x,
        - Outer call: takes a first derivative of that inner derivative.
    """
    if h == None:
        h = step(x0, config)

    if order == 1:
        return (f(x0 + h) - f(x0 - h)) / (2 * h)
    elif order == 2:
        return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h)
    
    return finite_diff(lambda x: finite_diff(f, x, order - 1, config, h = h), x0, 1, config, h = h)