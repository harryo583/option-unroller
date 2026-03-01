from dataclasses import dataclass 
from typing import List, Tuple 

DerivativeSpec = List[Tuple[str, int]]

@dataclass(frozen=True)
class Greek:
    key: str  # greek id
    spec: DerivativeSpec  # [] for price (no derivatives)
    theta_market: bool = False  # if True return -dP/dT instead of dP/dT