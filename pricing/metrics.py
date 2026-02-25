from dataclasses import dataclass 
from typing import List, Tuple 

DerivativeSpec = List[Tuple[str, int]]

@dataclass(frozen=True)
class Greek:
    key: str  # stable internal id
    label: str  # ui label
    spec: DerivativeSpec  # [] for price (no derivatives)
    theta_market: bool = False  # return -dP/dT instead of dP/dT if True