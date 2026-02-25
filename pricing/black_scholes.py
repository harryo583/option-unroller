import math
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass

from pricing.diff import DiffConfig, finite_diff, step
from pricing.metrics import Greek, DerivativeSpec

