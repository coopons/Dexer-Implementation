from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class DetectionParams:
    selected_attributes: List[str]
    threshold: float
    alpha: float
    k_min: int
    k_max: int
    mode: str

@dataclass
class DetectionResult:
    groups: Any
    visited: int
    elapsed: float
    explanations: Any = None