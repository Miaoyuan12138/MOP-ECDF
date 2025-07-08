from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EncodingStats:
    message: int = 0
    distinct_keys: int = 0
    disordered_keys: int = 0
    duplicate_key_request: int = 0
    percent_ordered: float = 1.0

@dataclass
class WorkerMsg:
    dict_delta: Optional[Dict[str, int]] = None
    samples: Optional[List[float]] = None
    flush: bool = False



