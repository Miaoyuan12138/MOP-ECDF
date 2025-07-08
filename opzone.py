from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class OPZone:
    start_key: int = 0
    end_key:   int = 0
    next_key:  int = 0  
    kv_map:    Dict[str, int] = field(default_factory=dict)
    # last: bool = True
