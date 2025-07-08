from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


class OrderedDictEncoder:

    def __init__(self, start: int = 0, *, slack: int | None = None) -> None:
        if slack is not None:
            start = slack
        self._next_key = start
        self._kv: Dict[str, int] = {}

    def encode(self, words: Sequence[str]) -> Dict[str, int]:
        delta: Dict[str, int] = {}
        for w in words:
            if w not in self._kv:
                self._kv[w] = self._next_key
                delta[w] = self._next_key
                self._next_key += 1
        return delta

    @property
    def kv_map(self) -> Dict[str, int]:
        return self._kv
    
    def insert_batch(self, words):
        return self.encode(words)

import numpy as np
from typing import Sequence, List

class EcdfSampler:
    """
    • sample(values, k)           —— Worker：抽 k 个等分位点
    • merge_worker_samples(pool)  —— Leader：合并后再抽 k 点
    • ecdf_lookup(samples, x)     —— 线性插值求 F̂(x)，误差 ≤ ½/k
    """

    @staticmethod
    def sample(values: Sequence[float], k: int) -> List[float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return [0.0] * k
        arr.sort()
        pos = np.linspace(0, arr.size - 1, k)
        return np.interp(pos, np.arange(arr.size), arr).tolist()

    @staticmethod
    def merge_worker_samples(pool: List[np.ndarray], k: int) -> np.ndarray:
        if not pool:
            return np.zeros(k, dtype=float)
        big = np.sort(np.concatenate(pool))
        pos = np.linspace(0, big.size - 1, k)
        return np.interp(pos, np.arange(big.size), big)

    @staticmethod
    def ecdf_lookup(samples: np.ndarray, x: float) -> float:
        if samples.size == 0:
            return 0.0
        idx = np.searchsorted(samples, x, side="right")
        k = samples.size - 1
        if idx == 0:
            return 0.0
        if idx >= samples.size:
            return 1.0
        x0, x1 = samples[idx - 1], samples[idx]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return ((idx - 1) + t) / k

