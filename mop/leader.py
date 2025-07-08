from __future__ import annotations
import json, logging, threading
from typing import Dict, List, Set

import numpy as np, zmq

from .config    import DictionaryConfig
from .algorithm import OrderedDictEncoder, EcdfSampler
from .          import net
from .base_types import EncodingStats
from .snapshot import dump_snapshot

_LOG = logging.getLogger(__name__)


class MOPDictLeader(threading.Thread):
    def __init__(self, cfg: DictionaryConfig, *, k: int = 256, workers: int = 1):
        super().__init__(daemon=True)
        self.cfg, self.k, self.workers = cfg, k, workers

        slack = max(1, round(cfg.pitch / cfg.look_ahead))
        self.encoder = OrderedDictEncoder(start=slack)

        self.sample_pool: List[np.ndarray] = []
        self.global_samples: np.ndarray | None = None
        self.done_workers: Set[int] = set() 

        self.stats = EncodingStats()
        self.pub, self.pull = net.bind_pub_pull(cfg)
        self._stop_evt = threading.Event()

    def run(self) -> None:
        poller = zmq.Poller(); poller.register(self.pull, zmq.POLLIN)

        while len(self.done_workers) < self.workers:
            events = dict(poller.poll(self.cfg.batch_time_ms))
            if self.pull in events:
                msg = json.loads(self.pull.recv())
                self._handle_worker_msg(msg)

        _LOG.info("All workers finished, leader shutting down")
        dump_snapshot(self, "snapshots/26.7.6 feather")
        self.pub.close(); self.pull.close()

    def ecdf(self, x: float) -> float:
        if self.global_samples is None:
            return 0.0
        return EcdfSampler.ecdf_lookup(self.global_samples, x)

    def _handle_worker_msg(self, msg: Dict) -> None:
        mtype = msg.get("type")
        if mtype == "HANDSHAKE":
            self._broadcast_dict({})
            return
        if mtype == "SAMPLE":
            self._merge_words(msg.get("proposal", []))
            if (s := msg.get("samples")):
                self.sample_pool.append(np.asarray(s, dtype=float))
            if msg.get("flush"):
                self._flush_batch()
        elif mtype == "END":   
            self.done_workers.add(msg["wid"])
            _LOG.info("Leader received END from %s", msg["wid"])
        elif mtype == "NEW":
            self._merge_words(msg.get("words", []))
            if (s := msg.get("samples")): 
                self.sample_pool.append(np.asarray(s, dtype=float))
            if msg.get("flush"):
                self._flush_batch()

    def _merge_words(self, words: List[str]) -> None:
        if not words:
            return
        delta = self.encoder.insert_batch(words)
        if delta:
            self._broadcast_dict(delta)
            self.stats.distinct_keys = len(self.encoder.kv_map)

    def _flush_batch(self) -> None:
        if not self.sample_pool:
            return
        self.global_samples = EcdfSampler.merge_worker_samples(
            self.sample_pool, self.k
        )
        #self.sample_pool.clear()
        _LOG.info("Global ECDF rebuilt (size=%d)", self.global_samples.size)

    def _broadcast_dict(self, delta: Dict[str, int]) -> None:
        self.pub.send_json({"type": "DICT", "dict": delta})
        self.stats.message += 1
