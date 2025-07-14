from __future__ import annotations
import json, logging, pathlib, threading, time
from typing import Dict, List, Sequence
from time import sleep

import numpy as np, zmq

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from .bitpacking import pack as bitpack

from .config    import DictionaryConfig
from .algorithm import EcdfSampler
from .          import net
from .base_types import EncodingStats

_LOG = logging.getLogger(__name__)


class MOPWorker(threading.Thread):
    def __init__(self, cfg: DictionaryConfig, *, k: int = 256, wid: int = 0):
        super().__init__(daemon=True)
        self.cfg, self.k, self.wid = cfg, k, wid

        self._input_path  = self._input_file()
        self._total_lines = sum(1 for _ in open(self._input_path))

        self.kv_map: Dict[str, int] = {}
        self.ecdf_vals: List[float] = []
        self.stats = EncodingStats()

        self.sub, self.push = net.connect_sub_push(cfg)

    def run(self) -> None:
        self._handshake()
        self._sampling_phase()
        tail_batch = self._main_loop()
        self._flush_batch(tail_batch)          # ← 字典已经最终同步完

            # -------- Bit-pack 编码列，并写出到 .bin + .txt --------
        print(">>> Writing bit-packed output...", flush=True)

        # 1. 重新按顺序遍历全部单词 → 映射成 id
        encoded_ids = np.fromiter(
            (self.kv_map[w] for w in self._all_words()),
            dtype=np.uint32
        )

        # 2. 使用自定义 pack() 函数做 bit packing
        bits, bw = bitpack(encoded_ids)   # bits: bytes, bw: bit width (e.g. 8)

        # 3. 写出 bitstream → binary 文件
        out_dir = self.cfg.worker_dir
        (out_dir / f"worker{self.wid}_bp.bin").write_bytes(bits)

        # 4. 写出元数据：使用了多少 bits
        (out_dir / f"worker{self.wid}_bw.txt").write_text(str(bw))

        print(f">>> Bit-packed column written: {len(bits)} bytes, bit-width: {bw}", flush=True)

        print(">>> Worker END about to send", flush=True)
        self.push.linger = 1000
        self.push.send_json({"type": "END", "wid": self.wid})
        sleep(0.05)
        print(">>> Worker END flushed", flush=True)
        self.push.close()
        self.sub.close()



    def _handshake(self):
        self.push.send_json({"type": "HANDSHAKE", "wid": self.wid})
        _LOG.info("Worker-%d handshake sent", self.wid)

    def _sampling_phase(self):
        ahead = int(self.cfg.look_ahead * self._line_count())
        proposal: List[str] = []
        with open(self._input_path) as fp:
            for _ in range(ahead):
                w = fp.readline().strip()
                if w:
                    proposal.append(w)
                    self.ecdf_vals.append(float(len(w)))

        samples = EcdfSampler.sample(self.ecdf_vals, self.k)
        self.push.send_json({"type": "SAMPLE",
                             "wid": self.wid,
                             "proposal": proposal,
                             "samples": samples})
        self._recv_kv_update()
        print(">>> SAMPLE size:", len(proposal))
        print(">>> samples len :", len(samples))

    def _main_loop(self) -> List[str]:
        batch: List[str] = []
        lines_read, t0 = 0, time.time()
        with open(self._input_path) as fp:
            for line in fp:
                lines_read += 1
                w = line.strip()
                if not w:
                    continue

                self.ecdf_vals.append(float(len(w))) 

                if w in self.kv_map:
                    continue

                batch.append(w)

                self.ecdf_vals.append(float(len(w)))

                if lines_read % 10_000 == 0:
                    _LOG.info("Worker-%d progressed %d / %d lines  (%.1f k/s)",
                              self.wid, lines_read, self._total_lines,
                              lines_read / max(time.time() - t0, 1e-3) / 1_000)

                if len(batch) >= self.cfg.worker_batch:
                    self._flush_batch(batch)
                    batch = []
        return batch   

    def _flush_batch(self, words: Sequence[str]):
        samples = EcdfSampler.sample(self.ecdf_vals, self.k)
        self.push.send_json({
            "type":   "NEW",
            "wid":    self.wid,
            "words":  list(words), 
            "samples": samples, 
            "flush":  True,
        })
        self.stats.message += 1
        self._recv_kv_update()  

    def _recv_kv_update(self, timeout_ms: int = 1000):
        poller = zmq.Poller(); poller.register(self.sub, zmq.POLLIN)
        events = dict(poller.poll(timeout_ms))
        if self.sub in events:
            msg = json.loads(self.sub.recv())
            if msg.get("type") == "DICT":
                self.kv_map.update(msg["dict"])

    def _input_file(self) -> pathlib.Path:
        return next(self.cfg.worker_dir.glob("*"))

    def _line_count(self) -> int:
        return sum(1 for _ in open(self._input_path))
    
    def _all_words(self):
        """Helper: iterate over *all* words in original order."""
        with open(self._input_path) as fp:
            for line in fp:
                w = line.strip()
                if w:
                    yield w