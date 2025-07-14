from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


def _abs_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


@dataclass
class DictionaryConfig:

    worker_dir: Path | str = _abs_path("data")
    look_ahead: float = 0.10
    worker_batch: int = 500

    batching: int = 50
    batch_time_ms: int = 500

    pub_port: int = 5556
    pull_port: int = 5557
    leader_ip: str = "127.0.0.1"
    test_leader: bool = False

    mop_levels: int = 4
    pitch: int = 20

    parquet_page: int = 128 * 1024

    timeout_ms: int = 5_000

    def __post_init__(self):
        self.worker_dir = _abs_path(self.worker_dir)