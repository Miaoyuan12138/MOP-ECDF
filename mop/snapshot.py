from __future__ import annotations
from pathlib import Path
import pyarrow as pa, pyarrow.feather as fe
import numpy as np
from typing import Dict, Tuple

def dump_snapshot(leader, file):
    kv       = leader.encoder.kv_map 
    samples  = leader.global_samples  
    n        = len(kv)

    word_col   = pa.array(list(kv.keys()),  pa.string())
    id_col     = pa.array(list(kv.values()), pa.uint32())

    sample_scalar = pa.scalar(samples.astype("float32").tolist(),
                              type=pa.list_(pa.float32()))
    sample_col    = pa.repeat(sample_scalar, n)

    tbl = pa.table({"word": word_col,
                    "id":   id_col,
                    "ecdf_samples": sample_col})

    fe.write_feather(tbl, file, compression="zstd")


def load_snapshot(file: str | Path) -> Tuple[Dict[str,int], np.ndarray]:
    tbl = fe.read_table(file)
    words   = tbl.column("word").to_pylist()
    ids     = tbl.column("id").to_pylist()
    kv      = dict(zip(words, ids))
    samples = np.asarray(tbl.column("ecdf_samples")[0].as_py(), dtype=float)
    return kv, samples
