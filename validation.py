'''''
find . -name '__pycache__' -exec rm -r {} +
python validation.py
'''''

from __future__ import annotations
import logging, random, string, tempfile
from pathlib import Path
from statistics import mean
from time import time

import numpy as np, pyarrow as pa, pyarrow.parquet as pq

from mop.config  import DictionaryConfig
from mop.leader  import MOPDictLeader
from mop.worker  import MOPWorker

from mop.worker import MOPWorker
import inspect, sys
import os
print(">>> Worker implementation from:", inspect.getsourcefile(MOPWorker), file=sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(threadName)-10s  %(levelname)s  %(message)s",
)
_LOG = logging.getLogger("VALIDATE")

TXT_SOURCE    = Path("20k.txt")
PARQUET_FILE  = Path("mix_20k.parquet")
TOTAL_ROWS   = 100_000
BOOT_ROWS    = 80_000 
K            = 256  

def rand_word() -> str:
    L = random.randint(3, 10)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(L))

def make_parquet(pq_file: Path, txt_file: Path):
    """
    Build a 100 k-row Parquet table from `txt_file` so that
      • 70 % lines come from the 1 000 most-common words  (hot set)
      • 30 % lines come from the remaining long-tail words
    """
    assert txt_file.exists(), f"{txt_file} not found"

    _LOG.info("Generating Parquet from %s …", txt_file)
    random.seed(42); np.random.seed(42)

    vocab = [w.strip() for w in txt_file.read_text().splitlines() if w.strip()]
    assert len(vocab) >= 1_000, "need at least 1 000 words in TXT_SOURCE"

    hot_pool  = vocab[:1_000]
    tail_pool = vocab[1:]

    rows: list[str] = []

    SEGMENT = 10_000
    for _ in range(TOTAL_ROWS // SEGMENT):
        seg_rows: list[str] = []

        hot_idx = np.random.zipf(1.3, int(SEGMENT * 0.70)) - 1
        hot_idx = np.clip(hot_idx, 0, len(hot_pool) - 1)
        seg_rows.extend(hot_pool[i] for i in hot_idx)

        seg_rows.extend(random.choices(tail_pool, k=SEGMENT - len(seg_rows)))

        random.shuffle(seg_rows)
        rows.extend(seg_rows)

    tbl = pa.table({"item_name": pa.array(rows, pa.string())})
    pq.write_table(tbl, pq_file,
                   compression="snappy",
                   use_dictionary=True,
                   row_group_size=32_000)

    _LOG.info("Parquet ready: %s (%.2f MiB)",
              pq_file, pq_file.stat().st_size / 2**20)

def parquet_to_txt(pfile: Path, txt_dir: Path) -> tuple[list[str], list[str]]:
    df = pq.read_table(pfile, columns=["item_name"]).to_pandas()
    words = df["item_name"].tolist()
    boot, hold = words[:BOOT_ROWS], words[BOOT_ROWS:]
    txt_dir.mkdir(parents=True, exist_ok=True)
    with (txt_dir / "words.txt").open("w") as fp:
        fp.write("\n".join(words)) 
    assert (txt_dir / "words.txt").stat().st_size > 2, "words.txt is empty!"          
    return boot, hold

def validate():
    if not PARQUET_FILE.exists():
        make_parquet(PARQUET_FILE, TXT_SOURCE)

    work_dir = Path(tempfile.mkdtemp()) / "data"
    boot_words, hold_words = parquet_to_txt(PARQUET_FILE, work_dir)

    cfg = DictionaryConfig(
        worker_dir   = work_dir,
        look_ahead   = 0.25, 
        worker_batch = 500,
        test_leader  = True,  
    )

    leader = MOPDictLeader(cfg, k=K)
    worker = MOPWorker(cfg, k=K)

    leader.start(); worker.start()
    worker.join(); leader.join()

    kv = leader.encoder.kv_map
    assert len(kv) == len(set(boot_words + hold_words)), "dict size mismatch"

    keys = list(kv.values())
    assert keys == sorted(keys), "keys not monotone increasing"

    lens_all  = np.sort([len(w) for w in boot_words + hold_words])
    lens_hold = [len(w) for w in hold_words]

    def true_ecdf(x: float) -> float:
        return (lens_all <= x).sum() / (len(lens_all) - 1)

    errs = [abs(leader.ecdf(float(l)) - true_ecdf(float(l))) for l in lens_hold]
    max_err, avg_err = max(errs), mean(errs)
    assert max_err < 0.03, f"ECDF max error {max_err:.4f} > 0.03"

    raw_txt = work_dir / "words.txt"
    raw_size = os.path.getsize(raw_txt)

    # Parquet 字典 + 最小整型列（非 bit-packing，仅压缩 4× 左右）
    par_file = work_dir / "worker0_mop.parquet"
    if par_file.exists():
        par_size = os.path.getsize(par_file)
        print(f"\nRaw txt size          : {raw_size/1024:.1f} KiB")
        print(f"Parquet uint encoding : {par_size/1024:.1f} KiB  "
            f"({par_size/raw_size:.2%} of raw)")
    else:
        print("No Parquet file found.")

    # Bit-packed 二进制位流列
    bin_file = work_dir / "worker0_bp.bin"
    bw_file = work_dir / "worker0_bw.txt"
    if bin_file.exists() and bw_file.exists():
        bin_size = os.path.getsize(bin_file)
        with open(bw_file) as f:
            bit_width = int(f.read().strip())

        print(f"Bit-packed .bin size  : {bin_size/1024:.1f} KiB  "
            f"({bin_size/raw_size:.2%} of raw)  [bit width = {bit_width}]")
    else:
        print("No bit-packed .bin found.")

    print("\n✓ All checks passed.")
    print(f"Dictionary size : {len(kv)}")
    print(f"ECDF max error  : {max_err:.4f}")
    print(f"ECDF mean error : {avg_err:.4f}")
    print("First 20 entries:")
    for w, k in list(kv.items())[:20]:
        print(f"  {w:<12s} → {k}")

if __name__ == "__main__":
    validate()