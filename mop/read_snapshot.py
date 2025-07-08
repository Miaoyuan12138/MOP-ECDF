import sys
from pprint import pprint
from mop.snapshot import load_snapshot

def main(path: str) -> None:
    kv_map, ecdf = load_snapshot(path)

    print(f"Loaded snapshot: {path}")
    print(f"Dictionary size : {len(kv_map):,}")
    print(f"ECDF sample size: {ecdf.size}")

    print("\nFirst 20 dictionary items:")
    for i, (word, key) in enumerate(kv_map.items()):
        if i == 20:
            break
        print(f"  {word:<12s} → {key}")

    if (ecdf[1:] < ecdf[:-1]).any():
        print("ECDF samples are not non-decreasing!")
    else:
        print("ECDF samples are monotone - looks good.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_snapshot.py <26.7.6 feather>")
        sys.exit(1)
    main(sys.argv[1])
