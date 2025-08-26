from __future__ import annotations
import argparse, csv, random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
from tqdm import tqdm

# -------- 15-puzzle basics --------
N = 4
GOAL = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0)
MOVES = {
    0:(1,4),1:(0,2,5),2:(1,3,6),3:(2,7),
    4:(0,5,8),5:(1,4,6,9),6:(2,5,7,10),7:(3,6,11),
    8:(4,9,12),9:(5,8,10,13),10:(6,9,11,14),11:(7,10,15),
    12:(8,13),13:(9,12,14),14:(10,13,15),15:(11,14)
}
TRAIN_COLS_15 = [f"pos{i}" for i in range(16)]

def state_key_tuple(s: tuple[int, ...]) -> str:
    return ",".join(map(str, s))

def random_walk_from_goal(steps: int, rng: random.Random) -> tuple[int, ...]:
    s = list(GOAL)
    z = s.index(0)
    prev = -1
    for _ in range(steps):
        opts = [nb for nb in MOVES[z] if nb != prev]  # avoid immediate backtrack
        nb = rng.choice(opts)
        s[z], s[nb] = s[nb], s[z]
        prev, z = z, nb
    return tuple(s)

# -------- training keys loading (chunked) --------
def row_to_key(row) -> str:
    return ",".join(str(int(row[c])) for c in TRAIN_COLS_15)

def load_train_keys(train_csv: str, chunksize: int) -> Set[str]:
    head = pd.read_csv(train_csv, nrows=1)
    missing = [c for c in TRAIN_COLS_15 if c not in head.columns]
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")
    keys: Set[str] = set()
    for chunk in tqdm(pd.read_csv(train_csv, usecols=TRAIN_COLS_15, chunksize=chunksize),
                      desc="Loading train keys", unit="rows"):
        for _, row in chunk.iterrows():
            keys.add(row_to_key(row))
    return keys

# -------- strata parsing --------
def parse_strata(spec: str) -> List[Tuple[int,int,int]]:
    """
    Parse 'a-b:k,c-d:m,...' -> list of (low, high, count).
    Example: '10-20:200,30-50:300,60-100:500'
    """
    out: List[Tuple[int,int,int]] = []
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        rng_s, cnt_s = p.split(":")
        lo_s, hi_s = rng_s.split("-")
        lo, hi, cnt = int(lo_s), int(hi_s), int(cnt_s)
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError(f"Bad range '{rng_s}'")
        if cnt <= 0:
            raise ValueError(f"Bad count '{cnt_s}'")
        out.append((lo, hi, cnt))
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Create stratified unseen 15-puzzle benchmark CSV (no overlap with training)."
    )
    ap.add_argument("--train-csv", required=True, help="15-p training CSV with pos0..pos15 (cost optional)")
    ap.add_argument("--out", required=True, help="Output CSV path (pos0..pos15)")
    ap.add_argument("--strata", required=False,
                    help="Depth bins as 'lo-hi:count,...'. Example: '10-20:200,30-50:300,60-100:500'")
    ap.add_argument("--n", type=int, default=0,
                    help="Total states if no --strata. Uses --walk-min/--walk-max uniformly.")
    ap.add_argument("--walk-min", type=int, default=20, help="Used only if --strata is not given")
    ap.add_argument("--walk-max", type=int, default=80, help="Used only if --strata is not given")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="Rows per chunk when reading training CSV")
    ap.add_argument("--max-attempts", type=int, default=10_000_000,
                    help="Safety cap on total attempts")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load training keys (unseen constraint)
    train_keys = load_train_keys(args.train_csv, chunksize=args.chunksize)
    print(f"[info] Loaded {len(train_keys):,} training keys")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [f"pos{i}" for i in range(16)]
    produced: Set[str] = set()
    total_goal = 0
    bins: List[Tuple[int,int,int]] = []

    if args.strata:
        bins = parse_strata(args.strata)
        total_goal = sum(cnt for _,_,cnt in bins)
        print(f"[info] Stratified generation, total target = {total_goal} "
              f"from bins: {bins}")
    else:
        if args.n <= 0:
            raise ValueError("Provide --strata or a positive --n")
        bins = [(args.walk_min, args.walk_max, args.n)]
        total_goal = args.n
        print(f"[info] Uniform generation {args.n} with depth in [{args.walk_min},{args.walk_max}]")

    attempts = 0
    written = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for (lo, hi, need) in bins:
            got = 0
            pbar = tqdm(total=need, desc=f"bin {lo}-{hi}", unit="state")
            while got < need and attempts < args.max_attempts:
                steps = rng.randint(lo, hi)
                s = random_walk_from_goal(steps, rng)
                k = state_key_tuple(s)
                attempts += 1
                if k in train_keys or k in produced:
                    continue
                w.writerow(list(s))
                produced.add(k)
                got += 1
                written += 1
                pbar.update(1)
            pbar.close()
            if got < need:
                print(f"[warn] bin {lo}-{hi}: produced {got}/{need}")

    if written < total_goal:
        print(f"[warn] Only generated {written}/{total_goal} unseen states after {attempts} attempts.")
    else:
        print(f"[done] Wrote {written} unseen states to {out_path.resolve()} after {attempts} attempts.")

if __name__ == "__main__":
    main()
