# -*- coding: utf-8 -*-
"""
Play 15-puzzle on unseen states with three heuristics:
1) Classic Manhattan
2) Classic Linear Conflict
3) XGB (transfer, uses model.json + encoder.joblib)
4) XGB (scratch,  uses model.json + encoder.joblib)

Hardcoded paths:
- Benchmark CSV (pos0..pos15): /home/itayraz/felner/bench_15p_unseen.csv
- Transfer model dir:           /home/itayraz/felner/models_for_15_puzzle/15_transfer_model
- Scratch model dir:            /home/itayraz/felner/models_for_15_puzzle/scratch_model
"""

import heapq
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import xgboost as xgb

# ---------------------------
# Hard-coded paths
# ---------------------------
BENCHMARK = "/home/itayraz/felner/bench_15p_unseen.csv"
MODEL_TRANSFER_DIR = "/home/itayraz/felner/models_for_15_puzzle/15_transfer_model"
MODEL_SCRATCH_DIR  = "/home/itayraz/felner/models_for_15_puzzle/scratch_model"

# file names expected inside those dirs:
MODEL_FILE = "model.json"
ENC_FILE   = "encoder.joblib"

# ---------------------------
# 15-puzzle basics (A*)
# ---------------------------
N = 4
GOAL = tuple(list(range(1,16)) + [0])

MOVES = {
    0:(1,4),1:(0,2,5),2:(1,3,6),3:(2,7),
    4:(0,5,8),5:(1,4,6,9),6:(2,5,7,10),7:(3,6,11),
    8:(4,9,12),9:(5,8,10,13),10:(6,9,11,14),11:(7,10,15),
    12:(8,13),13:(9,12,14),14:(10,13,15),15:(11,14)
}

def neighbors(state):
    z = state.index(0)
    for nb in MOVES[z]:
        lst = list(state); lst[z], lst[nb] = lst[nb], lst[z]
        yield tuple(lst), 1

def astar(start, h_func, max_expansions=1_000_000):
    """Standard A*; returns (ok, steps, nodes_expanded)."""
    if start == GOAL:
        return True, 0, 0
    g = {start: 0}
    closed = set()
    nodes = 0
    entry = 0
    pq = [(h_func(start), 0, entry, start)]; entry += 1
    parent = {}
    while pq:
        f, gc, _, s = heapq.heappop(pq)
        if s in closed:
            continue
        closed.add(s)
        if s == GOAL:
            steps = 0
            cur = s
            while cur in parent:
                cur = parent[cur]
                steps += 1
            return True, steps, nodes
        nodes += 1
        if nodes > max_expansions:
            return False, -1, nodes
        for nxt, cost in neighbors(s):
            ng = g[s] + cost
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = s
                heapq.heappush(pq, (ng + h_func(nxt), ng, entry, nxt))
                entry += 1
    return False, -1, nodes

# ---------------------------
# Classic heuristics
# ---------------------------
def h_manhattan(s):
    total = 0
    for i, v in enumerate(s):
        if v == 0: continue
        r, c = divmod(i, N)
        gr, gc = divmod(v-1, N)
        total += abs(r-gr) + abs(c-gc)
    return total

def h_linear_conflict(s):
    m = h_manhattan(s)
    conflicts = 0
    # rows
    for r in range(N):
        idx = [r*N + c for c in range(N)]
        tiles = [s[i] for i in idx if s[i] != 0 and divmod(s[i]-1, N)[0] == r]
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                if divmod(tiles[i]-1, N)[1] > divmod(tiles[j]-1, N)[1]:
                    conflicts += 1
    # cols
    for c in range(N):
        idx = [r*N + c for r in range(N)]
        tiles = [s[i] for i in idx if s[i] != 0 and divmod(s[i]-1, N)[1] == c]
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                if divmod(tiles[i]-1, N)[0] > divmod(tiles[j]-1, N)[0]:
                    conflicts += 1
    return m + 2*conflicts

# ---------------------------
# XGB heuristic loader (model.json + encoder.joblib)
# ---------------------------
def load_xgb_bundle(model_dir: str):
    """Load XGBRegressor from model.json and OneHotEncoder from encoder.joblib."""
    model_path = Path(model_dir) / MODEL_FILE
    enc_path   = Path(model_dir) / ENC_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost model file: {model_path}")
    if not enc_path.exists():
        raise FileNotFoundError(f"Missing encoder file: {enc_path}")

    # Load encoder
    enc = joblib.load(enc_path)

    # Load model.json into an XGBRegressor
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    return model, enc

def make_h_learned(model_dir: str):
    """Return h(s) that predicts with loaded model+encoder (one-hot only)."""
    model, enc = load_xgb_bundle(model_dir)
    cols = [f"pos{i}" for i in range(16)]
    def h(s):
        # build a single-row DataFrame and one-hot transform
        row = pd.DataFrame([list(s)], columns=cols)
        X = enc.transform(row)  # scipy sparse
        pred = model.predict(X)[0]
        return float(pred)
    return h

# ---------------------------
# Run benchmark
# ---------------------------
def run_benchmark(csv_path: str, heuristics: dict, limit: int = 100, max_expansions: int = 1_000_000):
    df = pd.read_csv(csv_path)
    # Expect columns pos0..pos15
    needed = [f"pos{i}" for i in range(16)]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Benchmark CSV missing columns: {miss}")

    # take first 'limit' states
    if len(df) > limit:
        df = df.head(limit).copy()

    results = []
    states = [tuple(int(df.iloc[i][f"pos{j}"]) for j in range(16)) for i in range(len(df))]

    for name, h in heuristics.items():
        succ = mv_sum = nodes_sum = 0
        for s in tqdm(states, desc=f"Solving with {name}", unit="puz"):
            ok, steps, nodes = astar(s, h, max_expansions=max_expansions)
            succ += int(ok)
            if ok and steps >= 0:
                mv_sum += steps
            nodes_sum += nodes
            results.append(dict(heuristic=name, ok=bool(ok), steps=steps, nodes=nodes,
                                state=",".join(map(str, s))))
        bench_n = len(states)
        avg_moves = (mv_sum / succ) if succ else -1
        avg_nodes = nodes_sum / bench_n
        print(f"[summary] {name:18s} | succ={succ/bench_n:.3f} | "
              f"avg_moves={avg_moves:.2f} | avg_nodes={avg_nodes:.1f}")
    return pd.DataFrame(results)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    heuristics = {
        "classic_manhattan": h_manhattan,
        "classic_linear":    h_linear_conflict,
        "xgb_transfer":      make_h_learned(MODEL_TRANSFER_DIR),
        "xgb_scratch":       make_h_learned(MODEL_SCRATCH_DIR),
        # If you ever want admissible learned estimates:
        # "xgb_transfer_clamped": lambda s: min(make_h_learned(MODEL_TRANSFER_DIR)(s), h_manhattan(s)),
    }

    df_res = run_benchmark(BENCHMARK, heuristics, limit=1000, max_expansions=1_000_000)

    # Print small sample of per-case results
    print("\n=== Sample of per-case results ===")
    print(df_res.head(20))
    df_res.to_csv("results_15p.csv", index=False)

    # Aggregated table
    print("\n=== Aggregated results ===")
    agg = df_res.groupby("heuristic").agg(
        success_rate=("ok", "mean"),
        avg_moves=("steps", lambda x: x[x >= 0].mean() if (x >= 0).any() else -1),
        avg_nodes=("nodes", "mean")
    )
    print(agg)
