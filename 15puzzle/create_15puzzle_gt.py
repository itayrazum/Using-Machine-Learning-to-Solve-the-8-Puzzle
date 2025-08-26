# -*- coding: utf-8 -*-
"""
Build a *sampled* ground truth CSV for the 15-puzzle (4x4, tiles 0..15; 0=blank).
We generate solvable instances by random walks from GOAL, then solve each optimally
with IDA* using Manhattan + Linear Conflict (admissible), and write:
    pos0..pos15,cost
to fifteen_puzzle_ground_truth.csv (or a path you specify).

WARNING: Solving hard 15p instances optimally can be slow. Start with small --samples.
"""

from __future__ import annotations
import argparse, random, sys, time
from collections import deque
from pathlib import Path
import csv

# 15-puzzle basics
GOAL = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0)
N = 4
POS_TO_RC = [(i//N, i%N) for i in range(N*N)]
GOAL_POS = {v: POS_TO_RC[i] for i,v in enumerate(GOAL)}
# blank moves
MOVES = {
    0:(1,4),1:(0,2,5),2:(1,3,6),3:(2,7),
    4:(0,5,8),5:(1,4,6,9),6:(2,5,7,10),7:(3,6,11),
    8:(4,9,12),9:(5,8,10,13),10:(6,9,11,14),11:(7,10,15),
    12:(8,13),13:(9,12,14),14:(10,13,15),15:(11,14)
}

def neighbors(state):
    z = state.index(0)
    for nb in MOVES[z]:
        lst = list(state)
        lst[z], lst[nb] = lst[nb], lst[z]
        yield tuple(lst)

# Classic admissible heuristics: Manhattan + Linear Conflict
def h_manhattan(s):
    total = 0
    for i,v in enumerate(s):
        if v == 0: continue
        r,c = POS_TO_RC[i]; gr,gc = GOAL_POS[v]
        total += abs(r-gr) + abs(c-gc)
    return total

def h_linear_conflict(s):
    m = h_manhattan(s)
    conflicts = 0
    # rows
    for r in range(N):
        row_idx = [r*N + c for c in range(N)]
        tiles = [s[i] for i in row_idx if s[i]!=0 and GOAL_POS[s[i]][0]==r]
        for i in range(len(tiles)):
            for j in range(i+1,len(tiles)):
                if GOAL_POS[tiles[i]][1] > GOAL_POS[tiles[j]][1]:
                    conflicts += 1
    # cols
    for c in range(N):
        col_idx = [r*N + c for r in range(N)]
        tiles = [s[i] for i in col_idx if s[i]!=0 and GOAL_POS[s[i]][1]==c]
        for i in range(len(tiles)):
            for j in range(i+1,len(tiles)):
                if GOAL_POS[tiles[i]][0] > GOAL_POS[tiles[j]][0]:
                    conflicts += 1
    return m + 2*conflicts

# IDA* for optimal cost
def ida_star(start):
    if start == GOAL: return 0
    bound = h_linear_conflict(start)
    path = [start]
    g = 0
    while True:
        t = _search(path, g, bound)
        if isinstance(t, int) and t >= 0:
            return t  # found optimal cost
        if t == float('inf'):
            return None
        bound = t

def _search(path, g, bound):
    s = path[-1]
    f = g + h_linear_conflict(s)
    if f > bound: return f
    if s == GOAL: return g
    min_bound = float('inf')
    z = s.index(0)
    for nb in MOVES[z]:
        lst = list(s); lst[z], lst[nb] = lst[nb], lst[z]
        nxt = tuple(lst)
        if len(path) >= 2 and nxt == path[-2]:  # avoid 2-step backtrack
            continue
        path.append(nxt)
        t = _search(path, g+1, bound)
        if isinstance(t, int) and t >= 0:
            return t
        if t < min_bound:
            min_bound = t
        path.pop()
    return min_bound

def random_walk_from_goal(steps, rng):
    s = list(GOAL)
    z = s.index(0)
    prev = -1
    for _ in range(steps):
        move_opts = [nb for nb in MOVES[z] if nb != prev]
        nb = rng.choice(move_opts)
        s[z], s[nb] = s[nb], s[z]
        prev, z = z, nb
    return tuple(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fifteen_puzzle_ground_truth.csv")
    ap.add_argument("--samples", type=int, default=500, help="number of states to sample & solve")
    ap.add_argument("--walk-min", type=int, default=10, help="min random-walk steps from goal")
    ap.add_argument("--walk-max", type=int, default=60, help="max random-walk steps from goal")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] sampling {args.samples} solvable 15p states (random walks {args.walk_min}-{args.walk_max})")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"pos{i}" for i in range(16)] + ["cost"])
        for i in range(args.samples):
            steps = rng.randint(args.walk_min, args.walk_max)
            s = random_walk_from_goal(steps, rng)
            t0 = time.perf_counter()
            cost = ida_star(s)
            dt = time.perf_counter() - t0
            if cost is None:
                print(f"[warn] IDA* failed (unexpected). Skipping state {i}.")
                continue
            w.writerow(list(s) + [cost])
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{args.samples}] cost={cost} time={dt:.3f}s")

    print(f"[done] wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
