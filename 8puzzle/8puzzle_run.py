# -*- coding: utf-8 -*-
"""
Evaluate learned A* heuristics (from predict_dict.json) vs. classic heuristics.
Paths are hard-coded for Itay's project (runs directory + dataset).
With tqdm progress bars + helpful prints.
"""

from __future__ import annotations
import json, time, heapq
from pathlib import Path
import pandas as pd

# NEW: tqdm
try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("Please install tqdm: pip install tqdm")

# ---------------------------
# HARD-CODED PATHS / SETTINGS
# ---------------------------
RUNS_ROOT = Path(r"C:\Users\itay\OneDrive - post.bgu.ac.il\Desktop\School general\year5\felner\runs")
DATA_CSV  = Path(r"C:\Users\itay\OneDrive - post.bgu.ac.il\Desktop\School general\year5\felner\eight_puzzle_ground_truth.csv")
OUT_SUMMARY = Path(r"C:\Users\itay\OneDrive - post.bgu.ac.il\Desktop\School general\year5\felner\summary.csv")
PER_STATE_LOG = ""   # e.g. r"C:\...\per_state.csv" to log every state result
CLASSIC_HEURISTICS = ["manhattan", "misplaced", "linear_conflict"]
SAMPLE_N = 50       # how many states to sample from DATA_CSV if no fixed benchmark
MAX_EXPANSIONS = 200000
VERBOSE_STATES = False  # set True to show per-state progress bars (noisy)

# Optional filters to speed up initial inspection
METHOD_FILTER = set()      # e.g. {"random_forest","mlp"}; empty=set() means all
PERC_FILTER   = set()      # e.g. {10,50,90}; empty means all
RUN_FILTER    = set()      # e.g. {"run_001"}; empty means all

# ---------------------------
# 8-puzzle basics
# ---------------------------
GOAL = (1,2,3,4,5,6,7,8,0)
POS_TO_RC = [(i//3, i%3) for i in range(9)]
GOAL_POS = {v: POS_TO_RC[i] for i,v in enumerate(GOAL)}
MOVES = {0:(1,3),1:(0,2,4),2:(1,5),3:(0,4,6),4:(1,3,5,7),5:(2,4,8),6:(3,7),7:(4,6,8),8:(5,7)}

def key_from_state(s): return ",".join(map(str, s))

def neighbors(state):
    z = state.index(0)
    for nb in MOVES[z]:
        lst = list(state)
        lst[z], lst[nb] = lst[nb], lst[z]
        yield tuple(lst), 1

# ---------------------------
# Classic heuristics
# ---------------------------
def h_manhattan(s):
    return sum(abs(POS_TO_RC[i][0]-GOAL_POS[v][0]) + abs(POS_TO_RC[i][1]-GOAL_POS[v][1])
               for i,v in enumerate(s) if v!=0)

def h_misplaced(s):
    return sum(1 for i,v in enumerate(s) if v!=0 and v!=GOAL[i])

def h_linear_conflict(s):
    m = h_manhattan(s); conflicts = 0
    for r in range(3):
        tiles=[s[3*r+c] for c in range(3) if s[3*r+c]!=0 and GOAL_POS[s[3*r+c]][0]==r]
        for i in range(len(tiles)):
            for j in range(i+1,len(tiles)):
                if GOAL_POS[tiles[i]][1] > GOAL_POS[tiles[j]][1]: conflicts += 1
    for c in range(3):
        tiles=[s[r*3+c] for r in range(3) if s[r*3+c]!=0 and GOAL_POS[s[r*3+c]][1]==c]
        for i in range(len(tiles)):
            for j in range(i+1,len(tiles)):
                if GOAL_POS[tiles[i]][0] > GOAL_POS[tiles[j]][0]: conflicts += 1
    return m + 2*conflicts

CLASSIC_MAP={"manhattan":h_manhattan,"misplaced":h_misplaced,"linear_conflict":h_linear_conflict}

# ---------------------------
# A* search
# ---------------------------
def astar(start,h_func,max_expansions=500000):
    t0=time.perf_counter()
    if start==GOAL: return True,0,0,0.0
    g={start:0}; parent={}; closed=set(); nodes_exp=0; entry_id=0
    pq=[(h_func(start),0,entry_id,start)]; entry_id+=1
    while pq:
        f,gc,_,s=heapq.heappop(pq)
        if s in closed: continue
        closed.add(s)
        if s==GOAL:
            steps=0; cur=s
            while cur in parent: cur=parent[cur]; steps+=1
            return True,steps,nodes_exp,time.perf_counter()-t0
        nodes_exp+=1
        if nodes_exp>max_expansions:
            return False,-1,nodes_exp,time.perf_counter()-t0
        for nxt,cost in neighbors(s):
            ng=g[s]+cost
            if nxt not in g or ng<g[nxt]:
                g[nxt]=ng; parent[nxt]=s
                heapq.heappush(pq,(ng+h_func(nxt),ng,entry_id,nxt)); entry_id+=1
    return False,-1,nodes_exp,time.perf_counter()-t0

# ---------------------------
# Helpers
# ---------------------------
def df_to_states(df):
    arr=df[[f"pos{i}" for i in range(9)]].astype(int).values
    return [tuple(row.tolist()) for row in arr]

def load_benchmark():
    df=pd.read_csv(DATA_CSV)
    if SAMPLE_N>0 and len(df)>SAMPLE_N:
        df=df.sample(n=SAMPLE_N,random_state=42)
    return df_to_states(df)

def iter_learned_runs(runs_root: Path):
    for method_dir in runs_root.iterdir():
        if not method_dir.is_dir(): continue
        method=method_dir.name
        if METHOD_FILTER and method not in METHOD_FILTER: continue
        for pdir in method_dir.glob("percent_*"):
            if not pdir.is_dir(): continue
            try: percent=int(pdir.name.split("_")[1])
            except: continue
            if PERC_FILTER and percent not in PERC_FILTER: continue
            for rdir in pdir.glob("run_*"):
                if RUN_FILTER and rdir.name not in RUN_FILTER: continue
                pred=rdir/"predict_dict.json"
                if pred.exists():
                    yield method,percent,rdir.name,pred

# ---------------------------
# Main
# ---------------------------
def main():
    bench_states=load_benchmark()
    print(f"[info] Benchmark states loaded: {len(bench_states)}")
    summary=[]

    # ---- Classic heuristics with a progress bar
    print(f"[info] Evaluating classic heuristics: {CLASSIC_HEURISTICS}")
    for cname in tqdm(CLASSIC_HEURISTICS, desc="Classic heuristics", unit="heur"):
        hfunc = CLASSIC_MAP[cname]
        succ,mv_sum,n_sum,t_sum=0,0,0,0.0
        iterable = (tqdm(bench_states, desc=f"{cname} states", leave=False) if VERBOSE_STATES else bench_states)
        for s in iterable:
            ok,steps,nodes,secs=astar(s,hfunc,MAX_EXPANSIONS)
            succ+=ok; mv_sum+=(steps if ok else 0); n_sum+=nodes; t_sum+=secs
        summary.append({"engine":"classic","method":cname,"percent":"","run_id":"",
                        "bench_n":len(bench_states),"success_rate":succ/len(bench_states),
                        "avg_moves":(mv_sum/succ if succ else -1),"avg_nodes":n_sum/len(bench_states),
                        "avg_time":t_sum/len(bench_states)})

    # ---- Learned heuristics list & progress bar
    learned_list = list(iter_learned_runs(RUNS_ROOT))
    print(f"[info] Found learned runs: {len(learned_list)} (method/percent/run)")
    for method,percent,run_id,pred_path in tqdm(learned_list, desc="Learned heuristics", unit="run"):
        # Small print header per run (not too spammy)
        tqdm.write(f"[run] {method} | {percent}% | {run_id} -> {pred_path.name}")
        with open(pred_path,"r",encoding="utf-8") as f:
            predict=json.load(f)

        def h_learned(s):
            # If every benchmark state is guaranteed to exist in predict_dict.json,
            # you can use: return float(predict[key_from_state(s)])
            return float(predict.get(key_from_state(s), h_manhattan(s)))

        succ,mv_sum,n_sum,t_sum=0,0,0,0.0
        iterable = (tqdm(bench_states, desc=f"{method} {percent}% {run_id}", leave=False)
                    if VERBOSE_STATES else bench_states)
        for s in iterable:
            ok,steps,nodes,secs=astar(s,h_learned,MAX_EXPANSIONS)
            succ+=ok; mv_sum+=(steps if ok else 0); n_sum+=nodes; t_sum+=secs

        summary.append({"engine":"learned","method":method,"percent":percent,"run_id":run_id,
                        "bench_n":len(bench_states),"success_rate":succ/len(bench_states),
                        "avg_moves":(mv_sum/succ if succ else -1),"avg_nodes":n_sum/len(bench_states),
                        "avg_time":t_sum/len(bench_states)})

    # Save outputs
    df=pd.DataFrame(summary).sort_values(by=["engine","method","percent","run_id"])
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SUMMARY,index=False)
    print(f"[saved] Summary -> {OUT_SUMMARY}")
    if PER_STATE_LOG:
        print("[note] You set PER_STATE_LOG, but per-state collection is disabled in this snippet to keep memory small.")

if __name__=="__main__":
    main()
