# Using-Machine-Learning-to-Solve-the-8-Puzzle

This repository contains code and experiments for learning and evaluating heuristics on the classic **8-puzzle** and **15-puzzle** search problems.  
The project explores how machine learning models (e.g., XGBoost) can be trained on ground-truth shortest distances and later used as heuristics for A* search.

---

## Repository Structure

### `15puzzle/`
Experiments with the 15-puzzle.

- **`create_15_puzzle_gt.py`**  
  Generates ground-truth states with their optimal distances.  
  - You control the maximum depth and number of states.  
  - Output is used for training ML models.

- **`create_15puzzle_benchmark.py`**  
  Creates benchmark states for evaluation.  
  - Ensures that states are **not** included in the training GT.  
  - Used to test heuristic performance.

- **`15puzzle_run.py`**  
  Runs experiments with different heuristics:  
  - Models trained only on 8-puzzle  
  - Models trained on both 8- and 15-puzzle  
  - Models trained only on 15-puzzle  
  Reports performance metrics (average nodes expanded, moves, success rate, etc.).

---

### `8puzzle/`
Experiments with the 8-puzzle.

- **`create_learned_hueristics.py`**  
  Trains ML models on ground-truth (GT) distances of the 8-puzzle.  
  - For each model, trains on varying fractions of the data (10%, 20%, … 90%).  
  - For each fraction, creates **X random models** (different splits).  
  - Predictions of all states are precomputed and stored (used as a heuristic hash map, not runtime prediction).

- **`8puzzle_run.py`**  
  Runs A* search with all heuristics (classic + learned).  
  Outputs summary statistics:  
  - Average nodes expanded  
  - Average moves  
  - Success rate

---

### `GT_data/`
- **`15puzzle_benchmark.csv`** – Benchmark states used in experiments.  
- **`GT_data.zip`** – Ground-truth distances for:  
  - 8-puzzle (complete database)  
  - 15-puzzle (partial database, limited depth due to size)  
  > ⚠️ Too large to store uncompressed, provided as a ZIP file.

---

### `results/`
Contains the results of experiments on both the 8-puzzle and 15-puzzle.


