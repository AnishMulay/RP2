# SDD: Creating a Python library out of the clustered push relabel solver

### 1. Context & Primary Goals

**Context:**
This repository houses a clustered push-relabel solver developed under tight time constraints for a research thesis. Consequently, the codebase currently reflects a "research script" paradigm rather than a production-grade library. Core components—specifically a K-level clustering algorithm (with GPU memory batching) and two distinct push-relabel implementations (Min Cost Bipartite Matching and Optimal Transport)—are heavily coupled in monolithic files. Additionally, experimental scripts necessary for thesis reproducibility exist alongside the core logic.

**Primary Goals:**

1. **Professional Library Architecture:** Restructure the repository into a production-ready, pip-installable Python library adhering to senior-engineering standards (e.g., modularity, separation of concerns, clean public interfaces).
2. **Intuitive Public API:** Redesign the user-facing API taking inspiration from industry standards like `ott-jax` and `POT`. The API must be simple to invoke for end-users.
3. **Component Decoupling:** * Expose the **K-level clustering** (with its batching logic) as a standalone, reusable module.
    - Separate the **Min Cost Bipartite Matching** solver from the **Optimal Transport** solver into distinct, easily importable classes/functions.
4. **Preserve Reproducibility:** Isolate the experimental setups into a dedicated `experiments/` directory. These must remain fully functional and reproducible for the upcoming research paper without cluttering the library's core source code.
5. **Strict AI-Refactoring Guardrails:** The restructuring will be executed by an AI agent. The design must provide rigid, granular boundaries to ensure the AI *moves and refactors* code without *rewriting or breaking* the underlying mathematical and algorithmic logic.

---

## 2. Target Low-Level Design (LLD) & Repository Architecture

**Crucial AI Directive:** The core mathematical logic and GPU operations inside the solver and clustering classes are proven to work. **Do not alter the internal math or tensor operations.** Your job is to decouple these classes into separate files and build a clean, functional API wrapper around them.

### 2.1 Target Repository Architecture

We will migrate from a flat "research script" layout to a standard Python package layout (`src/` layout).

Plaintext

`clustered-push-relabel/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Automated pytest and linting runner
├── src/
│   └── clustered_push_relabel/
│       ├── __init__.py               # Exposes the clean public API
│       ├── clustering/
│       │   ├── __init__.py
│       │   ├── k_level.py            # FastGPUMultiLevelClustering
│       │   └── two_level.py          # FastGPUClustering
│       ├── solvers/
│       │   ├── __init__.py
│       │   ├── bipartite.py          # GPUClusteredSolver (Min Cost)
│       │   └── transport.py          # GPUClusteredOTSolver (Optimal Transport)
│       └── utils/
│           └── distance.py           # Any shared distance/metric helpers
├── tests/                            # THE TESTING HARNESS
│   ├── conftest.py                   # Shared pytest fixtures (e.g., dummy tensors)
│   ├── test_clustering.py
│   ├── test_bipartite.py
│   └── test_transport.py
├── experiments/                      # Isolated Research Code
│   ├── runners/                      # The various e1_, e3_, experiment*.py scripts
│   ├── batch_tooling/                # SLURM/Batch generation scripts
│   └── data/                         # README on how to fetch data (do not commit data)
├── pyproject.toml
└── README.md`

### 2.2 Low-Level Design: The `ott-jax` / `POT` Inspired Public API

To provide a clean user experience without breaking the existing stateful classes, we will implement **functional wrappers** in the `__init__.py` files. Users will call a function that instantiates the class, runs `solve()`, extracts the result from the class state, and returns a clean dictionary or Dataclass.

**1. The Clustering Interface (`src/clustered_push_relabel/clustering/k_level.py`)**

- **Keep:** `FastGPUMultiLevelClustering` class exactly as is.
- **Add Wrapper:** ```python
def k_level_cluster(x, y, epsilon, k=4, batch_size=2048, metric="L2"):
"""Public functional interface for K-Level Clustering."""
model = FastGPUMultiLevelClustering(epsilon=epsilon, k=k, batch_size=batch_size, metric=metric)
blue_coo, red_coo, levels_red, levels_blue = model.run(x, y)
return {
"blue_cover": blue_coo,
"red_cover": red_coo,
"levels_red": levels_red,
"levels_blue": levels_blue
}

**2. The Bipartite Matching Solver (`src/clustered_push_relabel/solvers/bipartite.py`)**

- **Keep:** Both 2-level and K-level `GPUClusteredSolver` classes.
- **Add Wrapper:** ```python
def solve_bipartite_matching(x, y, epsilon, k=4, batch_size=None, metric="L2"):
"""Public interface. Returns the assignment vector."""
solver = GPUClusteredSolver(x, y, epsilon, k=k, batch_size=batch_size, metric=metric)
solver.solve()
# Extract the internal state (self.MB) to return to the user
return {"assignment_vector": solver.MB}

**3. The Optimal Transport Solver (`src/clustered_push_relabel/solvers/transport.py`)**

- **Keep:** `GPUClusteredOTSolver` class.
- **Add Wrapper:**Python
    
    `def solve_optimal_transport(x, y, mass_x, mass_y, epsilon, k=4, batch_size=2048, metric="L2"):
        """Public interface. Extracts the sparse flow plan from solver state."""
        solver = GPUClusteredOTSolver(x, y, mass_x, mass_y, epsilon, k, batch_size, metric)
        solver.solve()
        solver.de_scale_and_cleanup()
        # AI MUST identify the exact flow tensors (active_edges_u, active_edges_v, active_flow)
        return {
            "source_edges": solver.active_edges_u,
            "target_edges": solver.active_edges_v,
            "flow": solver.active_flow
        }`
    

### 2.3 Preserving the Experimental Setup

The experiment scripts (e.g., `experiment_mnist.py`, `e1_synthetic_vs_exact.py`) currently rely on hardcoded paths and are mixed with library code.

- **Action:** Move all experiment files into the `experiments/` directory.
- **Action:** Update imports inside these scripts to use the newly packaged library (e.g., `from clustered_push_relabel.solvers import solve_bipartite_matching`).
- **Action:** Standardize data paths. Ensure that scripts utilize a central `BASE_DIR` dynamically generated via `pathlib` so that running scripts from different directories does not cause `FileNotFound` errors for `data/train-images-idx3-ubyte.gz` or `./siftsmall`.

### 2.4 The Automated Testing Harness & CI (Strict Prerequisite)

Because the codebase is currently functional, **testing must be established before refactoring**. We cannot rely on manual testing during the AI refactor.

1. **Baseline Extraction:** The AI must look at the outputs of `experiment.py` and `experiment_runner.py` to understand the expected shapes and valid outputs of the current algorithms.
2. **Test Suite Creation (`pytest`):**
    - Create `tests/test_clustering.py`: Generate small synthetic `P_red` and `P_blue` tensors. Assert that the returned COO tuples match expected tensor dimensions and types.
    - Create `tests/test_bipartite.py`: Assert that `solver.MB` contains valid indices within the bounds of `P_red`.
    - Create `tests/test_transport.py`: Assert that the total flow sums to the expected mass (mass conservation).
3. **Continuous Integration:** Create a `.github/workflows/ci.yml` file that installs the package via `pip install .[test]`, installs `pytest` and `torch`, and runs the test suite on every commit.

---

## 3. Step-by-Step Execution Plan (AI Directives)

**Operating Principle:** *Test-Driven Migration.* The AI must verify the state of the solver after every structural change. If a test fails, the AI must revert the last move and fix the import/dependency before proceeding.

### Phase 1: The "Safety Net" (Baseline Testing)

*Goal: Capture the current working state of the research code.*

1. **Extract Baseline Data:** Run `experiment.py` or `experiment_pot.py` on a small synthetic dataset. Capture the output (e.g., the `MB` assignment vector or the total flow value).
2. **Create `tests/` Directory:** Create the folder structure and a `conftest.py` with a PyTorch device fixture (CPU/CUDA detection).
3. **Write Baseline Tests:** Create `tests/test_legacy.py`. This test should import the classes from the current root files (e.g., `from clustered_push_relabel import GPUClusteredSolver`) and assert that they produce the same output captured in Step 1.
4. **Verification:** Run `pytest`. Proceed only if tests pass.

### Phase 2: Package Scaffolding

*Goal: Create the professional container.*

1. **Initialize `pyproject.toml`:** Define the project name, version `0.1.0`, and dependencies (torch, numpy, pot, etc.).
2. **Create `src/` Layout:** Create `src/clustered_push_relabel/` and all subdirectories (`clustering/`, `solvers/`, `utils/`) with empty `__init__.py` files.
3. **Install Editable:** Run `pip install -e .`. This ensures the AI can use `import clustered_push_relabel` during the migration.

### Phase 3: Incremental Code Migration (The "Slicing")

*The AI must follow this order to handle dependencies correctly:*

1. **Step 3.1: Clustering Migration**
    - Move `FastGPUMultiLevelClustering` to `clustering/k_level.py`.
    - Move `FastGPUClustering` to `clustering/two_level.py`.
    - **Test:** Update `tests/test_clustering.py` to import from the new location. Run `pytest`.
2. **Step 3.2: Solver Migration**
    - Move `GPUClusteredSolver` (Bipartite) to `solvers/bipartite.py`.
    - Move `GPUClusteredOTSolver` (Transport) to `solvers/transport.py`.
    - **Crucial:** The AI must fix the internal imports (e.g., the solvers now need to import clustering from `..clustering.k_level`).
    - **Test:** Run `pytest`. Ensure the solvers still work in their new homes.
3. **Step 3.3: Functional Wrappers**
    - Implement the clean `k_level_cluster()`, `solve_bipartite_matching()`, and `solve_optimal_transport()` functions in the respective `__init__.py` or module files as defined in the LLD (Section 2.2).
    - **Test:** Add a new test file `tests/test_public_api.py` to ensure the clean functions return the correct dictionaries/data.

### Phase 4: Experiment Isolation & Cleanup

1. **Move Experiments:** Relocate all `e1_`, `e3_`, and `experiment_` files to `experiments/runners/`.
2. **Fix Experiment Imports:** Update all relocated scripts to use the new library format:
    - *Change:* `import clustered_push_relabel`
    - *To:* `from clustered_push_relabel.solvers import solve_bipartite_matching`.
3. **Path Standardization:** Replace hardcoded strings like `"data/..."` with `pathlib` logic that finds the data directory relative to the project root.
4. **Final Cleanup:** Remove the redundant monolithic `.py` files from the root directory.

### Phase 5: CI/CD Setup

1. **Create `.github/workflows/ci.yml`:**
    - Trigger: Push to `main` or Pull Request.
    - Steps: Checkout code, Setup Python, Install dependencies, Run `pytest`.
2. **Verification:** The AI must successfully trigger a passing "Green Checkmark" on a test commit.