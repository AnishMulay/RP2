# SDD: documentation

## 1. Context & Primary Goals

**Context:**
Following a successful repository refactor, the clustered push-relabel solver is now structured as a modular Python package (`src/` layout). However, the current documentation still reflects its origins as a private research repository. To drive adoption and make the library accessible to the broader scientific and engineering community, the documentation must be elevated to the standards of top-tier libraries like `POT` (Python Optimal Transport) and `ott-jax`.

**Primary Goals:**

1. **Professional Storefront (README):** Overhaul the `README.md` to include standard open-source badges (CI/CD, Python versions, License), clear installation instructions, and a minimal Quickstart guide. *(Note: The DOI/Zenodo badge will be added manually by the author later).*
2. **Comprehensive Docstrings:** Ensure all public APIs (wrappers in `__init__.py` and core solver classes) have standard, typed docstrings (NumPy or Google style) explaining inputs, outputs, and tensor shapes.
3. **Usage Examples:** Create isolated, copy-pasteable usage examples demonstrating how to use the clustering, bipartite matching, and optimal transport modules.
4. **Documentation Site Generation:** Scaffold a documentation generator (e.g., MkDocs + Material theme) so the docstrings and usage examples can be compiled into a beautiful, readable website.

**Strict AI Execution Guardrails:**

- **No Math Hallucinations:** When writing docstrings, the AI must strictly describe the code as it is. It must not invent mathematical properties or parameters that do not exist in the source code.
- **Validate Docs as Code:** Just like the code refactor, documentation must be tested. The AI must build the documentation site locally to ensure there are no rendering errors before moving to the next step.

---

## 2. Target Documentation Architecture (LLD)

### 2.1 The README Layout

The `README.md` must be rewritten to follow this exact structure:

1. **Header & Badges:** * Title and short description.
    - Badges: GitHub Actions CI status, Supported Python Versions (e.g., 3.9 - 3.12), License, and a placeholder for PyPI `[![PyPI version](...)](...)`.
2. **Installation:** Instructions for `pip install .` and `pip install -e .[test]`.
3. **Quickstart:** A 10-line code block showing how to import and run `solve_bipartite_matching` on synthetic data.
4. **Modules Overview:** Brief bullet points explaining Clustering, Bipartite Matching, and Optimal Transport.
5. **Research & Citation:** A section preserving the existing research context and providing a BibTeX citation block.

### 2.2 Docstring Standard (Google Style)

All public-facing functions in `src/clustered_push_relabel/` must adopt Google Style docstrings.
*Example standard for the AI to follow:*

Python

`def solve_bipartite_matching(x, y, epsilon, k=4, batch_size=None, metric="L2"):
    """
    Solves min-cost bipartite matching using K-level clustered push-relabel.

    Args:
        x (torch.Tensor): Source point cloud of shape (N, D).
        y (torch.Tensor): Target point cloud of shape (N, D).
        epsilon (float): Discretization / stopping threshold parameter.
        k (int, optional): Number of hierarchy levels. Defaults to 4.
        batch_size (int, optional): GPU batch size for clustering. Defaults to None.
        metric (str, optional): Distance metric ("L2" or "L1"). Defaults to "L2".

    Returns:
        dict: A dictionary containing the 'assignment_vector' (torch.Tensor) 
              mapping each target point to a source point.
    """`

### 2.3 The Documentation Site (MkDocs)

We will use `mkdocs` with the `mkdocs-material` theme and `mkdocstrings` plugin to auto-generate API docs from the Python code.

**Target File Structure:**

Plaintext

`clustered-push-relabel/
├── mkdocs.yml                # Configuration for the docs site
├── docs/                     # Markdown files for the site
│   ├── index.md              # (Can be a symlink or copy of README.md)
│   ├── getting_started.md    # Deeper dive into installation and usage
│   └── api/                  
│       ├── clustering.md     # ::: clustered_push_relabel.clustering
│       └── solvers.md        # ::: clustered_push_relabel.solvers
└── examples/                 # Clean python scripts for users to run
    ├── 01_basic_clustering.py
    └── 02_optimal_transport.py`

---

## 3. Step-by-Step Execution Plan (AI Directives)

**Operating Principle:** *Read -> Write -> Build -> Verify.* The AI must read the existing codebase to ensure accurate types, write the documentation, and then actually run the documentation builder to check for syntax errors.

### Phase 1: Codebase Docstrings

*Goal: Ensure the source code is self-documenting.*

1. **Analyze Public API:** Review the `__init__.py` files and the main functional wrappers (`k_level_cluster`, `solve_bipartite_matching`, `solve_optimal_transport`).
2. **Apply Google Style:** Rewrite the docstrings for these functions, accurately reflecting the tensor shapes (`[N, D]`) and expected device configurations (e.g., CUDA).
3. **Review Core Classes:** Add brief class-level docstrings to the underlying `GPUClusteredSolver` and `GPUClusteredOTSolver` noting that they are the underlying stateful engines.

### Phase 2: Overhauling the README

*Goal: Create the "Storefront".*

1. **Extract Legacy Info:** Copy the research context and citation information from the current `README.md` to a temporary file so it isn't lost.
2. **Rewrite README.md:** Implement the structure defined in Section 2.1.
3. **Generate Badges:** Add markdown badge links for CI (pointing to the `.github/workflows/ci.yml` path), Python versions, and License.
4. **Restore Legacy Info:** Append the research context and citation to the bottom of the new README.

### Phase 3: Creating Usage Examples

*Goal: Provide copy-pasteable scripts for new users.*

1. **Create `examples/` Directory:** At the repository root.
2. **Write `01_basic_clustering.py`:** A simple script that generates two random `torch.Tensor` point clouds, runs `k_level_cluster`, and prints the output shapes.
3. **Write `02_optimal_transport.py`:** A simple script that runs `solve_optimal_transport` on random data and prints the total flow.
4. **Verify:** Run `python examples/01_basic_clustering.py` and `02_optimal_transport.py`. They **must** execute without errors.

### Phase 4: MkDocs Site Generation

*Goal: Compile a searchable, professional website.*

1. **Update `pyproject.toml`:** Add a `[project.optional-dependencies]` section for `docs = ["mkdocs-material", "mkdocstrings[python]"]`.
2. **Install Tooling:** Run `pip install -e .[docs]`.
3. **Create `mkdocs.yml`:** Configure the Material theme, repository URL, and navigation tree.
4. **Create `docs/` Markdown Files:** Write the `index.md`, `getting_started.md`, and the `api/*.md` files containing the `mkdocstrings` auto-generation tags (`:::`).
5. **Verify Build:** Run `mkdocs build`. The AI must ensure this command succeeds without fatal errors before completing the task.