# Clustered Push-Relabel

[![CI](https://github.com/YourOrg/RP2/actions/workflows/ci.yml/badge.svg)](https://github.com/YourOrg/RP2/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/clustered-push-relabel.svg)](https://pypi.org/project/clustered-push-relabel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance GPU library for discrete Optimal Transport and Bipartite Matching, accelerated by K-level spatial clustering.

## Installation

```bash
# Basic installation
pip install .

# For running tests
pip install -e .[test]

# For building documentation
pip install -e .[docs]
```

## Quickstart

```python
import torch
from clustered_push_relabel import solve_bipartite_matching

# Generate synthetic point clouds
N, D = 10000, 2
x = torch.rand(N, D, device='cuda')
y = torch.rand(N, D, device='cuda')

# Solve with K=4 level hierarchical clustering
result = solve_bipartite_matching(x, y, epsilon=0.01, k=4)
print("Matched pairs:", result["assignment_vector"])
```

## Modules

- **Clustering**: GPU-accelerated spatial decomposition routines (`k_level_cluster`).
- **Bipartite Matching**: Exact min-cost matching solvers using push-relabel over generated cover graphs.
- **Optimal Transport**: Unbalanced mass routing and fractional flow optimization.

---

## Research & Citation

> **⚠️ ACADEMIC NOTICE / COPYRIGHT WARNING**
>
> This repository contains the ongoing research code for an ongoing project.
>
> **Status:** Active Research / Work in Progress
>
> **Usage Policy:**
> This code is publicly accessible for **transparency and academic verification purposes only**.
> You are **NOT** authorized to:
> - Use this code in your own projects (commercial or personal).
> - Modify or redistribute this code.
> - Train AI models on this codebase.
>
> If you wish to use this code or reference it in your own work, please contact [Your Email] for permission.
