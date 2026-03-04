# Getting Started

The `clustered_push_relabel` library is designed to compute exact/approximate matchings and fractional transports on large point clouds.

## Installation

You can install `clustered_push_relabel` directly via pip:
```bash
pip install .
```

To run the examples locally, check out the `examples/` directory in the repository.

## Basic Example

```python
import torch
from clustered_push_relabel import k_level_cluster

# Using CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.rand(1000, 2, device=device)
y = torch.rand(1000, 2, device=device)

# Compute the clustering cover graph based on epsilon scale
res = k_level_cluster(x, y, epsilon=0.05, k=4)
```
