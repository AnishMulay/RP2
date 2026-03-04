import pytest
import torch
import sys
import os

# Ensure the root codebase is in sys.path so we can import legacy files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock CUDA functions to allow CPU tests on Mac
if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *args, **kwargs: None
    torch.cuda.empty_cache = lambda *args, **kwargs: None

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
