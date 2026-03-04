import pytest
import torch
from clustered_push_relabel.clustering.k_level import FastGPUMultiLevelClustering
from clustered_push_relabel.clustering.two_level import FastGPUClustering

def test_k_level_clustering(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    k = 4
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    
    clustering = FastGPUMultiLevelClustering(epsilon, k=k)
    blue_coo, red_coo, levels_red, levels_blue = clustering.run(P_red, P_blue)
    
    assert isinstance(blue_coo, tuple)
    assert len(blue_coo) == 3
    assert levels_red.shape == (n,)
    assert levels_blue.shape == (n,)

def test_two_level_clustering(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    
    clustering = FastGPUClustering(epsilon)
    blue_coo, red_coo, r_mask, b_mask = clustering.run(P_red, P_blue)
    
    assert isinstance(blue_coo, tuple)
    assert len(blue_coo) == 3
    assert r_mask.shape == (n,)
    assert b_mask.shape == (n,)
