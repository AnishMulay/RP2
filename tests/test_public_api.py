import pytest
import torch
from clustered_push_relabel import (
    k_level_cluster,
    solve_bipartite_matching,
    solve_optimal_transport
)

def test_k_level_cluster(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    
    result = k_level_cluster(P_red, P_blue, epsilon, k=4)
    
    assert "blue_cover" in result
    assert "red_cover" in result
    assert "levels_red" in result
    assert "levels_blue" in result

def test_solve_bipartite_matching_two_level(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    
    result = solve_bipartite_matching(P_red, P_blue, epsilon, k=1)
    assert "assignment_vector" in result
    assert result["assignment_vector"].shape == (n,)

def test_solve_bipartite_matching_k_level(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    k = 4
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    
    result = solve_bipartite_matching(P_red, P_blue, epsilon, k=k)
    assert "assignment_vector" in result
    assert result["assignment_vector"].shape == (n,)

def test_solve_optimal_transport(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    k = 4
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    mass_x = torch.ones(n, device=device) / n
    mass_y = torch.ones(n, device=device) / n
    
    result = solve_optimal_transport(P_red, P_blue, mass_x, mass_y, epsilon, k=k)
    
    assert "source_edges" in result
    assert "target_edges" in result
    assert "flow" in result
    
    assert result["source_edges"].dim() == 1
    assert result["target_edges"].dim() == 1
    assert result["flow"].dim() == 1
    assert result["source_edges"].shape == result["target_edges"].shape
    assert result["target_edges"].shape == result["flow"].shape

