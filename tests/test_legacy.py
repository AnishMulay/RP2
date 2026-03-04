import pytest
import torch
from legacy_two_level import GPUClusteredSolver as TwoLevelSolver
from legacy_k_level import GPUClusteredSolver as KLevelSolver
from legacy_transport import GPUClusteredOTSolver

def test_legacy_baselines(device):
    torch.manual_seed(42)
    n = 10
    dim = 2
    epsilon = 0.01
    k = 4
    
    P_red = torch.rand(n, dim, device=device)
    P_blue = torch.rand(n, dim, device=device)
    mass_x = torch.ones(n, device=device) / n
    mass_y = torch.ones(n, device=device) / n
    
    # 2-Level
    solver_2lvl = TwoLevelSolver(P_red, P_blue, epsilon)
    solver_2lvl.solve()
    mb_2lvl = solver_2lvl.MB.cpu().tolist()
    assert mb_2lvl == [9, 3, 2, 6, 5, 4, 8, 1, 0, 7], f"Expected [9, 3, 2, 6, 5, 4, 8, 1, 0, 7], got {mb_2lvl}"
    
    # k-Level
    solver_klvl = KLevelSolver(P_red, P_blue, epsilon, k=k)
    solver_klvl.solve()
    mb_klvl = solver_klvl.MB.cpu().tolist()
    assert mb_klvl == [9, 3, 0, 6, 5, 4, 8, 1, 2, 7], f"Expected [9, 3, 0, 6, 5, 4, 8, 1, 2, 7], got {mb_klvl}"
    
    # OT
    solver_ot = GPUClusteredOTSolver(P_red, P_blue, mass_x, mass_y, epsilon, k=k, batch_size=2048, metric="L2")
    solver_ot.solve()
    solver_ot.de_scale_and_cleanup()
    
    ot_edges_u = solver_ot.active_edges_u.cpu().tolist() if solver_ot.active_edges_u is not None else None
    ot_edges_v = solver_ot.active_edges_v.cpu().tolist() if solver_ot.active_edges_v is not None else None
    ot_flow = solver_ot.active_flow.cpu().tolist() if solver_ot.active_flow is not None else None
    
    assert ot_edges_u == [4, 5, 8, 9, 7, 6, 0, 1, 3, 2]
    assert ot_edges_v == [3, 9, 7, 0, 6, 8, 1, 2, 5, 4]
    assert ot_flow == [48, 48, 48, 48, 48, 48, 48, 48, 48, 48]
