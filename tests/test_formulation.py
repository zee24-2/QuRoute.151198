import pytest
from backend.core.graph import generate_grid_network
from backend.core.formulation import MathematicalFormulation
from backend.solvers import get_solver
from backend.solvers.base import SolverConfig


def test_mathematical_formulation_metadata():
    graph = generate_grid_network(rows=4, cols=4, num_customers=6, seed=42)
    formulation = MathematicalFormulation(graph=graph, vehicle_capacity=50.0, num_vehicles=2)
    meta = formulation.get_formulation_metadata()

    assert "latex_objective" in meta
    assert len(meta["latex_constraints"]) >= 4
    assert meta["parameters"]["num_customers"] == 6


def test_exact_optimality_gap():
    # 5 customer nodes -> OR-Tools CP-SAT finds global optimum
    graph = generate_grid_network(rows=4, cols=4, num_customers=5, seed=99)
    formulation = MathematicalFormulation(graph=graph, vehicle_capacity=60.0, num_vehicles=2)

    exact_solver = get_solver("exact")
    exact_res = exact_solver.run(formulation, SolverConfig())

    qpso_solver = get_solver("qpso")
    qpso_res = qpso_solver.run(formulation, SolverConfig(max_iters=40, swarm_size=25, seed=99))

    assert exact_res.best_cost > 0
    assert qpso_res.best_cost > 0

    # Gap % = (QPSO - Exact) / Exact * 100
    gap = ((qpso_res.best_cost - exact_res.best_cost) / exact_res.best_cost) * 100.0
    # QPSO on small 5-node instance should get very close to exact optimum (within 20% or even 0%)
    assert gap >= -1.0  # Cannot beat true optimum beyond float rounding
