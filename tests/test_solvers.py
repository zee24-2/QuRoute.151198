import pytest
from backend.core.graph import generate_grid_network
from backend.core.formulation import MathematicalFormulation
from backend.solvers import get_solver
from backend.solvers.base import SolverConfig


@pytest.fixture
def sample_problem():
    graph = generate_grid_network(rows=4, cols=4, num_customers=6, seed=42)
    formulation = MathematicalFormulation(
        graph=graph,
        vehicle_capacity=50.0,
        num_vehicles=2,
    )
    return formulation


@pytest.mark.parametrize("solver_name", ["qpso", "pso", "ga", "aco", "exact"])
def test_all_solvers_produce_valid_results(sample_problem, solver_name):
    solver = get_solver(solver_name)
    config = SolverConfig(max_iters=25, swarm_size=15, seed=123)

    iterations_logged = []
    def callback(event):
        iterations_logged.append(event.iteration)

    result = solver.run(sample_problem, config, callback=callback)

    assert result.solver_name is not None
    assert result.best_cost > 0
    assert len(result.best_route) == sample_problem.num_vehicles
    assert len(result.convergence_history) > 0
    assert result.wall_clock_time_seconds > 0
    assert len(iterations_logged) > 0


def test_qpso_beats_or_matches_initial():
    graph = generate_grid_network(rows=4, cols=4, num_customers=8, seed=42)
    formulation = MathematicalFormulation(graph=graph, vehicle_capacity=60.0, num_vehicles=3)
    solver = get_solver("qpso")
    config = SolverConfig(max_iters=35, swarm_size=20, seed=42)
    result = solver.run(formulation, config)

    # Initial cost should be >= final cost
    assert result.convergence_history[0] >= result.convergence_history[-1]
