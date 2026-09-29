import pytest
from backend.core.graph import generate_grid_network, generate_indian_city_network
from backend.core.astar import astar_shortest_path, expand_vrp_routes_to_street_paths
from backend.core.formulation import MathematicalFormulation
from backend.solvers.astar_greedy import AStarGreedySolver
from backend.solvers.base import SolverConfig


def test_astar_point_to_point():
    graph = generate_grid_network(rows=5, cols=5, num_customers=6, seed=42)
    start_id = 0  # Corner
    goal_id = 24  # Opposite corner

    res = astar_shortest_path(graph, start_id, goal_id)
    assert res["path"][0] == start_id
    assert res["path"][-1] == goal_id
    assert len(res["path"]) >= 5
    assert res["total_travel_time"] > 0
    assert len(res["coordinates"]) == len(res["path"])
    assert res["nodes_explored"] > 0


def test_vrp_route_expansion_to_street_paths():
    graph = generate_grid_network(rows=4, cols=4, num_customers=4, seed=42)
    depot = graph.depot_ids[0]
    c1, c2 = graph.customer_ids[0], graph.customer_ids[1]
    routes = [[depot, c1, c2, depot], [depot, depot]]

    expanded = expand_vrp_routes_to_street_paths(graph, routes)
    assert len(expanded) == 2
    assert expanded[0]["vehicle_id"] == 0
    assert len(expanded[0]["street_node_path"]) >= len(routes[0])
    assert expanded[0]["total_travel_time"] > 0
    assert len(expanded[0]["street_coordinates"]) == len(expanded[0]["street_node_path"])


def test_astar_greedy_solver_runs():
    graph = generate_grid_network(rows=4, cols=4, num_customers=5, seed=42)
    formulation = MathematicalFormulation(graph=graph, vehicle_capacity=50.0, num_vehicles=2)
    solver = AStarGreedySolver()
    cfg = SolverConfig(seed=42)

    result = solver.run(formulation, cfg)
    assert result.solver_name is not None
    assert result.best_cost > 0
    assert len(result.best_route) == 2
    assert result.evaluation is not None
