"""
Comprehensive verification suite for QuRoute (SIH26137 | Team 151198):
- BPR (1964) congestion formula & 15-minute time-bucket Dijkstra caching
- Live Traffic Data Blackout fallback (Slide 4 Risk Mitigation)
- Solomon (1987) VRPTW benchmark loader (C101, R101, RC101)
- Equal evaluation budget (eval_budget) & Warm-started QPSO dynamic re-routing
- Paired Wilcoxon signed-rank test, Claims-to-Tests verifier, and UN SDG 11/13 Impact
- Headless REST API (POST /network, /traffic, /solve, /benchmark, GET /results/{id})
"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.core.graph import generate_grid_network
from backend.core.traffic import DynamicTrafficEngine
from backend.core.formulation import MathematicalFormulation
from backend.core.solomon import generate_solomon_instance
from backend.solvers import get_solver
from backend.solvers.base import SolverConfig
from backend.benchmark.runner import BenchmarkRunner


def test_bpr_congestion_and_time_bucket_caching():
    """Verify BPR (1964) travel time calculation and 15-min bucket Dijkstra caching."""
    g = generate_grid_network(rows=3, cols=3, num_customers=4, seed=42)

    edge = next(iter(g.edges.values()))
    edge.free_flow_time = 10.0
    edge.capacity = 1000.0
    edge.flow = 1000.0  # v/c = 1.0 -> multiplier = 1 + 0.15 * (1.0)^4 = 1.15
    edge.bpr_alpha = 0.15
    edge.bpr_beta = 4.0

    mult = edge.compute_bpr_multiplier()
    assert pytest.approx(mult, rel=1e-3) == 1.15

    # Verify 15-minute time bucket caching
    nodes = [0] + g.customer_ids
    m1 = g.get_travel_time_matrix(nodes, time_minutes=480.0)
    m2 = g.get_travel_time_matrix(nodes, time_minutes=485.0)  # Same 15-min bucket (32)
    assert (32, tuple(nodes)) in g._bucket_matrix_cache
    assert (m1 == m2).all()


def test_live_data_blackout_fallback():
    """Verify Slide 4 Risk Mitigation: Live Data Blackout falls back to predictive historical BPR."""
    g = generate_grid_network(rows=4, cols=4, num_customers=5, seed=42)
    sim = DynamicTrafficEngine(g)
    sim.set_time(8 * 60 + 30)  # 08:30 AM peak

    # Trigger severe incident
    inc = sim.trigger_random_incident(severity=5.0)
    u, v = inc.edge
    weight_with_incident = g.edges[(u, v)].live_weight

    # Activate Live Data Blackout -> suppresses incident spikes, uses historical diurnal BPR
    sim.set_blackout_mode(True)
    assert sim.live_data_blackout is True
    weight_during_blackout = g.edges[(u, v)].live_weight
    assert weight_during_blackout < weight_with_incident


def test_solomon_vrptw_loader():
    """Verify Solomon (1987) C101, R101, and RC101 benchmark instances."""
    for name in ["C101", "R101", "RC101"]:
        g, meta = generate_solomon_instance(instance_name=name, num_customers=15)
        assert meta["instance_name"] == name
        assert len(g.customer_ids) == 15
        assert g.depot_ids == [0]


def test_warm_start_qpso_and_equal_eval_budget():
    """Verify equal evaluation budget tracking and warm-started QPSO dynamic re-routing."""
    g = generate_grid_network(rows=4, cols=4, num_customers=8, seed=42)
    form = MathematicalFormulation(g, num_vehicles=3, vehicle_capacity=80.0)

    cold_solver = get_solver("qpso")
    cold_res = cold_solver.run(form, SolverConfig(max_iters=40, swarm_size=20, eval_budget=600, seed=42))
    assert cold_res.total_evaluations <= 650
    assert len(cold_res.eval_history) > 0

    # Warm-start from previous best route
    warm_solver = get_solver("qpso")
    warm_res = warm_solver.run(
        form,
        SolverConfig(
            max_iters=20,
            swarm_size=15,
            eval_budget=300,
            warm_start_routes=cold_res.best_route,
            seed=42,
        ),
    )
    assert warm_res.warm_started is True
    assert warm_res.best_cost <= cold_res.best_cost * 1.05


def test_claims_to_tests_and_wilcoxon():
    """Verify Wilcoxon significance, Claims-to-Tests matrix, and UN SDG 11/13 sustainability calculator."""
    g = generate_grid_network(rows=4, cols=4, num_customers=8, seed=42)
    form = MathematicalFormulation(g, num_vehicles=3, vehicle_capacity=80.0)
    runner = BenchmarkRunner(form)

    res = runner.run_statistical_benchmark(
        solver_names=["qpso", "pso", "ga"],
        num_trials=6,
        max_iters=25,
        base_seed=42,
        include_exact=True,
    )
    assert "claims_to_tests_table" in res
    assert len(res["claims_to_tests_table"]) == 4
    assert "wilcoxon_vs_qpso" in res["summary"]["pso"]
    assert "sdg_impact" in res
    assert res["sdg_impact"]["co2_avoided_kg"] >= 0.0


def test_headless_rest_api():
    """Verify the 5 headless REST endpoints from the QuRoute specification."""
    client = TestClient(app)

    # 1. POST /network
    r_net = client.post("/network", json={"source": "solomon", "solomon_instance": "C101", "num_customers": 10})
    assert r_net.status_code == 200
    assert r_net.json()["graph"]["mode"] == "solomon"

    # 2. POST /traffic
    r_traf = client.post("/traffic", json={"minutes_from_midnight": 510, "bpr_alpha": 0.15, "bpr_beta": 4.0, "live_data_blackout": False})
    assert r_traf.status_code == 200

    # 3. POST /solve
    r_solve = client.post("/solve", json={"solver_name": "qpso", "max_iters": 20, "swarm_size": 15, "eval_budget": 300})
    assert r_solve.status_code == 200
    run_id = r_solve.json()["run_id"]

    # 4. GET /results/{id}
    r_res = client.get(f"/results/{run_id}")
    assert r_res.status_code == 200
    assert r_res.json()["type"] == "solve"

    # 5. POST /benchmark
    r_bench = client.post("/benchmark", json={"solver_names": ["qpso", "pso"], "num_trials": 4, "max_iters": 15, "include_exact": True})
    assert r_bench.status_code == 200
    bench_id = r_bench.json()["run_id"]
    assert client.get(f"/results/{bench_id}").status_code == 200
