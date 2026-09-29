import pytest
from backend.core.graph import generate_grid_network
from backend.core.formulation import MathematicalFormulation
from backend.benchmark.runner import BenchmarkRunner
from backend.benchmark.scalability import ScalabilityExperiment


def test_benchmark_runner_and_pitch():
    graph = generate_grid_network(rows=4, cols=4, num_customers=5, seed=42)
    formulation = MathematicalFormulation(graph=graph, vehicle_capacity=50.0, num_vehicles=2)
    runner = BenchmarkRunner(formulation)

    # Fast test with 3 trials
    results = runner.run_statistical_benchmark(
        solver_names=["qpso", "pso"],
        num_trials=3,
        max_iters=20,
        base_seed=10,
        include_exact=True,
    )

    summary = results["summary"]
    assert "qpso" in summary
    assert "pso" in summary
    assert "exact" in summary
    assert summary["qpso"]["num_trials"] == 3
    assert len(summary["qpso"]["box_plot_data"]) == 3
    assert summary["qpso"]["optimality_gap_pct"] is not None

    pitch = results["empirical_pitch"]
    assert len(pitch) > 20
    assert "QPSO" in pitch


def test_scalability_sweep():
    experiment = ScalabilityExperiment(node_sizes=[5, 8])
    results = experiment.run_sweep(
        solver_names=["qpso", "pso"],
        trials_per_size=2,
        max_iters=15,
        seed=42,
    )

    assert results["sizes_evaluated"] == [5, 8]
    assert len(results["algorithms"]["qpso"]["mean_costs"]) == 2
    assert len(results["algorithms"]["qpso"]["mean_runtimes"]) == 2
