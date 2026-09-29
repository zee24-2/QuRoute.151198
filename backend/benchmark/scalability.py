"""
Scalability Sweep Framework.
Evaluates cost, runtime, and convergence scaling curves across problem sizes
N in [10, 25, 50, 100, 200+].
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np
from ..core.graph import generate_grid_network
from ..core.formulation import MathematicalFormulation
from ..solvers import get_solver
from ..solvers.base import SolverConfig


class ScalabilityExperiment:
    def __init__(self, node_sizes: Optional[List[int]] = None):
        if node_sizes is None:
            node_sizes = [10, 25, 50, 100, 150]
        self.node_sizes = node_sizes

    def run_sweep(
        self,
        solver_names: Optional[List[str]] = None,
        trials_per_size: int = 3,
        max_iters: int = 60,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs scalability sweep across problem sizes.
        Returns data points formatted for scaling curve plotting (Runtime vs N, Cost vs N).
        """
        if solver_names is None:
            solver_names = ["qpso", "pso", "ga", "aco"]

        results: Dict[str, Dict[str, List[float]]] = {
            name: {
                "sizes": [],
                "mean_costs": [],
                "mean_runtimes": [],
                "mean_convergence_iters": [],
            }
            for name in solver_names
        }

        for size in self.node_sizes:
            # Create a grid sized proportionally to accommodate 'size' customers
            grid_dim = max(4, int(np.ceil(np.sqrt(size * 1.6))))
            vehicles = max(2, int(np.ceil(size / 6.0)))
            capacity = 70.0

            graph = generate_grid_network(
                rows=grid_dim,
                cols=grid_dim,
                num_customers=size,
                seed=seed,
            )
            formulation = MathematicalFormulation(
                graph=graph,
                vehicle_capacity=capacity,
                num_vehicles=vehicles,
            )

            for name in solver_names:
                solver = get_solver(name)
                costs = []
                runtimes = []
                conv_iters = []

                for t in range(trials_per_size):
                    cfg = SolverConfig(max_iters=max_iters, seed=seed + t)
                    res = solver.run(formulation, cfg)
                    costs.append(res.best_cost)
                    runtimes.append(res.wall_clock_time_seconds)
                    conv_iters.append(res.iterations_to_convergence)

                results[name]["sizes"].append(size)
                results[name]["mean_costs"].append(round(float(np.mean(costs)), 2))
                results[name]["mean_runtimes"].append(round(float(np.mean(runtimes)), 4))
                results[name]["mean_convergence_iters"].append(round(float(np.mean(conv_iters)), 1))

        return {
            "sizes_evaluated": self.node_sizes,
            "algorithms": results,
            "trials_per_size": trials_per_size,
        }
