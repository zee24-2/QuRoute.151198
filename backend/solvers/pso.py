"""
Classical Particle Swarm Optimization (PSO) with Velocity Clamping and Inertia Weight.
Uses identical random-key encoding, objective evaluator, and equal evaluation budget (eval_budget)
to isolate the quantum update contribution.
"""

from typing import Optional, Callable, List
import time
import numpy as np
from .base import BaseRoutingSolver, SolverConfig, SolverResult, IterationEvent
from ..core.formulation import MathematicalFormulation
from ..core.encoding import RandomKeyDecoder


class ClassicalPSOSolver(BaseRoutingSolver):
    def __init__(self):
        super().__init__(name="Classical PSO")

    def run(
        self,
        formulation: MathematicalFormulation,
        config: SolverConfig,
        callback: Optional[Callable[[IterationEvent], None]] = None,
    ) -> SolverResult:
        start_time = time.perf_counter()
        rng = np.random.default_rng(config.seed)

        decoder = RandomKeyDecoder(formulation)
        num_customers = len(decoder.customer_ids)
        swarm_size = max(4, config.swarm_size)
        max_iters = config.resolve_max_iters(swarm_size)

        time_matrix = formulation.graph.get_travel_time_matrix(formulation.all_interest_nodes)

        x = rng.uniform(0.0, 1.0, size=(swarm_size, num_customers))
        v_max = 0.2
        v = rng.uniform(-v_max, v_max, size=(swarm_size, num_customers))

        pbest_x = np.copy(x)
        pbest_costs = np.full(swarm_size, np.inf)

        gbest_cost = np.inf
        gbest_x = np.zeros(num_customers)
        best_eval = None
        total_evals = 0

        for i in range(swarm_size):
            cost, ev = decoder.evaluate_continuous_vector(x[i], time_matrix)
            total_evals += 1
            pbest_costs[i] = cost
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_x = np.copy(x[i])
                best_eval = ev

        convergence_history: List[float] = [float(gbest_cost)]
        eval_history: List[List[float]] = [[float(total_evals), float(gbest_cost)]]

        if callback and best_eval:
            callback(IterationEvent(
                iteration=0,
                current_best_cost=float(gbest_cost),
                best_route=best_eval.vehicle_routes,
                timestamp_ms=(time.perf_counter() - start_time) * 1000.0,
                solver_name=self.name,
                evaluations=total_evals,
            ))

        w = config.w_inertia
        c1 = config.c1_cognitive
        c2 = config.c2_social

        for it in range(1, max_iters + 1):
            r1 = rng.uniform(0.0, 1.0, size=(swarm_size, num_customers))
            r2 = rng.uniform(0.0, 1.0, size=(swarm_size, num_customers))

            v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
            v = np.clip(v, -v_max, v_max)
            x = x + v

            for i in range(swarm_size):
                cost, ev = decoder.evaluate_continuous_vector(x[i], time_matrix)
                total_evals += 1

                if cost < pbest_costs[i]:
                    pbest_costs[i] = cost
                    pbest_x[i] = np.copy(x[i])

                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_x = np.copy(x[i])
                        best_eval = ev

            convergence_history.append(float(gbest_cost))
            eval_history.append([float(total_evals), float(gbest_cost)])

            if callback and best_eval and (it % 1 == 0 or it == max_iters):
                callback(IterationEvent(
                    iteration=it,
                    current_best_cost=float(gbest_cost),
                    best_route=best_eval.vehicle_routes,
                    timestamp_ms=(time.perf_counter() - start_time) * 1000.0,
                    solver_name=self.name,
                    evaluations=total_evals,
                ))

        elapsed_time = time.perf_counter() - start_time
        if best_eval is None:
            _, best_eval = decoder.evaluate_continuous_vector(gbest_x, time_matrix)

        conv_iter = self.calculate_iterations_to_convergence(convergence_history)

        return SolverResult(
            solver_name=self.name,
            best_route=best_eval.vehicle_routes,
            best_cost=float(gbest_cost),
            convergence_history=convergence_history,
            eval_history=eval_history,
            total_evaluations=total_evals,
            wall_clock_time_seconds=elapsed_time,
            feasibility=best_eval.is_feasible,
            iterations_to_convergence=conv_iter,
            evaluation=best_eval,
            parameters={
                "inertia_w": w,
                "c1": c1,
                "c2": c2,
                "swarm_size": swarm_size,
                "max_iters": max_iters,
                "eval_budget": total_evals,
            },
        )
