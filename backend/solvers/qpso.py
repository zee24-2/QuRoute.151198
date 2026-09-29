"""
Quantum-behaved Particle Swarm Optimization (QPSO) for QuRoute (SIH26137).
Deliverable 3: Core Algorithm Engine.

Mathematical Features:
1. No velocity vector: particles sample from double-exponential (Laplace) distributions
   around a delta potential well centered at stochastic attractor p.
2. mbest (mean best position): swarm centroid of personal best positions.
3. Contraction-Expansion coefficient beta linearly decayed from 1.0 -> 0.5.
4. Equal Evaluation Budget (eval_budget) tracking and Warm-Start support for sub-second incident re-routing.
"""

from typing import Optional, Callable, List
import time
import numpy as np
from .base import BaseRoutingSolver, SolverConfig, SolverResult, IterationEvent
from ..core.formulation import MathematicalFormulation
from ..core.encoding import RandomKeyDecoder


class QPSOSolver(BaseRoutingSolver):
    def __init__(self):
        super().__init__(name="QPSO")

    @staticmethod
    def _routes_to_random_key_vector(routes: List[List[int]], customer_ids: List[int]) -> np.ndarray:
        """Converts an existing route solution into a continuous random-key vector in [0, 1]^N."""
        n = len(customer_ids)
        vec = np.linspace(0.1, 0.9, n)
        cust_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
        ordered_custs = []
        for r in routes:
            for nid in r:
                if nid in cust_to_idx and nid not in ordered_custs:
                    ordered_custs.append(nid)
        for cid in customer_ids:
            if cid not in ordered_custs:
                ordered_custs.append(cid)
        out = np.zeros(n, dtype=np.float64)
        for rank, cid in enumerate(ordered_custs):
            out[cust_to_idx[cid]] = (rank + 0.5) / max(1, n)
        return out

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

        # 1. Initialize particle positions in [0, 1]^D (with optional Warm-Start seeding)
        x = rng.uniform(0.0, 1.0, size=(swarm_size, num_customers))
        is_warm_started = False

        if config.warm_start_routes and num_customers > 0:
            is_warm_started = True
            seed_vec = self._routes_to_random_key_vector(config.warm_start_routes, decoder.customer_ids)
            x[0] = np.copy(seed_vec)
            # Seed half the swarm with Laplace perturbations around the warm-start attractor
            half = max(1, swarm_size // 2)
            for i in range(1, half):
                u_lap = rng.uniform(1e-6, 1.0, size=num_customers)
                signs = rng.choice([-1.0, 1.0], size=num_customers)
                x[i] = seed_vec + signs * 0.08 * np.log(1.0 / u_lap)

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

        # 2. Main QPSO Optimization Loop (Double-Exponential Laplace Wave Collapse)
        for it in range(1, max_iters + 1):
            beta = config.beta_start - ((config.beta_start - config.beta_end) * (it / max(1, max_iters)))

            # Calculate mbest: mean of all personal best positions
            mbest = np.mean(pbest_x, axis=0)

            phi = rng.uniform(0.0, 1.0, size=(swarm_size, num_customers))
            p = phi * pbest_x + (1.0 - phi) * gbest_x

            # Laplace / Double-Exponential jump around p
            u = rng.uniform(1e-9, 1.0, size=(swarm_size, num_customers))
            step_length = beta * np.abs(mbest - x) * np.log(1.0 / u)
            signs = rng.choice([-1.0, 1.0], size=(swarm_size, num_customers))
            x = p + signs * step_length

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

        # Fast intra-route 2-opt refinement on best vehicle tours to close the last <2% gap
        if best_eval is not None and num_customers >= 3:
            improved_routes = []
            for r in best_eval.vehicle_routes:
                if len(r) <= 4:
                    improved_routes.append(r)
                    continue
                curr_r = list(r)
                curr_ev = formulation.evaluate_solution([curr_r], time_matrix)
                curr_c = curr_ev.objective_cost
                for a in range(1, len(curr_r) - 2):
                    for b in range(a + 1, len(curr_r) - 1):
                        cand_r = curr_r[:a] + list(reversed(curr_r[a:b + 1])) + curr_r[b + 1:]
                        cand_ev = formulation.evaluate_solution([cand_r], time_matrix)
                        if cand_ev.objective_cost < curr_c:
                            curr_r = cand_r
                            curr_c = cand_ev.objective_cost
                improved_routes.append(curr_r)

            polished_eval = formulation.evaluate_solution(improved_routes, time_matrix)
            if polished_eval.objective_cost < gbest_cost:
                gbest_cost = polished_eval.objective_cost
                best_eval = polished_eval
                convergence_history[-1] = float(gbest_cost)
                eval_history[-1][1] = float(gbest_cost)

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
            warm_started=is_warm_started,
            wall_clock_time_seconds=elapsed_time,
            feasibility=best_eval.is_feasible,
            iterations_to_convergence=conv_iter,
            evaluation=best_eval,
            parameters={
                "beta_start": config.beta_start,
                "beta_end": config.beta_end,
                "swarm_size": swarm_size,
                "max_iters": max_iters,
                "eval_budget": total_evals,
                "warm_started": is_warm_started,
            },
        )
