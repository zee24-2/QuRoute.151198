"""
Ant Colony Optimization (ACO) Baseline for QuRoute (SIH26137).
Implements pheromone matrix, heuristic visibility, and equal evaluation budget tracking.
"""

from typing import Optional, Callable, List
import time
import numpy as np
from .base import BaseRoutingSolver, SolverConfig, SolverResult, IterationEvent
from ..core.formulation import MathematicalFormulation
from ..core.encoding import RandomKeyDecoder


class AntColonyOptimizationSolver(BaseRoutingSolver):
    def __init__(self):
        super().__init__(name="Ant Colony Optimization")

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
        num_ants = max(6, min(config.swarm_size, 30))
        max_iters = config.resolve_max_iters(num_ants)

        time_matrix = formulation.graph.get_travel_time_matrix(formulation.all_interest_nodes)

        cust_indices = [formulation.node_to_idx[cid] for cid in decoder.customer_ids]
        cust_time_submatrix = time_matrix[np.ix_(cust_indices, cust_indices)]
        with np.errstate(divide="ignore"):
            eta = 1.0 / (cust_time_submatrix + 1e-4)
            np.fill_diagonal(eta, 0.0)

        tau0 = 1.0 / (max(1, num_customers) * 10.0)
        tau = np.full((num_customers, num_customers), tau0)

        alpha = config.aco_alpha
        beta = config.aco_beta
        rho = config.aco_rho

        gbest_cost = np.inf
        gbest_perm = None
        best_eval = None
        convergence_history: List[float] = []
        eval_history: List[List[float]] = []
        total_evals = 0

        for it in range(max_iters + 1):
            for _ in range(num_ants):
                unvisited = set(range(num_customers))
                current = rng.choice(num_customers)
                unvisited.remove(current)
                tour = [current]

                while unvisited:
                    unvisited_list = list(unvisited)
                    tau_vals = tau[current, unvisited_list] ** alpha
                    eta_vals = eta[current, unvisited_list] ** beta
                    probs = tau_vals * eta_vals
                    prob_sum = np.sum(probs)

                    if prob_sum == 0 or np.isnan(prob_sum):
                        nxt = rng.choice(unvisited_list)
                    else:
                        probs = probs / prob_sum
                        nxt = rng.choice(unvisited_list, p=probs)

                    tour.append(nxt)
                    unvisited.remove(nxt)
                    current = nxt

                routes = decoder.decode_permutation(tour)
                ev = formulation.evaluate_solution(routes, time_matrix)
                total_evals += 1
                cost = ev.objective_cost

                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_perm = tour
                    best_eval = ev

            tau = (1.0 - rho) * tau
            if gbest_perm is not None and gbest_cost > 0:
                deposit = 100.0 / gbest_cost
                for k in range(len(gbest_perm) - 1):
                    u_node = gbest_perm[k]
                    v_node = gbest_perm[k + 1]
                    tau[u_node, v_node] += deposit
                    tau[v_node, u_node] += deposit

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
                "aco_alpha": alpha,
                "aco_beta": beta,
                "aco_rho": rho,
                "num_ants": num_ants,
                "max_iters": max_iters,
                "eval_budget": total_evals,
            },
        )
