"""
Genetic Algorithm (GA) Baseline for QuRoute (SIH26137).
Implements Order Crossover (OX), swap and 2-opt mutation, elitism, and equal evaluation budget tracking.
"""

from typing import Optional, Callable, List, Tuple
import time
import numpy as np
from .base import BaseRoutingSolver, SolverConfig, SolverResult, IterationEvent
from ..core.formulation import MathematicalFormulation
from ..core.encoding import RandomKeyDecoder


class GeneticAlgorithmSolver(BaseRoutingSolver):
    def __init__(self):
        super().__init__(name="Genetic Algorithm")

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
        pop_size = max(4, config.swarm_size)
        max_generations = config.resolve_max_iters(pop_size)

        time_matrix = formulation.graph.get_travel_time_matrix(formulation.all_interest_nodes)

        population = [rng.permutation(num_customers).tolist() for _ in range(pop_size)]
        fitnesses = np.zeros(pop_size)
        evaluations = [None] * pop_size

        gbest_cost = np.inf
        best_eval = None
        total_evals = 0

        for i in range(pop_size):
            routes = decoder.decode_permutation(population[i])
            ev = formulation.evaluate_solution(routes, time_matrix)
            total_evals += 1
            cost = ev.objective_cost
            fitnesses[i] = cost
            evaluations[i] = ev
            if cost < gbest_cost:
                gbest_cost = cost
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

        def tournament_select(k: int = 3) -> List[int]:
            k_eff = min(k, pop_size)
            selected_indices = rng.choice(pop_size, size=k_eff, replace=False)
            best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
            return list(population[best_idx])

        def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
            if rng.random() > config.crossover_rate or num_customers < 3:
                return list(parent1), list(parent2)

            pt1, pt2 = sorted(rng.choice(num_customers, size=2, replace=False))
            child1 = [-1] * num_customers
            child2 = [-1] * num_customers

            child1[pt1:pt2] = parent1[pt1:pt2]
            child2[pt1:pt2] = parent2[pt1:pt2]

            def fill_child(child, donor):
                donor_idx = pt2
                fill_idx = pt2
                while -1 in child:
                    candidate = donor[donor_idx % num_customers]
                    if candidate not in child:
                        child[fill_idx % num_customers] = candidate
                        fill_idx += 1
                    donor_idx += 1

            fill_child(child1, parent2)
            fill_child(child2, parent1)
            return child1, child2

        def mutate(perm: List[int]) -> List[int]:
            if rng.random() < config.mutation_rate and num_customers > 2:
                if rng.random() < 0.5:
                    i, j = rng.choice(num_customers, size=2, replace=False)
                    perm[i], perm[j] = perm[j], perm[i]
                else:
                    i, j = sorted(rng.choice(num_customers, size=2, replace=False))
                    perm[i:j + 1] = reversed(perm[i:j + 1])
            return perm

        for gen in range(1, max_generations + 1):
            new_population: List[List[int]] = []

            sorted_indices = np.argsort(fitnesses)
            new_population.append(list(population[sorted_indices[0]]))
            if pop_size > 1:
                new_population.append(list(population[sorted_indices[1]]))

            while len(new_population) < pop_size:
                p1 = tournament_select()
                p2 = tournament_select()
                c1, c2 = order_crossover(p1, p2)
                new_population.append(mutate(c1))
                if len(new_population) < pop_size:
                    new_population.append(mutate(c2))

            population = new_population

            for i in range(pop_size):
                routes = decoder.decode_permutation(population[i])
                ev = formulation.evaluate_solution(routes, time_matrix)
                total_evals += 1
                cost = ev.objective_cost
                fitnesses[i] = cost
                evaluations[i] = ev

                if cost < gbest_cost:
                    gbest_cost = cost
                    best_eval = ev

            convergence_history.append(float(gbest_cost))
            eval_history.append([float(total_evals), float(gbest_cost)])

            if callback and best_eval and (gen % 1 == 0 or gen == max_generations):
                callback(IterationEvent(
                    iteration=gen,
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
                "crossover_rate": config.crossover_rate,
                "mutation_rate": config.mutation_rate,
                "pop_size": pop_size,
                "max_generations": max_generations,
                "eval_budget": total_evals,
            },
        )
