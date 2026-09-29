"""
A* Greedy / Nearest-Neighbor Fleet Baseline Solver.
Demonstrates why pure point-to-point A* (as used by Google Maps) is INVIABLE for multi-vehicle swarm fleet logistics.

Why A* alone fails at swarm logistics:
1. Combinatorial State-Space Explosion: A* operates on 1-to-1 point search. Expanding it to N! vehicle permutations
   explodes to O(N! * K^N), causing memory exhaustion or timeouts even on 15 nodes.
2. The Greedy Trap: Naive A* nearest-neighbor dispatch prioritizes locally close customers early,
   leaving distant customers stranded and forcing massive backtracking detours later in the tour.
3. Temporal Blindness: Point-to-point A* has no global foresight of time windows, leading to massive lateness penalties.
4. Fleet Imbalance: Without swarm potential-field sharing, early vehicles consume easy pickups, leaving subsequent
   vehicles with disjointed, high-cost delivery legs.
"""

import time
from typing import Optional, Callable, List
import numpy as np
from .base import BaseRoutingSolver, SolverConfig, SolverResult, IterationEvent
from ..core.formulation import MathematicalFormulation
from ..core.astar import astar_shortest_path


class AStarGreedySolver(BaseRoutingSolver):
    def __init__(self):
        super().__init__(name="A* Greedy Fleet (Google Maps Baseline)")

    def run(
        self,
        formulation: MathematicalFormulation,
        config: SolverConfig,
        callback: Optional[Callable[[IterationEvent], None]] = None,
    ) -> SolverResult:
        start_time = time.perf_counter()

        graph = formulation.graph
        depot_id = formulation.depot_id
        unvisited = set(graph.customer_ids)
        num_vehicles = formulation.num_vehicles
        capacity = formulation.vehicle_capacity

        vehicle_routes: List[List[int]] = []
        convergence_history: List[float] = []

        # Construct routes greedily vehicle by vehicle
        for v in range(num_vehicles):
            route = [depot_id]
            curr_node = depot_id
            curr_load = 0.0

            while unvisited:
                # Find nearest unvisited customer using A* live travel time
                best_cust = None
                best_time = float("inf")

                for cand in unvisited:
                    cand_demand = graph.nodes[cand].demand
                    # Check capacity constraint
                    if curr_load + cand_demand <= capacity:
                        astar_res = astar_shortest_path(graph, curr_node, cand)
                        t = astar_res["total_travel_time"]
                        if t < best_time:
                            best_time = t
                            best_cust = cand

                # If no candidate fits capacity, finish this vehicle tour
                if best_cust is None:
                    break

                # Add candidate to route
                route.append(best_cust)
                curr_load += graph.nodes[best_cust].demand
                unvisited.remove(best_cust)
                curr_node = best_cust

            route.append(depot_id)
            vehicle_routes.append(route)

        # If customers still remain unvisited after all vehicles, append them to last vehicle (capacity violation)
        if unvisited:
            for rem in list(unvisited):
                vehicle_routes[-1].insert(-1, rem)

        # Pad with empty routes if fewer vehicles used
        while len(vehicle_routes) < num_vehicles:
            vehicle_routes.append([depot_id, depot_id])

        time_matrix = formulation.graph.get_travel_time_matrix(formulation.all_interest_nodes)
        evaluation = formulation.evaluate_solution(vehicle_routes, time_matrix)
        best_cost = evaluation.objective_cost
        convergence_history = [best_cost]

        elapsed_time = time.perf_counter() - start_time

        if callback:
            callback(IterationEvent(
                iteration=1,
                current_best_cost=best_cost,
                best_route=vehicle_routes,
                timestamp_ms=elapsed_time * 1000.0,
                solver_name=self.name,
            ))

        return SolverResult(
            solver_name=self.name,
            best_route=vehicle_routes,
            best_cost=best_cost,
            convergence_history=convergence_history,
            wall_clock_time_seconds=elapsed_time,
            feasibility=evaluation.is_feasible,
            iterations_to_convergence=1,
            evaluation=evaluation,
            parameters={
                "method": "Pure A* Greedy Nearest-Neighbor",
                "inviability_cause": "Local greedy traps & O(N!) combinatorial explosion",
            },
        )
