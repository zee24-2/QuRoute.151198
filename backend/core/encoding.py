"""
Standardized Random-Key Encoding and Decoder.
Ensures 100% identical solution decoding and fitness evaluation across:
- Quantum Particle Swarm Optimization (QPSO)
- Classical Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from .formulation import MathematicalFormulation, RouteEvaluation


class RandomKeyDecoder:
    """
    Decodes a continuous vector z in [0, 1]^N (where N is the number of customers)
    into a set of discrete vehicle routes:
    1. argsort(z) -> Customer permutation
    2. Greedy capacity packing: Group customers sequentially into vehicle routes until capacity is met
    3. Return structured vehicle routes [depot, c1, c2, ..., depot]
    """
    def __init__(self, formulation: MathematicalFormulation):
        self.formulation = formulation
        self.depot_id = formulation.depot_id
        self.customer_ids = list(formulation.graph.customer_ids)
        self.num_customers = len(self.customer_ids)
        self.num_vehicles = formulation.num_vehicles
        self.vehicle_capacity = formulation.vehicle_capacity

    def decode_permutation(self, permutation: List[int]) -> List[List[int]]:
        """
        Takes an ordered list of customer indices [0..N-1] and partitions them into vehicle routes.
        """
        routes: List[List[int]] = [[] for _ in range(self.num_vehicles)]
        current_vehicle = 0
        current_load = 0.0

        for cust_idx in permutation:
            cust_id = self.customer_ids[cust_idx]
            cust_demand = self.formulation.graph.nodes[cust_id].demand

            # Check if current vehicle can accommodate demand
            if current_load + cust_demand <= self.vehicle_capacity or current_vehicle == self.num_vehicles - 1:
                routes[current_vehicle].append(cust_id)
                current_load += cust_demand
            else:
                # Move to next vehicle
                current_vehicle = min(current_vehicle + 1, self.num_vehicles - 1)
                routes[current_vehicle].append(cust_id)
                current_load = cust_demand

        # Wrap each non-empty route with depot at start and end
        formatted_routes = []
        for r in routes:
            if r:
                formatted_routes.append([self.depot_id] + r + [self.depot_id])
            else:
                formatted_routes.append([self.depot_id, self.depot_id])

        return formatted_routes

    def decode_continuous_vector(self, continuous_vector: np.ndarray) -> List[List[int]]:
        """
        Decodes a continuous random-key vector into vehicle routes.
        """
        # argsort gives permutation order
        perm = np.argsort(continuous_vector).tolist()
        return self.decode_permutation(perm)

    def evaluate_continuous_vector(
        self,
        continuous_vector: np.ndarray,
        time_matrix: np.ndarray,
    ) -> Tuple[float, RouteEvaluation]:
        """
        Uniform objective evaluation for continuous particles.
        Returns: (objective_cost, route_evaluation_object)
        """
        routes = self.decode_continuous_vector(continuous_vector)
        eval_result = self.formulation.evaluate_solution(routes, time_matrix)
        return eval_result.objective_cost, eval_result
