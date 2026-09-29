"""
Mathematical Formulation Module for Dynamic Capacitated Vehicle Routing Problem with Time Windows (DCVRPTW).
Explicit mathematical specification object representing:
- Decision Variables x_{i,j,k} in {0, 1}
- Objective: Total Live Travel Time
- Constraints: Flow conservation, Fleet Depot start/finish, Capacity, Time Window bounds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .graph import TrafficGraph, Node


@dataclass
class RouteEvaluation:
    is_feasible: bool
    total_travel_time: float
    total_distance: float
    total_lateness: float
    total_waiting_time: float
    capacity_violations: int
    time_window_violations: int
    objective_cost: float
    vehicle_routes: List[List[int]]
    route_details: List[Dict[str, Any]]
    details_by_vehicle: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "total_travel_time": round(self.total_travel_time, 2),
            "total_distance": round(self.total_distance, 2),
            "total_lateness": round(self.total_lateness, 2),
            "total_waiting_time": round(self.total_waiting_time, 2),
            "capacity_violations": self.capacity_violations,
            "time_window_violations": self.time_window_violations,
            "objective_cost": round(self.objective_cost, 2),
            "num_vehicles_used": len([r for r in self.vehicle_routes if len(r) > 2]),
            "vehicle_routes": self.vehicle_routes,
            "route_details": self.route_details,
        }


class MathematicalFormulation:
    """
    Standalone explicit specification of the routing optimization problem.
    Provides mathematical inspection, constraint validation, and exact objective cost evaluation.
    """
    def __init__(
        self,
        graph: TrafficGraph,
        vehicle_capacity: float = 60.0,
        num_vehicles: int = 4,
        start_time_minutes: float = 540.0,  # 09:00 AM
        lateness_penalty_per_min: float = 5.0,
        capacity_penalty_per_unit: float = 50.0,
    ):
        self.graph = graph
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.start_time_minutes = start_time_minutes
        self.lateness_penalty_per_min = lateness_penalty_per_min
        self.capacity_penalty_per_unit = capacity_penalty_per_unit

        self.depot_id = graph.depot_ids[0] if graph.depot_ids else 0
        self.customers = [graph.nodes[cid] for cid in graph.customer_ids]
        self.all_interest_nodes = [self.depot_id] + graph.customer_ids
        self.node_to_idx = {nid: idx for idx, nid in enumerate(self.all_interest_nodes)}
        self.idx_to_node = {idx: nid for idx, nid in enumerate(self.all_interest_nodes)}

    def get_formulation_metadata(self) -> Dict[str, Any]:
        """Provides mathematical LaTeX formulation and problem dimensions for UI display."""
        return {
            "problem_name": "Dynamic Capacitated Vehicle Routing Problem with Time Windows (DCVRPTW)",
            "latex_objective": r"\min \sum_{k \in K} \sum_{i \in V} \sum_{j \in V} x_{ijk} \cdot W_{ij}(t_{ik}) + \mathcal{P}_{\text{cap}} + \mathcal{P}_{\text{tw}}",
            "latex_constraints": [
                r"\sum_{k \in K} \sum_{j \in V \setminus \{i\}} x_{ijk} = 1 \quad \forall i \in \mathcal{C} \quad \text{(Customer Visit)}",
                r"\sum_{j \in V \setminus \{0\}} x_{0jk} = 1 \quad \forall k \in K \quad \text{(Depot Departure)}",
                r"\sum_{i \in V \setminus \{0\}} x_{i0k} = 1 \quad \forall k \in K \quad \text{(Depot Return)}",
                r"\sum_{i \in \mathcal{C}} d_i \sum_{j \in V} x_{ijk} \le Q_k \quad \forall k \in K \quad \text{(Capacity Limit)}",
                r"t_{jk} \ge t_{ik} + s_i + W_{ij}(t_{ik}) - M(1 - x_{ijk}) \quad \forall k, i, j \quad \text{(Schedule Precedence)}",
                r"e_j \le t_{jk} \le l_j \quad \forall j \in \mathcal{C} \quad \text{(Time Windows)}",
            ],
            "parameters": {
                "num_customers": len(self.customers),
                "num_vehicles": self.num_vehicles,
                "vehicle_capacity": self.vehicle_capacity,
                "depot_id": self.depot_id,
                "fleet_start_time": f"{int(self.start_time_minutes)//60:02d}:{int(self.start_time_minutes)%60:02d}",
            },
        }

    def evaluate_solution(
        self,
        vehicle_routes: List[List[int]],
        time_matrix: Optional[np.ndarray] = None,
    ) -> RouteEvaluation:
        """
        Evaluates a candidate solution consisting of routes for each vehicle.
        Each route is a list: [depot_id, cust_1, cust_2, ..., depot_id]
        Computes exact live travel time, wait time, lateness, and constraint violations.
        """
        if time_matrix is None:
            time_matrix = self.graph.get_travel_time_matrix(self.all_interest_nodes)

        total_travel_time = 0.0
        total_distance = 0.0
        total_lateness = 0.0
        total_waiting_time = 0.0
        capacity_violations = 0
        time_window_violations = 0
        visited_customers = set()

        route_details = []

        for v_idx, route in enumerate(vehicle_routes):
            if not route or len(route) <= 2:
                continue

            current_load = 0.0
            current_time = self.start_time_minutes
            v_travel_time = 0.0
            v_lateness = 0.0
            v_waiting = 0.0

            route_steps = []

            for step_idx in range(len(route) - 1):
                u = route[step_idx]
                v = route[step_idx + 1]

                if v in self.node_to_idx and v != self.depot_id:
                    visited_customers.add(v)

                u_idx = self.node_to_idx.get(u, 0)
                v_idx_loc = self.node_to_idx.get(v, 0)
                leg_time = time_matrix[u_idx, v_idx_loc]
                v_travel_time += leg_time
                current_time += leg_time

                # Check customer node demands and windows
                if v in self.graph.nodes and v != self.depot_id:
                    node_v = self.graph.nodes[v]
                    current_load += node_v.demand

                    # Early arrival -> wait until window opens
                    if current_time < node_v.earliest_time:
                        wait = node_v.earliest_time - current_time
                        v_waiting += wait
                        current_time = node_v.earliest_time

                    # Late arrival -> penalize
                    if current_time > node_v.latest_time:
                        late = current_time - node_v.latest_time
                        v_lateness += late
                        time_window_violations += 1

                    # Add service time
                    current_time += node_v.service_time

                route_steps.append({
                    "from_node": u,
                    "to_node": v,
                    "departure_time": round(current_time - leg_time, 2),
                    "arrival_time": round(current_time, 2),
                    "leg_travel_time": round(leg_time, 2),
                })

            # Check capacity violation for vehicle
            cap_excess = max(0.0, current_load - self.vehicle_capacity)
            if cap_excess > 0:
                capacity_violations += int(np.ceil(cap_excess / 10.0))

            total_travel_time += v_travel_time
            total_lateness += v_lateness
            total_waiting_time += v_waiting

            route_details.append({
                "vehicle_id": v_idx,
                "route": route,
                "load": current_load,
                "capacity": self.vehicle_capacity,
                "is_capacity_exceeded": cap_excess > 0,
                "travel_time": round(v_travel_time, 2),
                "lateness": round(v_lateness, 2),
                "waiting_time": round(v_waiting, 2),
                "steps": route_steps,
            })

        # Check customer coverage
        missing_customers = len(self.graph.customer_ids) - len(visited_customers)
        unvisited_penalty = missing_customers * 1000.0

        # Calculate penalized objective cost
        penalty = (
            (capacity_violations * self.capacity_penalty_per_unit)
            + (total_lateness * self.lateness_penalty_per_min)
            + unvisited_penalty
        )
        objective_cost = total_travel_time + penalty

        is_feasible = (
            capacity_violations == 0
            and missing_customers == 0
            and time_window_violations == 0
        )

        return RouteEvaluation(
            is_feasible=is_feasible,
            total_travel_time=total_travel_time,
            total_distance=total_travel_time * 0.65,  # Approx km
            total_lateness=total_lateness,
            total_waiting_time=total_waiting_time,
            capacity_violations=capacity_violations + missing_customers,
            time_window_violations=time_window_violations,
            objective_cost=objective_cost,
            vehicle_routes=vehicle_routes,
            route_details=route_details,
        )
