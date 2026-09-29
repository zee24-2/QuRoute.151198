"""
Common Solver Interface and Result Contracts for QuRoute (SIH26137).
Guarantees identical inputs, equal fitness evaluation budgets (eval_budget),
warm-starting support, and output reporting across all optimization algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from ..core.formulation import MathematicalFormulation, RouteEvaluation


@dataclass
class SolverConfig:
    max_iters: int = 150
    swarm_size: int = 40
    seed: int = 42
    eval_budget: Optional[int] = None  # Equal evaluation budget across algorithms
    warm_start_routes: Optional[List[List[int]]] = None  # Previous best solution for sub-second dynamic re-routing
    # Algorithm-specific parameters
    beta_start: float = 1.0    # QPSO contraction-expansion start
    beta_end: float = 0.5      # QPSO contraction-expansion end
    w_inertia: float = 0.72    # Classical PSO inertia
    c1_cognitive: float = 1.49  # Classical PSO cognitive
    c2_social: float = 1.49    # Classical PSO social
    crossover_rate: float = 0.85  # GA crossover rate
    mutation_rate: float = 0.15   # GA mutation rate
    aco_alpha: float = 1.0     # ACO pheromone importance
    aco_beta: float = 2.5      # ACO heuristic visibility importance
    aco_rho: float = 0.15      # ACO pheromone evaporation rate

    def resolve_max_iters(self, evals_per_iter: int) -> int:
        """If eval_budget is provided, computes the exact iteration count so total evaluations match."""
        if self.eval_budget is not None and self.eval_budget > 0:
            return max(1, (self.eval_budget // max(1, evals_per_iter)) - 1)
        return self.max_iters


@dataclass
class IterationEvent:
    iteration: int
    current_best_cost: float
    best_route: List[List[int]]
    timestamp_ms: float
    solver_name: str
    evaluations: int = 0


@dataclass
class SolverResult:
    solver_name: str
    best_route: List[List[int]]
    best_cost: float
    convergence_history: List[float]
    wall_clock_time_seconds: float
    feasibility: bool
    iterations_to_convergence: int
    evaluation: RouteEvaluation
    parameters: Dict[str, Any] = field(default_factory=dict)
    detailed_street_routes: List[Dict[str, Any]] = field(default_factory=list)
    eval_history: List[List[float]] = field(default_factory=list)  # [[eval_count, best_cost], ...]
    total_evaluations: int = 0
    warm_started: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "best_route": self.best_route,
            "best_cost": round(self.best_cost, 2),
            "convergence_history": [round(c, 2) for c in self.convergence_history],
            "eval_history": [[int(e), round(c, 2)] for e, c in self.eval_history],
            "total_evaluations": self.total_evaluations,
            "warm_started": self.warm_started,
            "wall_clock_time": round(self.wall_clock_time_seconds, 4),
            "feasibility": self.feasibility,
            "iterations_to_convergence": self.iterations_to_convergence,
            "evaluation": self.evaluation.to_dict(),
            "parameters": self.parameters,
            "detailed_street_routes": self.detailed_street_routes,
        }


class BaseRoutingSolver(ABC):
    """
    Abstract Base Class for all routing solvers (QPSO, PSO, GA, ACO, OR-Tools GLS, CP-SAT Exact).
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self,
        formulation: MathematicalFormulation,
        config: SolverConfig,
        callback: Optional[Callable[[IterationEvent], None]] = None,
    ) -> SolverResult:
        """Executes the optimization algorithm on the problem formulation."""
        pass

    @staticmethod
    def calculate_iterations_to_convergence(history: List[float], threshold_pct: float = 0.01) -> int:
        """Computes the first iteration/step where the objective is within 1% of the final converged cost."""
        if not history:
            return 0
        final_val = history[-1]
        target_val = final_val * (1.0 + threshold_pct)
        for i, val in enumerate(history):
            if val <= target_val:
                return i + 1
        return len(history)
