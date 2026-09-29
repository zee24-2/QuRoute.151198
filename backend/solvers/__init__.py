"""
Solver registry and factory for QuRoute (SIH26137).
"""

from typing import Dict, Type
from .base import BaseRoutingSolver, SolverConfig, SolverResult, IterationEvent
from .qpso import QPSOSolver
from .pso import ClassicalPSOSolver
from .ga import GeneticAlgorithmSolver
from .aco import AntColonyOptimizationSolver
from .exact import ExactRoutingSolver, ORToolsGLSSolver
from .astar_greedy import AStarGreedySolver

AVAILABLE_SOLVERS: Dict[str, Type[BaseRoutingSolver]] = {
    "qpso": QPSOSolver,
    "pso": ClassicalPSOSolver,
    "ga": GeneticAlgorithmSolver,
    "aco": AntColonyOptimizationSolver,
    "exact": ExactRoutingSolver,
    "cpsat_exact": ExactRoutingSolver,
    "ortools_gls": ORToolsGLSSolver,
    "astar_greedy": AStarGreedySolver,
}


def get_solver(name: str) -> BaseRoutingSolver:
    key = name.lower().strip()
    if key in AVAILABLE_SOLVERS:
        return AVAILABLE_SOLVERS[key]()
    raise ValueError(f"Unknown solver '{name}'. Available: {list(AVAILABLE_SOLVERS.keys())}")
