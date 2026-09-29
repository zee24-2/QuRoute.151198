# QPSO-Powered Traffic Route Optimizer — Complete Program Description

> **Purpose of this document**: A thorough, code-level description of every component in this project, written so that a downstream AI agent can fully understand, extend, or debug the codebase without access to the source files.

---

## 1. High-Level Summary

This is a **full-stack, single-page web application** for benchmarking and visualizing metaheuristic optimization algorithms on a **Dynamic Capacitated Vehicle Routing Problem with Time Windows (DCVRPTW)**. The core claim is that **Quantum-behaved Particle Swarm Optimization (QPSO)** outperforms Classical PSO, Genetic Algorithm (GA), and Ant Colony Optimization (ACO) — and this is verified empirically against a ground-truth **Google OR-Tools CP-SAT exact solver**.

**Technology stack:**
- **Backend**: Python 3.x, FastAPI + Uvicorn (ASGI), NetworkX, NumPy, OR-Tools, OSMnx (optional)
- **Frontend**: Vanilla HTML5/CSS/JavaScript, HTML5 Canvas, Chart.js
- **Communication**: REST (JSON) + WebSocket streaming (real-time iteration-by-iteration updates)
- **Entry point**: `run.py` — starts Uvicorn and auto-opens the browser at `http://127.0.0.1:8000`

---

## 2. Repository Layout

```
QPSO/
├── run.py                          # One-click launcher: starts Uvicorn + opens browser
├── README.md
├── backend/
│   ├── app.py                      # FastAPI application, all REST + WS endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph.py                # Node, Edge, TrafficGraph, grid + OSMnx generators
│   │   ├── traffic.py              # DynamicTrafficEngine (3-layer weight hierarchy)
│   │   ├── formulation.py          # MathematicalFormulation, RouteEvaluation
│   │   └── encoding.py             # RandomKeyDecoder (shared across QPSO/PSO/GA)
│   ├── solvers/
│   │   ├── __init__.py             # Solver registry + get_solver() factory
│   │   ├── base.py                 # SolverConfig, IterationEvent, SolverResult, BaseRoutingSolver
│   │   ├── qpso.py                 # QPSOSolver  ← primary algorithm
│   │   ├── pso.py                  # ClassicalPSOSolver
│   │   ├── ga.py                   # GeneticAlgorithmSolver
│   │   ├── aco.py                  # AntColonyOptimizationSolver
│   │   └── exact.py                # ExactRoutingSolver (OR-Tools CP-SAT)
│   └── benchmark/
│       ├── __init__.py
│       ├── runner.py               # BenchmarkRunner: 30-trial statistical harness
│       └── scalability.py          # ScalabilityExperiment: cost/runtime vs N sweep
├── frontend/
│   ├── index.html                  # Single-page app: 5 screen tabs, all HTML/CSS/JS
│   ├── css/
│   │   └── styles.css              # Dark-theme UI (navy/cyan/slate palette)
│   └── js/
│       ├── api.js                  # API client: REST fetch wrappers + WebSocket factory
│       ├── canvas.js               # NetworkCanvas class: HTML5 Canvas renderer
│       └── main.js                 # App controller: 5 screen initializers + appState
└── tests/
    ├── test_graph.py               # Grid/OSMnx generation, edge weights, connectivity
    ├── test_benchmark.py           # BenchmarkRunner produces valid stats
    ├── test_formulation.py         # LaTeX metadata, optimality gap vs exact
    └── test_solvers.py             # All 5 solvers run, produce valid SolverResult
```

---

## 3. Backend — Core Layer (`backend/core/`)

### 3.1 `graph.py` — Traffic Graph Data Model

#### Dataclasses

**`Node`**
```
id: int
name: str
node_type: str          # "depot" | "customer" | "intersection"
x, y: float             # Spatial coordinates (km for grid, lon/lat for OSM)
demand: float           # Delivery demand (customers only, 0 for others)
earliest_time: float    # Service window open (minutes from midnight, e.g. 480 = 08:00)
latest_time: float      # Service window close
service_time: float     # Unloading duration in minutes
```

**`Edge`**
```
u, v: int               # Directed edge endpoints (node IDs)
base_distance: float    # Fixed distance in km
base_travel_time: float # Fixed traversal time in minutes at free-flow speed
speed_limit: float      # km/h
congestion_multiplier: float  # Live multiplier applied by traffic engine
is_closed: bool         # If True, live_weight = 1e9 (impassable)

Property:
  live_weight = 1e9 if is_closed else base_travel_time * max(0.1, congestion_multiplier)
```

**`TrafficGraph`**
- Stores `nodes: Dict[int, Node]`, `edges: Dict[Tuple[int,int], Edge]`
- Maintains a `networkx.DiGraph` (`nx_graph`) mirroring the same structure for Dijkstra queries
- Tracks `depot_ids: List[int]` and `customer_ids: List[int]`
- **Key method**: `get_travel_time_matrix(target_node_ids) -> np.ndarray`
  - Syncs `nx_graph` edge weights from `Edge.live_weight`
  - Runs `nx.single_source_dijkstra_path_length` from every target node
  - Returns an `(n, n)` matrix of shortest live travel times
  - Disconnected pairs get penalty `1e6`
- `update_edge_multiplier(u, v, multiplier, is_closed)` — updates a single edge and marks the graph dirty
- `to_dict()` — serializes full graph to JSON-friendly dict for API responses

#### Graph Generators

**`generate_grid_network(rows, cols, cell_size_km, num_depots, num_customers, seed, diagonals)`**
- Creates an `N×M` lattice with node IDs `0..rows*cols-1`
- All nodes start as `"intersection"`, then:
  - Center node (`rows//2, cols//2`) promoted to `"depot"`, named `"Central_Depot"`
  - `num_customers` random nodes (excluding depot) promoted to `"customer"` with:
    - `demand`: uniform integer in [5, 25]
    - `service_time`: choice of [8, 10, 15] minutes
    - `earliest_time`: uniform in [480, 840] (8 AM–2 PM)
    - `latest_time`: `earliest + uniform(90, 240)` minutes
- Horizontal and vertical edges at `cell_size_km` distance, `speed_limit=40 km/h`
- Optional diagonals at `cell_size_km * 1.414`, `speed_limit=35 km/h`
- Default params: 6×6 grid, 1.5 km cells, 12 customers, seed 42

**`generate_indian_city_network(city_name, num_customers, seed)`**
- **Primary path**: Downloads real OSM data via `osmnx.graph_from_place(city_name + ", India", network_type="drive")`
  - Takes the largest strongly connected component
  - Trims to ≤120 nodes via BFS from center node (cutoff=12 hops)
  - Maps OSM node IDs to sequential integer IDs
  - Adds directed edges with `length_m / 1000` km and `speed_limit=35`
- **Fallback path** (if OSMnx unavailable/offline): Procedurally models **Connaught Place, New Delhi**:
  - 3 concentric circles: Inner (r=0.35 km, 12 pts), Middle (r=0.65 km, 16 pts), Outer (r=1.05 km, 20 pts)
  - Each circle has perimeter chord edges at `speed_limit=30`
  - 8 radial spokes connect nearest nodes between circles at `speed_limit=45`
  - Total: 48 intersection nodes
- After network construction (both paths): first node → depot, `num_customers` random nodes → customers

---

### 3.2 `traffic.py` — Dynamic Traffic Engine

**`TrafficIncident` dataclass**
```
id: str
edge: Tuple[int, int]
severity: float           # Initial congestion boost (e.g. 5.0 = +5x on top of base)
start_time: float         # Simulation clock time when incident began
decay_half_life: float    # Minutes for incident to decay to half severity (default 30)
description: str

Method: current_multiplier_boost(current_time) -> float
  = severity * (0.5) ^ (dt / half_life)   [exponential decay]
  Returns 0.0 if current_time < start_time
```

**`DynamicTrafficEngine`**

Constructor: `__init__(graph: TrafficGraph, initial_time_minutes=540.0)`
- Holds `current_time_minutes` (0..1440, representing midnight to midnight)
- `incidents: Dict[str, TrafficIncident]` — active incidents
- `manual_multipliers: Dict[Tuple[int,int], float]` — user overrides
- `closed_edges: set` — user-closed edge pairs
- Calls `update_all_edge_weights()` on init

**3-Layer Weight Hierarchy** (applied multiplicatively):

**Layer 1 — Cyclical diurnal curve** (`calculate_cyclical_multiplier(t_mins)`):
```python
hour = t_mins / 60.0
base = 0.8 + 0.3 * (0.5 + 0.5 * sin(π * (hour - 3.0) / 12.0))
peak_morning = 1.4 * exp(-((hour - 9.0)² / (2 * 0.9²)))   # 09:00 AM peak
peak_evening = 1.6 * exp(-((hour - 18.5)² / (2 * 1.1²)))  # 18:30 PM peak
return max(0.8, base + peak_morning + peak_evening)
# Range: ~0.8x (night) to ~3.0x (morning rush)
```

**Layer 2 — Incident boosts** (`trigger_random_incident(edge_key, severity, half_life, description)`):
- Picks a random edge if `edge_key=None`
- Allocates incident ID: `inc_{counter}_{time_minutes}`
- Reverse edge gets 70% of the boost
- Incidents with boost < 0.05 are automatically pruned on each `update_all_edge_weights()` call

**Layer 3 — Manual override** (`set_manual_edge_congestion(u, v, multiplier, is_closed)`):
- Sets `manual_multipliers[(u,v)]` and `(v,u)` symmetrically
- Adds to `closed_edges` if `is_closed=True`

**`update_all_edge_weights()`** — The central update that propagates all three layers:
```
final_multiplier = (cyclical_factor × manual_factor) + incident_boost
graph.update_edge_multiplier(u, v, final_multiplier, is_closed)
```
Called by `set_time()`, `trigger_random_incident()`, `set_manual_edge_congestion()`.

`get_status_summary()` returns a dict with `current_time`, `time_minutes`, `cyclical_multiplier`, `active_incidents_count`, `closed_edges_count`, and incident detail list.

---

### 3.3 `formulation.py` — Mathematical Problem Specification

**`RouteEvaluation` dataclass** — result of evaluating one candidate solution:
```
is_feasible: bool
total_travel_time: float
total_distance: float             # ≈ total_travel_time * 0.65 (approx km)
total_lateness: float             # Sum of lateness across all vehicles
total_waiting_time: float
capacity_violations: int
time_window_violations: int
objective_cost: float             # The scalar fitness value minimized by all solvers
vehicle_routes: List[List[int]]   # Per-vehicle node sequences
route_details: List[Dict]         # Per-step breakdown per vehicle
```

**`MathematicalFormulation`**

Constructor parameters:
- `graph: TrafficGraph`
- `vehicle_capacity: float` (default 60.0)
- `num_vehicles: int` (default 4)
- `start_time_minutes: float` (default 540.0 = 09:00)
- `lateness_penalty_per_min: float` (default 5.0)
- `capacity_penalty_per_unit: float` (default 50.0)

Internal state:
- `depot_id = graph.depot_ids[0]`
- `customers = [graph.nodes[cid] for cid in graph.customer_ids]`
- `all_interest_nodes = [depot_id] + customer_ids` — the only nodes that matter for routing
- `node_to_idx / idx_to_node` — bidirectional mapping for matrix indexing

**`get_formulation_metadata()`** — Returns LaTeX strings for the UI:
- `latex_objective`: the DCVRPTW objective function formula
- `latex_constraints`: 6 constraint strings (customer visit, depot departure, depot return, capacity, schedule precedence, time windows)
- `parameters`: problem dimensions dict

**`evaluate_solution(vehicle_routes, time_matrix)`** — Core fitness evaluator:
- Input: list of routes, each `[depot_id, cust1, cust2, ..., depot_id]`
- For each vehicle:
  - Simulates travel leg by leg using `time_matrix[u_idx, v_idx]`
  - Tracks `current_time` (initialized to `start_time_minutes`)
  - If arriving early → waits until `earliest_time`
  - If arriving late → accumulates lateness, increments `time_window_violations`
  - Adds `service_time` at each customer stop
  - Accumulates `current_load`, checks against `vehicle_capacity`
- After all vehicles:
  - `missing_customers` penalty: 1000 per unvisited customer
  - `capacity_violations` penalty: `capacity_penalty_per_unit * violations`
  - `lateness` penalty: `lateness_penalty_per_min * total_lateness`
  - `objective_cost = total_travel_time + all_penalties`
- `is_feasible = True` only if zero capacity violations, zero missing customers, zero TW violations

---

### 3.4 `encoding.py` — Shared Solution Representation

**`RandomKeyDecoder`**

The **central design decision** that ensures 100% fair comparison across QPSO, PSO, and GA. All three algorithms operate in the same continuous space $[0,1]^N$ (where N = number of customers) and use identical decoding.

**`decode_continuous_vector(z: np.ndarray) -> List[List[int]]`**:
1. `perm = argsort(z)` — converts the float vector to a customer visit permutation
2. `decode_permutation(perm)` — greedy capacity-based partitioning:
   - Iterate customers in permutation order
   - Add to current vehicle if capacity allows; else advance to next vehicle
   - Wrap each non-empty route with `[depot_id, ..., depot_id]`

**`evaluate_continuous_vector(z, time_matrix) -> (cost, RouteEvaluation)`**:
- Calls `decode_continuous_vector(z)` then `formulation.evaluate_solution(routes, time_matrix)`
- Returns `(objective_cost, eval_result)` — used by all population-based solvers

---

## 4. Backend — Solvers (`backend/solvers/`)

### 4.1 `base.py` — Solver Interface

**`SolverConfig` dataclass** — unified config for all algorithms:
```python
max_iters: int = 150
swarm_size: int = 40
seed: int = 42
beta_start: float = 1.0     # QPSO: CE coefficient start
beta_end: float = 0.5       # QPSO: CE coefficient end
w_inertia: float = 0.72     # PSO: inertia weight
c1_cognitive: float = 1.49  # PSO: cognitive coefficient
c2_social: float = 1.49     # PSO: social coefficient
crossover_rate: float = 0.85 # GA
mutation_rate: float = 0.15  # GA
aco_alpha: float = 1.0      # ACO: pheromone exponent
aco_beta: float = 2.5       # ACO: heuristic exponent
aco_rho: float = 0.15       # ACO: pheromone evaporation rate
```

**`IterationEvent` dataclass** — streamed per iteration over WebSocket:
```python
iteration: int
current_best_cost: float
best_route: List[List[int]]
timestamp_ms: float
solver_name: str
```

**`SolverResult` dataclass** — final output of every solver:
```python
solver_name: str
best_route: List[List[int]]
best_cost: float
convergence_history: List[float]   # best_cost at each iteration
wall_clock_time_seconds: float
feasibility: bool
iterations_to_convergence: int     # First iter within 1% of final cost
evaluation: RouteEvaluation
parameters: Dict                   # Algorithm-specific hyperparameters
```

**`BaseRoutingSolver`** (ABC):
- Abstract `run(formulation, config, callback) -> SolverResult`
- Static `calculate_iterations_to_convergence(history, threshold_pct=0.01)` — finds first iteration where `cost ≤ final_cost * 1.01`

**`get_solver(name)` factory** in `__init__.py`:
- Registry: `{"qpso": QPSOSolver, "pso": ClassicalPSOSolver, "ga": GeneticAlgorithmSolver, "aco": AntColonyOptimizationSolver, "exact": ExactRoutingSolver}`

---

### 4.2 `qpso.py` — QPSO Solver (Primary Algorithm)

**Algorithm**: Quantum-behaved Particle Swarm Optimization (Sun et al.)

**Initialization**:
- `x = rng.uniform(0, 1, size=(swarm_size, num_customers))` — particle positions in random-key space
- `pbest_x = copy(x)` — personal best positions
- `pbest_costs = [∞] * swarm_size`
- Evaluate all initial particles; track `gbest_x`, `gbest_cost`, `best_eval`

**Main loop** (per iteration `it = 1..max_iters`):
1. **Beta decay**: `β = beta_start - (beta_start - beta_end) * (it / max_iters)` — linearly decays from 1.0 to 0.5
2. **Mean best**: `mbest = mean(pbest_x, axis=0)` — centroid of all personal bests
3. **Local attractor**: `φ ~ U(0,1)^D`; `p = φ * pbest_x + (1-φ) * gbest_x` — per-particle stochastic blend
4. **Quantum position update**:
   - `u ~ U(1e-9, 1)^(swarm × D)`
   - `step_length = β * |mbest - x| * ln(1/u)` — samples from double-exponential (Laplace) distribution
   - `signs ~ {-1, +1}^(swarm × D)` — random sign
   - `x = p + signs * step_length`
5. Evaluate each new particle; update `pbest`, `gbest` if improved

**Key property**: No velocity vector. The quantum position update directly samples from a probability density whose width is controlled by `β`. This eliminates velocity clamping issues and enables more effective global search.

---

### 4.3 `pso.py` — Classical PSO Solver

**Algorithm**: Standard velocity-position PSO with inertia weight and clamping.

- Initializes both `x` (positions) and `v` (velocities in `[-v_max, v_max]`, `v_max=0.2`)
- Per iteration:
  - `v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)` (vectorized)
  - `v = clip(v, -v_max, v_max)` — velocity clamping
  - `x = x + v` — position update

Uses **identical** `RandomKeyDecoder` and `formulation.evaluate_solution` — the only difference from QPSO is the position update rule.

---

### 4.4 `ga.py` — Genetic Algorithm Solver

**Representation**: Direct permutation of `[0..num_customers-1]` (not random-key continuous).

**Components**:
- **Tournament selection** (`k=3`): picks 3 random individuals, returns best
- **Order Crossover (OX)**: Two-point crossover preserving relative order of genes. Copies a slice from parent1 to child1, fills remaining genes in parent2's order (rotating from crossover point)
- **Mutation** (applied at `mutation_rate=0.15`):
  - 50% chance: swap two genes
  - 50% chance: 2-opt sub-tour inversion (reverse a random sub-segment)
- **Elitism**: top 2 individuals survive each generation unchanged

**Decoding**: Uses `RandomKeyDecoder.decode_permutation(perm)` — same greedy capacity-pack decoder as QPSO/PSO.

---

### 4.5 `aco.py` — Ant Colony Optimization Solver

**Pheromone structure**: `τ: (num_customers × num_customers)` matrix, initialized to `1 / (num_customers * 10)`

**Heuristic visibility**: `η[i,j] = 1 / (customer_time_submatrix[i,j] + 1e-4)`, diagonal zero

**Per iteration**:
- Each ant starts at a random customer
- Constructs a complete permutation using probabilistic selection:
  - `probs[j] ∝ τ[current, j]^α * η[current, j]^β`
  - If all probs zero (numerically): random fallback
- Decoded to vehicle routes via `RandomKeyDecoder.decode_permutation`

**Pheromone update**:
- **Evaporation**: `τ = (1 - ρ) * τ`
- **Elitist deposit**: global best tour deposits `100 / gbest_cost` symmetrically on all edges

**Ant population**: capped at `min(swarm_size, 30)`, minimum 10

---

### 4.6 `exact.py` — OR-Tools CP-SAT Exact Solver

**Purpose**: Ground-truth optimum for benchmark gap calculation. Used for ≤15 customers.

**Implementation**:
- Converts `time_matrix * 100` to integer for OR-Tools
- Creates `RoutingIndexManager` and `RoutingModel`
- Registers transit callback (time matrix) and demand callback
- Capacity dimension with `vehicle_capacities`
- Search strategy: `PATH_CHEAPEST_ARC` first solution + `GUIDED_LOCAL_SEARCH` local search
- **Time limit: 5 seconds** (to keep UI responsive)
- If no solution found, returns trivial `[[depot, depot]] * num_vehicles`
- Returns a single-element `convergence_history = [best_cost]`

---

## 5. Backend — Benchmarking (`backend/benchmark/`)

### 5.1 `runner.py` — Statistical Benchmark Harness

**`BenchmarkRunner.run_statistical_benchmark(solver_names, num_trials, max_iters, base_seed, include_exact)`**

1. Optionally run exact solver once (if `num_customers <= 15`) for ground truth
2. For each trial `t in 0..num_trials-1`:
   - All algorithms get `seed = base_seed + t` (synchronized seeds for fairness)
   - Each solver runs on the same `FORMULATION` (same traffic snapshot)
3. Compute per-algorithm statistics:
   - `mean_cost`, `std_cost`, `min_cost`, `max_cost`, `median_cost`
   - `mean_runtime_seconds`, `mean_convergence_iterations`
   - `feasibility_rate_pct`
   - `optimality_gap_pct = (mean_cost - exact_cost) / exact_cost * 100`
   - `average_convergence_curve`: pointwise mean of all trial histories
   - `box_plot_data`: raw cost array for all trials
4. Calls `generate_why_qpso_pitch()` — auto-generates a plain-English narrative comparing QPSO vs Classical PSO, citing actual measured numbers

**Return structure**:
```json
{
  "summary": {
    "qpso": { "mean_cost": ..., "std_cost": ..., "optimality_gap_pct": ..., ... },
    "pso":  { ... },
    "ga":   { ... },
    "aco":  { ... },
    "exact": { ... }
  },
  "exact_optimal_cost": ...,
  "problem_size_customers": ...,
  "trials_conducted": ...,
  "empirical_pitch": "Across 30 trials..."
}
```

### 5.2 `scalability.py` — Scalability Sweep

**`ScalabilityExperiment.run_sweep(solver_names, trials_per_size, max_iters, seed)`**

For each node size `N` in `self.node_sizes` (default `[10, 25, 50, 100, 150]`):
1. Builds a grid with `grid_dim = ceil(sqrt(N * 1.6))` rows/cols and `N` customers
2. Sets `num_vehicles = ceil(N / 6)`, `vehicle_capacity = 70`
3. Runs `trials_per_size` trials of each solver, averages cost/runtime/convergence_iters

**Return structure**:
```json
{
  "sizes_evaluated": [10, 20, 35, 50],
  "algorithms": {
    "qpso": { "sizes": [...], "mean_costs": [...], "mean_runtimes": [...], "mean_convergence_iters": [...] }
  },
  "trials_per_size": 2
}
```

---

## 6. Backend — API (`backend/app.py`)

FastAPI app `"QPSO Traffic Route Optimizer API"` version `2.0.0`.

**Global mutable state** (module-level singletons, shared across all requests):
```python
CURRENT_GRAPH: TrafficGraph         # Default: 6x6 grid, 12 customers, seed=42
TRAFFIC_ENGINE: DynamicTrafficEngine  # Default: 09:00 AM
FORMULATION: MathematicalFormulation  # Default: capacity=60, 3 vehicles
LATEST_BENCHMARK_RESULTS: Optional[Dict]
LATEST_SCALABILITY_RESULTS: Optional[Dict]
```

`rebuild_formulation()` — called whenever graph or config changes; recreates `FORMULATION` with current engine time.

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/graph/current` | Returns current graph, traffic summary, formulation metadata |
| `POST` | `/api/graph/generate/grid` | Generates new grid network (replaces global state) |
| `POST` | `/api/graph/generate/city` | Generates Indian city network (replaces global state) |
| `POST` | `/api/traffic/time` | Sets simulation clock `{minutes: float}` |
| `POST` | `/api/traffic/incident` | Triggers traffic incident (optionally on specific edge) |
| `POST` | `/api/traffic/edge` | Sets manual congestion/closure on an edge |
| `POST` | `/api/graph/node/edit` | Edits node type/demand/time-window, rebuilds formulation |
| `POST` | `/api/formulation/config` | Updates vehicle_capacity and num_vehicles |
| `GET` | `/api/formulation` | Returns formulation metadata (LaTeX + parameters) |
| `POST` | `/api/benchmark/statistical` | Runs multi-trial benchmark (blocking) |
| `GET` | `/api/benchmark/latest` | Returns last benchmark results |
| `POST` | `/api/benchmark/scalability` | Runs scalability sweep |
| `GET` | `/api/benchmark/scalability/latest` | Returns last scalability results |
| `GET` | `/` | Serves `frontend/index.html` |
| Static | `/static/...` | Serves `frontend/` directory |

### WebSocket Endpoint `/ws/solve`

Handles two modes sent as JSON after connection:

**Mode `"single"`** — One solver, live streaming:
```json
{ "mode": "single", "solver_name": "qpso", "max_iters": 100, "swarm_size": 30, "seed": 42, "beta_start": 1.0, "beta_end": 0.5 }
```
- Runs solver in `asyncio.to_thread()` (non-blocking)
- Per iteration: `{"type": "iteration_update", "solver_name", "iteration", "cost", "route", "timestamp_ms"}`
- On completion: `{"type": "finished", "result": {...}}`

**Mode `"race"`** — All solvers simultaneously:
```json
{ "mode": "race", "solvers": ["qpso","pso","ga","aco"], "max_iters": 100, "swarm_size": 30, "seed": 42 }
```
- Runs each solver in a separate `asyncio.to_thread()`, all gathered concurrently via `asyncio.gather()`
- Each iteration: `{"type": "race_step", "solver_name", "solver_key", "iteration", "cost", "route"}`
- Each solver finish: `{"type": "race_solver_finished", "solver_name", "solver_key", "result"}`
- All done: `{"type": "race_finished"}`

---

## 7. Frontend (`frontend/`)

### 7.1 Architecture

Single `index.html` with 5 tab-based "screen" divs. Navigation is pure CSS `display` toggling — no page reloads. `main.js` is the master controller.

**Global `appState` object** (in `main.js`):
```javascript
{
  activeScreen: 'screen1',
  graphData: null,          // Serialized TrafficGraph (nodes[], edges[])
  trafficData: null,        // Traffic engine status summary
  formulationData: null,    // Formulation metadata
  networkCanvas: null,      // NetworkCanvas instance
  convergenceChart: null,   // Chart.js (Screen 3)
  raceChart: null,          // Chart.js (Screen 4)
  radarChart: null,         // Chart.js (Screen 4)
  boxPlotChart: null,       // Chart.js (Screen 5)
  scalabilityChart: null,   // Chart.js (Screen 5)
  raceResults: {},
}
```

### 7.2 `api.js` — API Client

The `API` object wraps all backend calls with `fetch()` and a WebSocket factory:
- `createSolverWebSocket(onMessage, onError, onClose)` → opens `ws://host/ws/solve`, parses JSON messages, returns the `WebSocket` object

### 7.3 `canvas.js` — `NetworkCanvas` Class

HTML5 Canvas renderer with pan/zoom interactivity.

**Coordinate transforms**: `toScreen(worldX, worldY)` and `toWorld(screenX, screenY)` — apply `scale` and `offset`.

**Auto-fit** (`autoFit()`): On new graph load, computes scale and offset to fit all nodes within canvas bounds with 80px padding.

**Events**:
- `mousedown` → starts drag + calls `handleClick()`
- `mousemove` → pans when dragging
- `wheel` → zoom centered on mouse pointer (factor 1.15 in, 0.85 out)

**`handleClick(screenX, screenY)`**:
- Within 16px of a node → fires `onNodeSelect(node)` callback
- Within 12px of an edge line segment (point-to-segment distance) → fires `onEdgeSelect(edge)` callback

**`draw()` pipeline**:
1. `drawEdges()`: Colors by `getCongestionColor(multiplier)`:
   - `≤1.05`: green; `≤1.6`: yellow; `≤2.8`: orange; `>2.8`: red; closed: dashed red
   - Incident dot at midpoint if `congestion > 3.0`
2. `drawRoutes()`: Thick colored polylines with glow shadow per vehicle (cyan, purple, emerald, amber, rose, blue)
3. `drawNodes()`:
   - `depot`: Gold diamond with glow + "DEPOT" label
   - `customer`: Blue circle with demand number inside + `C{id}` label
   - `intersection`: Small dark 3px dot

### 7.4 `main.js` — 5-Screen Controller

**Screen 1 — Network Builder**:
- Toggle between "grid" and "city" parameter groups
- "Generate Network" → calls `API.generateGrid()` or `API.generateCity()`, updates canvas
- Node edit panel shown on node click; "Save Node" → `API.editNode()`

**Screen 2 — Traffic Simulator**:
- Time slider (0..1440 min) → `API.setSimTime()` on every `input`; refreshes canvas with updated edge colors
- "Trigger Incident" button → `API.triggerIncident()`
- Selected-edge panel (congestion slider + close toggle) → `API.setManualEdge()` on change

**Screen 3 — Single Solver**:
- Controls: solver dropdown, iterations, swarm size, beta start/end
- Opens WebSocket `mode: "single"`
- `iteration_update` events → append to Chart.js convergence chart + redraw canvas routes
- `finished` event → show final cost/runtime/feasibility

**Screen 4 — Benchmark Arena (Race)**:
- Opens WebSocket `mode: "race"` with all 4 solvers
- `race_step` events → append to per-solver Chart.js dataset
- `race_solver_finished` → update scoreboard table (sorted by best_cost, winner highlighted with ★)
- Static Radar Chart: hardcoded qualitative scores for QPSO vs PSO vs GA on 5 dimensions (Solution Quality, Convergence Speed, Stability, Scalability, Feasibility)

**Screen 5 — Results Dashboard**:
- Default charts: placeholder bar chart + placeholder scalability line chart
- "Run 30-Trial Statistical Harness" → `API.runStatisticalBenchmark()` → updates bar chart with live means + displays auto-generated empirical pitch text
- "Run Scalability Sweep" → `API.runScalabilitySweep()` → updates scalability line chart with live measured runtimes

---

## 8. Mathematical Formulation Summary

**Problem type**: Dynamic Capacitated VRP with Time Windows (DCVRPTW)

**Decision variables**: $x_{ijk} \in \{0,1\}$ — 1 if vehicle $k$ traverses arc $(i \to j)$

**Objective** (minimized by all solvers):
$$\min \sum_{k \in K} \sum_{i \in V} \sum_{j \in V} x_{ijk} \cdot W_{ij}(t_{ik}) + \mathcal{P}_{\text{cap}} + \mathcal{P}_{\text{tw}}$$

Where $W_{ij}(t_{ik})$ is the live travel time (dynamic, reflecting current congestion), and penalties are:
- $\mathcal{P}_{\text{cap}} = 50 \times \text{capacity\_violations}$
- $\mathcal{P}_{\text{tw}} = 5 \times \text{total\_lateness\_minutes} + 1000 \times \text{unvisited\_customers}$

**QPSO Update Rule** (no velocity):
1. $\text{mbest} = \frac{1}{M}\sum_{i=1}^M \mathbf{pbest}_i$
2. $\mathbf{p}_i = \phi \odot \mathbf{pbest}_i + (1-\phi) \odot \mathbf{gbest},\quad \phi \sim U(0,1)$
3. $x_{id}(t+1) = p_{id} \pm \beta \cdot |\text{mbest}_d - x_{id}(t)| \cdot \ln(1/u),\quad u \sim U(0,1)$
4. $\beta$ decays linearly from $1.0 \to 0.5$ over iterations

---

## 9. Test Suite (`tests/`)

13 total tests across 4 files, run via `pytest tests/ -v`:

| File | Tests |
|------|-------|
| `test_graph.py` | Grid generation dimensions, node/edge counts, depot placement, OSMnx fallback |
| `test_traffic.py` | Diurnal multiplier at rush hour times, incident decay, edge closure |
| `test_formulation.py` | LaTeX metadata fields, constraint count, optimality gap QPSO vs exact |
| `test_solvers.py` | All 5 solvers produce valid `SolverResult` with correct structure; QPSO convergence monotonicity |

---

## 10. Data Flow (End-to-End)

```
User action on UI
        │
        ▼
API.js REST/WS call
        │
        ▼
FastAPI endpoint (app.py)
        │
        ├──► TrafficGraph (graph.py)
        │          │
        │          ▼
        │    DynamicTrafficEngine (traffic.py)   ← 3-layer multiplier
        │          │
        │          ▼
        ├──► MathematicalFormulation (formulation.py)
        │          │
        │          ▼
        │    RandomKeyDecoder (encoding.py)
        │          │
        │    ┌─────┴──────────────────────────────────────┐
        │    ▼                  ▼             ▼            ▼
        │  QPSOSolver  ClassicalPSOSolver  GASolver  ACOSolver  ExactSolver
        │    │                                              (OR-Tools)
        │    ▼
        │  IterationEvent stream (WebSocket callback)
        │    │
        ▼    ▼
  SolverResult → BenchmarkRunner → stats summary → JSON response
        │
        ▼
   canvas.js + Chart.js renders results
```

---

## 11. Key Design Decisions

1. **Shared random-key encoding**: QPSO, PSO, and GA all operate in $[0,1]^N$ and use the same `RandomKeyDecoder.decode_continuous_vector`. This means differences in results are **purely algorithmic**, not encoding artifacts.

2. **Single global state**: `CURRENT_GRAPH`, `TRAFFIC_ENGINE`, `FORMULATION` are module-level singletons in `app.py`. All endpoints read/mutate these. There is **no user session isolation** — designed for single-user demo.

3. **Async solver execution**: Solvers run in `asyncio.to_thread()` (thread pool) so the WebSocket event loop stays responsive for streaming callbacks.

4. **Three-layer traffic**: The multiplicative composition `(cyclical × manual) + incident_boost` allows realistic real-time simulation without requiring any external traffic data feed.

5. **OSMnx with fallback**: The Indian city mode gracefully degrades to a procedurally modeled Connaught Place if OSMnx is offline, ensuring demos always work.

6. **Exact solver as baseline**: OR-Tools runs for exactly 5 seconds, giving a practical optimum that makes the optimality gap metric meaningful even on medium instances.
