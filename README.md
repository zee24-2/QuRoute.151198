# QPSO-Powered Traffic Route Optimizer

A high-performance, mathematically rigorous benchmarking platform and interactive simulation environment that evaluates **Quantum-behaved Particle Swarm Optimization (QPSO)** against classical heuristic and exact baselines (**Classical PSO**, **Genetic Algorithm**, **Ant Colony Optimization**, and **Google OR-Tools CP-SAT**) on dynamic urban traffic networks.

---

## Deliverables Checklist

| Deliverable | Component in Codebase | Feature Description |
| :--- | :--- | :--- |
| **Deliverable 1: Graph & Dynamic Traffic Simulation** | `backend/core/graph.py`<br>`backend/core/traffic.py` | • Dual mode: $N \times M$ Synthetic Grid & Real-world Indian City District (Connaught Place, New Delhi via OSMnx)<br>• 3-layer dynamic weight hierarchy: Diurnal rush-hour curve, Poisson incidents with half-life decay, and manual live road closure/congestion slider. |
| **Deliverable 2: Mathematical Formulation** | `backend/core/formulation.py` | • Standalone formal mathematical specification object for Dynamic Capacitated VRP with Time Windows (DCVRPTW).<br>• Real-time objective function and constraint violation validator. |
| **Deliverable 3: QPSO Core Engine & Baselines** | `backend/solvers/qpso.py`<br>`backend/solvers/pso.py`<br>`backend/solvers/ga.py`<br>`backend/solvers/aco.py`<br>`backend/solvers/exact.py` | • Vectorized QPSO with mean-best position ($\mathbf{mbest}$), delta potential-well quantum update, and linear $\beta$ decay.<br>• Identical random-key continuous encoding across QPSO, PSO, and GA for 100% fair algorithmic comparison.<br>• Ground truth global optimum computed via OR-Tools CP-SAT. |
| **Deliverable 4: Interactive 5-Screen UI/UX** | `frontend/index.html`<br>`frontend/js/canvas.js`<br>`frontend/js/main.js` | • **Screen 1**: Grid / Indian Road Network Builder<br>• **Screen 2**: Dynamic Traffic Simulator with timeline scrubber & live road block<br>• **Screen 3**: Single-Run Solver with live route animation & convergence curve<br>• **Screen 4**: Benchmark Arena (simultaneous multi-algorithm live race)<br>• **Screen 5**: Results Dashboard, Box Plots, Radar Chart, & Formulation Inspector. |
| **Deliverable 5: Systematic Benchmarking** | `backend/benchmark/runner.py`<br>`backend/benchmark/scalability.py` | • 30-trial statistical harness with identical seed per trial.<br>• Optimality gap % relative to OR-Tools exact solver.<br>• Scalability sweep from $N=10$ to $N=150+$ nodes.<br>• Automated empirical evidence pitch generator. |

---

## Mathematical Formulation

### Objective Function
Minimize the total fleet live travel time subject to dynamic congestion weights:
$$\min \sum_{k \in \mathcal{K}} \sum_{i \in \mathcal{V}} \sum_{j \in \mathcal{V}} x_{ijk} \cdot W_{ij}(t_{ik}) + \mathcal{P}_{\text{capacity}} + \mathcal{P}_{\text{time\_window}}$$

### Quantum Position Update Rule (QPSO)
Unlike classical PSO, QPSO discards velocity vectors entirely, eliminating velocity clamping issues:
1. **Swarm Mean Best**:
   $$\mathbf{mbest} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{pbest}_i$$
2. **Local Attractor**:
   $$\mathbf{p}_i = \phi \odot \mathbf{pbest}_i + (1 - \phi) \odot \mathbf{gbest}, \quad \phi \sim \mathcal{U}(0, 1)$$
3. **Quantum Wave Collapse**:
   $$x_{id}(t+1) = p_{id} \pm \beta \cdot |mbest_d - x_{id}(t)| \cdot \ln(1 / u), \quad u \sim \mathcal{U}(0, 1)$$
   where Contraction-Expansion coefficient $\beta$ linearly decays from $1.0 \to 0.5$.

---

## Quickstart

### 1. Run Automated Test Suite
```bash
python -m pytest tests/ -v
```
All 13 unit and integration tests validate the graph generator, dynamic traffic decay, constraint validations, exact solver gap, and algorithmic fairness.

### 2. Launch Interactive Platform
```bash
python run.py
```
This automatically boots the FastAPI backend and opens the dashboard at `http://127.0.0.1:8000`.

---

## Live Demonstration Guide for Judges
1. **Screen 1 (Builder)**: Switch between Synthetic Grid and the Connaught Place, New Delhi arterial road network.
2. **Screen 2 (Traffic Sim)**: Drag the time slider to 09:00 (morning rush) and trigger an accident incident. Click any edge on the canvas and hit "Close this road" to see live weight recomputation.
3. **Screen 3 (Single Solver)**: Select QPSO and hit "Run Optimization". Watch candidate routes redraw on the canvas while the real-time convergence curve drops.
4. **Screen 4 (Benchmark Arena)**: Click "START LIVE BENCHMARK RACE" to watch QPSO, Classical PSO, GA, and ACO compete in real time on the identical traffic snapshot over WebSocket streaming.
5. **Screen 5 (Dashboard)**: Click "Run 30-Trial Statistical Harness" to generate the automated empirical pitch from measured trial numbers.
