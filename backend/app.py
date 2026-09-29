"""
FastAPI Server for QuRoute (SIH26137 - Visionaries for Change).
Exposes:
1. Headless REST API endpoints (POST /network, POST /traffic, POST /solve, POST /benchmark, GET /results/{id})
2. Interactive Dashboard REST endpoints (/api/...)
3. WebSocket live streaming (/ws/solve) with Equal Evaluation Budget (eval_budget) tracking.
"""

import os
import uuid
import asyncio
import json
import traceback
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .core.graph import (
    TrafficGraph,
    generate_grid_network,
    generate_indian_city_network,
    generate_connaught_place_procedural,
)
from .core.solomon import generate_solomon_instance
from .core.traffic import DynamicTrafficEngine
from .core.formulation import MathematicalFormulation
from .core.astar import astar_shortest_path, expand_vrp_routes_to_street_paths
from .solvers import get_solver
from .solvers.base import SolverConfig, IterationEvent, SolverResult
from .benchmark.runner import BenchmarkRunner
from .benchmark.scalability import ScalabilityExperiment

app = FastAPI(
    title="QuRoute API — SIH26137 (Visionaries for Change)",
    description="Quantum-Inspired Intelligent Traffic Route Optimization Platform (Team 151198)",
    version="3.0.0",
)

# Global application state shared across screens
CURRENT_GRAPH: TrafficGraph = generate_grid_network(rows=6, cols=6, num_customers=12, seed=42)
TRAFFIC_ENGINE: DynamicTrafficEngine = DynamicTrafficEngine(CURRENT_GRAPH, initial_time_minutes=540.0)
FORMULATION: MathematicalFormulation = MathematicalFormulation(
    graph=CURRENT_GRAPH,
    vehicle_capacity=60.0,
    num_vehicles=3,
    start_time_minutes=540.0,
)
LATEST_ROUTE: Optional[List[List[int]]] = None
LATEST_BENCHMARK_RESULTS: Optional[Dict[str, Any]] = None
LATEST_SCALABILITY_RESULTS: Optional[Dict[str, Any]] = None
STORED_RESULTS: Dict[str, Dict[str, Any]] = {}


def rebuild_formulation():
    global FORMULATION
    num_cust = len(CURRENT_GRAPH.customer_ids)
    vehicles = max(3, int((num_cust / 4.5) + 0.9))
    FORMULATION = MathematicalFormulation(
        graph=CURRENT_GRAPH,
        vehicle_capacity=max(60.0, FORMULATION.vehicle_capacity),
        num_vehicles=vehicles,
        start_time_minutes=TRAFFIC_ENGINE.current_time_minutes,
    )


# --- Request Models ---
class GridGenRequest(BaseModel):
    rows: int = 6
    cols: int = 6
    num_customers: int = 12
    cell_size_km: float = 1.5
    seed: int = 42
    diagonals: bool = False


class CityGenRequest(BaseModel):
    city_name: str = "Connaught Place, New Delhi"
    num_customers: int = 12
    seed: int = 42


class SolomonGenRequest(BaseModel):
    instance_name: str = "C101"
    num_customers: int = 15
    seed: int = 42


class HeadlessNetworkRequest(BaseModel):
    mode: str = "grid"  # "grid", "cp", "osm", "solomon"
    source: Optional[str] = None
    rows: int = 6
    cols: int = 6
    city_name: str = "Connaught Place, New Delhi"
    instance_name: str = "C101"
    solomon_instance: Optional[str] = None
    num_customers: int = 12
    seed: int = 42


class HeadlessTrafficRequest(BaseModel):
    time_minutes: Optional[float] = None
    minutes_from_midnight: Optional[float] = None
    bpr_alpha: Optional[float] = None
    bpr_beta: Optional[float] = None
    live_data_blackout: Optional[bool] = None
    incident_severity: Optional[float] = None
    override_u: Optional[int] = None
    override_v: Optional[int] = None
    override_multiplier: Optional[float] = None
    override_closed: Optional[bool] = False


class TimeRequest(BaseModel):
    minutes: float


class BPRConfigRequest(BaseModel):
    alpha: float = 0.15
    beta: float = 4.0
    live_data_blackout: bool = False


class IncidentRequest(BaseModel):
    severity: float = 5.0
    half_life: float = 25.0
    edge_u: Optional[int] = None
    edge_v: Optional[int] = None
    description: str = "Major Collision / Traffic Block"


class ManualEdgeRequest(BaseModel):
    u: int
    v: int
    multiplier: float = 1.0
    is_closed: bool = False


class NodeEditRequest(BaseModel):
    id: int
    node_type: str
    name: Optional[str] = None
    demand: float = 10.0
    earliest_time: float = 480.0
    latest_time: float = 720.0


class ConfigUpdateRequest(BaseModel):
    vehicle_capacity: float = 60.0
    num_vehicles: int = 3


class BenchmarkRequest(BaseModel):
    solver_names: List[str] = ["qpso", "pso", "ga", "aco"]
    num_trials: int = 12
    max_iters: int = 80
    eval_budget: Optional[int] = 1200
    seed: int = 100
    include_exact: bool = True


class ScalabilityRequest(BaseModel):
    sizes: List[int] = [10, 25, 50, 100]
    trials_per_size: int = 2
    max_iters: int = 50
    seed: int = 42


class AStarInspectRequest(BaseModel):
    start_id: int
    goal_id: int


class QuickSolveRequest(BaseModel):
    solver_name: str = "qpso"
    max_iters: int = 80
    swarm_size: int = 30
    eval_budget: Optional[int] = None
    warm_start: bool = False
    seed: int = 42
    beta_start: float = 1.0
    beta_end: float = 0.5


class RerouteRequest(BaseModel):
    solver_name: str = "qpso"
    max_iters: int = 40
    eval_budget: int = 240
    warm_start: bool = True
    seed: int = 42


# --- Headless REST API Endpoints (Section 5 of QuRoute Plan) ---

@app.post("/network")
def headless_post_network(req: HeadlessNetworkRequest):
    global CURRENT_GRAPH, TRAFFIC_ENGINE
    mode = (req.source or req.mode).lower().strip()
    inst_name = req.solomon_instance or req.instance_name
    solomon_meta = None
    if mode == "cp":
        CURRENT_GRAPH = generate_connaught_place_procedural(num_customers=req.num_customers, seed=req.seed)
    elif mode in ("osm", "map", "city"):
        CURRENT_GRAPH = generate_indian_city_network(city_name=req.city_name, num_customers=req.num_customers, seed=req.seed)
    elif mode == "solomon":
        CURRENT_GRAPH, solomon_meta = generate_solomon_instance(
            instance_name=inst_name, num_customers=req.num_customers, seed=req.seed
        )
    else:
        CURRENT_GRAPH = generate_grid_network(
            rows=req.rows, cols=req.cols, num_customers=req.num_customers, seed=req.seed
        )
    TRAFFIC_ENGINE = DynamicTrafficEngine(CURRENT_GRAPH, initial_time_minutes=540.0)
    rebuild_formulation()
    payload = get_current_graph()
    if solomon_meta:
        payload["solomon_metadata"] = solomon_meta
    return payload


@app.post("/traffic")
def headless_post_traffic(req: HeadlessTrafficRequest):
    if req.bpr_alpha is not None or req.bpr_beta is not None:
        TRAFFIC_ENGINE.set_bpr_parameters(
            alpha=req.bpr_alpha if req.bpr_alpha is not None else TRAFFIC_ENGINE.bpr_alpha,
            beta=req.bpr_beta if req.bpr_beta is not None else TRAFFIC_ENGINE.bpr_beta,
        )
    if req.live_data_blackout is not None:
        TRAFFIC_ENGINE.set_blackout_mode(req.live_data_blackout)
    t_val = req.time_minutes if req.time_minutes is not None else req.minutes_from_midnight
    if t_val is not None:
        TRAFFIC_ENGINE.set_time(t_val)
        FORMULATION.start_time_minutes = t_val
    if req.incident_severity is not None:
        TRAFFIC_ENGINE.trigger_random_incident(severity=req.incident_severity)
    if req.override_u is not None and req.override_v is not None:
        TRAFFIC_ENGINE.set_manual_edge_congestion(
            req.override_u,
            req.override_v,
            req.override_multiplier or 2.0,
            is_closed=bool(req.override_closed),
        )
    return TRAFFIC_ENGINE.get_status_summary()


@app.post("/solve")
def headless_post_solve(req: QuickSolveRequest):
    res = quick_solve(req)
    res_id = f"solve_{uuid.uuid4().hex[:8]}"
    res["result_id"] = res_id
    res["run_id"] = res_id
    res["type"] = "solve"
    STORED_RESULTS[res_id] = res
    return res


@app.post("/benchmark")
def headless_post_benchmark(req: BenchmarkRequest):
    res = run_benchmark(req)
    res_id = f"bench_{uuid.uuid4().hex[:8]}"
    res["result_id"] = res_id
    res["run_id"] = res_id
    res["type"] = "benchmark"
    STORED_RESULTS[res_id] = res
    return res


@app.get("/results/{result_id}")
def headless_get_result(result_id: str):
    if result_id in STORED_RESULTS:
        return STORED_RESULTS[result_id]
    if result_id == "latest" and LATEST_BENCHMARK_RESULTS:
        return LATEST_BENCHMARK_RESULTS
    raise HTTPException(status_code=404, detail=f"Result ID '{result_id}' not found.")


# --- Interactive UI REST Endpoints ---

@app.get("/api/graph/current")
def get_current_graph():
    return {
        "brand": {
            "product": "QuRoute",
            "ps_id": "SIH26137",
            "team_id": "151198",
            "team_name": "Visionaries for Change",
        },
        "graph": CURRENT_GRAPH.to_dict(),
        "traffic": TRAFFIC_ENGINE.get_status_summary(),
        "formulation": FORMULATION.get_formulation_metadata(),
    }


@app.post("/api/graph/generate/grid")
def generate_grid(req: GridGenRequest):
    global CURRENT_GRAPH, TRAFFIC_ENGINE, LATEST_ROUTE
    CURRENT_GRAPH = generate_grid_network(
        rows=req.rows,
        cols=req.cols,
        cell_size_km=req.cell_size_km,
        num_customers=req.num_customers,
        seed=req.seed,
        diagonals=req.diagonals,
    )
    TRAFFIC_ENGINE = DynamicTrafficEngine(CURRENT_GRAPH, initial_time_minutes=540.0)
    LATEST_ROUTE = None
    rebuild_formulation()
    return get_current_graph()


@app.post("/api/graph/generate/city")
def generate_city(req: CityGenRequest):
    global CURRENT_GRAPH, TRAFFIC_ENGINE, LATEST_ROUTE
    if "connaught" in req.city_name.lower():
        CURRENT_GRAPH = generate_connaught_place_procedural(num_customers=req.num_customers, seed=req.seed)
    else:
        CURRENT_GRAPH = generate_indian_city_network(
            city_name=req.city_name,
            num_customers=req.num_customers,
            seed=req.seed,
        )
    TRAFFIC_ENGINE = DynamicTrafficEngine(CURRENT_GRAPH, initial_time_minutes=540.0)
    LATEST_ROUTE = None
    rebuild_formulation()
    return get_current_graph()


@app.post("/api/graph/generate/solomon")
def generate_solomon(req: SolomonGenRequest):
    global CURRENT_GRAPH, TRAFFIC_ENGINE, LATEST_ROUTE
    CURRENT_GRAPH, meta = generate_solomon_instance(
        instance_name=req.instance_name,
        num_customers=req.num_customers,
        seed=req.seed,
    )
    TRAFFIC_ENGINE = DynamicTrafficEngine(CURRENT_GRAPH, initial_time_minutes=540.0)
    LATEST_ROUTE = None
    rebuild_formulation()
    resp = get_current_graph()
    resp["solomon_metadata"] = meta
    return resp


@app.post("/api/traffic/time")
def set_sim_time(req: TimeRequest):
    TRAFFIC_ENGINE.set_time(req.minutes)
    FORMULATION.start_time_minutes = req.minutes
    return TRAFFIC_ENGINE.get_status_summary()


@app.post("/api/traffic/bpr")
def set_bpr_and_blackout(req: BPRConfigRequest):
    TRAFFIC_ENGINE.set_bpr_parameters(alpha=req.alpha, beta=req.beta)
    TRAFFIC_ENGINE.set_blackout_mode(req.live_data_blackout)
    return {
        "traffic": TRAFFIC_ENGINE.get_status_summary(),
        "graph": CURRENT_GRAPH.to_dict(),
    }


@app.post("/api/traffic/incident")
def trigger_incident(req: IncidentRequest):
    edge_key = None
    if req.edge_u is not None and req.edge_v is not None:
        edge_key = (req.edge_u, req.edge_v)
    inc = TRAFFIC_ENGINE.trigger_random_incident(
        edge_key=edge_key,
        severity=req.severity,
        half_life=req.half_life,
        description=req.description,
    )
    return {
        "incident": {
            "id": inc.id,
            "edge": list(inc.edge),
            "severity": inc.severity,
            "description": inc.description,
        },
        "traffic": TRAFFIC_ENGINE.get_status_summary(),
    }


@app.post("/api/traffic/edge")
def set_manual_edge(req: ManualEdgeRequest):
    TRAFFIC_ENGINE.set_manual_edge_congestion(req.u, req.v, req.multiplier, is_closed=req.is_closed)
    return {
        "u": req.u,
        "v": req.v,
        "multiplier": req.multiplier,
        "is_closed": req.is_closed,
        "traffic": TRAFFIC_ENGINE.get_status_summary(),
    }


@app.post("/api/graph/node/edit")
def edit_node(req: NodeEditRequest):
    if req.id not in CURRENT_GRAPH.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    node = CURRENT_GRAPH.nodes[req.id]
    node.node_type = req.node_type
    if req.name:
        node.name = req.name
    node.demand = req.demand
    node.earliest_time = req.earliest_time
    node.latest_time = req.latest_time

    CURRENT_GRAPH.depot_ids = [nid for nid, n in CURRENT_GRAPH.nodes.items() if n.node_type == "depot"]
    CURRENT_GRAPH.customer_ids = [nid for nid, n in CURRENT_GRAPH.nodes.items() if n.node_type == "customer"]
    rebuild_formulation()
    return get_current_graph()


@app.post("/api/formulation/config")
def update_config(req: ConfigUpdateRequest):
    FORMULATION.vehicle_capacity = req.vehicle_capacity
    FORMULATION.num_vehicles = req.num_vehicles
    rebuild_formulation()
    return FORMULATION.get_formulation_metadata()


@app.get("/api/formulation")
def get_formulation():
    return FORMULATION.get_formulation_metadata()


@app.post("/api/benchmark/statistical")
def run_benchmark(req: BenchmarkRequest):
    global LATEST_BENCHMARK_RESULTS
    runner = BenchmarkRunner(FORMULATION)
    results = runner.run_statistical_benchmark(
        solver_names=req.solver_names,
        num_trials=req.num_trials,
        max_iters=req.max_iters,
        eval_budget=req.eval_budget,
        base_seed=req.seed,
        include_exact=req.include_exact,
    )
    LATEST_BENCHMARK_RESULTS = results
    return results


@app.get("/api/benchmark/latest")
def get_latest_benchmark():
    return LATEST_BENCHMARK_RESULTS or {"status": "none"}


@app.post("/api/benchmark/scalability")
def run_scalability(req: ScalabilityRequest):
    global LATEST_SCALABILITY_RESULTS
    exp = ScalabilityExperiment(node_sizes=req.sizes)
    results = exp.run_sweep(
        solver_names=["qpso", "pso", "ga", "aco"],
        trials_per_size=req.trials_per_size,
        max_iters=req.max_iters,
        seed=req.seed,
    )
    LATEST_SCALABILITY_RESULTS = results
    return results


@app.get("/api/benchmark/scalability/latest")
def get_latest_scalability():
    return LATEST_SCALABILITY_RESULTS or {"status": "none"}


@app.post("/api/route/astar_inspect")
def inspect_astar(req: AStarInspectRequest):
    return astar_shortest_path(CURRENT_GRAPH, req.start_id, req.goal_id)


@app.post("/api/solve/quick")
def quick_solve(req: QuickSolveRequest):
    global LATEST_ROUTE
    solver = get_solver(req.solver_name)
    cfg = SolverConfig(
        max_iters=req.max_iters,
        swarm_size=req.swarm_size,
        eval_budget=req.eval_budget,
        warm_start_routes=LATEST_ROUTE if req.warm_start else None,
        seed=req.seed,
        beta_start=req.beta_start,
        beta_end=req.beta_end,
    )
    result = solver.run(FORMULATION, cfg)
    LATEST_ROUTE = result.best_route
    street_routes = expand_vrp_routes_to_street_paths(CURRENT_GRAPH, result.best_route)
    result.detailed_street_routes = street_routes
    res_dict = result.to_dict()
    res_dict["detailed_street_routes"] = street_routes
    return res_dict


@app.post("/api/traffic/reroute")
def live_reroute(req: RerouteRequest):
    """
    Executes Warm-Started QPSO dynamic re-routing from the previous active route,
    demonstrating sub-second (<50 ms) recovery around live traffic roadblocks.
    """
    global LATEST_ROUTE
    solver = get_solver(req.solver_name)
    cfg = SolverConfig(
        max_iters=req.max_iters,
        swarm_size=16,
        eval_budget=req.eval_budget,
        warm_start_routes=LATEST_ROUTE if req.warm_start else None,
        seed=req.seed,
    )
    result = solver.run(FORMULATION, cfg)
    LATEST_ROUTE = result.best_route
    street_routes = expand_vrp_routes_to_street_paths(CURRENT_GRAPH, result.best_route)
    result.detailed_street_routes = street_routes
    res_dict = result.to_dict()
    res_dict["detailed_street_routes"] = street_routes
    res_dict["warm_start_ms"] = round(result.wall_clock_time_seconds * 1000.0, 2)
    return res_dict


# --- WebSocket Streaming Endpoint for Live Route Animations & Multi-Solver Race ---

@app.websocket("/ws/solve")
async def websocket_solver_stream(websocket: WebSocket):
    global LATEST_ROUTE
    await websocket.accept()
    try:
        data_text = await websocket.receive_text()
        req = json.loads(data_text)
        mode = req.get("mode", "single")
        max_iters = int(req.get("max_iters", 80))
        swarm_size = int(req.get("swarm_size", 24))
        eval_budget = req.get("eval_budget", max_iters * swarm_size)
        seed = int(req.get("seed", 42))

        loop = asyncio.get_running_loop()

        if mode == "single":
            solver_name = req.get("solver_name", "qpso")
            beta_start = float(req.get("beta_start", 1.0))
            beta_end = float(req.get("beta_end", 0.5))

            solver = get_solver(solver_name)
            cfg = SolverConfig(
                max_iters=max_iters,
                swarm_size=swarm_size,
                eval_budget=int(eval_budget) if eval_budget else None,
                seed=seed,
                beta_start=beta_start,
                beta_end=beta_end,
            )

            def streaming_callback(event: IterationEvent):
                payload = {
                    "type": "iteration_update",
                    "solver_name": solver.name,
                    "iteration": event.iteration,
                    "evaluations": event.evaluations,
                    "cost": round(event.current_best_cost, 2),
                    "route": event.best_route,
                    "timestamp_ms": round(event.timestamp_ms, 2),
                }
                asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)

            result: SolverResult = await asyncio.to_thread(
                solver.run, FORMULATION, cfg, streaming_callback
            )
            LATEST_ROUTE = result.best_route
            street_routes = expand_vrp_routes_to_street_paths(CURRENT_GRAPH, result.best_route)
            result.detailed_street_routes = street_routes
            res_dict = result.to_dict()
            res_dict["detailed_street_routes"] = street_routes

            await websocket.send_json({
                "type": "finished",
                "result": res_dict,
            })

        elif mode == "race":
            solvers_to_run = req.get("solvers", ["qpso", "pso", "ga", "aco"])

            async def run_single_race_solver(s_name: str):
                s_obj = get_solver(s_name)
                s_cfg = SolverConfig(
                    max_iters=max_iters,
                    swarm_size=swarm_size,
                    eval_budget=int(eval_budget) if eval_budget else (max_iters * swarm_size),
                    seed=seed,
                )

                def race_callback(event: IterationEvent):
                    payload = {
                        "type": "race_step",
                        "solver_name": s_obj.name,
                        "solver_key": s_name,
                        "iteration": event.iteration,
                        "evaluations": event.evaluations,
                        "cost": round(event.current_best_cost, 2),
                        "route": event.best_route,
                    }
                    asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)

                res = await asyncio.to_thread(s_obj.run, FORMULATION, s_cfg, race_callback)
                if s_name == "qpso":
                    global LATEST_ROUTE
                    LATEST_ROUTE = res.best_route
                street_routes = expand_vrp_routes_to_street_paths(CURRENT_GRAPH, res.best_route)
                res.detailed_street_routes = street_routes
                res_dict = res.to_dict()
                res_dict["detailed_street_routes"] = street_routes
                await websocket.send_json({
                    "type": "race_solver_finished",
                    "solver_name": s_obj.name,
                    "solver_key": s_name,
                    "result": res_dict,
                })
                return res

            tasks = [run_single_race_solver(name) for name in solvers_to_run]
            await asyncio.gather(*tasks)

            await websocket.send_json({
                "type": "race_finished",
                "message": "All solvers completed equal-budget race.",
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e), "trace": traceback.format_exc()})
        except Exception:
            pass


frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>QuRoute — SIH26137</h1>")
