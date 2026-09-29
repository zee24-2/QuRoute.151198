"""
Graph simulation layer for QuRoute (SIH26137 - Visionaries for Change).
Supports:
1. Synthetic N x M Grid Lattice with configurable node types, demands, and time windows.
2. Connaught Place procedural concentric rings + radial spokes (offline fallback).
3. Real-world Indian City District Networks via OSMnx with persistent disk caching.
4. Bureau of Public Roads (BPR, 1964) congestion function:
   t(e) = t_0 * (1 + alpha * (v / c)^beta)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import math
import random
import networkx as nx
import numpy as np


@dataclass
class Node:
    id: int
    name: str
    node_type: str  # "depot", "customer", "intersection"
    x: float
    y: float
    demand: float = 0.0  # Units to deliver (customers only)
    earliest_time: float = 0.0  # Earliest service time window (simulated minutes)
    latest_time: float = 1440.0  # Latest service time window
    service_time: float = 10.0  # Unloading/service duration (minutes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "demand": self.demand,
            "time_window": [round(self.earliest_time, 1), round(self.latest_time, 1)],
            "service_time": self.service_time,
        }


@dataclass
class Edge:
    u: int
    v: int
    base_distance: float  # length in kilometers or grid units
    base_travel_time: float  # free-flow time t_0 in simulated minutes
    speed_limit: float = 40.0  # km/h
    capacity: float = 100.0  # c: edge road capacity (vehicles/hour)
    flow: float = 50.0  # v: current traffic volume/flow
    bpr_alpha: float = 0.15  # Bureau of Public Roads (1964) default alpha = 0.15
    bpr_beta: float = 4.0  # Bureau of Public Roads (1964) default beta = 4.0
    congestion_multiplier: float = 1.0  # Combined BPR / override multiplier
    is_closed: bool = False  # If closed, cannot be traversed

    def compute_bpr_multiplier(self, extra_override: float = 1.0) -> float:
        """
        Bureau of Public Roads (1964) Congestion Function:
        t(e) = t_0 * (1 + alpha * (v / c)^beta)
        Returns the multiplier (1 + alpha * (v / c)^beta) * extra_override.
        """
        vc_ratio = max(0.0, self.flow / max(1.0, self.capacity))
        bpr_factor = 1.0 + self.bpr_alpha * math.pow(vc_ratio, self.bpr_beta)
        self.congestion_multiplier = bpr_factor * max(0.1, extra_override)
        return self.congestion_multiplier

    @property
    def live_weight(self) -> float:
        if self.is_closed:
            return 1e9
        return self.base_travel_time * max(0.1, self.congestion_multiplier)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "u": self.u,
            "v": self.v,
            "base_distance": round(self.base_distance, 3),
            "base_travel_time": round(self.base_travel_time, 2),
            "free_flow_time": round(self.base_travel_time, 2),
            "capacity": round(self.capacity, 1),
            "flow": round(self.flow, 1),
            "vc_ratio": round(self.flow / max(1.0, self.capacity), 2),
            "bpr_alpha": self.bpr_alpha,
            "bpr_beta": self.bpr_beta,
            "congestion_multiplier": round(self.congestion_multiplier, 3),
            "live_weight": round(self.live_weight, 2),
            "is_closed": self.is_closed,
        }


class TrafficGraph:
    """
    Unified representation for Grid lattice networks, Connaught Place radial networks,
    OSMnx road subgraphs, and Solomon VRPTW benchmark instances.
    Maintains shortest-path Dijkstra cost matrices with 15-minute time-bucket caching.
    """
    def __init__(self, name: str = "TrafficNetwork", mode: str = "grid"):
        self.name = name
        self.mode = mode  # "grid", "map", "cp", "solomon"
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.nx_graph: nx.DiGraph = nx.DiGraph()
        self.depot_ids: List[int] = []
        self.customer_ids: List[int] = []
        self._bucket_matrix_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
        self._cache_dirty: bool = True
        self.current_time_bucket: int = 36  # 15-min bucket index (e.g. 540 // 15 = 36 for 09:00)

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.nx_graph.add_node(
            node.id,
            name=node.name,
            type=node.node_type,
            x=node.x,
            y=node.y,
            demand=node.demand,
            earliest=node.earliest_time,
            latest=node.latest_time,
            service_time=node.service_time,
        )
        if node.node_type == "depot":
            if node.id not in self.depot_ids:
                self.depot_ids.append(node.id)
        elif node.node_type == "customer":
            if node.id not in self.customer_ids:
                self.customer_ids.append(node.id)
        self.mark_dirty()

    def add_edge(
        self,
        u: int,
        v: int,
        base_distance: float,
        base_travel_time: Optional[float] = None,
        speed_limit: float = 40.0,
        capacity: float = 100.0,
        flow: float = 50.0,
        bidirectional: bool = True,
    ):
        if base_travel_time is None:
            base_travel_time = (base_distance / max(10.0, speed_limit)) * 60.0

        edge_uv = Edge(
            u=u,
            v=v,
            base_distance=base_distance,
            base_travel_time=base_travel_time,
            speed_limit=speed_limit,
            capacity=capacity,
            flow=flow,
        )
        edge_uv.compute_bpr_multiplier()
        self.edges[(u, v)] = edge_uv
        self.nx_graph.add_edge(u, v, weight=edge_uv.live_weight, base_dist=base_distance, base_time=base_travel_time)

        if bidirectional:
            edge_vu = Edge(
                u=v,
                v=u,
                base_distance=base_distance,
                base_travel_time=base_travel_time,
                speed_limit=speed_limit,
                capacity=capacity,
                flow=flow,
            )
            edge_vu.compute_bpr_multiplier()
            self.edges[(v, u)] = edge_vu
            self.nx_graph.add_edge(v, u, weight=edge_vu.live_weight, base_dist=base_distance, base_time=base_travel_time)

        self.mark_dirty()

    def update_edge_multiplier(
        self,
        u: int,
        v: int,
        multiplier: float,
        is_closed: bool = False,
        flow: Optional[float] = None,
        bpr_alpha: Optional[float] = None,
        bpr_beta: Optional[float] = None,
    ):
        if (u, v) in self.edges:
            edge = self.edges[(u, v)]
            if flow is not None:
                edge.flow = flow
            if bpr_alpha is not None:
                edge.bpr_alpha = bpr_alpha
            if bpr_beta is not None:
                edge.bpr_beta = bpr_beta
            edge.congestion_multiplier = multiplier
            edge.is_closed = is_closed
            self.nx_graph[u][v]["weight"] = edge.live_weight
            self.mark_dirty()

    def mark_dirty(self):
        self._cache_dirty = True
        self._bucket_matrix_cache.clear()

    def get_travel_time_matrix(self, target_node_ids: Optional[List[int]] = None, time_minutes: Optional[float] = None) -> np.ndarray:
        """
        Computes the pairwise live travel time matrix between depot and customer nodes
        using NetworkX Dijkstra shortest paths on the BPR-weighted graph.
        Uses 15-minute time-bucket snapshot caching so solvers read a compact matrix without touching the graph on every evaluation.
        """
        if target_node_ids is None:
            target_node_ids = sorted(self.nodes.keys())

        bucket = int(time_minutes // 15) if time_minutes is not None else self.current_time_bucket
        cache_key = (bucket, tuple(target_node_ids))

        if not self._cache_dirty and cache_key in self._bucket_matrix_cache:
            return self._bucket_matrix_cache[cache_key]

        for (u, v), edge in self.edges.items():
            if self.nx_graph.has_edge(u, v):
                self.nx_graph[u][v]["weight"] = edge.live_weight

        n = len(target_node_ids)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i, src in enumerate(target_node_ids):
            lengths = nx.single_source_dijkstra_path_length(self.nx_graph, src, weight="weight")
            for j, dst in enumerate(target_node_ids):
                if i == j:
                    matrix[i, j] = 0.0
                elif dst in lengths:
                    matrix[i, j] = lengths[dst]
                else:
                    matrix[i, j] = 1e6

        self._bucket_matrix_cache[cache_key] = matrix
        self._cache_dirty = False
        return matrix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "depot_ids": self.depot_ids,
            "customer_ids": self.customer_ids,
        }


def generate_grid_network(
    rows: int = 6,
    cols: int = 6,
    cell_size_km: float = 1.5,
    num_depots: int = 1,
    num_customers: int = 15,
    seed: int = 42,
    diagonals: bool = False,
) -> TrafficGraph:
    """
    Generates an N x M lattice graph with BPR road capacities and customer time windows.
    """
    rng = random.Random(seed)
    graph = TrafficGraph(name=f"LatticeGrid_{rows}x{cols}", mode="grid")

    node_id = 0
    grid_coords: Dict[Tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size_km
            y = (rows - 1 - r) * cell_size_km
            graph.add_node(Node(
                id=node_id,
                name=f"Int_{r}_{c}",
                node_type="intersection",
                x=x,
                y=y,
            ))
            grid_coords[(r, c)] = node_id
            node_id += 1

    for r in range(rows):
        for c in range(cols):
            u = grid_coords[(r, c)]
            if c + 1 < cols:
                v = grid_coords[(r, c + 1)]
                graph.add_edge(u, v, base_distance=cell_size_km, speed_limit=40.0, capacity=120.0, flow=65.0)
            if r + 1 < rows:
                v = grid_coords[(r + 1, c)]
                graph.add_edge(u, v, base_distance=cell_size_km, speed_limit=40.0, capacity=120.0, flow=65.0)
            if diagonals:
                if r + 1 < rows and c + 1 < cols:
                    v = grid_coords[(r + 1, c + 1)]
                    graph.add_edge(u, v, base_distance=cell_size_km * 1.414, speed_limit=35.0, capacity=90.0, flow=50.0)
                if r + 1 < rows and c - 1 >= 0:
                    v = grid_coords[(r + 1, c - 1)]
                    graph.add_edge(u, v, base_distance=cell_size_km * 1.414, speed_limit=35.0, capacity=90.0, flow=50.0)

    all_node_ids = list(graph.nodes.keys())
    center_r, center_c = rows // 2, cols // 2
    depot_node_id = grid_coords[(center_r, center_c)]
    depot_node = graph.nodes[depot_node_id]
    depot_node.node_type = "depot"
    depot_node.name = "Central_Depot"
    graph.depot_ids = [depot_node_id]

    candidate_ids = [nid for nid in all_node_ids if nid != depot_node_id]
    num_customers = min(num_customers, len(candidate_ids))
    chosen_cust_ids = rng.sample(candidate_ids, num_customers)

    for idx, cid in enumerate(chosen_cust_ids):
        cust = graph.nodes[cid]
        cust.node_type = "customer"
        cust.name = f"Cust_{idx + 1}"
        cust.demand = float(rng.randint(5, 25))
        cust.service_time = float(rng.choice([8.0, 10.0, 15.0]))
        earliest = float(rng.randint(480, 840))
        window_duration = float(rng.randint(90, 240))
        cust.earliest_time = earliest
        cust.latest_time = earliest + window_duration
        graph.customer_ids.append(cid)

    return graph


def generate_connaught_place_procedural(
    num_customers: int = 15,
    seed: int = 42,
) -> TrafficGraph:
    """
    Generates the Connaught Place, New Delhi concentric rings + radial spokes network.
    Acts as the deterministic offline fallback promised on Slide 3 of the QuRoute deck.
    """
    rng = random.Random(seed)
    graph = TrafficGraph(name="Connaught_Place_New_Delhi", mode="cp")
    rad_circles = [
        ("Inner_Circle", 0.35, 12, 100.0),
        ("Middle_Circle", 0.65, 16, 120.0),
        ("Outer_Circle", 1.05, 20, 160.0),
    ]
    node_id = 0
    circle_node_ids: List[List[int]] = []

    for c_name, radius_km, num_pts, cap in rad_circles:
        curr_c_ids = []
        for pt_idx in range(num_pts):
            angle = (2 * math.pi * pt_idx) / num_pts
            x = radius_km * math.cos(angle)
            y = radius_km * math.sin(angle)
            graph.add_node(Node(
                id=node_id,
                name=f"{c_name}_{pt_idx}",
                node_type="intersection",
                x=round(x, 4),
                y=round(y, 4),
            ))
            curr_c_ids.append(node_id)
            node_id += 1
        circle_node_ids.append(curr_c_ids)

        for k in range(num_pts):
            u = curr_c_ids[k]
            v = curr_c_ids[(k + 1) % num_pts]
            chord_dist = 2 * radius_km * math.sin(math.pi / num_pts)
            graph.add_edge(u, v, base_distance=chord_dist, speed_limit=30.0, capacity=cap, flow=cap * 0.65, bidirectional=True)

    # Connect 8 radial spokes (Janpath, Barakhamba Rd, Sansad Marg, Panchkuian Rd, etc.)
    for spoke_idx in range(8):
        angle = (2 * math.pi * spoke_idx) / 8

        def find_closest(c_ids, ang):
            best_id = c_ids[0]
            best_diff = 100.0
            for cid in c_ids:
                node = graph.nodes[cid]
                node_ang = math.atan2(node.y, node.x)
                diff = abs((node_ang - ang + math.pi) % (2 * math.pi) - math.pi)
                if diff < best_diff:
                    best_diff = diff
                    best_id = cid
            return best_id

        n_inner = find_closest(circle_node_ids[0], angle)
        n_mid = find_closest(circle_node_ids[1], angle)
        n_outer = find_closest(circle_node_ids[2], angle)

        graph.add_edge(n_inner, n_mid, base_distance=0.3, speed_limit=45.0, capacity=140.0, flow=85.0, bidirectional=True)
        graph.add_edge(n_mid, n_outer, base_distance=0.4, speed_limit=45.0, capacity=140.0, flow=85.0, bidirectional=True)

    all_nids = list(graph.nodes.keys())
    depot_nid = all_nids[0]
    graph.nodes[depot_nid].node_type = "depot"
    graph.nodes[depot_nid].name = "Central_Depot_CP"
    graph.depot_ids = [depot_nid]

    cand_nids = [nid for nid in all_nids if nid != depot_nid]
    num_cust = min(num_customers, len(cand_nids))
    cust_ids = rng.sample(cand_nids, num_cust)

    for i, cid in enumerate(cust_ids):
        c = graph.nodes[cid]
        c.node_type = "customer"
        c.name = f"Delivery_Point_{i+1}"
        c.demand = float(rng.randint(6, 24))
        c.service_time = float(rng.choice([8.0, 12.0, 15.0]))
        earliest = float(rng.randint(480, 840))
        duration = float(rng.randint(120, 240))
        c.earliest_time = earliest
        c.latest_time = earliest + duration
        graph.customer_ids.append(cid)

    return graph


def generate_indian_city_network(
    city_name: str = "Connaught Place, New Delhi",
    num_customers: int = 15,
    seed: int = 42,
) -> TrafficGraph:
    """
    Creates an authentic Indian urban road network via OSMnx with persistent disk caching
    in backend/data/indian_cities/, falling back to Connaught Place concentric rings if offline.
    """
    rng = random.Random(seed)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indian_cities")
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = city_name.lower().replace(",", "").replace(" ", "_")
    cache_file = os.path.join(cache_dir, f"{safe_name}.json")

    graph = TrafficGraph(name=f"OSM_{city_name}", mode="map")
    loaded = False

    # 1. Try reading from local disk cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            for n in cached_data["nodes"]:
                graph.add_node(Node(
                    id=n["id"],
                    name=n["name"],
                    node_type="intersection",
                    x=n["x"],
                    y=n["y"],
                ))
            for e in cached_data["edges"]:
                graph.add_edge(
                    u=e["u"],
                    v=e["v"],
                    base_distance=e["base_distance"],
                    speed_limit=35.0,
                    capacity=120.0,
                    flow=70.0,
                    bidirectional=False,
                )
            loaded = len(graph.nodes) >= 10
        except Exception:
            loaded = False

    # 2. If not cached, try live OSMnx fetch and save to disk cache
    if not loaded:
        try:
            import osmnx as ox
            G_osm = ox.graph_from_place(f"{city_name}, India", network_type="drive", simplify=True)
            if len(G_osm.nodes) >= 10:
                largest_scc = max(nx.strongly_connected_components(G_osm), key=len)
                G_sub = G_osm.subgraph(largest_scc).copy()
                if len(G_sub.nodes) > 120:
                    center_node = list(G_sub.nodes)[0]
                    sub_nodes = list(nx.single_source_shortest_path_length(G_sub, center_node, cutoff=12).keys())
                    if len(sub_nodes) >= 25:
                        G_sub = G_sub.subgraph(sub_nodes).copy()

                osm_to_id = {nid: i for i, nid in enumerate(G_sub.nodes)}
                cache_nodes = []
                cache_edges = []
                for osm_node, data in G_sub.nodes(data=True):
                    nid = osm_to_id[osm_node]
                    x = float(data.get("x", 0.0))
                    y = float(data.get("y", 0.0))
                    graph.add_node(Node(id=nid, name=f"OSM_{nid}", node_type="intersection", x=x, y=y))
                    cache_nodes.append({"id": nid, "name": f"OSM_{nid}", "x": x, "y": y})

                for u, v, data in G_sub.edges(data=True):
                    if u in osm_to_id and v in osm_to_id:
                        uid, vid = osm_to_id[u], osm_to_id[v]
                        dist_km = float(data.get("length", 200.0)) / 1000.0
                        graph.add_edge(uid, vid, base_distance=dist_km, speed_limit=35.0, capacity=120.0, flow=70.0, bidirectional=False)
                        cache_edges.append({"u": uid, "v": vid, "base_distance": dist_km})

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({"nodes": cache_nodes, "edges": cache_edges}, f)
                loaded = True
        except Exception:
            loaded = False

    # 3. Fallback to Connaught Place procedural network (and cache it so offline runs always work)
    if not loaded:
        cp_graph = generate_connaught_place_procedural(num_customers=num_customers, seed=seed)
        cp_graph.mode = "map"
        return cp_graph

    # Assign Depot and Customers on loaded OSM network
    all_nids = list(graph.nodes.keys())
    depot_nid = all_nids[0]
    graph.nodes[depot_nid].node_type = "depot"
    graph.nodes[depot_nid].name = "Central_Depot_OSM"
    graph.depot_ids = [depot_nid]

    cand_nids = [nid for nid in all_nids if nid != depot_nid]
    num_cust = min(num_customers, len(cand_nids))
    cust_ids = rng.sample(cand_nids, num_cust)

    for i, cid in enumerate(cust_ids):
        c = graph.nodes[cid]
        c.node_type = "customer"
        c.name = f"Delivery_Point_{i+1}"
        c.demand = float(rng.randint(6, 24))
        c.service_time = float(rng.choice([8.0, 12.0, 15.0]))
        earliest = float(rng.randint(480, 840))
        duration = float(rng.randint(120, 240))
        c.earliest_time = earliest
        c.latest_time = earliest + duration
        graph.customer_ids.append(cid)

    return graph
