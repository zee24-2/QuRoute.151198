"""
A* Shortest Path Algorithm & Route Expansion Engine.
Implements:
1. Point-to-Point A* Pathfinding (as used by Google Maps) on live dynamic traffic graphs.
   Evaluation function: f(n) = g(n) + h(n)
   - g(n): live travel time accumulated from start to n along dynamic edges
   - h(n): Euclidean distance from n to goal / maximum free-flow speed (admissible heuristic)
2. Route Expansion: converts high-level customer sequences [depot, c1, c2, depot]
   into exact contiguous street-level node and coordinate paths [depot, int_1, int_2, c1, ..., depot].
"""

import math
import heapq
from typing import Dict, List, Tuple, Optional, Any
from .graph import TrafficGraph, Node


def euclidean_distance(n1: Node, n2: Node) -> float:
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


def astar_shortest_path(
    graph: TrafficGraph,
    start_id: int,
    goal_id: int,
    max_speed_kmh: float = 45.0,
) -> Dict[str, Any]:
    """
    Computes the optimal point-to-point path between start_id and goal_id using A*.
    Returns:
      - path: list of node IDs [start_id, ..., goal_id]
      - coordinates: list of [x, y] coordinates
      - total_travel_time: sum of live edge travel times in minutes
      - total_distance: sum of physical edge distances in km
      - nodes_explored: number of nodes popped from priority queue
      - edges_traversed: list of [u, v] pairs
    """
    if start_id not in graph.nodes or goal_id not in graph.nodes:
        return {
            "path": [start_id, goal_id],
            "coordinates": [],
            "total_travel_time": 1e6,
            "total_distance": 0.0,
            "nodes_explored": 0,
            "edges_traversed": [],
        }

    if start_id == goal_id:
        node = graph.nodes[start_id]
        return {
            "path": [start_id],
            "coordinates": [[node.x, node.y]],
            "total_travel_time": 0.0,
            "total_distance": 0.0,
            "nodes_explored": 0,
            "edges_traversed": [],
        }

    goal_node = graph.nodes[goal_id]

    def heuristic(node_id: int) -> float:
        curr = graph.nodes[node_id]
        dist_km = euclidean_distance(curr, goal_node)
        # Admissible heuristic: optimistic travel time in minutes at maximum free-flow speed
        return (dist_km / max(10.0, max_speed_kmh)) * 60.0

    # Priority queue stores tuples: (f_score, counter, node_id)
    pq = []
    counter = 0
    g_score: Dict[int, float] = {start_id: 0.0}
    f_score: Dict[int, float] = {start_id: heuristic(start_id)}
    came_from: Dict[int, int] = {}
    explored_nodes = 0

    heapq.heappush(pq, (f_score[start_id], counter, start_id))
    visited = set()

    while pq:
        _, _, current = heapq.heappop(pq)
        explored_nodes += 1

        if current == goal_id:
            # Reconstruct path
            path = []
            curr_step = goal_id
            while curr_step in came_from:
                path.append(curr_step)
                curr_step = came_from[curr_step]
            path.append(start_id)
            path.reverse()

            # Compute physical metrics
            total_dist = 0.0
            edges_traversed = []
            coords = [[graph.nodes[nid].x, graph.nodes[nid].y] for nid in path]

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edges_traversed.append([u, v])
                if (u, v) in graph.edges:
                    total_dist += graph.edges[(u, v)].base_distance
                elif (v, u) in graph.edges:
                    total_dist += graph.edges[(v, u)].base_distance

            return {
                "path": path,
                "coordinates": coords,
                "total_travel_time": round(g_score[goal_id], 2),
                "total_distance": round(total_dist, 3),
                "nodes_explored": explored_nodes,
                "edges_traversed": edges_traversed,
            }

        if current in visited:
            continue
        visited.add(current)

        # Explore outgoing neighbors from nx_graph
        if current in graph.nx_graph:
            for neighbor in graph.nx_graph.neighbors(current):
                edge_data = graph.nx_graph.get_edge_data(current, neighbor)
                live_w = edge_data.get("weight", 1e6) if edge_data else 1e6
                tentative_g = g_score[current] + live_w

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(pq, (f, counter, neighbor))

    # Path not found (disconnected or closed roads)
    start_node = graph.nodes[start_id]
    return {
        "path": [start_id, goal_id],
        "coordinates": [[start_node.x, start_node.y], [goal_node.x, goal_node.y]],
        "total_travel_time": 1e6,
        "total_distance": euclidean_distance(start_node, goal_node),
        "nodes_explored": explored_nodes,
        "edges_traversed": [],
        "is_disconnected": True,
    }


def expand_vrp_routes_to_street_paths(
    graph: TrafficGraph,
    vehicle_routes: List[List[int]],
) -> List[Dict[str, Any]]:
    """
    Expands high-level vehicle routes [[depot, c1, c2, depot], ...]
    into complete street-level trajectories through real intersection nodes.
    Returns for each vehicle:
      - vehicle_id
      - high_level_stops: original customer IDs
      - street_node_path: full contiguous node IDs along streets
      - street_coordinates: list of [x, y] coordinates along the streets
      - legs: step-by-step navigation legs with travel times and distances
      - total_travel_time
      - total_distance
    """
    detailed_routes = []

    for v_idx, route in enumerate(vehicle_routes):
        if not route or len(route) <= 2:
            detailed_routes.append({
                "vehicle_id": v_idx,
                "high_level_stops": route,
                "street_node_path": route,
                "street_coordinates": [[graph.nodes[n].x, graph.nodes[n].y] for n in route if n in graph.nodes],
                "legs": [],
                "total_travel_time": 0.0,
                "total_distance": 0.0,
                "is_empty": True,
            })
            continue

        full_node_path = []
        full_coords = []
        legs = []
        total_time = 0.0
        total_dist = 0.0

        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]

            # Use A* to find street path for this leg
            astar_res = astar_shortest_path(graph, u, v)
            sub_path = astar_res["path"]
            sub_coords = astar_res["coordinates"]
            leg_time = astar_res["total_travel_time"]
            leg_dist = astar_res["total_distance"]

            total_time += leg_time
            total_dist += leg_dist

            legs.append({
                "from_node": u,
                "to_node": v,
                "street_path": sub_path,
                "travel_time": leg_time,
                "distance_km": leg_dist,
            })

            # Append to full path without duplicating joining nodes
            if not full_node_path:
                full_node_path.extend(sub_path)
                full_coords.extend(sub_coords)
            else:
                full_node_path.extend(sub_path[1:])
                full_coords.extend(sub_coords[1:])

        detailed_routes.append({
            "vehicle_id": v_idx,
            "high_level_stops": route,
            "street_node_path": full_node_path,
            "street_coordinates": full_coords,
            "legs": legs,
            "total_travel_time": round(total_time, 2),
            "total_distance": round(total_dist, 3),
            "is_empty": False,
        })

    return detailed_routes
