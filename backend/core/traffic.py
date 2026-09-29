"""
Dynamic Traffic Simulation Engine for QuRoute (SIH26137 - Visionaries for Change).
Implements the Bureau of Public Roads (BPR, 1964) congestion function:
    t(e) = t_0 * (1 + alpha * (v / c)^beta)
with flow v aggregated from a 3-layer dynamic hierarchy:
1. Diurnal Curve: Time-of-day demand profile (morning & evening rush peaks).
   Also serves as the predictive fallback during Live Traffic Data Blackouts (Slide 4 Risk Mitigation).
2. Random Incident Layer: Poisson-triggered localized flow spikes decaying over simulated minutes.
3. Manual Layer: User edge congestion overrides and road closures.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import random
from .graph import TrafficGraph


@dataclass
class TrafficIncident:
    id: str
    edge: Tuple[int, int]
    severity: float  # flow/multiplier boost
    start_time: float  # simulated minutes from midnight (0.0 to 1440.0)
    decay_half_life: float = 30.0  # minutes until incident severity halves
    description: str = "Vehicle Breakdown / Incident"

    def current_multiplier_boost(self, current_time: float) -> float:
        dt = current_time - self.start_time
        if dt < 0:
            return 0.0
        factor = math.pow(0.5, dt / self.decay_half_life)
        return self.severity * factor


class DynamicTrafficEngine:
    def __init__(
        self,
        graph: TrafficGraph,
        initial_time_minutes: float = 540.0,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
    ):
        self.graph = graph
        self.current_time_minutes: float = initial_time_minutes  # Default 09:00 AM
        self.bpr_alpha: float = bpr_alpha  # Standard BPR (1964) alpha = 0.15
        self.bpr_beta: float = bpr_beta    # Standard BPR (1964) beta = 4.0
        self.live_data_blackout: bool = False  # Slide 4 Risk Mitigation Toggle
        self.incidents: Dict[str, TrafficIncident] = {}
        self.manual_multipliers: Dict[Tuple[int, int], float] = {}
        self.closed_edges: set = set()
        self.incident_counter: int = 0
        self.update_all_edge_weights()

    def get_time_string(self) -> str:
        mins = int(self.current_time_minutes) % 1440
        hours = mins // 60
        m = mins % 60
        return f"{hours:02d}:{m:02d}"

    def set_time(self, minutes_from_midnight: float):
        """Set simulated clock (0 to 1440 minutes) and update 15-min bucket."""
        self.current_time_minutes = float(minutes_from_midnight) % 1440.0
        self.graph.current_time_bucket = int(self.current_time_minutes // 15)
        self.update_all_edge_weights()

    def step_time(self, delta_minutes: float = 5.0):
        """Advance simulation clock by delta minutes."""
        self.set_time(self.current_time_minutes + delta_minutes)

    def set_bpr_parameters(self, alpha: float = 0.15, beta: float = 4.0):
        """Update tunable Bureau of Public Roads (1964) parameters."""
        self.bpr_alpha = max(0.01, float(alpha))
        self.bpr_beta = max(1.0, float(beta))
        self.update_all_edge_weights()

    def set_blackout_mode(self, blackout: bool):
        """
        Slide 4 Risk Mitigation: Live Traffic Data Blackout Toggle.
        When True, ignores noisy live incident feeds and falls back to predictive historical diurnal curves.
        """
        self.live_data_blackout = bool(blackout)
        self.update_all_edge_weights()

    def calculate_cyclical_multiplier(self, t_mins: float) -> float:
        """
        Layer 1: Diurnal curve.
        Morning peak centered at 09:00 (540m), evening peak at 18:30 (1110m).
        """
        hour = t_mins / 60.0
        base = 0.8 + 0.3 * (0.5 + 0.5 * math.sin(math.pi * (hour - 3.0) / 12.0))
        peak_morning = 1.4 * math.exp(-((hour - 9.0) ** 2) / (2 * (0.9 ** 2)))
        peak_evening = 1.6 * math.exp(-((hour - 18.5) ** 2) / (2 * (1.1 ** 2)))
        return max(0.8, round(base + peak_morning + peak_evening, 3))

    def trigger_random_incident(
        self,
        edge_key: Optional[Tuple[int, int]] = None,
        severity: float = 4.0,
        half_life: float = 30.0,
        description: str = "Accident Reported",
    ) -> TrafficIncident:
        """Layer 2: Triggers a Poisson incident on a specified or random edge."""
        if not self.graph.edges:
            raise ValueError("Graph has no edges to place incident.")

        if edge_key is None or edge_key not in self.graph.edges:
            edge_key = random.choice(list(self.graph.edges.keys()))

        self.incident_counter += 1
        incident_id = f"inc_{self.incident_counter}_{int(self.current_time_minutes)}"
        inc = TrafficIncident(
            id=incident_id,
            edge=edge_key,
            severity=severity,
            start_time=self.current_time_minutes,
            decay_half_life=half_life,
            description=description,
        )
        self.incidents[incident_id] = inc
        self.update_all_edge_weights()
        return inc

    def set_manual_edge_congestion(self, u: int, v: int, multiplier: float, is_closed: bool = False):
        """Layer 3: Direct user interaction override."""
        key = (u, v)
        rev_key = (v, u)

        if is_closed:
            self.closed_edges.add(key)
            if rev_key in self.graph.edges:
                self.closed_edges.add(rev_key)
        else:
            self.closed_edges.discard(key)
            self.closed_edges.discard(rev_key)

        self.manual_multipliers[key] = max(0.5, multiplier)
        if rev_key in self.graph.edges:
            self.manual_multipliers[rev_key] = max(0.5, multiplier)

        self.update_all_edge_weights()

    def update_all_edge_weights(self):
        """
        Recomputes live_weight on all edges using the BPR (1964) congestion function:
            t(e) = t_0 * (1 + alpha * (v / c)^beta)
        During Live Data Blackout mode, incident sensor spikes are suppressed in favor of
        the historical diurnal curve fallback.
        """
        cyclical_factor = self.calculate_cyclical_multiplier(self.current_time_minutes)

        incident_boosts: Dict[Tuple[int, int], float] = {}
        active_incidents = {}
        for inc_id, inc in self.incidents.items():
            boost = inc.current_multiplier_boost(self.current_time_minutes)
            if boost > 0.05:
                active_incidents[inc_id] = inc
                if not self.live_data_blackout:
                    incident_boosts[inc.edge] = incident_boosts.get(inc.edge, 0.0) + boost
                    rev = (inc.edge[1], inc.edge[0])
                    if rev in self.graph.edges:
                        incident_boosts[rev] = incident_boosts.get(rev, 0.0) + (boost * 0.7)
        self.incidents = active_incidents

        for (u, v), edge in self.graph.edges.items():
            manual_factor = self.manual_multipliers.get((u, v), 1.0) if not self.live_data_blackout else 1.0
            inc_boost = incident_boosts.get((u, v), 0.0)
            is_closed = ((u, v) in self.closed_edges) and (not self.live_data_blackout)

            # Compute flow v from diurnal ratio + incident spike
            vc_ratio = (0.75 * cyclical_factor) + (inc_boost * 0.35)
            flow_v = edge.capacity * vc_ratio
            bpr_term = 1.0 + self.bpr_alpha * math.pow(max(0.0, vc_ratio), self.bpr_beta)

            # Combine BPR factor with diurnal & incident boost so both BPR physics and incident spikes are responsive
            final_multiplier = max(0.8, (bpr_term * 0.55 + cyclical_factor * 0.45) * manual_factor + inc_boost)
            self.graph.update_edge_multiplier(
                u,
                v,
                final_multiplier,
                is_closed=is_closed,
                flow=flow_v,
                bpr_alpha=self.bpr_alpha,
                bpr_beta=self.bpr_beta,
            )

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "current_time": self.get_time_string(),
            "time_minutes": self.current_time_minutes,
            "time_bucket_15min": int(self.current_time_minutes // 15),
            "cyclical_multiplier": self.calculate_cyclical_multiplier(self.current_time_minutes),
            "bpr_alpha": self.bpr_alpha,
            "bpr_beta": self.bpr_beta,
            "bpr_formula": f"t(e) = t0 * (1 + {self.bpr_alpha} * (v/c)^{self.bpr_beta})",
            "live_data_blackout": self.live_data_blackout,
            "data_source_mode": "Historical Diurnal Fallback (Blackout Active)" if self.live_data_blackout else "Live BPR Sensor Stream + Diurnal",
            "active_incidents_count": len(self.incidents),
            "closed_edges_count": len(self.closed_edges),
            "incidents": [
                {
                    "id": inc.id,
                    "edge": list(inc.edge),
                    "description": inc.description,
                    "active_boost": round(inc.current_multiplier_boost(self.current_time_minutes), 2),
                }
                for inc in self.incidents.values()
            ],
        }
