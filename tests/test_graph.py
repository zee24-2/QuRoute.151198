import pytest
from backend.core.graph import generate_grid_network, generate_indian_city_network
from backend.core.traffic import DynamicTrafficEngine


def test_grid_generation():
    graph = generate_grid_network(rows=5, cols=5, num_customers=8, seed=42)
    assert len(graph.nodes) == 25
    assert len(graph.depot_ids) >= 1
    assert len(graph.customer_ids) == 8
    assert len(graph.edges) > 0


def test_indian_city_generation():
    graph = generate_indian_city_network(num_customers=10, seed=42)
    assert len(graph.nodes) >= 15
    assert len(graph.depot_ids) >= 1
    assert len(graph.customer_ids) == 10


def test_dynamic_traffic_cyclical_and_incident():
    graph = generate_grid_network(rows=4, cols=4, num_customers=5, seed=42)
    engine = DynamicTrafficEngine(graph, initial_time_minutes=540.0)  # 09:00 AM

    # Morning rush should have high cyclical multiplier
    mult_morning = engine.calculate_cyclical_multiplier(540.0)
    mult_night = engine.calculate_cyclical_multiplier(180.0)  # 03:00 AM
    assert mult_morning > mult_night

    # Trigger incident
    incident = engine.trigger_random_incident(severity=5.0, half_life=20.0)
    assert incident.id in engine.incidents

    # Edge multiplier should be boosted
    edge_obj = graph.edges[incident.edge]
    assert edge_obj.congestion_multiplier > 4.0

    # Advance time by 40 minutes (2 half-lives) and verify decay
    engine.step_time(40.0)
    decayed_boost = incident.current_multiplier_boost(engine.current_time_minutes)
    assert decayed_boost < 2.0
