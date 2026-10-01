"""Tests for Orbital Graph — graph-based orbital relationship modeling."""

from __future__ import annotations

import math
import pytest

from src.fusion.orbital_graph import (
    OrbitalEdge,
    OrbitalGraph,
    OrbitalObject,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj(
    obj_id: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **kwargs,
) -> OrbitalObject:
    return OrbitalObject(
        object_id=obj_id,
        position=pos,
        velocity=vel,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Construction & basic object management
# ---------------------------------------------------------------------------

class TestOrbitalGraphConstruction:
    def test_empty_graph_has_zero_objects(self):
        graph = OrbitalGraph()
        assert len(graph.get_all_objects()) == 0

    def test_empty_graph_has_zero_clusters(self):
        graph = OrbitalGraph()
        assert graph.get_cluster_count() == 0

    def test_default_proximity_threshold(self):
        graph = OrbitalGraph()
        assert graph.proximity_threshold > 0


class TestAddObject:
    def test_add_single_object(self):
        graph = OrbitalGraph()
        obj = _make_obj("SAT-1")
        graph.add_object(obj)
        assert len(graph.get_all_objects()) == 1

    def test_add_multiple_objects(self):
        graph = OrbitalGraph()
        for i in range(5):
            graph.add_object(_make_obj(f"SAT-{i}"))
        assert len(graph.get_all_objects()) == 5

    def test_add_duplicate_id_replaces(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("SAT-1", pos=(0, 0, 0)))
        graph.add_object(_make_obj("SAT-1", pos=(100, 0, 0)))
        assert len(graph.get_all_objects()) == 1
        assert graph.get_object("SAT-1").position == (100, 0, 0)


class TestRemoveObject:
    def test_remove_existing_object(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("SAT-1"))
        assert graph.remove_object("SAT-1") is True
        assert len(graph.get_all_objects()) == 0

    def test_remove_nonexistent_object(self):
        graph = OrbitalGraph()
        assert graph.remove_object("NOPE") is False

    def test_remove_object_clears_edges(self):
        graph = OrbitalGraph(proximity_threshold=1000.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(10, 0, 0)))
        graph.compute_proximity_edges()
        graph.remove_object("A")
        edges = graph.compute_proximity_edges()
        assert all(e.source_id != "A" and e.target_id != "A" for e in edges)


class TestGetObject:
    def test_get_existing_object(self):
        graph = OrbitalGraph()
        obj = _make_obj("SAT-1", pos=(1, 2, 3))
        graph.add_object(obj)
        retrieved = graph.get_object("SAT-1")
        assert retrieved is not None
        assert retrieved.position == (1, 2, 3)

    def test_get_nonexistent_object_returns_none(self):
        graph = OrbitalGraph()
        assert graph.get_object("NOPE") is None


# ---------------------------------------------------------------------------
# 2. Proximity analysis
# ---------------------------------------------------------------------------

class TestProximityEdges:
    def test_close_objects_create_edge(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        edges = graph.compute_proximity_edges()
        assert len(edges) == 1
        assert edges[0].distance == pytest.approx(50.0)

    def test_distant_objects_no_edge(self):
        graph = OrbitalGraph(proximity_threshold=10.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(100, 0, 0)))
        edges = graph.compute_proximity_edges()
        assert len(edges) == 0

    def test_edge_distance_is_symmetric(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(30, 40, 0)))
        edges = graph.compute_proximity_edges()
        assert len(edges) == 1
        assert edges[0].distance == pytest.approx(50.0)

    def test_multiple_proximity_pairs(self):
        graph = OrbitalGraph(proximity_threshold=200.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.add_object(_make_obj("C", pos=(100, 0, 0)))
        edges = graph.compute_proximity_edges()
        assert len(edges) == 3  # AB, AC, BC

    def test_custom_threshold_overrides_default(self):
        graph = OrbitalGraph(proximity_threshold=10.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        edges = graph.compute_proximity_edges(threshold=100.0)
        assert len(edges) == 1


class TestFindProximityPairs:
    def test_find_pairs_within_threshold(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.add_object(_make_obj("C", pos=(500, 0, 0)))
        pairs = graph.find_proximity_pairs()
        assert len(pairs) == 1
        pair_ids = {pairs[0][0], pairs[0][1]}
        assert pair_ids == {"A", "B"}

    def test_find_pairs_empty_graph(self):
        graph = OrbitalGraph()
        assert graph.find_proximity_pairs() == []

    def test_find_pairs_custom_threshold(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        pairs = graph.find_proximity_pairs(threshold=100.0)
        assert len(pairs) == 1


class TestNearestNeighbors:
    def test_nearest_neighbor_basic(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(10, 0, 0)))
        graph.add_object(_make_obj("C", pos=(100, 0, 0)))
        neighbors = graph.find_nearest_neighbors("A", k=1)
        assert len(neighbors) == 1
        assert neighbors[0][0] == "B"
        assert neighbors[0][1] == pytest.approx(10.0)

    def test_nearest_neighbors_k_larger_than_graph(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(10, 0, 0)))
        neighbors = graph.find_nearest_neighbors("A", k=5)
        assert len(neighbors) == 1

    def test_nearest_neighbors_sorted_by_distance(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(30, 0, 0)))
        graph.add_object(_make_obj("C", pos=(10, 0, 0)))
        neighbors = graph.find_nearest_neighbors("A", k=2)
        assert neighbors[0][0] == "C"
        assert neighbors[1][0] == "B"

    def test_nearest_neighbors_nonexistent_object(self):
        graph = OrbitalGraph()
        assert graph.find_nearest_neighbors("NOPE", k=3) == []


# ---------------------------------------------------------------------------
# 3. Cluster detection
# ---------------------------------------------------------------------------

class TestClusterDetection:
    def test_single_cluster(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.add_object(_make_obj("C", pos=(80, 0, 0)))
        clusters = graph.detect_clusters()
        assert len(clusters) == 1
        assert set(clusters[0]) == {"A", "B", "C"}

    def test_multiple_clusters(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        # Cluster 1
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        # Cluster 2 — far away
        graph.add_object(_make_obj("C", pos=(1000, 0, 0)))
        graph.add_object(_make_obj("D", pos=(1050, 0, 0)))
        clusters = graph.detect_clusters()
        assert len(clusters) == 2

    def test_isolated_objects_form_singleton_clusters(self):
        graph = OrbitalGraph(proximity_threshold=10.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(1000, 0, 0)))
        clusters = graph.detect_clusters()
        assert len(clusters) == 2
        assert all(len(c) == 1 for c in clusters)

    def test_cluster_count_matches_detect(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.add_object(_make_obj("C", pos=(1000, 0, 0)))
        clusters = graph.detect_clusters()
        assert graph.get_cluster_count() == len(clusters)

    def test_get_objects_in_cluster(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.add_object(_make_obj("C", pos=(1000, 0, 0)))
        clusters = graph.detect_clusters()
        # Find the cluster containing A
        cluster_a = next(c for c in clusters if "A" in c)
        objects = graph.get_objects_in_cluster(clusters.index(cluster_a))
        assert set(objects) == {"A", "B"}

    def test_clusters_empty_graph(self):
        graph = OrbitalGraph()
        assert graph.detect_clusters() == []


# ---------------------------------------------------------------------------
# 4. Orbital similarity
# ---------------------------------------------------------------------------

class TestOrbitalSimilarity:
    def test_identical_orbits_similarity_one(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", semi_major_axis=7000, eccentricity=0.01, inclination=45.0))
        graph.add_object(_make_obj("B", semi_major_axis=7000, eccentricity=0.01, inclination=45.0))
        sim = graph.compute_orbital_similarity("A", "B")
        assert sim == pytest.approx(1.0)

    def test_different_orbits_similarity_less_than_one(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", semi_major_axis=7000, eccentricity=0.01, inclination=45.0))
        graph.add_object(_make_obj("B", semi_major_axis=8000, eccentricity=0.2, inclination=60.0))
        sim = graph.compute_orbital_similarity("A", "B")
        assert sim < 1.0
        assert sim >= 0.0

    def test_similarity_nonexistent_object(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A"))
        assert graph.compute_orbital_similarity("A", "NOPE") == 0.0

    def test_find_similar_orbits(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A", semi_major_axis=7000, eccentricity=0.01, inclination=45.0))
        graph.add_object(_make_obj("B", semi_major_axis=7000, eccentricity=0.01, inclination=45.0))
        graph.add_object(_make_obj("C", semi_major_axis=20000, eccentricity=0.5, inclination=80.0))
        similar = graph.find_similar_orbits(threshold=0.95)
        assert len(similar) == 1
        pair_ids = {similar[0][0], similar[0][1]}
        assert pair_ids == {"A", "B"}


# ---------------------------------------------------------------------------
# 5. Graph stats & reset
# ---------------------------------------------------------------------------

class TestGraphStats:
    def test_stats_empty_graph(self):
        graph = OrbitalGraph()
        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 0
        assert stats["total_edges"] == 0
        assert stats["cluster_count"] == 0

    def test_stats_with_objects_and_edges(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.add_object(_make_obj("C", pos=(1000, 0, 0)))
        graph.compute_proximity_edges()
        graph.detect_clusters()
        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 3
        assert stats["total_edges"] == 1
        assert stats["cluster_count"] == 2


class TestReset:
    def test_reset_clears_all(self):
        graph = OrbitalGraph(proximity_threshold=100.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        graph.compute_proximity_edges()
        graph.detect_clusters()
        graph.reset()
        assert len(graph.get_all_objects()) == 0
        assert graph.get_cluster_count() == 0
        assert graph.compute_proximity_edges() == []


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_object_no_edges(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A"))
        assert graph.compute_proximity_edges() == []

    def test_single_object_single_cluster(self):
        graph = OrbitalGraph()
        graph.add_object(_make_obj("A"))
        clusters = graph.detect_clusters()
        assert len(clusters) == 1
        assert clusters[0] == ["A"]

    def test_objects_at_exact_threshold(self):
        graph = OrbitalGraph(proximity_threshold=50.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(50, 0, 0)))
        edges = graph.compute_proximity_edges()
        assert len(edges) == 1

    def test_negative_coordinates(self):
        graph = OrbitalGraph(proximity_threshold=200.0)
        graph.add_object(_make_obj("A", pos=(-50, -50, -50)))
        graph.add_object(_make_obj("B", pos=(50, 50, 50)))
        edges = graph.compute_proximity_edges()
        expected_dist = math.sqrt(3 * 100**2)
        assert len(edges) == 1
        assert edges[0].distance == pytest.approx(expected_dist)

    def test_3d_distance_correct(self):
        graph = OrbitalGraph(proximity_threshold=1000.0)
        graph.add_object(_make_obj("A", pos=(0, 0, 0)))
        graph.add_object(_make_obj("B", pos=(3, 4, 0)))
        edges = graph.compute_proximity_edges()
        assert len(edges) == 1
        assert edges[0].distance == pytest.approx(5.0)
