"""Orbital Graph — graph-based orbital relationship modeling.

This module provides:
- OrbitalObject: a space object with position, velocity, and orbital elements
- OrbitalEdge: a proximity relationship between two objects
- OrbitalGraph: graph structure for proximity analysis, cluster detection,
  and orbital similarity computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OrbitalObject:
    """A space object with state and orbital elements."""

    object_id: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    semi_major_axis: float = 0.0  # km
    eccentricity: float = 0.0
    inclination: float = 0.0  # degrees
    object_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrbitalEdge:
    """A proximity edge between two orbital objects."""

    source_id: str
    target_id: str
    distance: float


class OrbitalGraph:
    """Graph-based model of orbital relationships between space objects.

    Supports proximity analysis, cluster detection, and orbital similarity
    computation for space situational awareness.
    """

    def __init__(self, proximity_threshold: float = 100.0):
        """Initialize orbital graph.

        Args:
            proximity_threshold: Default distance threshold (km) for
                proximity edge creation.
        """
        self.proximity_threshold = proximity_threshold
        self._objects: dict[str, OrbitalObject] = {}
        self._edges: list[OrbitalEdge] = []
        self._clusters: list[list[str]] = []

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(self, obj: OrbitalObject) -> None:
        """Add or replace an orbital object."""
        self._objects[obj.object_id] = obj

    def remove_object(self, object_id: str) -> bool:
        """Remove an object by ID. Returns True if found and removed."""
        if object_id not in self._objects:
            return False
        del self._objects[object_id]
        # Invalidate cached edges and clusters
        self._edges = []
        self._clusters = []
        return True

    def get_object(self, object_id: str) -> OrbitalObject | None:
        """Get an object by ID."""
        return self._objects.get(object_id)

    def get_all_objects(self) -> list[OrbitalObject]:
        """Get all objects in the graph."""
        return list(self._objects.values())

    # ------------------------------------------------------------------
    # Proximity analysis
    # ------------------------------------------------------------------

    def compute_proximity_edges(
        self, threshold: float | None = None
    ) -> list[OrbitalEdge]:
        """Compute proximity edges between all object pairs.

        Args:
            threshold: Distance threshold (km). Uses instance default if None.

        Returns:
            List of OrbitalEdge for all pairs within threshold.
        """
        thresh = threshold if threshold is not None else self.proximity_threshold
        edges: list[OrbitalEdge] = []
        objs = list(self._objects.values())

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                dist = self._euclidean_distance(objs[i].position, objs[j].position)
                if dist <= thresh:
                    edges.append(
                        OrbitalEdge(
                            source_id=objs[i].object_id,
                            target_id=objs[j].object_id,
                            distance=dist,
                        )
                    )

        self._edges = edges
        return edges

    def find_proximity_pairs(
        self, threshold: float | None = None
    ) -> list[tuple[str, str]]:
        """Find all object pairs within proximity threshold.

        Returns:
            List of (id_a, id_b) tuples.
        """
        edges = self.compute_proximity_edges(threshold)
        return [(e.source_id, e.target_id) for e in edges]

    def find_nearest_neighbors(
        self, object_id: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """Find k nearest neighbors of an object.

        Args:
            object_id: The reference object ID.
            k: Number of neighbors to return.

        Returns:
            List of (neighbor_id, distance) tuples sorted by distance.
        """
        ref = self._objects.get(object_id)
        if ref is None:
            return []

        distances: list[tuple[str, float]] = []
        for oid, obj in self._objects.items():
            if oid == object_id:
                continue
            dist = self._euclidean_distance(ref.position, obj.position)
            distances.append((oid, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    # ------------------------------------------------------------------
    # Cluster detection
    # ------------------------------------------------------------------

    def detect_clusters(self) -> list[list[str]]:
        """Detect clusters using connected components on proximity graph.

        Returns:
            List of clusters, each a list of object IDs.
        """
        if not self._objects:
            self._clusters = []
            return []

        # Build adjacency from proximity edges
        if not self._edges:
            self.compute_proximity_edges()

        adjacency: dict[str, set[str]] = {oid: set() for oid in self._objects}
        for edge in self._edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        # BFS to find connected components
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for oid in self._objects:
            if oid in visited:
                continue
            cluster: list[str] = []
            queue = [oid]
            visited.add(oid)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(sorted(cluster))

        self._clusters = clusters
        return clusters

    def get_cluster_count(self) -> int:
        """Get the number of detected clusters."""
        if not self._clusters:
            self.detect_clusters()
        return len(self._clusters)

    def get_objects_in_cluster(self, cluster_index: int) -> list[str]:
        """Get object IDs in a specific cluster by index."""
        if not self._clusters:
            self.detect_clusters()
        if cluster_index < 0 or cluster_index >= len(self._clusters):
            return []
        return self._clusters[cluster_index]

    # ------------------------------------------------------------------
    # Orbital similarity
    # ------------------------------------------------------------------

    def compute_orbital_similarity(self, id_a: str, id_b: str) -> float:
        """Compute orbital similarity between two objects (0.0 to 1.0).

        Uses normalized differences in semi-major axis, eccentricity,
        and inclination.
        """
        obj_a = self._objects.get(id_a)
        obj_b = self._objects.get(id_b)
        if obj_a is None or obj_b is None:
            return 0.0

        # Normalize each parameter to [0, 1] difference
        # Semi-major axis: typical range 6500–42000 km
        sma_diff = abs(obj_a.semi_major_axis - obj_b.semi_major_axis) / 35500.0
        # Eccentricity: range 0–1
        ecc_diff = abs(obj_a.eccentricity - obj_b.eccentricity)
        # Inclination: range 0–180 degrees
        inc_diff = abs(obj_a.inclination - obj_b.inclination) / 180.0

        # Weighted average similarity
        total_diff = 0.4 * sma_diff + 0.3 * ecc_diff + 0.3 * inc_diff
        similarity = max(0.0, 1.0 - total_diff)
        return similarity

    def find_similar_orbits(
        self, threshold: float = 0.9
    ) -> list[tuple[str, str]]:
        """Find all pairs with orbital similarity >= threshold.

        Returns:
            List of (id_a, id_b) tuples.
        """
        pairs: list[tuple[str, str]] = []
        objs = list(self._objects.values())
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                sim = self.compute_orbital_similarity(
                    objs[i].object_id, objs[j].object_id
                )
                if sim >= threshold:
                    pairs.append((objs[i].object_id, objs[j].object_id))
        return pairs

    # ------------------------------------------------------------------
    # Stats & lifecycle
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        if not self._edges:
            self.compute_proximity_edges()
        if not self._clusters:
            self.detect_clusters()
        return {
            "total_objects": len(self._objects),
            "total_edges": len(self._edges),
            "cluster_count": len(self._clusters),
        }

    def reset(self) -> None:
        """Clear all objects, edges, and clusters."""
        self._objects.clear()
        self._edges = []
        self._clusters = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean_distance(
        p1: tuple[float, float, float], p2: tuple[float, float, float]
    ) -> float:
        """Compute Euclidean distance between two 3D points."""
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )
