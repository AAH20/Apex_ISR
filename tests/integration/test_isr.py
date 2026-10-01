"""ISR Integration Tests — full pipeline (collection → processing → exploitation → dissemination).

TDD enforced: these tests define the expected behavior of the ISR pipeline
when multiple components are combined.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.fusion.ekf import BearingOnlyEKF, RangeBearingEKF
from src.fusion.orbital_graph import OrbitalGraph, OrbitalObject
from src.fusion.track import TrackConfig, TrackManager, TrackStatus
from src.isr.collection import (
    CollectionPlan,
    ISRCollectionManager,
    Priority,
    Sensor,
    SensorStatus,
    SensorType,
    TaskStatus,
)
from src.isr.dissemination import (
    Classification,
    CoalitionPartner,
    CoalitionSharing,
    PriorityRouter,
    ProductFormatter,
    ProductGenerator,
    ProductType,
)
from src.isr.entity_resolution import (
    EntityObservation,
    EntityResolver,
    EntityType,
    INTSource,
    ResolvedEntity,
    normalize_name,
)
from src.isr.dissemination import Priority as DisseminationPriority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sensor(
    sensor_id: str,
    sensor_type: SensorType,
    max_concurrent: int = 1,
    location: tuple[float, float] = (0.0, 0.0),
) -> Sensor:
    return Sensor(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        max_concurrent=max_concurrent,
        location=location,
    )


def _make_track_manager() -> TrackManager:
    return TrackManager(TrackConfig(m_of_n=3, n_of_n=5, max_misses=3))


def _make_collection_manager() -> ISRCollectionManager:
    return ISRCollectionManager()


def _make_orbital_graph(proximity_threshold: float = 100.0) -> OrbitalGraph:
    return OrbitalGraph(proximity_threshold=proximity_threshold)


def _make_product_generator() -> ProductGenerator:
    return ProductGenerator()


def _make_product_formatter() -> ProductFormatter:
    return ProductFormatter()


def _make_priority_router() -> PriorityRouter:
    return PriorityRouter()


def _make_coalition_sharing() -> CoalitionSharing:
    return CoalitionSharing()


def _make_entity_resolver() -> EntityResolver:
    return EntityResolver()


# ---------------------------------------------------------------------------
# 1. Collection → Track Management Pipeline
# ---------------------------------------------------------------------------


class TestCollectionToTrackPipeline:
    """Integration: sensor tasking → track initiation and management."""

    def test_collection_task_creates_track(self):
        """A completed collection task should produce a track."""
        cm = _make_collection_manager()
        tm = _make_track_manager()

        # Register a radar sensor
        sensor = _make_sensor("RADAR-1", SensorType.RADAR)
        cm.register_sensor(sensor)

        # Create and assign a collection task
        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # Simulate detection from the task → create track
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(10.0, 5.0))
        assert track is not None
        assert track.status == TrackStatus.TENTATIVE
        assert track.hits == 1

    def test_multiple_collection_tasks_produce_multiple_tracks(self):
        """Multiple collection tasks should produce multiple tracks."""
        cm = _make_collection_manager()
        tm = _make_track_manager()

        # Register sensors
        for i in range(3):
            cm.register_sensor(_make_sensor(f"RADAR-{i}", SensorType.RADAR))

        # Create tasks for different targets
        targets = ["TGT-A", "TGT-B", "TGT-C"]
        for tgt in targets:
            task = cm.create_task(SensorType.RADAR, tgt, Priority.MEDIUM)
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

        # Create tracks from detections
        tracks = []
        for i, tgt in enumerate(targets):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            tracks.append(track)

        assert len(tracks) == 3
        assert len(tm.get_all_tracks()) == 3
        assert all(t.status == TrackStatus.TENTATIVE for t in tracks)

    def test_collection_priority_affects_track_priority(self):
        """Higher priority collection tasks should be processed first."""
        cm = _make_collection_manager()
        tm = _make_track_manager()

        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR, max_concurrent=3))

        # Create tasks with different priorities
        critical_task = cm.create_task(SensorType.RADAR, "TGT-CRIT", Priority.CRITICAL)
        low_task = cm.create_task(SensorType.RADAR, "TGT-LOW", Priority.LOW)
        high_task = cm.create_task(SensorType.RADAR, "TGT-HIGH", Priority.HIGH)

        # Allocate plan
        plan = cm.create_plan("PLAN-001")
        cm.add_task_to_plan(plan.plan_id, critical_task)
        cm.add_task_to_plan(plan.plan_id, low_task)
        cm.add_task_to_plan(plan.plan_id, high_task)

        result = cm.allocate_plan(plan.plan_id)

        # Critical task should be assigned first
        assert result["assigned"] == 3
        # Verify order: critical first
        details = result["details"]
        assert details[0]["task_id"] == critical_task.task_id

    def test_track_confirmation_from_collection_detections(self):
        """Tracks should be confirmed after M-of-N detections from collection."""
        cm = _make_collection_manager()
        tm = _make_track_manager()

        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        # Create a collection task
        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # Initiate track
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(10.0, 5.0))
        track_id = track.track_id

        # Simulate additional detections (hits) to confirm track
        for i in range(3):
            tm.update_track(track_id, position=(100.0 + i * 10, 200.0 + i * 5))

        # Track should be confirmed after 3 hits (M-of-N = 3)
        confirmed_track = tm.get_track(track_id)
        assert confirmed_track.status == TrackStatus.CONFIRMED

    def test_track_deletion_after_missed_collections(self):
        """Tracks should be deleted after consecutive misses."""
        cm = _make_collection_manager()
        tm = _make_track_manager()

        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        # Create and complete a collection task
        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # Initiate track
        track = tm.initiate_track(position=(100.0, 200.0))
        track_id = track.track_id

        # Simulate misses (no detection)
        for i in range(4):  # max_misses = 3
            tm.update_track(track_id, position=None)

        # Track should be deleted
        deleted_track = tm.get_track(track_id)
        assert deleted_track.status == TrackStatus.DELETED


# ---------------------------------------------------------------------------
# 2. Collection → Entity Resolution Pipeline
# ---------------------------------------------------------------------------


class TestCollectionToEntityResolutionPipeline:
    """Integration: sensor tasking → multi-INT entity resolution."""

    def test_collection_task_produces_entity_observation(self):
        """A completed collection task should produce an entity observation."""
        cm = _make_collection_manager()
        resolver = _make_entity_resolver()

        # Register sensors of different types
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))
        cm.register_sensor(_make_sensor("SIGINT-1", SensorType.SIGINT))
        cm.register_sensor(_make_sensor("HUMINT-1", SensorType.HUMINT))

        # Create collection tasks
        radar_task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        sigint_task = cm.create_task(SensorType.SIGINT, "TGT-001", Priority.MEDIUM)
        humint_task = cm.create_task(SensorType.HUMINT, "TGT-001", Priority.LOW)

        # Complete all tasks
        for task in [radar_task, sigint_task, humint_task]:
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

        # Create entity observations from collection results
        radar_obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VEHICLE,
            name="Vehicle-Alpha",
            attributes={"type": "truck"},
            location=(40.0, -73.0),
            reliability=0.9,
        )
        sigint_obs = EntityObservation(
            source=IGINTSource.SIGINT,
            entity_type=EntityType.VEHICLE,
            name="Vehicle-Alpha",
            attributes={"type": "truck"},
            location=(40.001, -73.001),
            reliability=0.8,
        )
        humint_obs = EntityObservation(
            source=IGINTSource.HUMINT,
            entity_type=EntityType.VEHICLE,
            name="Vehicle Alpha",
            attributes={"type": "truck"},
            location=(40.002, -73.002),
            reliability=0.7,
        )

        # Resolve entity
        entity = ResolvedEntity(
            entity_id="e-vehicle-alpha",
            entity_type=EntityType.VEHICLE,
            names={"Vehicle-Alpha"},
            attributes={"type": {"truck"}},
            observations=[radar_obs],
        )

        # Score observations against entity
        radar_score = resolver.score_observation(radar_obs, entity)
        sigint_score = resolver.score_observation(sigint_obs, entity)
        humint_score = resolver.score_observation(humint_obs, entity)

        # All scores should be high (same entity)
        assert radar_score > 0.5
        assert sigint_score > 0.5
        assert humint_score > 0.5

    def test_multi_int_collection_resolves_single_entity(self):
        """Multiple INT sources collecting on the same target should resolve to one entity."""
        cm = _make_collection_manager()
        resolver = _make_entity_resolver()

        # Register sensors
        for st in [SensorType.RADAR, SensorType.SIGINT, SensorType.EO_IR, SensorType.SAR]:
            cm.register_sensor(_make_sensor(f"{st.value.upper()}-1", st))

        # Create collection tasks for the same target
        tasks = []
        for st in [SensorType.RADAR, SensorType.SIGINT, SensorType.EO_IR, SensorType.SAR]:
            task = cm.create_task(st, "TGT-X", Priority.HIGH)
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)
            tasks.append(task)

        # Create observations from each source
        observations = [
            EntityObservation(
                source=IGINTSource.GEOINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.0, -0.1),
                reliability=0.9,
            ),
            EntityObservation(
                source=IGINTSource.SIGINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.001, -0.101),
                reliability=0.85,
            ),
            EntityObservation(
                source=IGINTSource.GEOINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.002, -0.102),
                reliability=0.8,
            ),
            EntityObservation(
                source=IGINTSource.GEOINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.003, -0.103),
                reliability=0.75,
            ),
        ]

        # Create resolved entity
        entity = ResolvedEntity(
            entity_id="e-aircraft-1",
            entity_type=EntityType.AIRCRAFT,
            names={"Aircraft-1"},
            attributes={"type": {"fighter"}},
            observations=observations[:1],
        )

        # All observations should score high
        for obs in observations:
            score = resolver.score_observation(obs, entity)
            assert score > 0.5

    def test_collection_plan_produces_resolved_entities(self):
        """A collection plan should produce observations that resolve to entities."""
        cm = _make_collection_manager()
        resolver = _make_entity_resolver()

        # Register sensors
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))
        cm.register_sensor(_make_sensor("EO-1", SensorType.EO_IR))

        # Create a collection plan
        plan = cm.create_plan("PLAN-MULTI")

        # Add tasks to plan
        task1 = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        task2 = cm.create_task(SensorType.EO_IR, "TGT-001", Priority.HIGH)
        cm.add_task_to_plan(plan.plan_id, task1)
        cm.add_task_to_plan(plan.plan_id, task2)

        # Allocate plan
        result = cm.allocate_plan(plan.plan_id)
        assert result["assigned"] == 2

        # Complete tasks
        cm.start_task(task1.task_id)
        cm.complete_task(task1.task_id)
        cm.start_task(task2.task_id)
        cm.complete_task(task2.task_id)

        # Create observations
        obs1 = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VESSEL,
            name="Ship-1",
            attributes={"type": "destroyer"},
            location=(35.0, 140.0),
            reliability=0.9,
        )
        obs2 = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VESSEL,
            name="Ship-1",
            attributes={"type": "destroyer"},
            location=(35.001, 140.001),
            reliability=0.85,
        )

        # Resolve
        entity = ResolvedEntity(
            entity_id="e-ship-1",
            entity_type=EntityType.VESSEL,
            names={"Ship-1"},
            attributes={"type": {"destroyer"}},
            observations=[obs1],
        )

        score = resolver.score_observation(obs2, entity)
        assert score > 0.5


# ---------------------------------------------------------------------------
# 3. Track Management → Dissemination Pipeline
# ---------------------------------------------------------------------------


class TestTrackToDisseminationPipeline:
    """Integration: track management → product generation and dissemination."""

    def test_confirmed_tracks_generate_report(self):
        """Confirmed tracks should be includable in a report product."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        # Create and confirm tracks
        tracks = []
        for i in range(3):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            # Confirm each track
            for _ in range(3):
                tm.update_track(
                    track.track_id,
                    position=(100.0 * i + 10, 200.0 * i + 5),
                )
            tracks.append(tm.get_track(track.track_id))

        # Generate report from confirmed tracks
        confirmed = [t for t in tracks if t.status == TrackStatus.CONFIRMED]
        track_dicts = [
            {"track_id": t.track_id, "position": t.position, "status": t.status.value}
            for t in confirmed
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        assert product.product_type == ProductType.REPORT
        assert product.source == "GEOINT"
        assert "3" in product.content  # Track count

    def test_critical_track_generates_alert(self):
        """A critical track should generate an alert product."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        # Create a track
        track = tm.initiate_track(position=(500.0, 500.0), velocity=(50.0, 50.0))

        # Generate alert for critical track
        product = gen.generate_alert(
            f"Critical track detected: {track.track_id} at {track.position}",
            source="GEOINT",
        )

        assert product.product_type == ProductType.ALERT
        assert product.priority == Priority.CRITICAL
        assert track.track_id in product.content

    def test_track_summary_generates_summary_product(self):
        """A summary of tracks should generate a summary product."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        # Create multiple tracks
        for i in range(5):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            for _ in range(3):
                tm.update_track(
                    track.track_id,
                    position=(100.0 * i + 10, 200.0 * i + 5),
                )

        # Generate summary
        all_tracks = tm.get_all_tracks()
        track_dicts = [
            {"track_id": t.track_id, "position": t.position, "status": t.status.value}
            for t in all_tracks
        ]
        product = gen.generate_summary(track_dicts, source="GEOINT")

        assert product.product_type == ProductType.SUMMARY
        assert "5" in product.content

    def test_track_products_route_correctly(self):
        """Products generated from tracks should route to correct channels."""
        tm = _make_track_manager()
        gen = _make_product_generator()
        router = _make_priority_router()

        # Create tracks
        track = tm.initiate_track(position=(100.0, 200.0))

        # Generate products of different types
        report = gen.generate_report(
            [{"track_id": track.track_id, "position": track.position}]
        )
        alert = gen.generate_alert(f"Track {track.track_id} critical")
        brief = gen.generate_brief(f"Track {track.track_id} summary")
        summary = gen.generate_summary([{"track_id": track.track_id}])

        # Route products
        assert router.route(report) == "routine"
        assert router.route(alert) == "immediate"
        assert router.route(brief) == "routine"
        assert router.route(summary) == "deferred"

    def test_track_products_format_correctly(self):
        """Products from tracks should format correctly for dissemination."""
        tm = _make_track_manager()
        gen = _make_product_generator()
        formatter = _make_product_formatter()

        # Create and confirm a track
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(10.0, 5.0))
        for _ in range(3):
            tm.update_track(track.track_id, position=(110.0, 205.0))

        confirmed = tm.get_track(track.track_id)
        track_dicts = [
            {"track_id": confirmed.track_id, "position": confirmed.position}
        ]

        # Generate and format products
        report = gen.generate_report(track_dicts)
        text_output = formatter.format_text(report)
        json_output = formatter.format_json(report)
        xml_output = formatter.format_xml(report)

        assert "ISR Report" in text_output
        assert "GEOINT" in text_output
        assert "product_id" in json_output
        assert "<product>" in xml_output


# ---------------------------------------------------------------------------
# 4. Entity Resolution → Dissemination Pipeline
# ---------------------------------------------------------------------------


class TestEntityToDisseminationPipeline:
    """Integration: entity resolution → product generation and dissemination."""

    def test_resolved_entity_generates_report(self):
        """A resolved entity should be includable in a report product."""
        resolver = _make_entity_resolver()
        gen = _make_product_generator()

        # Create observations
        obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.PERSON,
            name="John Smith",
            attributes={"rank": "captain"},
            location=(40.0, -73.0),
            reliability=0.9,
        )

        # Resolve entity
        entity = ResolvedEntity(
            entity_id="e-john-smith",
            entity_type=EntityType.PERSON,
            names={"John Smith"},
            attributes={"rank": {"captain"}},
            observations=[obs],
        )

        # Generate report
        track_dicts = [{"entity_id": entity.entity_id, "name": list(entity.names)[0]}]
        product = gen.generate_report(track_dicts, source="HUMINT")

        assert product.product_type == ProductType.REPORT
        assert product.source == "HUMINT"

    def test_entity_confidence_generates_brief(self):
        """Entity confidence score should be includable in a brief."""
        resolver = _make_entity_resolver()
        gen = _make_product_generator()

        # Create observations with different reliability
        high_rel_obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VESSEL,
            name="Ship-1",
            reliability=0.95,
        )
        low_rel_obs = EntityObservation(
            source=IGINTSource.OSINT,
            entity_type=EntityType.VESSEL,
            name="Ship-1",
            reliability=0.3,
        )

        entity = ResolvedEntity(
            entity_id="e-ship-1",
            entity_type=EntityType.VESSEL,
            names={"Ship-1"},
            observations=[],
        )

        high_score = resolver.score_observation(high_rel_obs, entity)
        low_score = resolver.score_observation(low_rel_obs, entity)

        # Generate brief
        brief_content = f"Entity confidence: high={high_score:.2f}, low={low_score:.2f}"
        product = gen.generate_brief(brief_content, source="GEOINT")

        assert product.product_type == ProductType.BRIEF
        assert f"{high_score:.2f}" in product.content

    def test_entity_sharing_respects_classification(self):
        """Entity products should only be shared with authorized partners."""
        resolver = _make_entity_resolver()
        gen = _make_product_generator()
        sharing = _make_coalition_sharing()

        # Create a secret entity product
        obs = EntityObservation(
            source=IGINTSource.SIGINT,
            entity_type=EntityType.PERSON,
            name="Agent-X",
            reliability=0.9,
        )
        entity = ResolvedEntity(
            entity_id="e-agent-x",
            entity_type=EntityType.PERSON,
            names={"Agent-X"},
            observations=[obs],
        )

        product = gen.generate_report(
            [{"entity_id": entity.entity_id}], source="SIGINT"
        )

        # Try sharing with different partners
        nato = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        partner_x = CoalitionPartner(
            name="PartnerX", clearance=Classification.UNCLASSIFIED
        )

        assert sharing.can_share(product, nato) is True
        assert sharing.can_share(product, partner_x) is False

        # Share with NATO
        record = sharing.share(product, nato)
        assert record["partner"] == "NATO"
        assert record["product_id"] == product.product_id


# ---------------------------------------------------------------------------
# 5. Orbital Graph → Dissemination Pipeline
# ---------------------------------------------------------------------------


class TestOrbitalGraphToDisseminationPipeline:
    """Integration: orbital graph analysis → product generation and dissemination."""

    def test_proximity_cluster_generates_alert(self):
        """A proximity cluster should generate an alert product."""
        graph = _make_orbital_graph(proximity_threshold=100.0)
        gen = _make_product_generator()

        # Add objects in close proximity
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(0.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-2",
                position=(50.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )

        # Detect clusters
        clusters = graph.detect_clusters()
        assert len(clusters) == 1

        # Generate alert
        product = gen.generate_alert(
            f"Proximity cluster detected: {clusters[0]}",
            source="GEOINT",
        )
        assert product.product_type == ProductType.ALERT
        assert product.priority == Priority.CRITICAL

    def test_orbital_similarity_generates_report(self):
        """Orbital similarity analysis should generate a report."""
        graph = _make_orbital_graph()
        gen = _make_product_generator()

        # Add objects with similar orbits
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-2",
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-3",
                semi_major_axis=20000,
                eccentricity=0.5,
                inclination=80.0,
            )
        )

        # Find similar orbits
        similar = graph.find_similar_orbits(threshold=0.9)
        assert len(similar) == 1

        # Generate report
        track_dicts = [{"pair": pair} for pair in similar]
        product = gen.generate_report(track_dicts, source="GEOINT")
        assert product.product_type == ProductType.REPORT

    def test_graph_stats_generate_summary(self):
        """Graph statistics should generate a summary product."""
        graph = _make_orbital_graph(proximity_threshold=100.0)
        gen = _make_product_generator()

        # Add objects
        for i in range(5):
            graph.add_object(
                OrbitalObject(
                    object_id=f"SAT-{i}",
                    position=(100.0 * i, 0.0, 0.0),
                )
            )

        # Get stats
        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 5

        # Generate summary
        product = gen.generate_summary(
            [{"stats": stats}], source="GEOINT"
        )
        assert product.product_type == ProductType.SUMMARY


# ---------------------------------------------------------------------------
# 6. Full ISR Pipeline: Collection → Processing → Exploitation → Dissemination
# ---------------------------------------------------------------------------


class TestFullISRPipeline:
    """Integration: full ISR pipeline from collection to dissemination."""

    def test_full_pipeline_single_target(self):
        """Full pipeline: collect → track → resolve → disseminate."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))
        cm.register_sensor(_make_sensor("EO-1", SensorType.EO_IR))

        task1 = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        task2 = cm.create_task(SensorType.EO_IR, "TGT-001", Priority.HIGH)

        cm.auto_assign_task(task1.task_id)
        cm.auto_assign_task(task2.task_id)
        cm.start_task(task1.task_id)
        cm.start_task(task2.task_id)
        cm.complete_task(task1.task_id)
        cm.complete_task(task2.task_id)

        # 2. PROCESSING (Track Management)
        tm = _make_track_manager()
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(10.0, 5.0))
        for _ in range(3):
            tm.update_track(track.track_id, position=(110.0, 205.0))

        confirmed_track = tm.get_track(track.track_id)
        assert confirmed_track.status == TrackStatus.CONFIRMED

        # 3. PROCESSING (Entity Resolution)
        resolver = _make_entity_resolver()
        obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VEHICLE,
            name="Vehicle-1",
            attributes={"type": "truck"},
            location=(40.0, -73.0),
            reliability=0.9,
        )
        entity = ResolvedEntity(
            entity_id="e-vehicle-1",
            entity_type=EntityType.VEHICLE,
            names={"Vehicle-1"},
            attributes={"type": {"truck"}},
            observations=[obs],
        )
        score = resolver.score_observation(obs, entity)
        assert score > 0.5

        # 4. EXPLOITATION (Orbital Graph)
        graph = _make_orbital_graph(proximity_threshold=500.0)
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(0.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-2",
                position=(100.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        clusters = graph.detect_clusters()
        assert len(clusters) == 1

        # 5. DISSEMINATION
        gen = _make_product_generator()
        formatter = _make_product_formatter()
        router = _make_priority_router()

        track_dicts = [
            {
                "track_id": confirmed_track.track_id,
                "position": confirmed_track.position,
                "entity_id": entity.entity_id,
            }
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        # Format and route
        text_output = formatter.format_text(product)
        channel = router.route(product)

        assert "ISR Report" in text_output
        assert channel == "routine"
        assert product.product_type == ProductType.REPORT

    def test_full_pipeline_multiple_targets(self):
        """Full pipeline with multiple targets and sensors."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        for i in range(3):
            cm.register_sensor(_make_sensor(f"RADAR-{i}", SensorType.RADAR))

        targets = ["TGT-A", "TGT-B", "TGT-C"]
        for tgt in targets:
            task = cm.create_task(SensorType.RADAR, tgt, Priority.HIGH)
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        tracks = []
        for i, tgt in enumerate(targets):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            for _ in range(3):
                tm.update_track(
                    track.track_id,
                    position=(100.0 * i + 10, 200.0 * i + 5),
                )
            tracks.append(tm.get_track(track.track_id))

        confirmed = [t for t in tracks if t.status == TrackStatus.CONFIRMED]
        assert len(confirmed) == 3

        # 3. DISSEMINATION
        gen = _make_product_generator()
        track_dicts = [
            {"track_id": t.track_id, "position": t.position} for t in confirmed
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        assert "3" in product.content

    def test_full_pipeline_with_coalition_sharing(self):
        """Full pipeline ending with coalition sharing."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.CRITICAL)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(50.0, 50.0))
        for _ in range(3):
            tm.update_track(track.track_id, position=(150.0, 250.0))

        # 3. DISSEMINATION
        gen = _make_product_generator()
        sharing = _make_coalition_sharing()

        track_dicts = [{"track_id": track.track_id, "position": track.position}]
        product = gen.generate_alert(
            f"Critical track {track.track_id} detected", source="GEOINT"
        )

        # Share with coalition
        nato = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        if sharing.can_share(product, nato):
            record = sharing.share(product, nato)
            assert record["partner"] == "NATO"

    def test_full_pipeline_with_orbital_analysis(self):
        """Full pipeline with orbital graph analysis."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        task = cm.create_task(SensorType.RADAR, "SAT-1", Priority.HIGH)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        track = tm.initiate_track(position=(7000.0, 0.0), velocity=(0.0, 7.5))
        for _ in range(3):
            tm.update_track(track.track_id, position=(7000.0, 100.0))

        # 3. EXPLOITATION
        graph = _make_orbital_graph(proximity_threshold=1000.0)
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(7000.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-2",
                position=(7100.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )

        clusters = graph.detect_clusters()
        stats = graph.get_graph_stats()

        # 4. DISSEMINATION
        gen = _make_product_generator()
        product = gen.generate_report(
            [
                {
                    "track_id": track.track_id,
                    "clusters": clusters,
                    "stats": stats,
                }
            ],
            source="GEOINT",
        )

        assert product.product_type == ProductType.REPORT
        assert stats["total_objects"] == 2


# ---------------------------------------------------------------------------
# 7. EKF → Track Management Pipeline
# ---------------------------------------------------------------------------


class TestEKFToTrackPipeline:
    """Integration: EKF sensor fusion → track management."""

    def test_bearing_only_ekf_updates_track(self):
        """Bearing-only EKF should provide measurements for track updates."""
        tm = _make_track_manager()

        # Create a bearing-only EKF
        x0 = np.array([100.0, 100.0, 10.0, 10.0])
        P0 = np.eye(4) * 100.0
        Q = np.eye(4) * 0.1
        R = np.array([[0.01]])
        sensor_pos = np.array([0.0, 0.0])

        ekf = BearingOnlyEKF(x0, P0, Q, R, sensor_pos)

        # Create a track
        track = tm.initiate_track(position=(100.0, 100.0), velocity=(10.0, 10.0))
        track_id = track.track_id

        # Simulate EKF updates
        for i in range(5):
            dt = 1.0
            ekf.predict(dt)
            # Simulate bearing measurement
            bearing = math.atan2(ekf.x[1], ekf.x[0])
            ekf.update(np.array([bearing]))

            # Update track with EKF state
            tm.update_track(
                track_id,
                position=(float(ekf.x[0]), float(ekf.x[1])),
                velocity=(float(ekf.x[2]), float(ekf.x[3])),
            )

        # Track should be confirmed
        final_track = tm.get_track(track_id)
        assert final_track.status == TrackStatus.CONFIRMED
        assert final_track.hits >= 3

    def test_range_bearing_ekf_updates_track(self):
        """Range-bearing EKF should provide measurements for track updates."""
        tm = _make_track_manager()

        # Create a range-bearing EKF
        x0 = np.array([200.0, 200.0, 15.0, 15.0])
        P0 = np.eye(4) * 200.0
        Q = np.eye(4) * 0.5
        R = np.eye(2) * 1.0
        sensor_pos = np.array([0.0, 0.0])

        ekf = RangeBearingEKF(x0, P0, Q, R, sensor_pos)

        # Create a track
        track = tm.initiate_track(position=(200.0, 200.0), velocity=(15.0, 15.0))
        track_id = track.track_id

        # Simulate EKF updates
        for i in range(5):
            dt = 1.0
            ekf.predict(dt)
            # Simulate range-bearing measurement
            dx = ekf.x[0] - sensor_pos[0]
            dy = ekf.x[1] - sensor_pos[1]
            range_meas = math.sqrt(dx**2 + dy**2)
            bearing_meas = math.atan2(dy, dx)
            ekf.update(np.array([range_meas, bearing_meas]))

            # Update track
            tm.update_track(
                track_id,
                position=(float(ekf.x[0]), float(ekf.x[1])),
                velocity=(float(ekf.x[2]), float(ekf.x[3])),
            )

        final_track = tm.get_track(track_id)
        assert final_track.status == TrackStatus.CONFIRMED

    def test_ekf_track_quality_improves(self):
        """Track quality should improve with EKF updates."""
        tm = _make_track_manager()

        x0 = np.array([100.0, 100.0, 10.0, 10.0])
        P0 = np.eye(4) * 100.0
        Q = np.eye(4) * 0.1
        R = np.array([[0.01]])
        sensor_pos = np.array([0.0, 0.0])

        ekf = BearingOnlyEKF(x0, P0, Q, R, sensor_pos)

        track = tm.initiate_track(position=(100.0, 100.0), velocity=(10.0, 10.0))
        track_id = track.track_id

        initial_quality = track.quality_score

        for i in range(5):
            ekf.predict(1.0)
            bearing = math.atan2(ekf.x[1], ekf.x[0])
            ekf.update(np.array([bearing]))
            tm.update_track(
                track_id,
                position=(float(ekf.x[0]), float(ekf.x[1])),
            )

        final_track = tm.get_track(track_id)
        assert final_track.quality_score >= initial_quality


# ---------------------------------------------------------------------------
# 8. Collection → Orbital Graph Pipeline
# ---------------------------------------------------------------------------


class TestCollectionToOrbitalGraphPipeline:
    """Integration: sensor tasking → orbital graph analysis."""

    def test_collection_task_creates_orbital_object(self):
        """A completed collection task should create an orbital object."""
        cm = _make_collection_manager()
        graph = _make_orbital_graph()

        # Register a radar sensor
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        # Create and complete a collection task
        task = cm.create_task(SensorType.RADAR, "SAT-1", Priority.HIGH)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # Create orbital object from collection result
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(7000.0, 0.0, 0.0),
                velocity=(0.0, 7.5, 0.0),
                semi_major_axis=7000.0,
                eccentricity=0.01,
                inclination=45.0,
                object_type="satellite",
            )
        )

        assert len(graph.get_all_objects()) == 1
        obj = graph.get_object("SAT-1")
        assert obj.object_id == "SAT-1"
        assert obj.semi_major_axis == 7000.0

    def test_multiple_collections_create_orbital_graph(self):
        """Multiple collection tasks should create a graph of orbital objects."""
        cm = _make_collection_manager()
        graph = _make_orbital_graph(proximity_threshold=500.0)

        # Register sensors
        for i in range(3):
            cm.register_sensor(_make_sensor(f"RADAR-{i}", SensorType.RADAR))

        # Create collection tasks for satellites
        satellites = ["SAT-1", "SAT-2", "SAT-3"]
        positions = [
            (7000.0, 0.0, 0.0),
            (7100.0, 0.0, 0.0),
            (7200.0, 0.0, 0.0),
        ]

        for sat, pos in zip(satellites, positions):
            task = cm.create_task(SensorType.RADAR, sat, Priority.HIGH)
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

            graph.add_object(
                OrbitalObject(
                    object_id=sat,
                    position=pos,
                    semi_major_axis=7000.0,
                    eccentricity=0.01,
                    inclination=45.0,
                )
            )

        # Analyze graph
        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 3
        assert stats["total_edges"] > 0

    def test_collection_plan_creates_proximity_analysis(self):
        """A collection plan should enable proximity analysis."""
        cm = _make_collection_manager()
        graph = _make_orbital_graph(proximity_threshold=200.0)

        # Register sensors
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))
        cm.register_sensor(_make_sensor("RADAR-2", SensorType.RADAR))

        # Create a collection plan
        plan = cm.create_plan("PLAN-SAT")

        task1 = cm.create_task(SensorType.RADAR, "SAT-1", Priority.HIGH)
        task2 = cm.create_task(SensorType.RADAR, "SAT-2", Priority.HIGH)
        cm.add_task_to_plan(plan.plan_id, task1)
        cm.add_task_to_plan(plan.plan_id, task2)

        # Allocate and complete
        cm.allocate_plan(plan.plan_id)
        cm.start_task(task1.task_id)
        cm.complete_task(task1.task_id)
        cm.start_task(task2.task_id)
        cm.complete_task(task2.task_id)

        # Add to orbital graph
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(7000.0, 0.0, 0.0),
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-2",
                position=(7100.0, 0.0, 0.0),
            )
        )

        # Find proximity pairs
        pairs = graph.find_proximity_pairs()
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# 9. Track Management → Orbital Graph Pipeline
# ---------------------------------------------------------------------------


class TestTrackToOrbitalGraphPipeline:
    """Integration: track management → orbital graph analysis."""

    def test_track_position_updates_orbital_object(self):
        """Track position should update an orbital object in the graph."""
        tm = _make_track_manager()
        graph = _make_orbital_graph()

        # Create a track
        track = tm.initiate_track(position=(7000.0, 0.0), velocity=(0.0, 7.5))
        track_id = track.track_id

        # Add orbital object
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(7000.0, 0.0, 0.0),
                velocity=(0.0, 7.5, 0.0),
            )
        )

        # Update track
        tm.update_track(track_id, position=(7000.0, 100.0))

        # Update orbital object position
        obj = graph.get_object("SAT-1")
        obj.position = (7000.0, 100.0, 0.0)

        assert obj.position == (7000.0, 100.0, 0.0)

    def test_multiple_tracks_create_orbital_graph(self):
        """Multiple tracks should create a graph of orbital objects."""
        tm = _make_track_manager()
        graph = _make_orbital_graph(proximity_threshold=1000.0)

        # Create tracks
        tracks = []
        for i in range(3):
            track = tm.initiate_track(
                position=(7000.0 + 100 * i, 0.0), velocity=(0.0, 7.5)
            )
            tracks.append(track)

        # Add orbital objects from tracks
        for i, track in enumerate(tracks):
            graph.add_object(
                OrbitalObject(
                    object_id=f"SAT-{i}",
                    position=(7000.0 + 100 * i, 0.0, 0.0),
                    velocity=(0.0, 7.5, 0.0),
                )
            )

        # Analyze
        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 3
        assert stats["total_edges"] > 0

    def test_track_confirmation_updates_orbital_object(self):
        """Confirmed tracks should update orbital objects."""
        tm = _make_track_manager()
        graph = _make_orbital_graph()

        # Create and confirm a track
        track = tm.initiate_track(position=(7000.0, 0.0), velocity=(0.0, 7.5))
        track_id = track.track_id

        for _ in range(3):
            tm.update_track(track_id, position=(7000.0, 100.0))

        confirmed = tm.get_track(track_id)
        assert confirmed.status == TrackStatus.CONFIRMED

        # Add and update orbital object
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=confirmed.position + (0.0,),
                velocity=confirmed.velocity + (0.0,),
            )
        )

        obj = graph.get_object("SAT-1")
        assert obj is not None


# ---------------------------------------------------------------------------
# 10. Entity Resolution → Orbital Graph Pipeline
# ---------------------------------------------------------------------------


class TestEntityToOrbitalGraphPipeline:
    """Integration: entity resolution → orbital graph analysis."""

    def test_resolved_entity_creates_orbital_object(self):
        """A resolved entity should create an orbital object."""
        resolver = _make_entity_resolver()
        graph = _make_orbital_graph()

        # Create observations
        obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VEHICLE,
            name="Vehicle-1",
            location=(40.0, -73.0),
            reliability=0.9,
        )

        # Resolve entity
        entity = ResolvedEntity(
            entity_id="e-vehicle-1",
            entity_type=EntityType.VEHICLE,
            names={"Vehicle-1"},
            observations=[obs],
        )

        # Create orbital object from entity
        graph.add_object(
            OrbitalObject(
                object_id=entity.entity_id,
                position=(40.0, -73.0, 0.0),
                object_type="vehicle",
            )
        )

        assert len(graph.get_all_objects()) == 1

    def test_multiple_entities_create_orbital_graph(self):
        """Multiple resolved entities should create a graph."""
        resolver = _make_entity_resolver()
        graph = _make_orbital_graph(proximity_threshold=1000.0)

        # Create observations for multiple entities
        entities_data = [
            ("e-1", "Entity-1", (40.0, -73.0)),
            ("e-2", "Entity-2", (40.01, -73.01)),
            ("e-3", "Entity-3", (51.0, -0.1)),
        ]

        for entity_id, name, location in entities_data:
            obs = EntityObservation(
                source=IGINTSource.GEOINT,
                entity_type=EntityType.VEHICLE,
                name=name,
                location=location,
                reliability=0.9,
            )
            entity = ResolvedEntity(
                entity_id=entity_id,
                entity_type=EntityType.VEHICLE,
                names={name},
                observations=[obs],
            )

            graph.add_object(
                OrbitalObject(
                    object_id=entity_id,
                    position=(location[0], location[1], 0.0),
                )
            )

        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 3


# ---------------------------------------------------------------------------
# 11. Full Pipeline with Multiple Sensors and Tracks
# ---------------------------------------------------------------------------


class TestFullPipelineMultipleSensors:
    """Integration: full pipeline with multiple sensors and tracks."""

    def test_multi_sensor_multi_target_pipeline(self):
        """Full pipeline with multiple sensors tracking multiple targets."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        sensor_types = [SensorType.RADAR, SensorType.EO_IR, SensorType.SAR, SensorType.SIGINT]
        for i, st in enumerate(sensor_types):
            cm.register_sensor(_make_sensor(f"{st.value.upper()}-{i}", st))

        # Create tasks for multiple targets
        targets = ["TGT-A", "TGT-B", "TGT-C", "TGT-D"]
        for tgt in targets:
            for st in sensor_types:
                task = cm.create_task(st, tgt, Priority.HIGH)
                cm.auto_assign_task(task.task_id)
                cm.start_task(task.task_id)
                cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        tracks = []
        for i, tgt in enumerate(targets):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            for _ in range(3):
                tm.update_track(
                    track.track_id,
                    position=(100.0 * i + 10, 200.0 * i + 5),
                )
            tracks.append(tm.get_track(track.track_id))

        confirmed = [t for t in tracks if t.status == TrackStatus.CONFIRMED]
        assert len(confirmed) == 4

        # 3. DISSEMINATION
        gen = _make_product_generator()
        track_dicts = [
            {"track_id": t.track_id, "position": t.position} for t in confirmed
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        assert "4" in product.content

    def test_multi_sensor_entity_resolution_pipeline(self):
        """Full pipeline with multi-sensor entity resolution."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        for st in [SensorType.RADAR, SensorType.SIGINT, SensorType.EO_IR]:
            cm.register_sensor(_make_sensor(f"{st.value.upper()}-1", st))

        # Create tasks for the same target
        for st in [SensorType.RADAR, SensorType.SIGINT, SensorType.EO_IR]:
            task = cm.create_task(st, "TGT-001", Priority.HIGH)
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

        # 2. PROCESSING (Entity Resolution)
        resolver = _make_entity_resolver()
        observations = [
            EntityObservation(
                source=IGINTSource.GEOINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.0, -0.1),
                reliability=0.9,
            ),
            EntityObservation(
                source=IGINTSource.SIGINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.001, -0.101),
                reliability=0.85,
            ),
            EntityObservation(
                source=IGINTSource.GEOINT,
                entity_type=EntityType.AIRCRAFT,
                name="Aircraft-1",
                attributes={"type": "fighter"},
                location=(51.002, -0.102),
                reliability=0.8,
            ),
        ]

        entity = ResolvedEntity(
            entity_id="e-aircraft-1",
            entity_type=EntityType.AIRCRAFT,
            names={"Aircraft-1"},
            attributes={"type": {"fighter"}},
            observations=observations[:1],
        )

        # Score all observations
        scores = [resolver.score_observation(obs, entity) for obs in observations]
        assert all(s > 0.5 for s in scores)

        # 3. DISSEMINATION
        gen = _make_product_generator()
        product = gen.generate_report(
            [{"entity_id": entity.entity_id, "scores": scores}],
            source="GEOINT",
        )
        assert product.product_type == ProductType.REPORT


# ---------------------------------------------------------------------------
# 12. Collection → Processing → Dissemination with Coalition Sharing
# ---------------------------------------------------------------------------


class TestFullPipelineWithCoalitionSharing:
    """Integration: full pipeline with coalition sharing."""

    def test_classified_pipeline_with_sharing(self):
        """Full pipeline with classified products and coalition sharing."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.CRITICAL)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(50.0, 50.0))
        for _ in range(3):
            tm.update_track(track.track_id, position=(150.0, 250.0))

        # 3. DISSEMINATION
        gen = _make_product_generator()
        sharing = _make_coalition_sharing()

        track_dicts = [{"track_id": track.track_id, "position": track.position}]
        product = gen.generate_alert(
            f"Critical track {track.track_id}", source="GEOINT"
        )

        # Share with coalition partners
        partners = [
            CoalitionPartner(name="NATO", clearance=Classification.SECRET),
            CoalitionPartner(name="FiveEyes", clearance=Classification.TOP_SECRET),
            CoalitionPartner(name="PartnerX", clearance=Classification.UNCLASSIFIED),
        ]

        shareable = sharing.get_shareable_products([product], partners[0])
        assert len(shareable) == 1

        # Share with NATO
        record = sharing.share(product, partners[0])
        assert record["partner"] == "NATO"

        # Cannot share with PartnerX
        assert sharing.can_share(product, partners[2]) is False

    def test_multi_product_pipeline_with_routing(self):
        """Full pipeline with multiple products and routing."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        for i in range(4):
            task = cm.create_task(SensorType.RADAR, f"TGT-{i}", Priority.HIGH)
            cm.auto_assign_task(task.task_id)
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        for i in range(4):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            for _ in range(3):
                tm.update_track(
                    track.track_id,
                    position=(100.0 * i + 10, 200.0 * i + 5),
                )

        # 3. DISSEMINATION
        gen = _make_product_generator()
        router = _make_priority_router()

        all_tracks = tm.get_all_tracks()
        track_dicts = [
            {"track_id": t.track_id, "position": t.position} for t in all_tracks
        ]

        # Generate different product types
        report = gen.generate_report(track_dicts)
        alert = gen.generate_alert("Critical detection")
        brief = gen.generate_brief("Daily summary")
        summary = gen.generate_summary(track_dicts)

        # Route all products
        products = [report, alert, brief, summary]
        routed = router.route_all(products)

        assert "immediate" in routed
        assert "routine" in routed
        assert "deferred" in routed
        assert len(routed["immediate"]) == 1  # alert
        assert len(routed["routine"]) == 2  # report + brief
        assert len(routed["deferred"]) == 1  # summary


# ---------------------------------------------------------------------------
# 13. EKF → Track → Dissemination Pipeline
# ---------------------------------------------------------------------------


class TestEKFTrackDisseminationPipeline:
    """Integration: EKF → track management → dissemination."""

    def test_ekf_track_generates_product(self):
        """EKF-updated track should generate a product."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        # Create EKF
        x0 = np.array([100.0, 100.0, 10.0, 10.0])
        P0 = np.eye(4) * 100.0
        Q = np.eye(4) * 0.1
        R = np.array([[0.01]])
        sensor_pos = np.array([0.0, 0.0])
        ekf = BearingOnlyEKF(x0, P0, Q, R, sensor_pos)

        # Create track
        track = tm.initiate_track(position=(100.0, 100.0), velocity=(10.0, 10.0))
        track_id = track.track_id

        # Update track with EKF
        for _ in range(5):
            ekf.predict(1.0)
            bearing = math.atan2(ekf.x[1], ekf.x[0])
            ekf.update(np.array([bearing]))
            tm.update_track(
                track_id,
                position=(float(ekf.x[0]), float(ekf.x[1])),
            )

        # Generate product
        final_track = tm.get_track(track_id)
        track_dicts = [
            {"track_id": final_track.track_id, "position": final_track.position}
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        assert product.product_type == ProductType.REPORT
        assert final_track.track_id in product.content

    def test_ekf_track_quality_in_product(self):
        """EKF track quality should be reflected in product."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        x0 = np.array([100.0, 100.0, 10.0, 10.0])
        P0 = np.eye(4) * 100.0
        Q = np.eye(4) * 0.1
        R = np.array([[0.01]])
        sensor_pos = np.array([0.0, 0.0])
        ekf = BearingOnlyEKF(x0, P0, Q, R, sensor_pos)

        track = tm.initiate_track(position=(100.0, 100.0), velocity=(10.0, 10.0))
        track_id = track.track_id

        for _ in range(5):
            ekf.predict(1.0)
            bearing = math.atan2(ekf.x[1], ekf.x[0])
            ekf.update(np.array([bearing]))
            tm.update_track(
                track_id,
                position=(float(ekf.x[0]), float(ekf.x[1])),
            )

        final_track = tm.get_track(track_id)
        track_dicts = [
            {
                "track_id": final_track.track_id,
                "quality": final_track.quality_score,
            }
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        assert final_track.quality_score > 0.3


# ---------------------------------------------------------------------------
# 14. Collection → Entity → Orbital → Dissemination Pipeline
# ---------------------------------------------------------------------------


class TestFullPipelineAllComponents:
    """Integration: all ISR components working together."""

    def test_complete_isr_cycle(self):
        """Complete ISR cycle: collect → process → exploit → disseminate."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))
        cm.register_sensor(_make_sensor("EO-1", SensorType.EO_IR))
        cm.register_sensor(_make_sensor("SIGINT-1", SensorType.SIGINT))

        # Create collection plan
        plan = cm.create_plan("PLAN-001")
        tasks = []
        for st in [SensorType.RADAR, SensorType.EO_IR, SensorType.SIGINT]:
            task = cm.create_task(st, "TGT-001", Priority.HIGH)
            cm.add_task_to_plan(plan.plan_id, task)
            tasks.append(task)

        # Allocate and execute plan
        cm.allocate_plan(plan.plan_id)
        for task in tasks:
            cm.start_task(task.task_id)
            cm.complete_task(task.task_id)

        # 2. PROCESSING (Track Management)
        tm = _make_track_manager()
        track = tm.initiate_track(position=(100.0, 200.0), velocity=(10.0, 5.0))
        for _ in range(3):
            tm.update_track(track.track_id, position=(110.0, 205.0))

        confirmed_track = tm.get_track(track.track_id)
        assert confirmed_track.status == TrackStatus.CONFIRMED

        # 3. PROCESSING (Entity Resolution)
        resolver = _make_entity_resolver()
        obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.VEHICLE,
            name="Vehicle-1",
            attributes={"type": "truck"},
            location=(40.0, -73.0),
            reliability=0.9,
        )
        entity = ResolvedEntity(
            entity_id="e-vehicle-1",
            entity_type=EntityType.VEHICLE,
            names={"Vehicle-1"},
            attributes={"type": {"truck"}},
            observations=[obs],
        )
        score = resolver.score_observation(obs, entity)
        assert score > 0.5

        # 4. EXPLOITATION (Orbital Graph)
        graph = _make_orbital_graph(proximity_threshold=500.0)
        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(0.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        graph.add_object(
            OrbitalObject(
                object_id="SAT-2",
                position=(100.0, 0.0, 0.0),
                semi_major_axis=7000,
                eccentricity=0.01,
                inclination=45.0,
            )
        )
        clusters = graph.detect_clusters()
        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 2

        # 5. DISSEMINATION
        gen = _make_product_generator()
        formatter = _make_product_formatter()
        router = _make_priority_router()
        sharing = _make_coalition_sharing()

        track_dicts = [
            {
                "track_id": confirmed_track.track_id,
                "entity_id": entity.entity_id,
                "clusters": clusters,
            }
        ]
        product = gen.generate_report(track_dicts, source="GEOINT")

        # Format
        text_output = formatter.format_text(product)
        assert "ISR Report" in text_output

        # Route
        channel = router.route(product)
        assert channel == "routine"

        # Share
        nato = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        if sharing.can_share(product, nato):
            record = sharing.share(product, nato)
            assert record["partner"] == "NATO"

    def test_complete_isr_cycle_with_alert(self):
        """Complete ISR cycle ending with critical alert."""
        # 1. COLLECTION
        cm = _make_collection_manager()
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        task = cm.create_task(SensorType.RADAR, "TGT-CRIT", Priority.CRITICAL)
        cm.auto_assign_task(task.task_id)
        cm.start_task(task.task_id)
        cm.complete_task(task.task_id)

        # 2. PROCESSING
        tm = _make_track_manager()
        track = tm.initiate_track(position=(500.0, 500.0), velocity=(100.0, 100.0))
        for _ in range(3):
            tm.update_track(track.track_id, position=(600.0, 600.0))

        # 3. DISSEMINATION
        gen = _make_product_generator()
        router = _make_priority_router()

        track_dicts = [{"track_id": track.track_id, "position": track.position}]
        product = gen.generate_alert(
            f"CRITICAL: Track {track.track_id} at high velocity",
            source="GEOINT",
        )

        channel = router.route(product)
        assert channel == "immediate"
        assert product.priority == Priority.CRITICAL


# ---------------------------------------------------------------------------
# 15. Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    """Integration: edge cases and error handling in the ISR pipeline."""

    def test_empty_collection_produces_empty_product(self):
        """Empty collection should produce an empty product."""
        cm = _make_collection_manager()
        gen = _make_product_generator()

        # No sensors registered, no tasks created
        stats = cm.get_stats()
        assert stats["total_tasks"] == 0

        # Generate product from empty tracks
        product = gen.generate_report([], source="GEOINT")
        assert "0" in product.content

    def test_deleted_track_not_in_product(self):
        """Deleted tracks should not appear in products."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        # Create and delete a track
        track = tm.initiate_track(position=(100.0, 200.0))
        track_id = track.track_id
        tm.delete_track(track_id)

        # Generate product from active tracks only
        active_tracks = tm.get_active_tracks()
        track_dicts = [{"track_id": t.track_id} for t in active_tracks]
        product = gen.generate_report(track_dicts)

        assert track_id not in product.content

    def test_offline_sensor_not_assigned(self):
        """Offline sensors should not be assigned tasks."""
        cm = _make_collection_manager()

        # Register an offline sensor
        sensor = _make_sensor("RADAR-1", SensorType.RADAR)
        sensor.status = SensorStatus.OFFLINE
        cm.register_sensor(sensor)

        # Create a task
        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)

        # Try to assign
        result = cm.auto_assign_task(task.task_id)
        assert result is None

    def test_maintenance_sensor_not_assigned(self):
        """Sensors in maintenance should not be assigned tasks."""
        cm = _make_collection_manager()

        sensor = _make_sensor("RADAR-1", SensorType.RADAR)
        sensor.status = SensorStatus.MAINTENANCE
        cm.register_sensor(sensor)

        task = cm.create_task(SensorType.RADAR, "TGT-001", Priority.HIGH)
        result = cm.auto_assign_task(task.task_id)
        assert result is None

    def test_wrong_sensor_type_not_assigned(self):
        """Tasks should not be assigned to sensors of the wrong type."""
        cm = _make_collection_manager()

        # Register a radar sensor
        cm.register_sensor(_make_sensor("RADAR-1", SensorType.RADAR))

        # Create a SIGINT task
        task = cm.create_task(SensorType.SIGINT, "TGT-001", Priority.HIGH)

        # Try to assign to radar sensor
        result = cm.auto_assign_task(task.task_id)
        assert result is None

    def test_entity_resolution_with_no_observations(self):
        """Entity resolution should handle entities with no observations."""
        resolver = _make_entity_resolver()

        entity = ResolvedEntity(
            entity_id="e-empty",
            entity_type=EntityType.PERSON,
            names={"Unknown"},
            observations=[],
        )

        obs = EntityObservation(
            source=IGINTSource.GEOINT,
            entity_type=EntityType.PERSON,
            name="Unknown",
        )

        score = resolver.score_observation(obs, entity)
        assert 0.0 <= score <= 1.0

    def test_orbital_graph_with_single_object(self):
        """Orbital graph should handle a single object."""
        graph = _make_orbital_graph()

        graph.add_object(
            OrbitalObject(
                object_id="SAT-1",
                position=(7000.0, 0.0, 0.0),
            )
        )

        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 1
        assert stats["total_edges"] == 0

        clusters = graph.detect_clusters()
        assert len(clusters) == 1

    def test_product_with_special_characters(self):
        """Products should handle special characters in content."""
        gen = _make_product_generator()
        formatter = _make_product_formatter()

        product = gen.generate_alert(
            "Target detected: <script>alert('xss')</script> & \"quotes\"",
            source="GEOINT",
        )

        # Should not raise
        text_output = formatter.format_text(product)
        json_output = formatter.format_json(product)
        xml_output = formatter.format_xml(product)

        assert "<script>" in text_output
        assert "product_id" in json_output


# ---------------------------------------------------------------------------
# 16. Performance and Scalability Tests
# ---------------------------------------------------------------------------


class TestPipelineScalability:
    """Integration: pipeline scalability with many objects."""

    def test_many_tracks_in_product(self):
        """Pipeline should handle many tracks in a product."""
        tm = _make_track_manager()
        gen = _make_product_generator()

        # Create many tracks
        for i in range(50):
            track = tm.initiate_track(
                position=(100.0 * i, 200.0 * i), velocity=(10.0, 5.0)
            )
            for _ in range(3):
                tm.update_track(
                    track.track_id,
                    position=(100.0 * i + 10, 200.0 * i + 5),
                )

        all_tracks = tm.get_all_tracks()
        track_dicts = [{"track_id": t.track_id} for t in all_tracks]
        product = gen.generate_report(track_dicts)

        assert "50" in product.content

    def test_many_orbital_objects(self):
        """Orbital graph should handle many objects."""
        graph = _make_orbital_graph(proximity_threshold=100.0)

        # Add many objects
        for i in range(20):
            graph.add_object(
                OrbitalObject(
                    object_id=f"SAT-{i}",
                    position=(100.0 * i, 0.0, 0.0),
                )
            )

        stats = graph.get_graph_stats()
        assert stats["total_objects"] == 20

    def test_many_sensors_and_tasks(self):
        """Collection manager should handle many sensors and tasks."""
        cm = _make_collection_manager()

        # Register many sensors
        for i in range(10):
            cm.register_sensor(
                _make_sensor(f"RADAR-{i}", SensorType.RADAR, max_concurrent=2)
            )

        # Create many tasks
        for i in range(20):
            task = cm.create_task(SensorType.RADAR, f"TGT-{i}", Priority.HIGH)
            cm.auto_assign_task(task.task_id)

        stats = cm.get_stats()
        assert stats["total_sensors"] == 10
        assert stats["total_tasks"] == 20


# ---------------------------------------------------------------------------
# 17. Coalition Sharing with Multiple Products
# ---------------------------------------------------------------------------


class TestCoalitionSharingMultipleProducts:
    """Integration: coalition sharing with multiple products."""

    def test_share_multiple_products_with_partner(self):
        """Multiple products should be shareable with a partner."""
        gen = _make_product_generator()
        sharing = _make_coalition_sharing()

        # Create products of different classifications
        products = [
            gen.generate_summary([{"track_id": "T1"}]),  # UNCLASSIFIED
            gen.generate_brief("Brief"),  # CONFIDENTIAL
            gen.generate_report([{"track_id": "T2"}]),  # CONFIDENTIAL
            gen.generate_alert("Alert"),  # SECRET
        ]

        # Share with SECRET-cleared partner
        nato = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        shareable = sharing.get_shareable_products(products, nato)

        # Should share all except TOP_SECRET (none in this case)
        assert len(shareable) == 4

    def test_filter_products_by_clearance(self):
        """Products should be filtered by partner clearance."""
        gen = _make_product_generator()
        sharing = _make_coalition_sharing()

        products = [
            gen.generate_summary([{"track_id": "T1"}]),  # UNCLASSIFIED
            gen.generate_alert("Alert"),  # SECRET
        ]

        # UNCLASSIFIED partner
        partner = CoalitionPartner(name="Low", clearance=Classification.UNCLASSIFIED)
        shareable = sharing.get_shareable_products(products, partner)
        assert len(shareable) == 1
        assert shareable[0].product_type == ProductType.SUMMARY

    def test_share_all_shareable_products(self):
        """All shareable products should be shareable."""
        gen = _make_product_generator()
        sharing = _make_coalition_sharing()

        products = [
            gen.generate_summary([{"track_id": "T1"}]),
            gen.generate_brief("Brief"),
            gen.generate_report([{"track_id": "T2"}]),
        ]

        nato = CoalitionPartner(name="NATO", clearance=Classification.SECRET)

        for product in products:
            if sharing.can_share(product, nato):
                record = sharing.share(product, nato)
                assert record["partner"] == "NATO"
                assert "timestamp" in record


# ---------------------------------------------------------------------------
# 18. Product Formatting Integration
# ---------------------------------------------------------------------------


class TestProductFormattingIntegration:
    """Integration: product formatting with various product types."""

    def test_format_all_product_types(self):
        """All product types should format correctly."""
        gen = _make_product_generator()
        formatter = _make_product_formatter()

        products = [
            gen.generate_report([{"track_id": "T1"}]),
            gen.generate_alert("Alert"),
            gen.generate_brief("Brief"),
            gen.generate_summary([{"track_id": "T2"}]),
        ]

        for product in products:
            text = formatter.format_text(product)
            json_str = formatter.format_json(product)
            xml_str = formatter.format_xml(product)

            assert len(text) > 0
            assert len(json_str) > 0
            assert len(xml_str) > 0
            assert product.product_id in text
            assert product.product_id in json_str
            assert product.product_id in xml_str

    def test_format_product_with_metadata(self):
        """Products with metadata should format correctly."""
        gen = _make_product_generator()
        formatter = _make_product_formatter()

        product = gen.generate_report([{"track_id": "T1"}])
        product.metadata = {"classification_reason": "test", "confidence": 0.95}

        json_str = formatter.format_json(product)
        assert "metadata" in json_str
        assert "confidence" in json_str


# ---------------------------------------------------------------------------
# 19. Priority Routing Integration
# ---------------------------------------------------------------------------


class TestPriorityRoutingIntegration:
    """Integration: priority routing with various products."""

    def test_route_products_by_priority(self):
        """Products should route to correct channels based on priority."""
        gen = _make_product_generator()
        router = _make_priority_router()

        # Create products with different priorities
        products = [
            gen.generate_alert("Critical"),  # CRITICAL → immediate
            gen.generate_report([{"track_id": "T1"}]),  # MEDIUM → routine
            gen.generate_brief("Brief"),  # MEDIUM → routine
            gen.generate_summary([{"track_id": "T2"}]),  # LOW → deferred
        ]

        routed = router.route_all(products)

        assert "immediate" in routed
        assert "routine" in routed
        assert "deferred" in routed
        assert len(routed["immediate"]) == 1
        assert len(routed["routine"]) == 2
        assert len(routed["deferred"]) == 1

    def test_route_empty_product_list(self):
        """Routing an empty product list should return empty channels."""
        router = _make_priority_router()
        routed = router.route_all([])
        assert routed == {}


# ---------------------------------------------------------------------------
# 20. Track Management Statistics Integration
# ---------------------------------------------------------------------------


class TestTrackStatisticsIntegration:
    """Integration: track management statistics in pipeline."""

    def test_track_stats_reflect_pipeline_state(self):
        """Track statistics should reflect the current pipeline state."""
        tm = _make_track_manager()

        # Create tracks
        for i in range(5):
            track = tm.initiate_track(position=(100.0 * i, 200.0 * i))
            if i < 3:
                # Confirm first 3 tracks
                for _ in range(3):
                    tm.update_track(track.track_id, position=(100.0 * i + 10, 200.0 * i + 5))
            if i == 4:
                # Delete last track
                tm.delete_track(track.track_id)

        stats = tm.get_stats()
        assert stats["total"] == 5
        assert stats["confirmed"] == 3
        assert stats["tentative"] == 1
        assert stats["deleted"] == 1

    def test_track_stats_after_misses(self):
        """Track statistics should update after misses."""
        tm = _make_track_manager()

        track = tm.initiate_track(position=(100.0, 200.0))
        track_id = track.track_id

        # Add misses
        for _ in range(4):
            tm.update_track(track_id, position=None)

        stats = tm.get_stats()
        assert stats["deleted"] == 1
