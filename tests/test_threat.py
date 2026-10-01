"""Tests for Threat Assessment Engine — TDD enforced."""

from __future__ import annotations

import math
import time

import pytest

from src.isr.threat import (
    Alert,
    AlertGenerator,
    BehaviorPattern,
    Intent,
    ThreatAssessor,
    ThreatConfig,
    ThreatLevel,
    ThreatScore,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_track(
    track_id: str = "TRK-001",
    position: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    status: str = "confirmed",
    hits: int = 5,
    misses: int = 0,
    quality_score: float = 0.8,
    **kwargs,
):
    """Create a track-like dict for testing."""
    return {
        "track_id": track_id,
        "position": position,
        "velocity": velocity,
        "status": status,
        "hits": hits,
        "misses": misses,
        "quality_score": quality_score,
        "total_updates": hits + misses,
        "created_at": time.time() - 100,
        "last_update": time.time(),
        "history": [],
        **kwargs,
    }


def _make_context(
    asset_position: tuple[float, float] = (100.0, 100.0),
    asset_radius: float = 50.0,
    **kwargs,
):
    """Create a context dict for testing."""
    return {
        "asset_position": asset_position,
        "asset_radius": asset_radius,
        "timestamp": time.time(),
        **kwargs,
    }


# ── ThreatLevel enum ────────────────────────────────────────────────


class TestThreatLevel:
    def test_threat_level_has_five_levels(self):
        assert len(ThreatLevel) == 5

    def test_threat_level_ordering(self):
        assert ThreatLevel.NONE < ThreatLevel.LOW
        assert ThreatLevel.LOW < ThreatLevel.MEDIUM
        assert ThreatLevel.MEDIUM < ThreatLevel.HIGH
        assert ThreatLevel.HIGH < ThreatLevel.CRITICAL

    def test_threat_level_values(self):
        assert ThreatLevel.NONE.value == 0
        assert ThreatLevel.LOW.value == 1
        assert ThreatLevel.MEDIUM.value == 2
        assert ThreatLevel.HIGH.value == 3
        assert ThreatLevel.CRITICAL.value == 4


# ── ThreatConfig ────────────────────────────────────────────────────


class TestThreatConfig:
    def test_default_config(self):
        config = ThreatConfig()
        assert config.speed_weight > 0
        assert config.proximity_weight > 0
        assert config.altitude_weight > 0
        assert config.heading_weight > 0
        assert config.maneuver_weight > 0
        assert config.maturity_weight > 0
        assert config.size_weight > 0

    def test_default_thresholds(self):
        config = ThreatConfig()
        assert config.low_threshold == 0.2
        assert config.medium_threshold == 0.4
        assert config.high_threshold == 0.6
        assert config.critical_threshold == 0.8

    def test_custom_config(self):
        config = ThreatConfig(speed_weight=2.0, low_threshold=0.1)
        assert config.speed_weight == 2.0
        assert config.low_threshold == 0.1


# ── ThreatAssessor: basic scoring ───────────────────────────────────


class TestThreatAssessorBasic:
    def test_assess_returns_threat_score(self):
        assessor = ThreatAssessor()
        track = _make_track()
        context = _make_context()
        score = assessor.assess(track, context)
        assert isinstance(score, ThreatScore)

    def test_assess_sets_track_id(self):
        assessor = ThreatAssessor()
        track = _make_track(track_id="TRK-42")
        context = _make_context()
        score = assessor.assess(track, context)
        assert score.track_id == "TRK-42"

    def test_assess_sets_timestamp(self):
        assessor = ThreatAssessor()
        track = _make_track()
        context = _make_context()
        before = time.time()
        score = assessor.assess(track, context)
        after = time.time()
        assert before <= score.timestamp <= after

    def test_assess_score_in_valid_range(self):
        assessor = ThreatAssessor()
        track = _make_track()
        context = _make_context()
        score = assessor.assess(track, context)
        assert 0.0 <= score.score <= 1.0

    def test_assess_level_matches_score(self):
        assessor = ThreatAssessor()
        track = _make_track()
        context = _make_context()
        score = assessor.assess(track, context)
        expected = assessor._score_to_level(score.score)
        assert score.level == expected


# ── ThreatAssessor: speed factor ────────────────────────────────────


class TestThreatSpeedFactor:
    def test_high_speed_increases_score(self):
        assessor = ThreatAssessor()
        slow_track = _make_track(velocity=(1.0, 0.0))
        fast_track = _make_track(velocity=(500.0, 0.0))
        context = _make_context()
        slow_score = assessor.assess(slow_track, context)
        fast_score = assessor.assess(fast_track, context)
        assert fast_score.score > slow_score.score

    def test_zero_speed_minimal_threat(self):
        assessor = ThreatAssessor()
        track = _make_track(velocity=(0.0, 0.0))
        context = _make_context()
        score = assessor.assess(track, context)
        assert score.score < 0.5


# ── ThreatAssessor: proximity factor ────────────────────────────────


class TestThreatProximityFactor:
    def test_close_proximity_increases_score(self):
        assessor = ThreatAssessor()
        far_track = _make_track(position=(1000.0, 1000.0))
        near_track = _make_track(position=(105.0, 105.0))
        context = _make_context(asset_position=(100.0, 100.0))
        far_score = assessor.assess(far_track, context)
        near_score = assessor.assess(near_track, context)
        assert near_score.score > far_score.score

    def test_very_close_proximity_critical(self):
        assessor = ThreatAssessor()
        track = _make_track(position=(101.0, 101.0), velocity=(-500.0, -500.0))
        context = _make_context(asset_position=(100.0, 100.0))
        score = assessor.assess(track, context)
        assert score.level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)


# ── ThreatAssessor: heading factor ──────────────────────────────────


class TestThreatHeadingFactor:
    def test_heading_toward_asset_increases_score(self):
        assessor = ThreatAssessor()
        # Asset at (100, 100), track at (0, 0) moving toward it
        toward_track = _make_track(
            position=(0.0, 0.0),
            velocity=(100.0, 100.0),
        )
        # Track moving away
        away_track = _make_track(
            position=(0.0, 0.0),
            velocity=(-100.0, -100.0),
        )
        context = _make_context(asset_position=(100.0, 100.0))
        toward_score = assessor.assess(toward_track, context)
        away_score = assessor.assess(away_track, context)
        assert toward_score.score > away_score.score


# ── ThreatAssessor: maturity factor ─────────────────────────────────


class TestThreatMaturityFactor:
    def test_confirmed_track_more_threatening(self):
        assessor = ThreatAssessor()
        tentative = _make_track(status="tentative", hits=1)
        confirmed = _make_track(status="confirmed", hits=10)
        context = _make_context()
        tentative_score = assessor.assess(tentative, context)
        confirmed_score = assessor.assess(confirmed, context)
        assert confirmed_score.score > tentative_score.score


# ── ThreatAssessor: intent inference ────────────────────────────────


class TestIntentInference:
    def test_high_speed_approaching_infers_intercept(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(0.0, 0.0),
            velocity=(500.0, 500.0),
        )
        context = _make_context(asset_position=(100.0, 100.0))
        intent = assessor.infer_intent(track, context)
        assert intent == Intent.INTERCEPT

    def test_low_speed_loitering_infers_surveillance(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(105.0, 105.0),
            velocity=(1.0, 0.0),
        )
        context = _make_context(asset_position=(100.0, 100.0))
        intent = assessor.infer_intent(track, context)
        assert intent == Intent.SURVEILLANCE

    def test_high_altitude_linear_infers_reconnaissance(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(500.0, 500.0),
            velocity=(200.0, 0.0),
            altitude=30000.0,
        )
        context = _make_context(asset_position=(100.0, 100.0))
        intent = assessor.infer_intent(track, context)
        assert intent == Intent.RECONNAISSANCE

    def test_stationary_infers_communication(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(200.0, 200.0),
            velocity=(0.0, 0.0),
        )
        context = _make_context(asset_position=(100.0, 100.0))
        intent = assessor.infer_intent(track, context)
        assert intent == Intent.COMMUNICATION

    def test_erratic_approaching_infers_attack(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(50.0, 50.0),
            velocity=(300.0, 300.0),
            history=[
                {"time": 0, "position": (0.0, 0.0)},
                {"time": 1, "position": (100.0, 0.0)},
                {"time": 2, "position": (100.0, 100.0)},
                {"time": 3, "position": (0.0, 100.0)},
                {"time": 4, "position": (50.0, 50.0)},
            ],
        )
        context = _make_context(asset_position=(100.0, 100.0))
        intent = assessor.infer_intent(track, context)
        assert intent == Intent.ATTACK

    def test_departing_track_infers_departure(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(500.0, 500.0),
            velocity=(50.0, 50.0),
        )
        context = _make_context(asset_position=(100.0, 100.0))
        intent = assessor.infer_intent(track, context)
        assert intent == Intent.DEPARTURE


# ── ThreatAssessor: behavior pattern ────────────────────────────────


class TestBehaviorPattern:
    def test_stationary_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(velocity=(0.0, 0.0))
        context = _make_context()
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.STATIONARY

    def test_linear_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(200.0, 500.0),
            velocity=(100.0, 0.0),
            history=[
                {"time": 0, "position": (0.0, 500.0)},
                {"time": 1, "position": (100.0, 500.0)},
                {"time": 2, "position": (200.0, 500.0)},
            ],
        )
        context = _make_context(asset_position=(200.0, 0.0))
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.LINEAR

    def test_approaching_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(150.0, 150.0),
            velocity=(-10.0, -10.0),
            history=[
                {"time": 0, "position": (300.0, 300.0)},
                {"time": 1, "position": (250.0, 250.0)},
                {"time": 2, "position": (200.0, 200.0)},
                {"time": 3, "position": (150.0, 150.0)},
            ],
        )
        context = _make_context(asset_position=(100.0, 100.0))
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.APPROACHING

    def test_departing_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(200.0, 200.0),
            velocity=(50.0, 50.0),
            history=[
                {"time": 0, "position": (110.0, 110.0)},
                {"time": 1, "position": (130.0, 130.0)},
                {"time": 2, "position": (160.0, 160.0)},
                {"time": 3, "position": (200.0, 200.0)},
            ],
        )
        context = _make_context(asset_position=(100.0, 100.0))
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.DEPARTING

    def test_loitering_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(
            position=(120.0, 120.0),
            velocity=(2.0, 0.0),
            history=[
                {"time": 0, "position": (110.0, 110.0)},
                {"time": 1, "position": (115.0, 118.0)},
                {"time": 2, "position": (120.0, 120.0)},
            ],
        )
        context = _make_context(asset_position=(100.0, 100.0))
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.LOITERING

    def test_orbital_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(
            velocity=(50.0, 50.0),
            history=[
                {"time": 0, "position": (100.0, 0.0)},
                {"time": 1, "position": (70.0, 70.0)},
                {"time": 2, "position": (0.0, 100.0)},
                {"time": 3, "position": (-70.0, 70.0)},
                {"time": 4, "position": (-100.0, 0.0)},
            ],
        )
        context = _make_context(asset_position=(0.0, 0.0))
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.ORBITAL

    def test_evasive_pattern(self):
        assessor = ThreatAssessor()
        track = _make_track(
            velocity=(200.0, 0.0),
            history=[
                {"time": 0, "position": (0.0, 0.0)},
                {"time": 1, "position": (100.0, 0.0)},
                {"time": 2, "position": (100.0, 100.0)},
                {"time": 3, "position": (0.0, 100.0)},
                {"time": 4, "position": (0.0, 0.0)},
            ],
        )
        context = _make_context()
        pattern = assessor.analyze_behavior(track, context)
        assert pattern == BehaviorPattern.EVASIVE


# ── ThreatAssessor: batch assessment ────────────────────────────────


class TestBatchAssessment:
    def test_batch_assess_returns_scores_for_all_tracks(self):
        assessor = ThreatAssessor()
        tracks = [
            _make_track(track_id="TRK-001"),
            _make_track(track_id="TRK-002"),
            _make_track(track_id="TRK-003"),
        ]
        context = _make_context()
        scores = assessor.batch_assess(tracks, context)
        assert len(scores) == 3
        assert all(isinstance(s, ThreatScore) for s in scores)

    def test_batch_assess_empty_list(self):
        assessor = ThreatAssessor()
        context = _make_context()
        scores = assessor.batch_assess([], context)
        assert scores == []


# ── Alert generation ────────────────────────────────────────────────


class TestAlertGeneration:
    def test_generate_alert_returns_alert(self):
        gen = AlertGenerator()
        track = _make_track(track_id="TRK-001")
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        alert = gen.generate(score)
        assert isinstance(alert, Alert)

    def test_alert_has_track_id(self):
        gen = AlertGenerator()
        track = _make_track(track_id="TRK-99")
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        alert = gen.generate(score)
        assert alert.track_id == "TRK-99"

    def test_alert_level_matches_score_level(self):
        gen = AlertGenerator()
        track = _make_track()
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        alert = gen.generate(score)
        assert alert.level == score.level

    def test_alert_has_message(self):
        gen = AlertGenerator()
        track = _make_track()
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        alert = gen.generate(score)
        assert len(alert.message) > 0

    def test_alert_has_recommended_action(self):
        gen = AlertGenerator()
        track = _make_track()
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        alert = gen.generate(score)
        assert len(alert.recommended_action) > 0

    def test_alert_has_timestamp(self):
        gen = AlertGenerator()
        track = _make_track()
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        before = time.time()
        alert = gen.generate(score)
        after = time.time()
        assert before <= alert.timestamp <= after

    def test_alert_has_unique_id(self):
        gen = AlertGenerator()
        track = _make_track()
        context = _make_context()
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        alert1 = gen.generate(score)
        alert2 = gen.generate(score)
        assert alert1.alert_id != alert2.alert_id

    def test_critical_alert_has_urgent_action(self):
        gen = AlertGenerator()
        track = _make_track(
            position=(101.0, 101.0),
            velocity=(500.0, 500.0),
        )
        context = _make_context(asset_position=(100.0, 100.0))
        assessor = ThreatAssessor()
        score = assessor.assess(track, context)
        if score.level == ThreatLevel.CRITICAL:
            alert = gen.generate(score)
            assert "evasive" in alert.recommended_action.lower() or "engage" in alert.recommended_action.lower()


# ── Alert prioritization ─────────────────────────────────────────────


class TestAlertPrioritization:
    def test_prioritize_sorts_by_level_descending(self):
        gen = AlertGenerator()
        assessor = ThreatAssessor()
        context = _make_context()

        low_track = _make_track(track_id="T1", velocity=(0.0, 0.0))
        high_track = _make_track(
            track_id="T2",
            position=(101.0, 101.0),
            velocity=(500.0, 500.0),
        )

        low_score = assessor.assess(low_track, context)
        high_score = assessor.assess(high_track, context)

        alerts = [gen.generate(low_score), gen.generate(high_score)]
        sorted_alerts = gen.prioritize(alerts)
        assert sorted_alerts[0].level.value >= sorted_alerts[1].level.value

    def test_prioritize_empty_list(self):
        gen = AlertGenerator()
        assert gen.prioritize([]) == []


# ── Alert filtering ──────────────────────────────────────────────────


class TestAlertFiltering:
    def test_filter_by_level(self):
        gen = AlertGenerator()
        assessor = ThreatAssessor()
        context = _make_context()

        tracks = [
            _make_track(track_id="T1", velocity=(0.0, 0.0)),
            _make_track(
                track_id="T2",
                position=(101.0, 101.0),
                velocity=(500.0, 500.0),
            ),
            _make_track(
                track_id="T3",
                position=(102.0, 102.0),
                velocity=(300.0, 300.0),
            ),
        ]

        alerts = [gen.generate(assessor.assess(t, context)) for t in tracks]
        high_alerts = gen.filter_by_level(alerts, ThreatLevel.HIGH)
        assert all(a.level.value >= ThreatLevel.HIGH.value for a in high_alerts)

    def test_filter_by_level_empty_result(self):
        gen = AlertGenerator()
        assert gen.filter_by_level([], ThreatLevel.LOW) == []


# ── ThreatScore dataclass ───────────────────────────────────────────


class TestThreatScore:
    def test_threat_score_creation(self):
        score = ThreatScore(
            track_id="TRK-001",
            level=ThreatLevel.HIGH,
            score=0.75,
            intent=Intent.INTERCEPT,
            behavior_pattern=BehaviorPattern.APPROACHING,
            confidence=0.9,
            timestamp=time.time(),
            indicators=[],
        )
        assert score.track_id == "TRK-001"
        assert score.level == ThreatLevel.HIGH
        assert score.score == 0.75
        assert score.intent == Intent.INTERCEPT
        assert score.behavior_pattern == BehaviorPattern.APPROACHING
        assert score.confidence == 0.9

    def test_threat_score_confidence_in_range(self):
        score = ThreatScore(
            track_id="TRK-001",
            level=ThreatLevel.LOW,
            score=0.1,
            intent=Intent.UNKNOWN,
            behavior_pattern=BehaviorPattern.STATIONARY,
            confidence=1.5,
            timestamp=time.time(),
            indicators=[],
        )
        assert 0.0 <= score.confidence <= 1.0


# ── Integration: full pipeline ───────────────────────────────────────


class TestFullPipeline:
    def test_full_pipeline_generates_alerts(self):
        assessor = ThreatAssessor()
        gen = AlertGenerator()
        context = _make_context(asset_position=(100.0, 100.0))

        tracks = [
            _make_track(
                track_id="TRK-001",
                position=(0.0, 0.0),
                velocity=(500.0, 500.0),
            ),
            _make_track(
                track_id="TRK-002",
                position=(500.0, 500.0),
                velocity=(0.0, 0.0),
            ),
            _make_track(
                track_id="TRK-003",
                position=(105.0, 105.0),
                velocity=(5.0, 0.0),
            ),
        ]

        scores = assessor.batch_assess(tracks, context)
        alerts = [gen.generate(s) for s in scores]
        prioritized = gen.prioritize(alerts)

        assert len(prioritized) == 3
        assert prioritized[0].level.value >= prioritized[-1].level.value

    def test_full_pipeline_filters_critical(self):
        assessor = ThreatAssessor()
        gen = AlertGenerator()
        context = _make_context(asset_position=(100.0, 100.0))

        tracks = [
            _make_track(
                track_id="TRK-001",
                position=(101.0, 101.0),
                velocity=(-500.0, -500.0),
            ),
            _make_track(
                track_id="TRK-002",
                position=(500.0, 500.0),
                velocity=(0.0, 0.0),
            ),
        ]

        scores = assessor.batch_assess(tracks, context)
        alerts = [gen.generate(s) for s in scores]
        critical = gen.filter_by_level(alerts, ThreatLevel.HIGH)

        assert len(critical) >= 1
        assert all(a.level.value >= ThreatLevel.HIGH.value for a in critical)
