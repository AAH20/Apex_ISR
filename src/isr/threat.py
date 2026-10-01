"""Threat Assessment Engine — scoring, intent inference, behavior patterns, alerts.

This module provides:
- Threat scoring: weighted multi-factor threat level computation
- Intent inference: classify track intent from kinematic and contextual cues
- Behavior pattern analysis: detect movement patterns (linear, orbital, evasive, etc.)
- Alert generation: produce actionable alerts with recommended responses
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class ThreatLevel(IntEnum):
    """Threat severity levels."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Intent(Enum):
    """Inferred intent of a tracked object."""

    UNKNOWN = "unknown"
    INTERCEPT = "intercept"
    SURVEILLANCE = "surveillance"
    RECONNAISSANCE = "reconnaissance"
    COMMUNICATION = "communication"
    ATTACK = "attack"
    DEPARTURE = "departure"


class BehaviorPattern(Enum):
    """Detected behavior pattern of a tracked object."""

    STATIONARY = "stationary"
    LINEAR = "linear"
    APPROACHING = "approaching"
    DEPARTING = "departing"
    LOITERING = "loitering"
    ORBITAL = "orbital"
    EVASIVE = "evasive"
    UNKNOWN = "unknown"


@dataclass
class ThreatConfig:
    """Configuration for threat assessment weights and thresholds."""

    # Factor weights (should sum to ~1.0)
    speed_weight: float = 0.20
    proximity_weight: float = 0.25
    altitude_weight: float = 0.10
    heading_weight: float = 0.15
    maneuver_weight: float = 0.10
    maturity_weight: float = 0.10
    size_weight: float = 0.10

    # Thresholds for threat levels
    low_threshold: float = 0.2
    medium_threshold: float = 0.4
    high_threshold: float = 0.6
    critical_threshold: float = 0.8

    # Speed reference (m/s) — above this is considered high threat
    high_speed_threshold: float = 300.0
    max_speed_reference: float = 800.0

    # Proximity reference (m) — inside this is considered close
    close_proximity_threshold: float = 50.0
    max_proximity_reference: float = 1000.0

    # Altitude reference (m)
    high_altitude_threshold: float = 20000.0

    # Maneuver detection
    maneuver_angle_threshold: float = 45.0  # degrees

    # Loitering detection
    loitering_speed_threshold: float = 10.0
    loitering_radius_threshold: float = 100.0

    # Evasive detection
    evasive_direction_changes: int = 3
    evasive_time_window: int = 5  # number of history points

    # Confirmation maturity
    confirmed_hit_threshold: int = 5


@dataclass
class ThreatScore:
    """Result of a threat assessment."""

    track_id: str
    level: ThreatLevel
    score: float
    intent: Intent
    behavior_pattern: BehaviorPattern
    confidence: float
    timestamp: float
    indicators: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Alert:
    """Generated threat alert."""

    alert_id: str
    track_id: str
    level: ThreatLevel
    message: str
    recommended_action: str
    timestamp: float
    score: float = 0.0
    intent: Intent = Intent.UNKNOWN
    behavior_pattern: BehaviorPattern = BehaviorPattern.UNKNOWN


class ThreatAssessor:
    """Assesses threat level of tracked objects."""

    def __init__(self, config: ThreatConfig | None = None):
        self.config = config or ThreatConfig()

    def assess(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> ThreatScore:
        """Assess the threat level of a single track."""
        track_id = track.get("track_id", "UNKNOWN")
        position = track.get("position", (0.0, 0.0))
        velocity = track.get("velocity", (0.0, 0.0))
        altitude = track.get("altitude", 0.0)
        status = track.get("status", "unknown")
        hits = track.get("hits", 0)
        history = track.get("history", [])

        asset_position = context.get("asset_position", (0.0, 0.0))
        asset_radius = context.get("asset_radius", 50.0)

        # Compute individual factors
        speed_factor = self._compute_speed_factor(velocity)
        proximity_factor = self._compute_proximity_factor(
            position, asset_position, asset_radius
        )
        altitude_factor = self._compute_altitude_factor(altitude)
        heading_factor = self._compute_heading_factor(position, velocity, asset_position)
        maneuver_factor = self._compute_maneuver_factor(history)
        maturity_factor = self._compute_maturity_factor(status, hits)
        size_factor = self._compute_size_factor(track)

        # Weighted sum
        score = (
            self.config.speed_weight * speed_factor
            + self.config.proximity_weight * proximity_factor
            + self.config.altitude_weight * altitude_factor
            + self.config.heading_weight * heading_factor
            + self.config.maneuver_weight * maneuver_factor
            + self.config.maturity_weight * maturity_factor
            + self.config.size_weight * size_factor
        )
        score = max(0.0, min(1.0, score))

        level = self._score_to_level(score)
        intent = self.infer_intent(track, context)
        behavior = self.analyze_behavior(track, context)
        confidence = self._compute_confidence(track, history)
        indicators = self._collect_indicators(
            speed_factor, proximity_factor, altitude_factor,
            heading_factor, maneuver_factor, maturity_factor,
        )

        return ThreatScore(
            track_id=track_id,
            level=level,
            score=score,
            intent=intent,
            behavior_pattern=behavior,
            confidence=confidence,
            timestamp=time.time(),
            indicators=indicators,
        )

    def batch_assess(
        self,
        tracks: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[ThreatScore]:
        """Assess threat for multiple tracks."""
        return [self.assess(t, context) for t in tracks]

    def infer_intent(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> Intent:
        """Infer the intent of a tracked object."""
        position = track.get("position", (0.0, 0.0))
        velocity = track.get("velocity", (0.0, 0.0))
        altitude = track.get("altitude", 0.0)
        history = track.get("history", [])

        asset_position = context.get("asset_position", (0.0, 0.0))

        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        distance = self._distance(position, asset_position)

        # Check for erratic/evasive behavior first
        if self._is_evasive(history) and distance < self.config.close_proximity_threshold * 2:
            return Intent.ATTACK

        # High speed approaching
        if speed > self.config.high_speed_threshold:
            if self._is_approaching(position, velocity, asset_position):
                return Intent.INTERCEPT
            return Intent.RECONNAISSANCE

        # Low speed near asset
        if speed < self.config.loitering_speed_threshold:
            if distance < self.config.close_proximity_threshold * 2:
                return Intent.SURVEILLANCE
            return Intent.COMMUNICATION

        # High altitude
        if altitude > self.config.high_altitude_threshold:
            return Intent.RECONNAISSANCE

        # Medium speed approaching
        if self._is_approaching(position, velocity, asset_position):
            return Intent.INTERCEPT

        # Departing
        if self._is_departing(position, velocity, asset_position):
            return Intent.DEPARTURE

        return Intent.UNKNOWN

    def analyze_behavior(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> BehaviorPattern:
        """Analyze the behavior pattern of a tracked object."""
        velocity = track.get("velocity", (0.0, 0.0))
        history = track.get("history", [])
        position = track.get("position", (0.0, 0.0))

        asset_position = context.get("asset_position", (0.0, 0.0))

        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        # Stationary
        if speed < 1.0:
            return BehaviorPattern.STATIONARY

        # Need history for pattern detection
        if len(history) < 3:
            if self._is_approaching(position, velocity, asset_position):
                return BehaviorPattern.APPROACHING
            if self._is_departing(position, velocity, asset_position):
                return BehaviorPattern.DEPARTING
            return BehaviorPattern.LINEAR

        # Check for evasive
        if self._is_evasive(history):
            return BehaviorPattern.EVASIVE

        # Check for orbital
        if self._is_orbital(history, asset_position):
            return BehaviorPattern.ORBITAL

        # Check for loitering
        if self._is_loitering(history, speed):
            return BehaviorPattern.LOITERING

        # Check approaching/departing
        if self._is_approaching(position, velocity, asset_position):
            return BehaviorPattern.APPROACHING
        if self._is_departing(position, velocity, asset_position):
            return BehaviorPattern.DEPARTING

        # Default to linear
        return BehaviorPattern.LINEAR

    def _score_to_level(self, score: float) -> ThreatLevel:
        """Convert a numeric score to a threat level."""
        if score >= self.config.critical_threshold:
            return ThreatLevel.CRITICAL
        if score >= self.config.high_threshold:
            return ThreatLevel.HIGH
        if score >= self.config.medium_threshold:
            return ThreatLevel.MEDIUM
        if score >= self.config.low_threshold:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _compute_speed_factor(self, velocity: tuple[float, float]) -> float:
        """Compute threat factor from speed."""
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        if speed <= 0:
            return 0.0
        return min(1.0, speed / self.config.max_speed_reference)

    def _compute_proximity_factor(
        self,
        position: tuple[float, float],
        asset_position: tuple[float, float],
        asset_radius: float,
    ) -> float:
        """Compute threat factor from proximity to asset."""
        distance = self._distance(position, asset_position)
        effective_distance = max(0.0, distance - asset_radius)
        if effective_distance >= self.config.max_proximity_reference:
            return 0.0
        return 1.0 - (effective_distance / self.config.max_proximity_reference)

    def _compute_altitude_factor(self, altitude: float) -> float:
        """Compute threat factor from altitude."""
        if altitude <= 0:
            return 0.0
        return min(1.0, altitude / self.config.high_altitude_threshold)

    def _compute_heading_factor(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float],
        asset_position: tuple[float, float],
    ) -> float:
        """Compute threat factor from heading toward asset."""
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        if speed < 1.0:
            return 0.0

        # Direction to asset
        dx = asset_position[0] - position[0]
        dy = asset_position[1] - position[1]
        dist_to_asset = math.sqrt(dx ** 2 + dy ** 2)
        if dist_to_asset < 1.0:
            return 1.0

        # Normalize
        dx /= dist_to_asset
        dy /= dist_to_asset

        # Velocity direction
        vx = velocity[0] / speed
        vy = velocity[1] / speed

        # Dot product: 1 = directly toward, -1 = directly away
        dot = vx * dx + vy * dy

        # Map [-1, 1] to [0, 1]
        return max(0.0, (dot + 1.0) / 2.0)

    def _compute_maneuver_factor(self, history: list[dict[str, Any]]) -> float:
        """Compute threat factor from maneuvering behavior."""
        if len(history) < 3:
            return 0.0

        direction_changes = 0
        for i in range(2, len(history)):
            p0 = history[i - 2].get("position", (0.0, 0.0))
            p1 = history[i - 1].get("position", (0.0, 0.0))
            p2 = history[i].get("position", (0.0, 0.0))

            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])

            speed1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            speed2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if speed1 < 1.0 or speed2 < 1.0:
                continue

            # Angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = max(-1.0, min(1.0, dot / (speed1 * speed2)))
            angle = math.degrees(math.acos(cos_angle))

            if angle > self.config.maneuver_angle_threshold:
                direction_changes += 1

        return min(1.0, direction_changes / 3.0)

    def _compute_maturity_factor(self, status: str, hits: int) -> float:
        """Compute threat factor from track maturity."""
        if status == "confirmed":
            return min(1.0, hits / self.config.confirmed_hit_threshold)
        if status == "tentative":
            return 0.3
        return 0.1

    def _compute_size_factor(self, track: dict[str, Any]) -> float:
        """Compute threat factor from object size."""
        size = track.get("size", 0.0)
        if size <= 0:
            return 0.0
        # Normalize: assume 100m is large
        return min(1.0, size / 100.0)

    def _compute_confidence(
        self,
        track: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> float:
        """Compute confidence in the assessment."""
        hits = track.get("hits", 0)
        total = track.get("total_updates", 0)

        if total == 0:
            return 0.0

        # Base confidence from hit ratio
        hit_ratio = hits / total

        # Bonus for history depth
        history_bonus = min(0.2, len(history) * 0.02)

        confidence = hit_ratio * 0.8 + history_bonus
        return max(0.0, min(1.0, confidence))

    def _collect_indicators(
        self,
        speed_factor: float,
        proximity_factor: float,
        altitude_factor: float,
        heading_factor: float,
        maneuver_factor: float,
        maturity_factor: float,
    ) -> list[str]:
        """Collect human-readable threat indicators."""
        indicators = []
        if speed_factor > 0.7:
            indicators.append("high_speed")
        if proximity_factor > 0.7:
            indicators.append("close_proximity")
        if altitude_factor > 0.7:
            indicators.append("high_altitude")
        if heading_factor > 0.8:
            indicators.append("heading_toward_asset")
        if maneuver_factor > 0.5:
            indicators.append("maneuvering")
        if maturity_factor > 0.7:
            indicators.append("confirmed_track")
        return indicators

    def _distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _is_approaching(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float],
        asset_position: tuple[float, float],
    ) -> bool:
        """Check if track is approaching the asset."""
        dx = asset_position[0] - position[0]
        dy = asset_position[1] - position[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 1.0:
            return True

        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        if speed < 1.0:
            return False

        # Dot product of velocity and direction to asset
        dot = velocity[0] * dx + velocity[1] * dy
        return dot > 0

    def _is_departing(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float],
        asset_position: tuple[float, float],
    ) -> bool:
        """Check if track is departing from the asset."""
        dx = asset_position[0] - position[0]
        dy = asset_position[1] - position[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 1.0:
            return False

        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        if speed < 1.0:
            return False

        dot = velocity[0] * dx + velocity[1] * dy
        return dot < 0

    def _is_evasive(self, history: list[dict[str, Any]]) -> bool:
        """Check if track shows evasive behavior."""
        if len(history) < self.config.evasive_time_window:
            return False

        recent = history[-self.config.evasive_time_window:]
        direction_changes = 0

        for i in range(2, len(recent)):
            p0 = recent[i - 2].get("position", (0.0, 0.0))
            p1 = recent[i - 1].get("position", (0.0, 0.0))
            p2 = recent[i].get("position", (0.0, 0.0))

            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])

            speed1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            speed2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if speed1 < 1.0 or speed2 < 1.0:
                continue

            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = max(-1.0, min(1.0, dot / (speed1 * speed2)))
            angle = math.degrees(math.acos(cos_angle))

            if angle > self.config.maneuver_angle_threshold:
                direction_changes += 1

        return direction_changes >= self.config.evasive_direction_changes

    def _is_orbital(
        self,
        history: list[dict[str, Any]],
        center: tuple[float, float],
    ) -> bool:
        """Check if track shows orbital behavior around a center point."""
        if len(history) < 5:
            return False

        # Check if distance from center is roughly constant
        distances = []
        for point in history:
            pos = point.get("position", (0.0, 0.0))
            distances.append(self._distance(pos, center))

        if len(distances) < 3:
            return False

        avg_dist = sum(distances) / len(distances)
        if avg_dist < 1.0:
            return False

        # Check variance
        variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
        cv = math.sqrt(variance) / avg_dist  # Coefficient of variation

        # Low CV means roughly constant distance = orbital
        return cv < 0.3

    def _is_loitering(
        self,
        history: list[dict[str, Any]],
        speed: float,
    ) -> bool:
        """Check if track is loitering (low speed, small area)."""
        if len(history) < 3:
            return False

        if speed > self.config.loitering_speed_threshold * 2:
            return False

        # Check if all points are within a small radius
        positions = [p.get("position", (0.0, 0.0)) for p in history]
        if not positions:
            return False

        # Compute centroid
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)

        # Check max distance from centroid
        max_dist = max(
            self._distance(p, (cx, cy)) for p in positions
        )

        return max_dist < self.config.loitering_radius_threshold


class AlertGenerator:
    """Generates alerts from threat assessments."""

    def __init__(self):
        self._alert_counter = 0

    def generate(self, score: ThreatScore) -> Alert:
        """Generate an alert from a threat score."""
        self._alert_counter += 1
        alert_id = f"ALERT-{self._alert_counter:06d}-{uuid.uuid4().hex[:8]}"

        message = self._compose_message(score)
        action = self._recommend_action(score)

        return Alert(
            alert_id=alert_id,
            track_id=score.track_id,
            level=score.level,
            message=message,
            recommended_action=action,
            timestamp=time.time(),
            score=score.score,
            intent=score.intent,
            behavior_pattern=score.behavior_pattern,
        )

    def prioritize(self, alerts: list[Alert]) -> list[Alert]:
        """Sort alerts by threat level (highest first), then by score."""
        return sorted(alerts, key=lambda a: (a.level.value, a.score), reverse=True)

    def filter_by_level(
        self,
        alerts: list[Alert],
        min_level: ThreatLevel,
    ) -> list[Alert]:
        """Filter alerts by minimum threat level."""
        return [a for a in alerts if a.level.value >= min_level.value]

    def _compose_message(self, score: ThreatScore) -> str:
        """Compose a human-readable alert message."""
        parts = [
            f"Threat {score.level.name}",
            f"Track {score.track_id}",
            f"Score {score.score:.2f}",
        ]

        if score.intent != Intent.UNKNOWN:
            parts.append(f"Intent: {score.intent.value}")

        if score.behavior_pattern != BehaviorPattern.UNKNOWN:
            parts.append(f"Pattern: {score.behavior_pattern.value}")

        if score.indicators:
            parts.append(f"Indicators: {', '.join(score.indicators)}")

        return " | ".join(parts)

    def _recommend_action(self, score: ThreatScore) -> str:
        """Recommend an action based on threat level and intent."""
        if score.level == ThreatLevel.CRITICAL:
            if score.intent == Intent.ATTACK:
                return "Engage countermeasures immediately; alert command authority"
            return "Evasive action required; engage defensive systems"

        if score.level == ThreatLevel.HIGH:
            if score.intent == Intent.INTERCEPT:
                return "Prepare defensive posture; track continuously"
            return "Increase tracking frequency; notify supervisor"

        if score.level == ThreatLevel.MEDIUM:
            return "Monitor closely; prepare response options"

        if score.level == ThreatLevel.LOW:
            return "Continue routine monitoring"

        return "No action required"
