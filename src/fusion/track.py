"""Track Management — M-of-N initiation, confirmation, deletion, and quality scoring.

This module implements classic M-of-N track management for sensor fusion:
- Track initiation: create tentative track from unassociated detection
- Confirmation: promote to confirmed after M hits out of N opportunities
- Deletion: remove track after consecutive misses exceed threshold
- Quality scoring: compute track quality from hit ratio and consistency
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TrackStatus(Enum):
    """Track lifecycle status."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track:
    """A single track with state and history."""

    track_id: str
    status: TrackStatus
    position: tuple[float, float]
    velocity: tuple[float, float]
    hits: int = 0
    misses: int = 0
    total_updates: int = 0
    created_at: float = 0.0
    last_update: float = 0.0
    quality_score: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrackConfig:
    """Configuration for track management."""

    m_of_n: int = 3  # M hits required to confirm
    n_of_n: int = 5  # out of N opportunities
    max_misses: int = 3  # delete after this many consecutive misses
    min_quality_score: float = 0.3  # minimum quality to confirm
    quality_decay: float = 0.95  # decay factor for quality score
    gating_threshold: float = 100.0  # max distance for association


class TrackManager:
    """Manages track lifecycle: initiation, confirmation, deletion, quality."""

    def __init__(self, config: TrackConfig | None = None):
        self.config = config or TrackConfig()
        self._tracks: dict[str, Track] = {}
        self._next_id: int = 1
        self._lock = threading.RLock()

    def _generate_id(self) -> str:
        track_id = f"TRK-{self._next_id:06d}"
        self._next_id += 1
        return track_id

    def initiate_track(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        timestamp: float | None = None,
    ) -> Track:
        """Create a new tentative track from a detection."""
        now = timestamp if timestamp is not None else time.time()
        track_id = self._generate_id()
        track = Track(
            track_id=track_id,
            status=TrackStatus.TENTATIVE,
            position=position,
            velocity=velocity,
            hits=1,  # First detection counts as first hit
            misses=0,
            total_updates=1,
            created_at=now,
            last_update=now,
            quality_score=0.1,  # Initial quality
            history=[{"time": now, "event": "initiated", "position": position}],
        )
        with self._lock:
            self._tracks[track_id] = track
        return track

    def update_track(
        self,
        track_id: str,
        position: tuple[float, float] | None = None,
        velocity: tuple[float, float] | None = None,
        timestamp: float | None = None,
    ) -> Track | None:
        """Update a track with a new detection (hit) or mark as miss."""
        now = timestamp if timestamp is not None else time.time()

        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return None

            if track.status == TrackStatus.DELETED:
                return None

            if position is not None:
                # Hit
                track.hits += 1
                track.misses = 0
                track.position = position
                if velocity is not None:
                    track.velocity = velocity
                track.total_updates += 1
                track.last_update = now
                track.history.append({"time": now, "event": "hit", "position": position})
            else:
                # Miss
                track.misses += 1
                track.total_updates += 1
                track.last_update = now
                track.history.append({"time": now, "event": "miss"})

            # Update quality score
            track.quality_score = self._calculate_quality_score(track)

            # Check for confirmation
            if track.status == TrackStatus.TENTATIVE:
                if self._should_confirm(track):
                    track.status = TrackStatus.CONFIRMED
                    track.history.append({"time": now, "event": "confirmed"})

            # Check for deletion
            if self._should_delete(track):
                track.status = TrackStatus.DELETED
                track.history.append({"time": now, "event": "deleted"})

            return track

    def confirm_track(self, track_id: str) -> bool:
        """Manually confirm a tentative track."""
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return False
            if track.status == TrackStatus.DELETED:
                return False
            track.status = TrackStatus.CONFIRMED
            track.history.append({"time": time.time(), "event": "manually_confirmed"})
            return True

    def delete_track(self, track_id: str) -> bool:
        """Manually delete a track."""
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return False
            track.status = TrackStatus.DELETED
            track.history.append({"time": time.time(), "event": "manually_deleted"})
            return True

    def get_track(self, track_id: str) -> Track | None:
        """Get a track by ID."""
        return self._tracks.get(track_id)

    def get_all_tracks(self) -> list[Track]:
        """Get all tracks."""
        return list(self._tracks.values())

    def get_confirmed_tracks(self) -> list[Track]:
        """Get all confirmed tracks."""
        return [t for t in self._tracks.values() if t.status == TrackStatus.CONFIRMED]

    def get_tentative_tracks(self) -> list[Track]:
        """Get all tentative tracks."""
        return [t for t in self._tracks.values() if t.status == TrackStatus.TENTATIVE]

    def get_active_tracks(self) -> list[Track]:
        """Get all non-deleted tracks."""
        return [t for t in self._tracks.values() if t.status != TrackStatus.DELETED]

    def process_detection(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        timestamp: float | None = None,
    ) -> tuple[Track, bool]:
        """Process a detection: associate with existing track or create new.

        Returns:
            Tuple of (track, is_new) where is_new is True if a new track was created.
        """
        now = timestamp if timestamp is not None else time.time()

        with self._lock:
            # Try to associate with existing track
            best_track = self._find_best_association(position)
            if best_track is not None:
                self.update_track(best_track.track_id, position, velocity, now)
                return best_track, False

            # Create new track
            track = self.initiate_track(position, velocity, now)
            return track, True

    def _find_best_association(self, position: tuple[float, float]) -> Track | None:
        """Find the best track to associate with a detection."""
        best_track = None
        best_distance = float("inf")

        for track in self._tracks.values():
            if track.status == TrackStatus.DELETED:
                continue
            dist = self._distance(track.position, position)
            if dist < best_distance:
                best_distance = dist
                best_track = track

        # Only associate if within gating threshold
        if best_distance < self.config.gating_threshold:
            return best_track
        return None

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _calculate_quality_score(self, track: Track) -> float:
        """Calculate quality score based on hit ratio and consistency."""
        if track.total_updates == 0:
            return 0.0

        hit_ratio = track.hits / track.total_updates

        # Consistency bonus: penalize high miss ratio
        consistency = 1.0 - (track.misses / max(track.total_updates, 1))

        # Age bonus: older tracks get higher quality
        age_factor = min(1.0, track.hits / self.config.m_of_n)

        # Combined score
        score = 0.5 * hit_ratio + 0.3 * consistency + 0.2 * age_factor

        # Apply decay for recent misses
        if track.misses > 0:
            score *= self.config.quality_decay ** track.misses

        return max(0.0, min(1.0, score))

    def _should_confirm(self, track: Track) -> bool:
        """Check if track meets M-of-N confirmation criterion."""
        if track.hits >= self.config.m_of_n:
            if track.quality_score >= self.config.min_quality_score:
                return True
        return False

    def _should_delete(self, track: Track) -> bool:
        """Check if track should be deleted."""
        if track.misses >= self.config.max_misses:
            return True
        return False

    def reset(self) -> None:
        """Clear all tracks."""
        with self._lock:
            self._tracks.clear()
            self._next_id = 1

    def get_stats(self) -> dict[str, Any]:
        """Get track management statistics."""
        all_tracks = list(self._tracks.values())
        return {
            "total": len(all_tracks),
            "tentative": sum(1 for t in all_tracks if t.status == TrackStatus.TENTATIVE),
            "confirmed": sum(1 for t in all_tracks if t.status == TrackStatus.CONFIRMED),
            "deleted": sum(1 for t in all_tracks if t.status == TrackStatus.DELETED),
        }
