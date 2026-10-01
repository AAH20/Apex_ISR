#!/usr/bin/env python3
"""
Apex_ISR Demo: Track Lifecycle Management
==========================================

Demonstrates the full lifecycle of a track in Apex_ISR's multi-target
tracking (MTT) pipeline:

    TENTATIVE ──(N consecutive hits)──▶ CONFIRMED ──(M misses)──▶ DELETED
        ▲                                   │
        └────────── (re-acquisition) ───────┘

Concepts shown:
  - Track initiation from unassociated detections
  - Track promotion (tentative → confirmed) after enough hits
  - Track maintenance: update on hit, coast (predict-only) on miss
  - Track deletion after too many consecutive misses
  - Nearest-neighbour data association with a gating threshold

Everything is pure standard-library Python so the demo runs anywhere.

Run:
    python3 demo_track_lifecycle.py
"""

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Lifecycle tuning parameters (mirrors Apex_ISR's tracker config).
# ---------------------------------------------------------------------------
PROMOTE_AFTER_HITS = 3      # consecutive hits needed to confirm a track
DELETE_AFTER_MISSES = 4     # consecutive misses before deletion
GATE_THRESHOLD = 5.0        # metres; detections beyond this are not associated
DT = 0.5                    # seconds between frames
N_FRAMES = 30


# ---------------------------------------------------------------------------
# Track model
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Track:
    """A single target track with a simple constant-velocity model."""

    track_id: int
    state: str = "TENTATIVE"     # TENTATIVE | CONFIRMED | DELETED
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 0
    misses: int = 0
    age: int = 0                 # frames since initiation
    history: list[str] = field(default_factory=list)

    def predict(self) -> None:
        """Coast the track forward one time step (constant velocity)."""
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.age += 1

    def update(self, zx: float, zy: float) -> None:
        """
        Update the track with an associated detection.

        A simple alpha-beta filter blends the detection into the state;
        velocity is estimated from the position delta.
        """
        alpha = 0.7   # position gain
        beta = 0.4    # velocity gain
        dx = zx - self.x
        dy = zy - self.y
        self.vx = (1 - beta) * self.vx + beta * dx / DT
        self.vy = (1 - beta) * self.vy + beta * dy / DT
        self.x = (1 - alpha) * self.x + alpha * zx
        self.y = (1 - alpha) * self.y + alpha * zy
        self.hits += 1
        self.misses = 0

    def mark_miss(self) -> None:
        """Record a frame with no associated detection."""
        self.misses += 1
        self.predict()  # coast

    def transition(self) -> None:
        """Apply lifecycle state transitions based on hit/miss counts."""
        if self.state == "TENTATIVE" and self.hits >= PROMOTE_AFTER_HITS:
            self.state = "CONFIRMED"
        elif self.state in ("TENTATIVE", "CONFIRMED") and \
                self.misses >= DELETE_AFTER_MISSES:
            self.state = "DELETED"

    def record(self) -> None:
        """Append the current state to the track's history log."""
        self.history.append(self.state)


# ---------------------------------------------------------------------------
# Track manager
# ---------------------------------------------------------------------------
class TrackManager:
    """Owns all tracks and runs association + lifecycle each frame."""

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self._next_id = 1

    def initiate(self, zx: float, zy: float) -> Track:
        """Create a new tentative track from an unassociated detection."""
        t = Track(track_id=self._next_id, x=zx, y=zy)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def associate(self, detections: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
        """
        Nearest-neighbour association with a distance gate.

        Returns a mapping track_id -> detection for accepted associations.
        """
        associations: dict[int, tuple[float, float]] = {}
        used_detections: set[int] = set()

        # Sort candidate pairs by distance so the closest pairs win first.
        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            if track.state == "DELETED":
                continue
            for di, (zx, zy) in enumerate(detections):
                d = math.hypot(zx - track.x, zy - track.y)
                if d <= GATE_THRESHOLD:
                    candidates.append((d, ti, di))
        candidates.sort()

        for d, ti, di in candidates:
            track = self.tracks[ti]
            if track.track_id in associations or di in used_detections:
                continue
            associations[track.track_id] = detections[di]
            used_detections.add(di)

        return associations

    def process_frame(self, detections: list[tuple[float, float]]) -> None:
        """Run one full cycle: associate, update, coast, transition, prune."""
        associations = self.associate(detections)

        for track in self.tracks:
            if track.state == "DELETED":
                continue
            if track.track_id in associations:
                zx, zy = associations[track.track_id]
                track.update(zx, zy)
            else:
                track.mark_miss()
            track.transition()
            track.record()

        # Remove deleted tracks from the active list.
        self.tracks = [t for t in self.tracks if t.state != "DELETED"]

        # Initiate new tentative tracks from leftover detections.
        associated_dets = set(associations.values())
        for det in detections:
            if det not in associated_dets:
                self.initiate(*det)


# ---------------------------------------------------------------------------
# Scenario: two targets crossing, with dropouts and a false alarm.
# ---------------------------------------------------------------------------
def build_scenario(rng: random.Random) -> list[list[tuple[float, float]]]:
    """
    Generate ground-truth trajectories and per-frame detections.

    Target A moves left-to-right; target B moves right-to-left.
    Each frame has a chance of dropping one target and a chance of a
    false alarm (clutter).
    """
    frames: list[list[tuple[float, float]]] = []
    ax, ay, avx, avy = 0.0, 0.0, 8.0, 2.0
    bx, by, bvx, bvy = 100.0, 30.0, -6.0, -1.0

    for _ in range(N_FRAMES):
        ax += avx * DT
        ay += avy * DT
        bx += bvx * DT
        by += bvy * DT

        dets: list[tuple[float, float]] = []
        # Target A: 10 % dropout.
        if rng.random() > 0.10:
            dets.append((ax + rng.gauss(0, 0.5), ay + rng.gauss(0, 0.5)))
        # Target B: 20 % dropout.
        if rng.random() > 0.20:
            dets.append((bx + rng.gauss(0, 0.5), by + rng.gauss(0, 0.5)))
        # Clutter: 15 % chance of a false alarm.
        if rng.random() < 0.15:
            dets.append((rng.uniform(0, 100), rng.uniform(0, 40)))

        frames.append(dets)
    return frames


def main() -> None:
    rng = random.Random(7)  # deterministic output for reproducible demos
    frames = build_scenario(rng)
    manager = TrackManager()

    print("=" * 78)
    print(" Apex_ISR Track Lifecycle Demo")
    print("=" * 78)
    print(f"Policy: promote after {PROMOTE_AFTER_HITS} hits | "
          f"delete after {DELETE_AFTER_MISSES} misses | "
          f"gate = {GATE_THRESHOLD} m")
    print(f"Scenario: 2 crossing targets + clutter, {N_FRAMES} frames, "
          f"dt = {DT}s")
    print("-" * 78)
    print(f"{'frame':>5} | {'#det':>4} | {'#tracks':>7} | "
          f"{'TENT':>4} {'CONF':>4} {'DEL':>3} | active track positions")
    print("-" * 78)

    for k, dets in enumerate(frames):
        manager.process_frame(dets)

        n_tent = sum(1 for t in manager.tracks if t.state == "TENTATIVE")
        n_conf = sum(1 for t in manager.tracks if t.state == "CONFIRMED")
        n_del = sum(1 for t in manager.tracks if t.state == "DELETED")
        # Deleted tracks are pruned, so count them from history instead.
        # (We recompute below from the full lifecycle log.)

        positions = "  ".join(
            f"T{t.track_id}({t.x:5.1f},{t.y:5.1f})" for t in manager.tracks
        )
        print(f"{k:5d} | {len(dets):4d} | {len(manager.tracks):7d} | "
              f"{n_tent:4d} {n_conf:4d} {n_del:3d} | {positions}")

    # ------------------------------------------------------------------
    # Lifecycle summary
    # ------------------------------------------------------------------
    print("-" * 78)
    print("Lifecycle summary (all tracks ever created):")
    # Re-run to collect the full history including deleted tracks.
    rng2 = random.Random(7)
    frames2 = build_scenario(rng2)
    manager2 = TrackManager()
    all_tracks: list[Track] = []
    for dets in frames2:
        before = set(manager2.tracks)
        manager2.process_frame(dets)
        for t in manager2.tracks:
            if t not in before:
                all_tracks.append(t)
        # Keep deleted tracks in the log.
        for t in list(manager2.tracks):
            if t.state == "DELETED" and t not in all_tracks:
                all_tracks.append(t)

    # Deduplicate while preserving order.
    seen: set[int] = set()
    unique_tracks: list[Track] = []
    for t in all_tracks:
        if t.track_id not in seen:
            seen.add(t.track_id)
            unique_tracks.append(t)

    for t in unique_tracks:
        transitions = " → ".join(t.history)
        print(f"  Track {t.track_id}: {transitions}")

    n_confirmed = sum(1 for t in unique_tracks
                      if "CONFIRMED" in t.history)
    n_deleted = sum(1 for t in unique_tracks
                     if t.history and t.history[-1] == "DELETED")
    print(f"\nTracks created: {len(unique_tracks)} | "
          f"confirmed: {n_confirmed} | deleted: {n_deleted}")
    print("=" * 78)
    print("Demo complete. Tracks were initiated from unassociated detections,")
    print("promoted to CONFIRMED after enough hits, coasted through dropouts,")
    print("and DELETED after too many consecutive misses.")


if __name__ == "__main__":
    main()
