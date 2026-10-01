"""Conjunction Detection — closest approach, collision probability, CDM generation.

This module provides:
- closest_approach: compute time and distance of closest approach between two objects
- collision_probability_1d/2d: collision probability using 1D and 2D methods
- analyze_conjunction: full conjunction analysis combining all metrics
- generate_cdm: Conjunction Data Message generation (CCSDS-compatible)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def closest_approach(
    r1: np.ndarray,
    v1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    t0: float = 0.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute closest approach between two objects on linear trajectories.

    Args:
        r1: Position of object 1 at epoch t0 (meters)
        v1: Velocity of object 1 (m/s)
        r2: Position of object 2 at epoch t0 (meters)
        v2: Velocity of object 2 (m/s)
        t0: Epoch time (seconds)

    Returns:
        Tuple of (tca, miss_distance, r1_at_tca, r2_at_tca) where:
        - tca: Time of closest approach relative to t0 (seconds, clamped to >= 0)
        - miss_distance: Minimum separation distance (meters)
        - r1_at_tca: Position of object 1 at TCA
        - r2_at_tca: Position of object 2 at TCA
    """
    r1 = np.asarray(r1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    dr = r1 - r2
    dv = v1 - v2

    dv_sq = float(dv @ dv)

    if dv_sq < 1e-20:
        # Relative velocity is zero — distance is constant
        tca_rel = 0.0
    else:
        tca_rel = -float(dr @ dv) / dv_sq
        if tca_rel < 0.0:
            tca_rel = 0.0

    tca = t0 + tca_rel
    r1_ca = r1 + v1 * tca_rel
    r2_ca = r2 + v2 * tca_rel
    miss = float(np.linalg.norm(r1_ca - r2_ca))

    return tca, miss, r1_ca, r2_ca


def collision_probability_1d(
    miss_distance: float,
    sigma1: float,
    sigma2: float,
) -> float:
    """Compute 1D collision probability (simplified Gaussian model).

    Uses the peak probability density of the relative position distribution:
    P = exp(-d² / (2(σ₁² + σ₂²)))

    Args:
        miss_distance: Minimum separation distance (meters)
        sigma1: Position uncertainty of object 1 (1-sigma, meters)
        sigma2: Position uncertainty of object 2 (1-sigma, meters)

    Returns:
        Collision probability in [0, 1]
    """
    sigma_sq = sigma1**2 + sigma2**2
    if sigma_sq < 1e-20:
        return 1.0 if miss_distance < 1e-10 else 0.0
    return float(math.exp(-miss_distance**2 / (2.0 * sigma_sq)))


def collision_probability_2d(
    miss_distance: float,
    sigma1: float,
    sigma2: float,
    hard_body_radius: float,
) -> float:
    """Compute 2D collision probability over a circular hard-body region.

    Integrates the 2D Gaussian relative-position density over a circle of
    radius ``hard_body_radius`` centred at the origin.  Uses the standard
    radial form with the modified Bessel function I₀:

        P = ∫₀ᴿ (r/σ²) exp(-(r²+d²)/(2σ²)) I₀(rd/σ²) dr

    where σ² = σ₁² + σ₂² and d is the miss distance.

    Args:
        miss_distance: Minimum separation distance (meters)
        sigma1: Position uncertainty of object 1 (1-sigma, meters)
        sigma2: Position uncertainty of object 2 (1-sigma, meters)
        hard_body_radius: Combined hard-body radius (meters)

    Returns:
        Collision probability in [0, 1]
    """
    if hard_body_radius <= 0.0:
        return 0.0

    sigma_sq = sigma1**2 + sigma2**2
    if sigma_sq < 1e-20:
        return 1.0 if miss_distance < hard_body_radius else 0.0

    sigma = math.sqrt(sigma_sq)
    R = hard_body_radius
    d = miss_distance

    # Closed form for zero miss distance
    if d < 1e-10:
        return float(1.0 - math.exp(-R**2 / (2.0 * sigma_sq)))

    # Numerical integration using Simpson's rule
    n = 200  # even number of intervals
    h = R / n
    total = 0.0

    for i in range(n + 1):
        r = i * h
        # Integrand: (r/σ²) * exp(-(r²+d²)/(2σ²)) * I₀(rd/σ²)
        arg = r * d / sigma_sq
        # I₀(x) for moderate x — use series or numpy
        i0_val = _i0_series(arg)
        f = (r / sigma_sq) * math.exp(-(r**2 + d**2) / (2.0 * sigma_sq)) * i0_val

        if i == 0 or i == n:
            total += f
        elif i % 2 == 1:
            total += 4.0 * f
        else:
            total += 2.0 * f

    return float(total * h / 3.0)


def _i0_series(x: float, terms: int = 30) -> float:
    """Modified Bessel function I₀(x) via series expansion."""
    total = 1.0
    term = 1.0
    for k in range(1, terms):
        term *= (x / (2.0 * k)) ** 2
        total += term
        if term < 1e-15 * total:
            break
    return total


@dataclass
class OrbitalState:
    """Orbital state of a space object at a given epoch.

    Attributes:
        position: Position vector [x, y, z] in meters (ECI frame)
        velocity: Velocity vector [vx, vy, vz] in m/s (ECI frame)
        covariance: Optional 3×3 position covariance matrix (m²)
        epoch: Epoch time in seconds
        object_id: Optional object identifier
    """
    position: np.ndarray
    velocity: np.ndarray
    covariance: Optional[np.ndarray] = None
    epoch: float = 0.0
    object_id: str = ""


@dataclass
class ConjunctionResult:
    """Result of a conjunction analysis between two objects.

    Attributes:
        tca: Time of closest approach (seconds from epoch)
        miss_distance: Minimum separation distance (meters)
        r1_at_tca: Position of object 1 at TCA (meters)
        r2_at_tca: Position of object 2 at TCA (meters)
        collision_probability: Collision probability [0, 1] or None if not computed
        hard_body_radius: Combined hard-body radius used (meters)
    """
    tca: float
    miss_distance: float
    r1_at_tca: np.ndarray
    r2_at_tca: np.ndarray
    collision_probability: Optional[float] = None
    hard_body_radius: float = 0.0


def analyze_conjunction(
    state1: OrbitalState,
    state2: OrbitalState,
    hard_body_radius: float = 0.0,
) -> ConjunctionResult:
    """Perform full conjunction analysis between two orbital states.

    Computes closest approach, miss distance, and optionally collision
    probability if covariance matrices are provided.

    Args:
        state1: Orbital state of object 1
        state2: Orbital state of object 2
        hard_body_radius: Combined hard-body radius for collision probability (meters)

    Returns:
        ConjunctionResult with TCA, miss distance, and collision probability
    """
    tca, miss, r1_ca, r2_ca = closest_approach(
        state1.position,
        state1.velocity,
        state2.position,
        state2.velocity,
        t0=state1.epoch,
    )

    collision_prob = None
    if state1.covariance is not None and state2.covariance is not None:
        # Extract position covariance (3×3 block)
        cov1 = state1.covariance[:3, :3] if state1.covariance.shape[0] >= 3 else state1.covariance
        cov2 = state2.covariance[:3, :3] if state2.covariance.shape[0] >= 3 else state2.covariance

        # Use trace/3 as scalar sigma for each object
        sigma1 = math.sqrt(float(np.trace(cov1)) / 3.0)
        sigma2 = math.sqrt(float(np.trace(cov2)) / 3.0)

        collision_prob = collision_probability_2d(
            miss, sigma1, sigma2, hard_body_radius
        )

    return ConjunctionResult(
        tca=tca,
        miss_distance=miss,
        r1_at_tca=r1_ca,
        r2_at_tca=r2_ca,
        collision_probability=collision_prob,
        hard_body_radius=hard_body_radius,
    )


@dataclass
class CDM:
    """Conjunction Data Message.

    Attributes:
        object1_id: Identifier of object 1
        object2_id: Identifier of object 2
        tca: Time of closest approach (seconds)
        miss_distance: Minimum separation distance (meters)
        r1_at_tca: Position of object 1 at TCA (meters)
        r2_at_tca: Position of object 2 at TCA (meters)
        collision_probability: Collision probability [0, 1] or None
        hard_body_radius: Combined hard-body radius (meters)
        epoch: Epoch time (seconds)
    """
    object1_id: str
    object2_id: str
    tca: float
    miss_distance: float
    r1_at_tca: np.ndarray
    r2_at_tca: np.ndarray
    collision_probability: Optional[float] = None
    hard_body_radius: float = 0.0
    epoch: float = 0.0


def generate_cdm(
    state1: OrbitalState,
    state2: OrbitalState,
    result: ConjunctionResult,
) -> CDM:
    """Generate a Conjunction Data Message from analysis results.

    Args:
        state1: Orbital state of object 1
        state2: Orbital state of object 2
        result: ConjunctionResult from analyze_conjunction

    Returns:
        CDM dataclass
    """
    return CDM(
        object1_id=state1.object_id,
        object2_id=state2.object_id,
        tca=result.tca,
        miss_distance=result.miss_distance,
        r1_at_tca=result.r1_at_tca,
        r2_at_tca=result.r2_at_tca,
        collision_probability=result.collision_probability,
        hard_body_radius=result.hard_body_radius,
        epoch=state1.epoch,
    )


def cdm_to_dict(cdm: CDM) -> dict[str, Any]:
    """Convert a CDM to a dictionary.

    Args:
        cdm: CDM dataclass

    Returns:
        Dictionary representation
    """
    return {
        "object1_id": cdm.object1_id,
        "object2_id": cdm.object2_id,
        "tca": cdm.tca,
        "miss_distance": cdm.miss_distance,
        "r1_at_tca": cdm.r1_at_tca.tolist(),
        "r2_at_tca": cdm.r2_at_tca.tolist(),
        "collision_probability": cdm.collision_probability,
        "hard_body_radius": cdm.hard_body_radius,
        "epoch": cdm.epoch,
    }


def cdm_to_ccsds(cdm: CDM) -> str:
    """Convert a CDM to a CCSDS-like text format.

    Args:
        cdm: CDM dataclass

    Returns:
        String in CCSDS-like format
    """
    lines = [
        "CCSDS_CONJUNCTION_DATA_MESSAGE",
        "OBJECT1_ID = " + cdm.object1_id,
        "OBJECT2_ID = " + cdm.object2_id,
        "TCA = " + f"{cdm.tca:.6f}",
        "MISS_DISTANCE = " + f"{cdm.miss_distance:.6f}",
        "R1_AT_TCA = [" + ", ".join(f"{v:.3f}" for v in cdm.r1_at_tca) + "]",
        "R2_AT_TCA = [" + ", ".join(f"{v:.3f}" for v in cdm.r2_at_tca) + "]",
        "COLLISION_PROBABILITY = " + (
            f"{cdm.collision_probability:.6e}"
            if cdm.collision_probability is not None
            else "N/A"
        ),
        "HARD_BODY_RADIUS = " + f"{cdm.hard_body_radius:.6f}",
        "EPOCH = " + f"{cdm.epoch:.6f}",
        "END_CDM",
    ]
    return "\n".join(lines)
