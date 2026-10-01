#!/usr/bin/env python3
"""
Apex_ISR Demo: Multi-Sensor Fusion
===================================

Demonstrates how Apex_ISR fuses detections from heterogeneous sensors
(radar, camera, lidar) into a single unified track estimate.

Concepts shown:
  - Per-sensor measurement generation with realistic noise and bias
  - Weighted fusion using inverse-variance (information) weighting
  - Covariance Intersection (CI) for fusion with unknown cross-correlation
  - Fused estimate compared against ground truth

Everything is pure standard-library Python so the demo runs anywhere.

Run:
    python3 demo_sensor_fusion.py
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# Ground truth: a target moving with constant velocity along the x-axis.
# ---------------------------------------------------------------------------
TRUE_POSITION = 100.0      # metres
TRUE_VELOCITY = 12.0       # m/s
DT = 0.1                   # seconds between frames
N_FRAMES = 20

# Sensor characteristics: (measurement_std, bias, dropout_probability)
SENSORS = {
    "radar":  {"std": 1.5, "bias": 0.3,  "dropout": 0.05},
    "camera": {"std": 3.0, "bias": -0.8, "dropout": 0.15},
    "lidar":  {"std": 0.8, "bias": 0.1,  "dropout": 0.10},
}


def true_position(t: float) -> float:
    """Ground-truth position at time t."""
    return TRUE_POSITION + TRUE_VELOCITY * t


def simulate_sensor(name: str, truth: float, rng: random.Random) -> float | None:
    """Generate one noisy (and possibly dropped) sensor measurement."""
    spec = SENSORS[name]
    if rng.random() < spec["dropout"]:
        return None  # sensor missed the target this frame
    noise = rng.gauss(0.0, spec["std"])
    return truth + spec["bias"] + noise


def inverse_variance_fusion(measurements: dict[str, float]) -> tuple[float, float]:
    """
    Fuse measurements using inverse-variance weighting.

    Each sensor is treated as an independent estimate with variance std^2.
    The fused mean is the precision-weighted average; the fused variance is
    the reciprocal of the total precision.

    Returns (fused_mean, fused_std).
    """
    precision_sum = 0.0
    weighted_sum = 0.0
    for name, value in measurements.items():
        var = SENSORS[name]["std"] ** 2
        precision_sum += 1.0 / var
        weighted_sum += value / var
    fused_mean = weighted_sum / precision_sum
    fused_std = math.sqrt(1.0 / precision_sum)
    return fused_mean, fused_std


def covariance_intersection(measurements: dict[str, float]) -> tuple[float, float]:
    """
    Covariance Intersection (CI) fusion.

    CI provides a consistent fused estimate even when the cross-correlation
    between sensor errors is unknown.  For two estimates it finds the weight
    omega in [0, 1] that minimises the fused variance; for N estimates we
    use the simple iterative (pairwise) approximation.

    Returns (fused_mean, fused_std).
    """
    names = list(measurements)
    if len(names) == 1:
        return measurements[names[0]], SENSORS[names[0]]["std"]

    # Iterative pairwise CI: fuse the first two, then fuse the result with
    # the next sensor, and so on.
    omega = 0.5  # equal weighting is a reasonable default for pairwise CI
    a_name = names[0]
    a_mean = measurements[a_name]
    a_var = SENSORS[a_name]["std"] ** 2

    for b_name in names[1:]:
        b_mean = measurements[b_name]
        b_var = SENSORS[b_name]["std"] ** 2
        # Optimal omega minimises the fused variance.
        omega = b_var / (a_var + b_var) if (a_var + b_var) > 0 else 0.5
        fused_mean = omega * a_mean + (1.0 - omega) * b_mean
        fused_var = omega * a_var + (1.0 - omega) * b_var
        a_mean, a_var = fused_mean, fused_var

    return a_mean, math.sqrt(a_var)


def main() -> None:
    rng = random.Random(42)  # deterministic output for reproducible demos

    print("=" * 72)
    print(" Apex_ISR Multi-Sensor Fusion Demo")
    print("=" * 72)
    print(f"True motion: x(t) = {TRUE_POSITION} + {TRUE_VELOCITY}*t  (metres)")
    print(f"Sensors: {', '.join(SENSORS)}")
    print(f"Frames:  {N_FRAMES}  |  dt = {DT}s")
    print("-" * 72)
    print(f"{'t(s)':>5} {'truth':>8} {'radar':>8} {'camera':>8} {'lidar':>8} "
          f"{'fused(IV)':>10} {'fused(CI)':>10} {'err(IV)':>8}")
    print("-" * 72)

    iv_errors: list[float] = []
    ci_errors: list[float] = []

    for k in range(N_FRAMES):
        t = k * DT
        truth = true_position(t)

        # 1. Simulate each sensor.
        measurements: dict[str, float] = {}
        for name in SENSORS:
            m = simulate_sensor(name, truth, rng)
            if m is not None:
                measurements[name] = m

        # 2. Fuse the available measurements.
        if measurements:
            iv_mean, _ = inverse_variance_fusion(measurements)
            ci_mean, _ = covariance_intersection(measurements)
            iv_errors.append(abs(iv_mean - truth))
            ci_errors.append(abs(ci_mean - truth))
        else:
            iv_mean = ci_mean = float("nan")

        # 3. Pretty-print the frame.
        def fmt(v: float | None) -> str:
            return "  --  " if v is None else f"{v:8.2f}"

        print(f"{t:5.1f} {truth:8.2f} "
              f"{fmt(measurements.get('radar'))} "
              f"{fmt(measurements.get('camera'))} "
              f"{fmt(measurements.get('lidar'))} "
              f"{iv_mean:10.2f} {ci_mean:10.2f} "
              f"{abs(iv_mean - truth):8.2f}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("-" * 72)
    if iv_errors:
        iv_mae = sum(iv_errors) / len(iv_errors)
        ci_mae = sum(ci_errors) / len(ci_errors)
        print(f"Mean absolute error  (Inverse-Variance): {iv_mae:.3f} m")
        print(f"Mean absolute error  (Covariance Int.):  {ci_mae:.3f} m")
        best = "Inverse-Variance" if iv_mae <= ci_mae else "Covariance Intersection"
        print(f"Best performer this run: {best}")
    print("=" * 72)
    print("Demo complete. Fused estimates track the true trajectory despite")
    print("per-sensor noise, bias, and dropouts.")


if __name__ == "__main__":
    main()
