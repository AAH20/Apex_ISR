#!/usr/bin/env python3
"""
Apex_ISR Demo: Kalman Filter
==============================

Demonstrates a 1D constant-velocity Kalman filter as used in Apex_ISR's
tracking pipeline to smooth noisy sensor measurements and predict target
state between detections.

Concepts shown:
  - State prediction (time update) using a constant-velocity model
  - Measurement update (correction) blending prediction with observation
  - Covariance evolution: grows during prediction, shrinks during update
  - Comparison of raw measurements vs. filtered estimates vs. ground truth

Everything is pure standard-library Python so the demo runs anywhere.

Run:
    python3 demo_kalman_filter.py
"""

import math
import random

# ---------------------------------------------------------------------------
# Ground truth: a target moving with constant velocity.
# ---------------------------------------------------------------------------
TRUE_POSITION = 50.0       # metres
TRUE_VELOCITY = 4.0        # m/s
DT = 0.2                   # seconds between measurements
N_STEPS = 30

# Process noise (acceleration uncertainty) and measurement noise.
PROCESS_NOISE = 0.5        # m/s^2  (how much the velocity can change)
MEASUREMENT_NOISE = 2.0    # metres (sensor standard deviation)


# ---------------------------------------------------------------------------
# 1D constant-velocity Kalman filter (implemented from scratch).
# ---------------------------------------------------------------------------
class KalmanFilter1D:
    """
    Kalman filter for a 1D constant-velocity model.

    State vector:  x = [position, velocity]^T
    State transition:  x_k = F * x_{k-1} + w,  w ~ N(0, Q)
    Measurement:      z_k = H * x_k + v,       v ~ N(0, R)

    where
        F = [[1, dt], [0, 1]]       (constant velocity)
        H = [[1, 0]]                (we measure position only)
        Q = process noise covariance
        R = measurement noise variance
    """

    def __init__(self, x0: float, v0: float, dt: float,
                 process_noise: float, measurement_noise: float) -> None:
        self.dt = dt
        # State estimate.
        self.x = x0
        self.v = v0
        # State covariance (position, velocity).
        self.P_pos = 10.0   # initial position uncertainty
        self.P_vel = 10.0   # initial velocity uncertainty
        self.P_cross = 0.0  # position-velocity covariance
        # Noise parameters.
        self.q = process_noise
        self.r = measurement_noise

    def predict(self) -> None:
        """Time update: project state and covariance forward."""
        # State prediction (constant velocity).
        self.x = self.x + self.v * self.dt
        # Covariance prediction: P = F P F^T + Q
        dt2 = self.dt * self.dt
        new_P_pos = self.P_pos + 2 * self.dt * self.P_cross \
            + dt2 * self.P_vel + self.q * dt2
        new_P_cross = self.P_cross + self.dt * self.P_vel
        new_P_vel = self.P_vel + self.q
        self.P_pos, self.P_cross, self.P_vel = new_P_pos, new_P_cross, new_P_vel

    def update(self, measurement: float) -> None:
        """Measurement update: correct state with a new observation."""
        # Innovation (residual).
        y = measurement - self.x
        # Innovation covariance.
        S = self.P_pos + self.r
        # Kalman gain.
        K_pos = self.P_pos / S
        K_vel = self.P_cross / S
        # State update.
        self.x = self.x + K_pos * y
        self.v = self.v + K_vel * y
        # Covariance update (Joseph form for numerical stability).
        self.P_pos = (1 - K_pos) * self.P_pos
        self.P_cross = (1 - K_pos) * self.P_cross
        self.P_vel = self.P_vel - K_vel * self.P_cross

    @property
    def position(self) -> float:
        return self.x

    @property
    def velocity(self) -> float:
        return self.v

    @property
    def position_std(self) -> float:
        return math.sqrt(self.P_pos)


def true_state(t: float) -> tuple[float, float]:
    """Ground-truth (position, velocity) at time t."""
    return TRUE_POSITION + TRUE_VELOCITY * t, TRUE_VELOCITY


def simulate_measurement(truth_pos: float, rng: random.Random) -> float:
    """Generate a noisy position measurement."""
    return truth_pos + rng.gauss(0.0, MEASUREMENT_NOISE)


def main() -> None:
    rng = random.Random(123)  # deterministic output for reproducible demos

    # Initialise the filter with a rough guess (deliberately offset).
    kf = KalmanFilter1D(
        x0=TRUE_POSITION + 5.0,   # 5 m off
        v0=0.0,                    # assume stationary
        dt=DT,
        process_noise=PROCESS_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
    )

    print("=" * 78)
    print(" Apex_ISR Kalman Filter Demo (1D Constant-Velocity Model)")
    print("=" * 78)
    print(f"True motion: x(t) = {TRUE_POSITION} + {TRUE_VELOCITY}*t")
    print(f"Process noise q = {PROCESS_NOISE} | Measurement noise "
          f"r = {MEASUREMENT_NOISE}")
    print(f"Initial guess: x0 = {kf.position:.1f} (5 m off), v0 = 0.0")
    print("-" * 78)
    print(f"{'step':>4} {'t(s)':>5} {'truth':>8} {'meas':>8} "
          f"{'est':>8} {'est_v':>7} {'std':>6} {'err':>7}")
    print("-" * 78)

    raw_errors: list[float] = []
    filtered_errors: list[float] = []

    for k in range(N_STEPS):
        t = k * DT
        truth_pos, truth_vel = true_state(t)

        # 1. Simulate a noisy measurement.
        z = simulate_measurement(truth_pos, rng)

        # 2. Predict (time update).
        kf.predict()

        # 3. Update (measurement correction).
        kf.update(z)

        # 4. Log.
        err = abs(kf.position - truth_pos)
        raw_err = abs(z - truth_pos)
        raw_errors.append(raw_err)
        filtered_errors.append(err)

        print(f"{k:4d} {t:5.1f} {truth_pos:8.2f} {z:8.2f} "
              f"{kf.position:8.2f} {kf.velocity:7.2f} "
              f"{kf.position_std:6.2f} {err:7.2f}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("-" * 78)
    raw_mae = sum(raw_errors) / len(raw_errors)
    filt_mae = sum(filtered_errors) / len(filtered_errors)
    improvement = (1 - filt_mae / raw_mae) * 100 if raw_mae > 0 else 0
    print(f"Mean absolute error (raw measurements):  {raw_mae:.3f} m")
    print(f"Mean absolute error (Kalman estimate):   {filt_mae:.3f} m")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Final position std (1-sigma): {kf.position_std:.3f} m")
    print(f"Final velocity estimate: {kf.velocity:.3f} m/s "
          f"(true: {TRUE_VELOCITY:.3f} m/s)")
    print("=" * 78)
    print("Demo complete. The filter converges from a poor initial guess,")
    print("smooths noisy measurements, and estimates velocity even though")
    print("only position is measured directly.")


if __name__ == "__main__":
    main()
