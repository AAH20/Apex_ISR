"""Extended Kalman Filter — non-linear sensor fusion.

This module provides:
- ExtendedKalmanFilter: general-purpose EKF for non-linear systems
- BearingOnlyEKF: specialized EKF for bearing-only sensor fusion
- RangeBearingEKF: specialized EKF for range-bearing sensor fusion
"""

import math
import numpy as np
from typing import Callable, Optional


class ExtendedKalmanFilter:
    """General-purpose Extended Kalman Filter for non-linear systems.

    The EKF linearizes the process and measurement models using Jacobians
    at each time step, then applies standard Kalman filter equations.

    State vector: x (n×1)
    Covariance: P (n×n)
    Process noise: Q (n×n)
    Measurement noise: R (m×m)
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        """Initialize EKF.

        Args:
            x0: Initial state vector (n×1)
            P0: Initial covariance matrix (n×n)
            Q: Process noise covariance (n×n)
            R: Measurement noise covariance (m×m)
        """
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.Q = Q.astype(float).copy()
        self.R = R.astype(float).copy()
        self.n = len(x0)
        self.m = R.shape[0]

    def predict(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        F_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: Optional[np.ndarray] = None,
    ) -> None:
        """Prediction step.

        Propagates state through non-linear process model and updates
        covariance using the Jacobian linearization.

        Args:
            f: Non-linear process model x_{k+1} = f(x_k)
            F_jacobian: Jacobian of f evaluated at current state: F = ∂f/∂x
            Q: Optional override for process noise covariance
        """
        F = F_jacobian(self.x)
        self.x = f(self.x)
        Q_eff = Q if Q is not None else self.Q
        self.P = F @ self.P @ F.T + Q_eff

    def update(
        self,
        z: np.ndarray,
        h: Callable[[np.ndarray], np.ndarray],
        H_jacobian: Callable[[np.ndarray], np.ndarray],
        R: Optional[np.ndarray] = None,
    ) -> None:
        """Update step.

        Incorporates measurement using non-linear measurement model
        and its Jacobian linearization.

        Args:
            z: Measurement vector (m×1)
            h: Non-linear measurement model z = h(x)
            H_jacobian: Jacobian of h evaluated at current state: H = ∂h/∂x
            R: Optional override for measurement noise covariance
        """
        z = z.astype(float)
        H = H_jacobian(self.x)
        R_eff = R if R is not None else self.R

        # Innovation
        y = z - h(self.x)

        # Innovation covariance
        S = H @ self.P @ H.T + R_eff

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T

    def predict_update(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        F_jacobian: Callable[[np.ndarray], np.ndarray],
        z: np.ndarray,
        h: Callable[[np.ndarray], np.ndarray],
        H_jacobian: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Convenience method for predict followed by update."""
        self.predict(f, F_jacobian)
        self.update(z, h, H_jacobian)


class BearingOnlyEKF(ExtendedKalmanFilter):
    """Extended Kalman Filter for bearing-only sensor fusion.

    Fuses bearing measurements from one or more sensors to estimate
    target position and velocity. The measurement model is:

        bearing = atan2(y - sensor_y, x - sensor_x)

    This is non-linear, requiring the EKF framework.
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        sensor_pos: np.ndarray,
    ):
        """Initialize bearing-only EKF.

        Args:
            x0: Initial state [x, y, vx, vy]
            P0: Initial covariance (4×4)
            Q: Process noise covariance (4×4)
            R: Measurement noise covariance (1×1)
            sensor_pos: Sensor position [sx, sy]
        """
        super().__init__(x0, P0, Q, R)
        self.sensor_pos = sensor_pos.astype(float).copy()

    def predict(self, dt: float, Q: Optional[np.ndarray] = None) -> None:
        """Predict with constant-velocity motion model.

        Args:
            dt: Time step
            Q: Optional process noise override
        """
        def f(x):
            return np.array([
                x[0] + dt * x[2],
                x[1] + dt * x[3],
                x[2],
                x[3],
            ])

        def F_jacobian(x):
            return np.array([
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])

        super().predict(f, F_jacobian, Q=Q)

    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None) -> None:
        """Update with bearing measurement.

        Args:
            z: Bearing measurement [bearing] in radians
            R: Optional measurement noise override
        """
        def h(x):
            dx = x[0] - self.sensor_pos[0]
            dy = x[1] - self.sensor_pos[1]
            return np.array([math.atan2(dy, dx)])

        def H_jacobian(x):
            dx = x[0] - self.sensor_pos[0]
            dy = x[1] - self.sensor_pos[1]
            r_sq = dx**2 + dy**2
            if r_sq < 1e-10:
                r_sq = 1e-10
            return np.array([[-dy / r_sq, dx / r_sq, 0.0, 0.0]])

        super().update(z, h, H_jacobian, R=R)

    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute measurement Jacobian at given state.

        Args:
            x: State vector at which to evaluate Jacobian

        Returns:
            1×4 Jacobian matrix
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        r_sq = dx**2 + dy**2
        if r_sq < 1e-10:
            r_sq = 1e-10
        return np.array([[-dy / r_sq, dx / r_sq, 0.0, 0.0]])


class RangeBearingEKF(ExtendedKalmanFilter):
    """Extended Kalman Filter for range-bearing sensor fusion.

    Fuses range and bearing measurements from one or more sensors to
    estimate target position and velocity. The measurement model is:

        range = sqrt((x - sensor_x)^2 + (y - sensor_y)^2)
        bearing = atan2(y - sensor_y, x - sensor_x)

    This is non-linear, requiring the EKF framework.
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        sensor_pos: np.ndarray,
    ):
        """Initialize range-bearing EKF.

        Args:
            x0: Initial state [x, y, vx, vy]
            P0: Initial covariance (4×4)
            Q: Process noise covariance (4×4)
            R: Measurement noise covariance (2×2)
            sensor_pos: Sensor position [sx, sy]
        """
        super().__init__(x0, P0, Q, R)
        self.sensor_pos = sensor_pos.astype(float).copy()

    def predict(self, dt: float, Q: Optional[np.ndarray] = None) -> None:
        """Predict with constant-velocity motion model.

        Args:
            dt: Time step
            Q: Optional process noise override
        """
        def f(x):
            return np.array([
                x[0] + dt * x[2],
                x[1] + dt * x[3],
                x[2],
                x[3],
            ])

        def F_jacobian(x):
            return np.array([
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])

        super().predict(f, F_jacobian, Q=Q)

    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None) -> None:
        """Update with range-bearing measurement.

        Args:
            z: Measurement [range, bearing] in meters and radians
            R: Optional measurement noise override
        """
        def h(x):
            dx = x[0] - self.sensor_pos[0]
            dy = x[1] - self.sensor_pos[1]
            r = math.sqrt(dx**2 + dy**2)
            bearing = math.atan2(dy, dx)
            return np.array([r, bearing])

        def H_jacobian(x):
            dx = x[0] - self.sensor_pos[0]
            dy = x[1] - self.sensor_pos[1]
            r_sq = dx**2 + dy**2
            r = math.sqrt(r_sq)
            if r_sq < 1e-10:
                r_sq = 1e-10
                r = math.sqrt(r_sq)
            return np.array([
                [dx / r, dy / r, 0.0, 0.0],
                [-dy / r_sq, dx / r_sq, 0.0, 0.0],
            ])

        super().update(z, h, H_jacobian, R=R)

    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute measurement Jacobian at given state.

        Args:
            x: State vector at which to evaluate Jacobian

        Returns:
            2×4 Jacobian matrix
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        r_sq = dx**2 + dy**2
        r = math.sqrt(r_sq)
        if r_sq < 1e-10:
            r_sq = 1e-10
            r = math.sqrt(r_sq)
        return np.array([
            [dx / r, dy / r, 0.0, 0.0],
            [-dy / r_sq, dx / r_sq, 0.0, 0.0],
        ])
