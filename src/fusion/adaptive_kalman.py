"""Adaptive, robust, and nonlinear Kalman filtering for sensor fusion.

This module provides:
- AdaptiveKalmanFilter: KF with online Q/R adaptation from innovation sequence
- RobustKalmanFilter: KF with Huber-based outlier rejection
- UnscentedKalmanFilter: sigma-point nonlinear estimation (UKF)
"""

import math
from typing import Callable, Optional

import numpy as np


# ─── Huber helpers ────────────────────────────────────────────────────────────

def huber_weight(residual: float, delta: float) -> float:
    """Compute Huber weight for a residual.

    Args:
        residual: Innovation (measurement residual)
        delta: Huber tuning parameter (threshold)

    Returns:
        Weight in (0, 1]: 1.0 for inliers, delta/|r| for outliers
    """
    abs_r = abs(residual)
    if abs_r <= delta:
        return 1.0
    return delta / abs_r


def huber_influence(residual: float, delta: float) -> float:
    """Compute Huber influence function psi(r).

    Args:
        residual: Innovation
        delta: Huber tuning parameter

    Returns:
        psi(r) = r if |r| <= delta, else delta * sign(r)
    """
    abs_r = abs(residual)
    if abs_r <= delta:
        return residual
    return delta * math.copysign(1.0, residual)


# ─── Adaptive Kalman Filter ──────────────────────────────────────────────────

class AdaptiveKalmanFilter:
    """Kalman filter with online adaptation of Q and R.

    Uses the innovation sequence to estimate the actual process and
    measurement noise levels, adapting Q and R when innovations are
    larger than expected.

    State: x (n×1), Covariance: P (n×n)
    Process noise: Q (n×n), Measurement noise: R (m×m)
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        adapt_Q: bool = True,
        adapt_R: bool = True,
        adaptation_window: int = 10,
        adaptation_rate: float = 0.1,
    ):
        """Initialize adaptive Kalman filter.

        Args:
            x0: Initial state vector (n×1)
            P0: Initial covariance matrix (n×n)
            Q: Process noise covariance (n×n)
            R: Measurement noise covariance (m×m)
            adapt_Q: Whether to adapt process noise
            adapt_R: Whether to adapt measurement noise
            adaptation_window: Window size for innovation averaging
            adaptation_rate: Rate of adaptation (0-1)
        """
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.Q = Q.astype(float).copy()
        self.R = R.astype(float).copy()
        self.n = len(x0)
        self.m = R.shape[0]
        self.adapt_Q = adapt_Q
        self.adapt_R = adapt_R
        self.adaptation_window = adaptation_window
        self.adaptation_rate = adaptation_rate
        self.innovation_history: list[np.ndarray] = []
        self.innovation_cov_history: list[np.ndarray] = []

    def predict(self, F: np.ndarray, Q: Optional[np.ndarray] = None) -> None:
        """Prediction step.

        Args:
            F: State transition matrix (n×n)
            Q: Optional process noise override
        """
        Q_eff = Q if Q is not None else self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q_eff

    def update(self, z: np.ndarray, H: np.ndarray, R: Optional[np.ndarray] = None) -> None:
        """Update step with optional noise adaptation.

        Args:
            z: Measurement vector (m×1)
            H: Measurement matrix (m×n)
            R: Optional measurement noise override
        """
        z = z.astype(float)
        R_eff = R if R is not None else self.R

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R_eff

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T

        # Adaptation
        self._adapt(y, S)

    def _adapt(self, y: np.ndarray, S: np.ndarray) -> None:
        """Adapt Q and R based on innovation statistics.

        Args:
            y: Innovation vector
            S: Innovation covariance
        """
        self.innovation_history.append(y.copy())
        self.innovation_cov_history.append(S.copy())

        if len(self.innovation_history) < 2:
            return

        # Use recent window of innovations
        window = min(len(self.innovation_history), self.adaptation_window)
        recent_y = self.innovation_history[-window:]

        # Estimated innovation covariance from sample
        y_mean = np.mean(recent_y, axis=0)
        y_centered = np.array([yi - y_mean for yi in recent_y])
        C_yy = (y_centered.T @ y_centered) / max(window - 1, 1)

        # Theoretical innovation covariance (average of recent S)
        C_S = np.mean(self.innovation_cov_history[-window:], axis=0)

        # Adapt R: if actual innovation cov > theoretical, increase R
        if self.adapt_R:
            R_delta = C_yy - C_S
            # Only increase R, don't decrease below original
            R_new = self.R + self.adaptation_rate * R_delta
            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(R_new)
            if np.all(eigvals > 0):
                self.R = R_new

        # Adapt Q: scale based on innovation magnitude
        if self.adapt_Q:
            # Normalized innovation squared
            nis = float(y.T @ np.linalg.inv(S) @ y)
            # Expected NIS is m (dimension of measurement)
            # If NIS >> m, increase Q
            if nis > 2.0 * self.m:
                scale = 1.0 + self.adaptation_rate * (nis / self.m - 1.0)
                self.Q = self.Q * scale
            elif nis < 0.5 * self.m:
                scale = max(0.5, 1.0 - self.adaptation_rate * (1.0 - nis / self.m))
                self.Q = self.Q * scale


# ─── Robust Kalman Filter ────────────────────────────────────────────────────

class RobustKalmanFilter:
    """Kalman filter with Huber-based robust outlier rejection.

    Uses the Huber influence function to downweight outlier measurements,
    preventing them from corrupting the state estimate.

    State: x (n×1), Covariance: P (n×n)
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        delta: float = 1.5,
    ):
        """Initialize robust Kalman filter.

        Args:
            x0: Initial state vector (n×1)
            P0: Initial covariance matrix (n×n)
            Q: Process noise covariance (n×n)
            R: Measurement noise covariance (m×m)
            delta: Huber tuning parameter (threshold for outlier detection)
        """
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.Q = Q.astype(float).copy()
        self.R = R.astype(float).copy()
        self.n = len(x0)
        self.m = R.shape[0]
        self.delta = delta

    def predict(self, F: np.ndarray, Q: Optional[np.ndarray] = None) -> None:
        """Prediction step.

        Args:
            F: State transition matrix (n×n)
            Q: Optional process noise override
        """
        Q_eff = Q if Q is not None else self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q_eff

    def update(self, z: np.ndarray, H: np.ndarray, R: Optional[np.ndarray] = None) -> None:
        """Update step with Huber-based robust weighting.

        Args:
            z: Measurement vector (m×1)
            H: Measurement matrix (m×n)
            R: Optional measurement noise override
        """
        z = z.astype(float)
        R_eff = R if R is not None else self.R

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R_eff

        # Compute Huber weights for each measurement component
        # Use normalized innovation for weight computation
        S_inv = np.linalg.inv(S)
        normalized_innovation = np.sqrt(np.abs(y.T @ S_inv @ y))

        outlier_detected = False
        if self.m == 1:
            # Scalar measurement: simple Huber weight
            weight = huber_weight(float(y[0]), self.delta * math.sqrt(S[0, 0]))
            if weight < 1.0:
                outlier_detected = True
            # Inflate R for outlier
            R_weighted = R_eff / max(weight, 1e-6)
        else:
            # Vector measurement: use per-component weights
            # Compute marginal standard deviations
            marginal_std = np.sqrt(np.diag(S))
            weights = np.array([
                huber_weight(float(y[i]), self.delta * marginal_std[i])
                for i in range(self.m)
            ])
            if np.any(weights < 1.0):
                outlier_detected = True
            # Inflate R component-wise
            R_weighted = R_eff.copy()
            for i in range(self.m):
                R_weighted[i, i] = R_eff[i, i] / max(weights[i], 1e-6)

        # Kalman gain with weighted R
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R_weighted)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_weighted @ K.T

        # Explicit covariance inflation on outlier
        if outlier_detected:
            self.P = self.P * (1.0 + self.delta)


# ─── Unscented Kalman Filter ─────────────────────────────────────────────────

class UnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear estimation.

    Uses the unscented transform (sigma points) to propagate mean and
    covariance through nonlinear functions, avoiding linearization.

    State: x (n×1), Covariance: P (n×n)
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """Initialize UKF.

        Args:
            x0: Initial state vector (n×1)
            P0: Initial covariance matrix (n×n)
            Q: Process noise covariance (n×n)
            R: Measurement noise covariance (m×m)
            alpha: Spread of sigma points (typically 1e-3 to 1)
            beta: Incorporates prior knowledge of distribution (2 for Gaussian)
            kappa: Secondary scaling parameter (typically 0 or 3-n)
        """
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.Q = Q.astype(float).copy()
        self.R = R.astype(float).copy()
        self.n = len(x0)
        self.m = R.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute weights
        self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute sigma point weights."""
        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_)))

        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wc[0] = lambda_ / (n + lambda_) + (1.0 - self.alpha**2 + self.beta)

        self.lambda_ = lambda_

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points.

        Args:
            x: Mean state (n×1)
            P: Covariance matrix (n×n)

        Returns:
            Sigma points (2n+1 × n)
        """
        n = len(x)
        points = np.zeros((2 * n + 1, n))
        points[0] = x

        # Matrix square root
        try:
            U = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh((n + self.lambda_) * P)
            eigvals = np.maximum(eigvals, 1e-10)
            U = eigvecs @ np.diag(np.sqrt(eigvals))

        for i in range(n):
            points[i + 1] = x + U[i]
            points[i + 1 + n] = x - U[i]

        return points

    def predict(self, f: Callable[[np.ndarray], np.ndarray], Q: Optional[np.ndarray] = None) -> None:
        """Prediction step with nonlinear process model.

        Args:
            f: Nonlinear process model x_{k+1} = f(x_k)
            Q: Optional process noise override
        """
        Q_eff = Q if Q is not None else self.Q

        # Generate sigma points
        sigma_points = self._sigma_points(self.x, self.P)

        # Propagate through process model
        propagated = np.array([f(sp) for sp in sigma_points])

        # Predicted mean
        self.x = np.sum(self.Wm[:, None] * propagated, axis=0)

        # Predicted covariance
        self.P = Q_eff.copy()
        for i in range(2 * self.n + 1):
            diff = propagated[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

    def update(self, z: np.ndarray, h: Callable[[np.ndarray], np.ndarray], R: Optional[np.ndarray] = None) -> None:
        """Update step with nonlinear measurement model.

        Args:
            z: Measurement vector (m×1)
            h: Nonlinear measurement model z = h(x)
            R: Optional measurement noise override
        """
        z = z.astype(float)
        R_eff = R if R is not None else self.R

        # Generate sigma points
        sigma_points = self._sigma_points(self.x, self.P)

        # Propagate through measurement model
        z_sigma = np.array([h(sp) for sp in sigma_points])

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * z_sigma, axis=0)

        # Measurement covariance
        S = R_eff.copy()
        for i in range(2 * self.n + 1):
            diff = z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)

        # Cross-covariance
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            diff_x = sigma_points[i] - self.x
            diff_z = z_sigma[i] - z_pred
            P_xz += self.Wc[i] * np.outer(diff_x, diff_z)

        # Kalman gain
        K = P_xz @ np.linalg.inv(S)

        # Innovation
        y = z - z_pred

        # State update
        self.x = self.x + K @ y

        # Covariance update with symmetrization for numerical stability
        self.P = self.P - K @ P_xz.T
        self.P = (self.P + self.P.T) / 2.0
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 1e-10):
            self.P += np.eye(self.n) * (1e-10 - min(eigvals))
