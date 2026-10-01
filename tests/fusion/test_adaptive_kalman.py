"""Tests for adaptive Kalman filter, robust Kalman filter, and nonlinear estimation.

Covers:
- AdaptiveKalmanFilter: Q/R adaptation from innovation sequence
- RobustKalmanFilter: Huber-based outlier rejection
- UnscentedKalmanFilter: sigma-point nonlinear estimation
"""
import math

import numpy as np
import pytest

from src.fusion.adaptive_kalman import (
    AdaptiveKalmanFilter,
    RobustKalmanFilter,
    UnscentedKalmanFilter,
    huber_influence,
    huber_weight,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cv_matrix(dt: float, q: float = 0.1) -> np.ndarray:
    """Constant-velocity process noise for 1D position+velocity state."""
    return q * np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt],
    ])


def _cv_jacobian(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix for [x, vx]."""
    return np.array([
        [1.0, dt],
        [0.0, 1.0],
    ])


# ─── AdaptiveKalmanFilter ────────────────────────────────────────────────────

class TestAdaptiveKalmanFilter:
    """Tests for adaptive Kalman filter with innovation-based Q/R estimation."""

    def test_initialization_state(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        np.testing.assert_array_equal(kf.x, x0)

    def test_initialization_covariance(self):
        x0 = np.array([0.0, 1.0])
        P0 = 2.0 * np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        np.testing.assert_array_equal(kf.P, P0)

    def test_predict_updates_state(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = _cv_matrix(1.0)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        F = _cv_jacobian(1.0)
        kf.predict(F)
        np.testing.assert_allclose(kf.x, [1.0, 1.0], atol=1e-10)

    def test_predict_increases_covariance(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = _cv_matrix(1.0)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        F = _cv_jacobian(1.0)
        kf.predict(F)
        assert kf.P[0, 0] > P0[0, 0]

    def test_update_reduces_covariance(self):
        x0 = np.array([0.0, 1.0])
        P0 = 5.0 * np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        kf.update(np.array([2.0]), H)
        assert kf.P[0, 0] < P0[0, 0]

    def test_update_corrects_state_toward_measurement(self):
        x0 = np.array([0.0, 0.0])
        P0 = 10.0 * np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[0.5]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        kf.update(np.array([5.0]), H)
        assert kf.x[0] > 0.0
        assert kf.x[0] < 5.0

    def test_adaptive_Q_increases_on_large_innovation(self):
        """Large innovations should cause Q to grow."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R, adapt_Q=True, adapt_R=False)
        H = np.array([[1.0, 0.0]])
        Q_before = kf.Q.copy()
        # Feed a large measurement to create a big innovation
        for _ in range(5):
            kf.predict(_cv_jacobian(1.0))
            kf.update(np.array([100.0]), H)
        assert kf.Q[0, 0] > Q_before[0, 0]

    def test_adaptive_R_increases_on_large_innovation(self):
        """Large innovations should cause R to grow."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R, adapt_Q=False, adapt_R=True)
        H = np.array([[1.0, 0.0]])
        R_before = kf.R.copy()
        for _ in range(5):
            kf.predict(_cv_jacobian(1.0))
            kf.update(np.array([100.0]), H)
        assert kf.R[0, 0] > R_before[0, 0]

    def test_no_adaptation_when_disabled(self):
        """Q and R should remain fixed when adaptation is off."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R, adapt_Q=False, adapt_R=False)
        H = np.array([[1.0, 0.0]])
        Q_before = kf.Q.copy()
        R_before = kf.R.copy()
        for _ in range(5):
            kf.predict(_cv_jacobian(1.0))
            kf.update(np.array([100.0]), H)
        np.testing.assert_array_equal(kf.Q, Q_before)
        np.testing.assert_array_equal(kf.R, R_before)

    def test_steady_state_convergence(self):
        """Filter should converge to true state with consistent measurements."""
        x0 = np.array([0.0, 0.0])
        P0 = 10.0 * np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[0.5]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        true_x = 10.0
        for _ in range(50):
            kf.predict(_cv_jacobian(1.0))
            kf.update(np.array([true_x + np.random.normal(0, 0.5)]), H)
        assert abs(kf.x[0] - true_x) < 1.0

    def test_zero_process_noise_deterministic_prediction(self):
        """With Q=0, prediction should be purely deterministic."""
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2)
        Q = np.zeros((2, 2))
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        F = _cv_jacobian(1.0)
        kf.predict(F)
        np.testing.assert_allclose(kf.x, [3.0, 2.0], atol=1e-10)

    def test_large_measurement_noise_small_gain(self):
        """Large R should result in small Kalman gain (trust prediction)."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1000.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        kf.update(np.array([10.0]), H)
        # State should barely move toward measurement
        assert kf.x[0] < 1.0

    def test_innovation_history_recorded(self):
        """Innovation sequence should be stored for analysis."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        kf.update(np.array([1.0]), H)
        kf.update(np.array([2.0]), H)
        assert len(kf.innovation_history) == 2

    def test_covariance_stays_symmetric(self):
        """P should remain symmetric after predict+update cycles."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        for _ in range(10):
            kf.predict(_cv_jacobian(1.0))
            kf.update(np.array([1.0]), H)
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-10)

    def test_covariance_stays_positive_semidefinite(self):
        """P should remain positive semi-definite."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R)
        H = np.array([[1.0, 0.0]])
        for _ in range(20):
            kf.predict(_cv_jacobian(1.0))
            kf.update(np.array([1.0]), H)
        eigenvalues = np.linalg.eigvalsh(kf.P)
        assert np.all(eigenvalues >= -1e-10)


# ─── RobustKalmanFilter ──────────────────────────────────────────────────────

class TestRobustKalmanFilter:
    """Tests for robust Kalman filter with Huber-based outlier rejection."""

    def test_huber_weight_inlier(self):
        """Inliers (|r| <= delta) should have weight 1.0."""
        assert huber_weight(0.5, delta=1.0) == 1.0
        assert huber_weight(1.0, delta=1.0) == 1.0

    def test_huber_weight_outlier(self):
        """Outliers (|r| > delta) should have weight < 1.0."""
        w = huber_weight(3.0, delta=1.0)
        assert 0.0 < w < 1.0

    def test_huber_weight_scales_inversely(self):
        """Weight should scale as delta / |r| for outliers."""
        w = huber_weight(4.0, delta=2.0)
        assert w == pytest.approx(0.5)

    def test_huber_influence_bounded(self):
        """Influence function should be bounded by delta."""
        assert abs(huber_influence(100.0, delta=1.0)) == pytest.approx(1.0)
        assert abs(huber_influence(-100.0, delta=1.0)) == pytest.approx(1.0)

    def test_huber_influence_linear_region(self):
        """Influence should equal residual for small residuals."""
        assert huber_influence(0.3, delta=1.0) == pytest.approx(0.3)

    def test_robust_update_accepts_inlier(self):
        """Inlier measurements should update state normally."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = RobustKalmanFilter(x0, P0, Q, R, delta=1.5)
        H = np.array([[1.0, 0.0]])
        kf.update(np.array([1.0]), H)
        assert kf.x[0] > 0.0

    def test_robust_update_ignores_outlier(self):
        """Extreme outliers should have minimal effect on state."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf_std = AdaptiveKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy())
        kf_rob = RobustKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), delta=1.5)
        H = np.array([[1.0, 0.0]])
        outlier = np.array([1000.0])
        kf_std.update(outlier, H)
        kf_rob.update(outlier, H)
        # Robust filter should be much closer to zero
        assert abs(kf_rob.x[0]) < abs(kf_std.x[0])

    def test_robust_beats_standard_on_outlier_contaminated(self):
        """Robust filter should track better with outlier contamination."""
        np.random.seed(42)
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = _cv_matrix(1.0)
        R = np.array([[0.5]])
        kf_std = AdaptiveKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy())
        kf_rob = RobustKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy(), delta=1.5)
        H = np.array([[1.0, 0.0]])
        F = _cv_jacobian(1.0)
        true_positions = []
        std_errors = []
        rob_errors = []
        for k in range(30):
            true_x = float(k)
            true_positions.append(true_x)
            # 20% outlier contamination
            if np.random.random() < 0.2:
                z = true_x + np.random.normal(0, 20.0)
            else:
                z = true_x + np.random.normal(0, 0.5)
            kf_std.predict(F)
            kf_std.update(np.array([z]), H)
            kf_rob.predict(F)
            kf_rob.update(np.array([z]), H)
            std_errors.append(abs(kf_std.x[0] - true_x))
            rob_errors.append(abs(kf_rob.x[0] - true_x))
        assert np.mean(rob_errors) < np.mean(std_errors)

    def test_robust_covariance_inflation_on_outlier(self):
        """Covariance should inflate when outlier detected."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = RobustKalmanFilter(x0, P0, Q, R, delta=1.5)
        H = np.array([[1.0, 0.0]])
        P_before = kf.P.copy()
        kf.update(np.array([100.0]), H)
        # Covariance should inflate due to outlier
        assert kf.P[0, 0] > P_before[0, 0]

    def test_robust_predict_matches_standard(self):
        """Prediction step should be identical to standard KF."""
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2)
        Q = _cv_matrix(1.0)
        R = np.array([[1.0]])
        kf_std = AdaptiveKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy())
        kf_rob = RobustKalmanFilter(x0.copy(), P0.copy(), Q.copy(), R.copy())
        F = _cv_jacobian(1.0)
        kf_std.predict(F)
        kf_rob.predict(F)
        np.testing.assert_allclose(kf_std.x, kf_rob.x)
        np.testing.assert_allclose(kf_std.P, kf_rob.P)


# ─── UnscentedKalmanFilter ───────────────────────────────────────────────────

class TestUnscentedKalmanFilter:
    """Tests for Unscented Kalman Filter (sigma-point nonlinear estimation)."""

    def test_initialization(self):
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)
        np.testing.assert_array_equal(ukf.x, x0)
        np.testing.assert_array_equal(ukf.P, P0)

    def test_sigma_points_symmetric(self):
        """Sigma points should be symmetric around the mean."""
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)
        points = ukf._sigma_points(x0, P0)
        # Mean of sigma points should equal the mean
        mean_of_points = np.mean(points, axis=0)
        np.testing.assert_allclose(mean_of_points, x0, atol=1e-10)

    def test_sigma_points_count(self):
        """Should have 2n+1 sigma points for n-dimensional state."""
        x0 = np.array([0.0, 0.0, 0.0])
        P0 = np.eye(3)
        Q = 0.1 * np.eye(3)
        R = np.array([[1.0]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)
        points = ukf._sigma_points(x0, P0)
        assert points.shape[0] == 7  # 2*3 + 1

    def test_ukf_predict_nonlinear(self):
        """UKF predict should handle nonlinear dynamics."""
        x0 = np.array([1.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)

        # Nonlinear: constant turn rate
        def f(x):
            return np.array([x[0] + x[1], x[1]])

        ukf.predict(f)
        np.testing.assert_allclose(ukf.x, [1.0, 0.0], atol=1e-10)

    def test_ukf_update_nonlinear(self):
        """UKF update should handle nonlinear measurement."""
        x0 = np.array([3.0, 4.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)

        # Nonlinear measurement: range = sqrt(x^2 + y^2)
        def h(x):
            return np.array([np.sqrt(x[0]**2 + x[1]**2)])

        z = np.array([5.0])  # True range
        ukf.update(z, h)
        # State should stay close to true value
        np.testing.assert_allclose(ukf.x, [3.0, 4.0], atol=0.5)

    def test_ukf_converges_range_bearing(self):
        """UKF should converge on range-bearing tracking."""
        np.random.seed(42)
        x0 = np.array([10.0, 10.0, 0.0, 0.0])
        P0 = np.eye(4) * 10.0
        Q = 0.1 * np.eye(4)
        R = np.diag([1.0, 0.01])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)
        sensor_pos = np.array([0.0, 0.0])

        def f(x):
            return np.array([
                x[0] + x[2],
                x[1] + x[3],
                x[2],
                x[3],
            ])

        def h(x):
            dx = x[0] - sensor_pos[0]
            dy = x[1] - sensor_pos[1]
            r = math.sqrt(dx**2 + dy**2)
            bearing = math.atan2(dy, dx)
            return np.array([r, bearing])

        # True target at (5, 5) stationary
        true_x, true_y = 5.0, 5.0
        for _ in range(30):
            ukf.predict(f)
            r = math.sqrt(true_x**2 + true_y**2)
            bearing = math.atan2(true_y, true_x)
            z = np.array([r + np.random.normal(0, 0.5), bearing + np.random.normal(0, 0.05)])
            ukf.update(z, h)

        # Should be closer to true position than initial guess
        dist_initial = math.sqrt((10 - true_x)**2 + (10 - true_y)**2)
        dist_final = math.sqrt((ukf.x[0] - true_x)**2 + (ukf.x[1] - true_y)**2)
        assert dist_final < dist_initial

    def test_ukf_covariance_stays_positive_semidefinite(self):
        """P should remain positive semi-definite through UKF cycles."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)

        def f(x):
            return np.array([x[0] + x[1], x[1]])

        def h(x):
            return np.array([x[0]])

        for _ in range(20):
            ukf.predict(f)
            ukf.update(np.array([1.0]), h)
        eigenvalues = np.linalg.eigvalsh(ukf.P)
        assert np.all(eigenvalues >= -1e-10)

    def test_ukf_covariance_symmetric(self):
        """P should remain symmetric through UKF cycles."""
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)

        def f(x):
            return np.array([x[0] + x[1], x[1]])

        def h(x):
            return np.array([x[0]])

        for _ in range(10):
            ukf.predict(f)
            ukf.update(np.array([1.0]), h)
        np.testing.assert_allclose(ukf.P, ukf.P.T, atol=1e-10)

    def test_ukf_handles_wrapping_bearing(self):
        """UKF should handle bearing wrap-around near ±π."""
        x0 = np.array([1.0, 0.001])
        P0 = np.eye(2)
        Q = 0.001 * np.eye(2)
        R = np.array([[0.01]])
        ukf = UnscentedKalmanFilter(x0, P0, Q, R)

        def h(x):
            return np.array([math.atan2(x[1], x[0])])

        # Bearing near π (negative x-axis)
        z = np.array([math.pi - 0.01])
        ukf.update(z, h)
        # Should not crash and should produce finite state
        assert np.all(np.isfinite(ukf.x))


# ─── Integration / Cross-filter tests ─────────────────────────────────────────

class TestCrossFilterIntegration:
    """Integration tests comparing filter behaviors."""

    def test_adaptive_tracks_maneuvering_target(self):
        """Adaptive filter should track a target that changes velocity."""
        np.random.seed(42)
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[0.5]])
        kf = AdaptiveKalmanFilter(x0, P0, Q, R, adapt_Q=True, adapt_R=False)
        H = np.array([[1.0, 0.0]])
        F = _cv_jacobian(1.0)
        errors = []
        for k in range(40):
            # Target accelerates at k=20
            if k == 20:
                kf.x[1] += 2.0  # velocity jump
            true_x = kf.x[0]
            z = true_x + np.random.normal(0, 0.5)
            kf.predict(F)
            kf.update(np.array([z]), H)
            errors.append(abs(kf.x[0] - true_x))
        # Should recover quickly after maneuver
        assert np.mean(errors[-10:]) < 2.0

    def test_robust_handles_spike_noise(self):
        """Robust filter should handle occasional spike measurements."""
        np.random.seed(42)
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.array([[1.0]])
        kf = RobustKalmanFilter(x0, P0, Q, R, delta=1.5)
        H = np.array([[1.0, 0.0]])
        F = _cv_jacobian(1.0)
        for k in range(30):
            true_x = float(k) * 0.5
            # Spike at k=15
            if k == 15:
                z = true_x + 50.0
            else:
                z = true_x + np.random.normal(0, 0.3)
            kf.predict(F)
            kf.update(np.array([z]), H)
        # After spike, should still be close to true trajectory
        assert abs(kf.x[0] - 14.5) < 3.0

    def test_ukf_outperforms_linear_on_nonlinear(self):
        """UKF should outperform linear KF on highly nonlinear measurement."""
        np.random.seed(42)
        # True state
        true_x = np.array([3.0, 4.0])
        # Nonlinear measurement: range squared (more nonlinear than range)
        true_meas = true_x[0]**2 + true_x[1]**2  # = 25

        # Linear KF with linearized H (poor approximation for range squared)
        x0_lin = np.array([0.0, 0.0])
        P0 = np.eye(2) * 10.0
        Q = np.zeros((2, 2))
        R = np.array([[0.1]])
        kf_lin = AdaptiveKalmanFilter(x0_lin, P0.copy(), Q.copy(), R.copy())
        H_lin = np.array([[1.0, 0.0]])  # Linearized at origin (poor approx)
        kf_lin.update(np.array([true_meas]), H_lin)

        # UKF
        ukf = UnscentedKalmanFilter(
            np.array([0.0, 0.0]), P0.copy(), Q.copy(), R.copy()
        )

        def h(x):
            return np.array([x[0]**2 + x[1]**2])

        ukf.update(np.array([true_meas]), h)

        # UKF should be closer to true state
        err_lin = np.linalg.norm(kf_lin.x - true_x)
        err_ukf = np.linalg.norm(ukf.x - true_x)
        assert err_ukf < err_lin
