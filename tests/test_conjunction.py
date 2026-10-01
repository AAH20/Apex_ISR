"""Tests for conjunction detection — TDD enforced."""

import math
import numpy as np
import pytest

from src.fusion.conjunction import (
    closest_approach,
    collision_probability_1d,
    collision_probability_2d,
    analyze_conjunction,
    generate_cdm,
    cdm_to_dict,
    cdm_to_ccsds,
    OrbitalState,
    ConjunctionResult,
    CDM,
)


class TestClosestApproach:
    def test_head_on_collision(self):
        """Two objects heading directly at each other."""
        r1 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([100.0, 0.0, 0.0])
        r2 = np.array([1000.0, 0.0, 0.0])
        v2 = np.array([-100.0, 0.0, 0.0])
        tca, miss, r1_ca, r2_ca = closest_approach(r1, v1, r2, v2)
        assert tca == pytest.approx(5.0)
        assert miss == pytest.approx(0.0, abs=1e-9)

    def test_parallel_trajectories(self):
        """Objects on parallel paths — miss distance is the offset."""
        r1 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([100.0, 0.0, 0.0])
        r2 = np.array([0.0, 50.0, 0.0])
        v2 = np.array([100.0, 0.0, 0.0])
        tca, miss, r1_ca, r2_ca = closest_approach(r1, v1, r2, v2)
        assert tca == pytest.approx(0.0)
        assert miss == pytest.approx(50.0)

    def test_stationary_objects(self):
        """Both objects stationary — miss distance is constant."""
        r1 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([30.0, 40.0, 0.0])
        v2 = np.array([0.0, 0.0, 0.0])
        tca, miss, r1_ca, r2_ca = closest_approach(r1, v1, r2, v2)
        assert tca == pytest.approx(0.0)
        assert miss == pytest.approx(50.0)

    def test_past_closest_approach_clamped(self):
        """If closest approach is in the past, tca clamps to 0."""
        r1 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([100.0, 0.0, 0.0])
        r2 = np.array([-1000.0, 0.0, 0.0])
        v2 = np.array([100.0, 0.0, 0.0])
        tca, miss, r1_ca, r2_ca = closest_approach(r1, v1, r2, v2)
        assert tca == pytest.approx(0.0)
        assert miss == pytest.approx(1000.0)

    def test_3d_trajectories(self):
        """Closest approach in 3D space."""
        r1 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([10.0, 20.0, 30.0])
        r2 = np.array([100.0, 200.0, 300.0])
        v2 = np.array([0.0, 0.0, 0.0])
        tca, miss, r1_ca, r2_ca = closest_approach(r1, v1, r2, v2)
        # tca = -(dr·dv)/(dv·dv) = -(-14000)/1400 = 10
        assert tca == pytest.approx(10.0)
        assert miss == pytest.approx(0.0, abs=1e-9)

    def test_custom_epoch(self):
        """TCA is relative to the provided epoch."""
        r1 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([100.0, 0.0, 0.0])
        r2 = np.array([1000.0, 0.0, 0.0])
        v2 = np.array([-100.0, 0.0, 0.0])
        tca, miss, r1_ca, r2_ca = closest_approach(r1, v1, r2, v2, t0=1000.0)
        assert tca == pytest.approx(1005.0)


class TestCollisionProbability1D:
    def test_zero_miss_distance(self):
        """Perfect alignment → probability 1."""
        p = collision_probability_1d(0.0, 10.0, 5.0)
        assert p == pytest.approx(1.0)

    def test_large_miss_distance(self):
        """Miss distance >> combined sigma → probability ~0."""
        p = collision_probability_1d(10000.0, 10.0, 5.0)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        """Probability is symmetric in the two sigmas."""
        p1 = collision_probability_1d(50.0, 10.0, 20.0)
        p2 = collision_probability_1d(50.0, 20.0, 10.0)
        assert p1 == pytest.approx(p2)

    def test_monotonic_decreasing(self):
        """Probability decreases as miss distance increases."""
        sig = 10.0
        p1 = collision_probability_1d(0.0, sig, sig)
        p2 = collision_probability_1d(50.0, sig, sig)
        p3 = collision_probability_1d(100.0, sig, sig)
        assert p1 > p2 > p3

    def test_known_value(self):
        """Known value: miss=0, sigma1=sigma2=10 → P = erf(0) + ... = 1."""
        p = collision_probability_1d(0.0, 10.0, 10.0)
        assert p == pytest.approx(1.0)

    def test_bounded(self):
        """Probability is always in [0, 1]."""
        for miss in [0.0, 10.0, 50.0, 100.0, 500.0]:
            p = collision_probability_1d(miss, 10.0, 10.0)
            assert 0.0 <= p <= 1.0


class TestCollisionProbability2D:
    def test_zero_miss_distance(self):
        """Known closed form: P = 1 - exp(-R²/(2σ²)) where σ² = σ₁² + σ₂²."""
        sigma = 10.0
        R = 5.0
        p = collision_probability_2d(0.0, sigma, sigma, R)
        combined_sigma_sq = sigma**2 + sigma**2
        expected = 1.0 - math.exp(-R**2 / (2 * combined_sigma_sq))
        assert p == pytest.approx(expected, rel=1e-3)

    def test_large_miss_distance(self):
        """Miss distance >> sigma → probability ~0."""
        p = collision_probability_2d(10000.0, 10.0, 10.0, 5.0)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_bounded(self):
        """Probability is always in [0, 1]."""
        for miss in [0.0, 10.0, 50.0, 100.0, 500.0]:
            p = collision_probability_2d(miss, 10.0, 10.0, 5.0)
            assert 0.0 <= p <= 1.0

    def test_monotonic_decreasing(self):
        """Probability decreases as miss distance increases."""
        p1 = collision_probability_2d(0.0, 10.0, 10.0, 5.0)
        p2 = collision_probability_2d(50.0, 10.0, 10.0, 5.0)
        p3 = collision_probability_2d(100.0, 10.0, 10.0, 5.0)
        assert p1 > p2 > p3

    def test_hard_body_radius_zero(self):
        """Zero hard-body radius → zero probability."""
        p = collision_probability_2d(0.0, 10.0, 10.0, 0.0)
        assert p == pytest.approx(0.0)

    def test_symmetric(self):
        """Probability is symmetric in the two sigmas."""
        p1 = collision_probability_2d(50.0, 10.0, 20.0, 5.0)
        p2 = collision_probability_2d(50.0, 20.0, 10.0, 5.0)
        assert p1 == pytest.approx(p2)


class TestOrbitalState:
    def test_creation(self):
        """OrbitalState can be created with position and velocity."""
        state = OrbitalState(
            position=np.array([7000e3, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 0.0]),
        )
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)

    def test_with_covariance(self):
        """OrbitalState can include a covariance matrix."""
        cov = np.eye(3) * 100.0
        state = OrbitalState(
            position=np.array([7000e3, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 0.0]),
            covariance=cov,
        )
        assert state.covariance is not None
        assert state.covariance.shape == (3, 3)

    def test_default_covariance_is_none(self):
        """Default covariance is None."""
        state = OrbitalState(
            position=np.array([7000e3, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 0.0]),
        )
        assert state.covariance is None


class TestAnalyzeConjunction:
    def test_basic_analysis(self):
        """analyze_conjunction returns a ConjunctionResult."""
        s1 = OrbitalState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
        )
        s2 = OrbitalState(
            position=np.array([1000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),
        )
        result = analyze_conjunction(s1, s2)
        assert isinstance(result, ConjunctionResult)
        assert result.tca == pytest.approx(5.0)
        assert result.miss_distance == pytest.approx(0.0, abs=1e-9)

    def test_with_covariance(self):
        """analyze_conjunction computes collision probability when covariance is provided."""
        cov1 = np.eye(3) * 100.0
        cov2 = np.eye(3) * 100.0
        s1 = OrbitalState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
            covariance=cov1,
        )
        s2 = OrbitalState(
            position=np.array([1000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),
            covariance=cov2,
        )
        result = analyze_conjunction(s1, s2, hard_body_radius=10.0)
        assert result.collision_probability is not None
        assert 0.0 <= result.collision_probability <= 1.0

    def test_without_covariance(self):
        """analyze_conjunction returns None collision probability without covariance."""
        s1 = OrbitalState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
        )
        s2 = OrbitalState(
            position=np.array([1000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),
        )
        result = analyze_conjunction(s1, s2)
        assert result.collision_probability is None

    def test_parallel_trajectories(self):
        """Parallel trajectories give correct miss distance."""
        s1 = OrbitalState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
        )
        s2 = OrbitalState(
            position=np.array([0.0, 50.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
        )
        result = analyze_conjunction(s1, s2)
        assert result.miss_distance == pytest.approx(50.0)
        assert result.tca == pytest.approx(0.0)

    def test_result_has_positions_at_tca(self):
        """Result includes positions at TCA."""
        s1 = OrbitalState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
        )
        s2 = OrbitalState(
            position=np.array([1000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),
        )
        result = analyze_conjunction(s1, s2)
        assert result.r1_at_tca is not None
        assert result.r2_at_tca is not None
        assert result.r1_at_tca.shape == (3,)
        assert result.r2_at_tca.shape == (3,)


class TestCDM:
    def _make_result(self):
        s1 = OrbitalState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
            object_id="OBJ-1",
        )
        s2 = OrbitalState(
            position=np.array([1000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),
            object_id="OBJ-2",
        )
        result = analyze_conjunction(s1, s2, hard_body_radius=10.0)
        return s1, s2, result

    def test_generate_cdm(self):
        """generate_cdm produces a CDM dataclass."""
        s1, s2, result = self._make_result()
        cdm = generate_cdm(s1, s2, result)
        assert isinstance(cdm, CDM)
        assert cdm.object1_id == "OBJ-1"
        assert cdm.object2_id == "OBJ-2"
        assert cdm.tca == pytest.approx(5.0)
        assert cdm.miss_distance == pytest.approx(0.0, abs=1e-9)

    def test_cdm_to_dict(self):
        """cdm_to_dict returns a dictionary with expected keys."""
        s1, s2, result = self._make_result()
        cdm = generate_cdm(s1, s2, result)
        d = cdm_to_dict(cdm)
        assert isinstance(d, dict)
        assert "object1_id" in d
        assert "object2_id" in d
        assert "tca" in d
        assert "miss_distance" in d
        assert "collision_probability" in d
        assert d["object1_id"] == "OBJ-1"
        assert d["object2_id"] == "OBJ-2"

    def test_cdm_to_ccsds(self):
        """cdm_to_ccsds returns a string with CCSDS-like format."""
        s1, s2, result = self._make_result()
        cdm = generate_cdm(s1, s2, result)
        text = cdm_to_ccsds(cdm)
        assert isinstance(text, str)
        assert "CDM" in text
        assert "OBJ-1" in text
        assert "OBJ-2" in text

    def test_cdm_contains_hard_body_radius(self):
        """CDM includes hard body radius."""
        s1, s2, result = self._make_result()
        cdm = generate_cdm(s1, s2, result)
        assert cdm.hard_body_radius == pytest.approx(10.0)

    def test_cdm_contains_positions_at_tca(self):
        """CDM includes positions at TCA."""
        s1, s2, result = self._make_result()
        cdm = generate_cdm(s1, s2, result)
        assert cdm.r1_at_tca is not None
        assert cdm.r2_at_tca is not None
        assert cdm.r1_at_tca.shape == (3,)
        assert cdm.r2_at_tca.shape == (3,)

    def test_cdm_to_dict_roundtrip(self):
        """cdm_to_dict values match the CDM dataclass fields."""
        s1, s2, result = self._make_result()
        cdm = generate_cdm(s1, s2, result)
        d = cdm_to_dict(cdm)
        assert d["tca"] == pytest.approx(cdm.tca)
        assert d["miss_distance"] == pytest.approx(cdm.miss_distance)
        assert d["hard_body_radius"] == pytest.approx(cdm.hard_body_radius)
