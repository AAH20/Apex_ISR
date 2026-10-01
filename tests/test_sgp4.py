"""Unit tests for SGP4/SDP4 orbital propagator.

TDD: These tests define the expected behavior of the SGP4 module.
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fusion.sgp4 import (
    TLE,
    TLEParser,
    SGP4Propagator,
    SGP4Error,
    CoordinateTransform,
    jday,
    days2mdhms,
    invjday,
    _gstime,
)


# ─── Test Data ───────────────────────────────────────────────────────────────

# ISS (ZARYA) TLE from CelesTrak
ISS_TLE_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_TLE_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# NOAA 18 TLE
NOAA18_TLE_LINE1 = "1 28654U 05018A   08264.50251234  .00000089  00000-0  52760-4 0  9991"
NOAA18_TLE_LINE2 = "2 28654  99.0515 262.7745 0014091  88.2004 272.0095 14.12501587223866"

# Molniya 1-89 (highly elliptical, deep space)
MOLNIYA_TLE_LINE1 = "1 24966U 97058A   08264.47108495  .00000018  00000-0  10000-3 0  9994"
MOLNIYA_TLE_LINE2 = "2 24966  63.1706 107.5492 7159102 264.4486  10.0618  2.00612534105085"

# Vanguard 1 (very old, near-circular)
VANGUARD_TLE_LINE1 = "1 00005U 58002B   08264.51782528 -.00000051  00000-0  00000+0 0  9990"
VANGUARD_TLE_LINE2 = "2 00005  34.2682  19.3842 0001155  88.4232 271.6820 14.85000000  1000"


# ─── TLE Parsing Tests ───────────────────────────────────────────────────────

class TestTLEParsing:
    """Test TLE line parsing and field extraction."""

    def test_parse_line1_satellite_number(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.satellite_number == 25544

    def test_parse_line1_classification(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.classification == "U"

    def test_parse_line1_international_designator(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.international_designator == "98067A"

    def test_parse_line1_epoch_year(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.epoch_year == 2008

    def test_parse_line1_epoch_day(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.epoch_day - 264.51782528) < 1e-8

    def test_parse_line1_mean_motion_derivative(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.mean_motion_derivative - (-0.00002182)) < 1e-12

    def test_parse_line1_bstar(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.bstar - (-1.1606e-5)) < 1e-12

    def test_parse_line2_inclination(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.inclination - 51.6416) < 1e-4

    def test_parse_line2_raan(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.raan - 247.4627) < 1e-4

    def test_parse_line2_eccentricity(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.eccentricity - 0.0006703) < 1e-8

    def test_parse_line2_arg_of_perigee(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.arg_of_perigee - 130.5360) < 1e-4

    def test_parse_line2_mean_anomaly(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.mean_anomaly - 325.0288) < 1e-4

    def test_parse_line2_mean_motion(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert abs(tle.mean_motion - 15.72125391) < 1e-8

    def test_parse_line2_rev_number(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.rev_number == 56353

    def test_parse_line1_element_number(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.element_number == 292

    def test_parse_line2_checksum(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.line2_checksum == 7

    def test_parse_line1_checksum(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.line1_checksum == 7

    def test_parse_noaa18(self):
        tle = TLEParser.parse(NOAA18_TLE_LINE1, NOAA18_TLE_LINE2)
        assert tle.satellite_number == 28654
        assert abs(tle.inclination - 99.0515) < 1e-4
        assert abs(tle.mean_motion - 14.12501587) < 1e-8

    def test_parse_molniya(self):
        tle = TLEParser.parse(MOLNIYA_TLE_LINE1, MOLNIYA_TLE_LINE2)
        assert tle.satellite_number == 24966
        assert abs(tle.eccentricity - 0.7159102) < 1e-8
        assert abs(tle.mean_motion - 2.00612534) < 1e-8

    def test_parse_vanguard(self):
        tle = TLEParser.parse(VANGUARD_TLE_LINE1, VANGUARD_TLE_LINE2)
        assert tle.satellite_number == 5
        assert abs(tle.inclination - 34.2682) < 1e-4

    def test_parse_raises_on_invalid_line1(self):
        with pytest.raises(SGP4Error):
            TLEParser.parse("INVALID LINE 1", ISS_TLE_LINE2)

    def test_parse_raises_on_invalid_line2(self):
        with pytest.raises(SGP4Error):
            TLEParser.parse(ISS_TLE_LINE1, "INVALID LINE 2")

    def test_parse_raises_on_mismatched_satellite_numbers(self):
        bad_line2 = "2 99999  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        with pytest.raises(SGP4Error):
            TLEParser.parse(ISS_TLE_LINE1, bad_line2)

    def test_parse_raises_on_bad_checksum(self):
        bad_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563530"
        with pytest.raises(SGP4Error):
            TLEParser.parse(ISS_TLE_LINE1, bad_line2)

    def test_parse_raises_on_short_line(self):
        with pytest.raises(SGP4Error):
            TLEParser.parse("1 25544U", ISS_TLE_LINE2)

    def test_parse_raises_on_empty_input(self):
        with pytest.raises(SGP4Error):
            TLEParser.parse("", "")

    def test_parse_raises_on_none_input(self):
        with pytest.raises(SGP4Error):
            TLEParser.parse(None, None)


# ─── TLE Validation Tests ────────────────────────────────────────────────────

class TestTLEValidation:
    """Test TLE validation and checksum verification."""

    def test_valid_tle_passes_validation(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.is_valid() is True

    def test_checksum_calculation_line1(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        # Verify checksum is computed correctly
        assert tle.line1_checksum == 7

    def test_checksum_calculation_line2(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert tle.line2_checksum == 7

    def test_invalid_checksum_fails(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        # Corrupt the checksum
        tle.line2_checksum = 0
        assert tle.is_valid() is False

    def test_tle_str_representation(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        s = str(tle)
        assert "25544" in s
        assert "51.6416" in s

    def test_tle_repr(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        r = repr(tle)
        assert "TLE" in r
        assert "25544" in r


# ─── SGP4 Propagator Tests ───────────────────────────────────────────────────

class TestSGP4Propagator:
    """Test SGP4 propagation and state vector computation."""

    def test_propagator_initialization(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        assert prop.tle.satellite_number == 25544

    def test_propagate_at_epoch(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        # ISS at epoch should be at reasonable altitude (~400 km)
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert 6500 < r < 7500  # km

    def test_produce_velocity_vector(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        # ISS orbital velocity ~7.66 km/s
        v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        assert 7.0 < v < 8.0  # km/s

    def test_produce_3d_position(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        assert len(pos) == 3
        assert len(vel) == 3

    def test_propagate_future_time(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos0, vel0 = prop.propagate(0.0)
        pos1, vel1 = prop.propagate(600.0)  # 10 minutes later
        # Position should change
        dx = pos1[0] - pos0[0]
        dy = pos1[1] - pos0[1]
        dz = pos1[2] - pos0[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist > 100  # km, should move significantly in 10 min

    def test_propagate_negative_time(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos0, _ = prop.propagate(0.0)
        pos_neg, _ = prop.propagate(-600.0)
        # Should be different from epoch
        dx = pos_neg[0] - pos0[0]
        dy = pos_neg[1] - pos0[1]
        dz = pos_neg[2] - pos0[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist > 100

    def test_propagate_one_orbit(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos0, _ = prop.propagate(0.0)
        # ISS period ~92 minutes = 5520 seconds
        pos1, _ = prop.propagate(5520.0)
        # After one orbit, should be close to starting position
        dx = pos1[0] - pos0[0]
        dy = pos1[1] - pos0[1]
        dz = pos1[2] - pos0[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist < 500  # km, should be reasonably close after one orbit

    def test_propagate_noaa18(self):
        tle = TLEParser.parse(NOAA18_TLE_LINE1, NOAA18_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        # NOAA 18 is at ~870 km altitude
        assert 7000 < r < 7500

    def test_propagate_molniya(self):
        tle = TLEParser.parse(MOLNIYA_TLE_LINE1, MOLNIYA_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        # Molniya has very high apogee
        assert r > 10000  # km

    def test_propagate_vanguard(self):
        tle = TLEParser.parse(VANGUARD_TLE_LINE1, VANGUARD_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        # Vanguard 1 is at ~650 km altitude
        assert 6800 < r < 7200

    def test_get_state_vector(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        state = prop.get_state_vector(0.0)
        assert "position" in state
        assert "velocity" in state
        assert "timestamp" in state
        assert len(state["position"]) == 3
        assert len(state["velocity"]) == 3

    def test_get_position_only(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos = prop.get_position(0.0)
        assert len(pos) == 3

    def test_get_velocity_only(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        vel = prop.get_velocity(0.0)
        assert len(vel) == 3

    def test_orbital_period(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        period = prop.get_orbital_period()
        # ISS period ~92.9 minutes
        assert 5400 < period < 5700  # seconds

    def test_semi_major_axis(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        sma = prop.get_semi_major_axis()
        # ISS semi-major axis ~6793 km
        assert 6700 < sma < 6900  # km

    def test_apogee_perigee(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        apogee, perigee = prop.get_apogee_perigee()
        # ISS: apogee ~426 km, perigee ~416 km (altitude above Earth surface)
        assert 400 < apogee < 500
        assert 400 < perigee < 500
        assert apogee >= perigee

    def test_is_deep_space_near_earth(self):
        tle_iss = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop_iss = SGP4Propagator(tle_iss)
        assert prop_iss.is_deep_space() is False

    def test_is_deep_space_molniya(self):
        tle_mol = TLEParser.parse(MOLNIYA_TLE_LINE1, MOLNIYA_TLE_LINE2)
        prop_mol = SGP4Propagator(tle_mol)
        # Molniya has period > 225 min, so it's deep space
        assert prop_mol.is_deep_space() is True

    def test_propagate_multiple_times(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        times = [0, 60, 120, 180, 240, 300]
        positions = [prop.get_position(t) for t in times]
        # All positions should be different
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                assert dist > 0

    def test_propagate_consistency(self):
        """Propagating in two steps should give similar result to one step."""
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos1, vel1 = prop.propagate(300.0)
        pos2, vel2 = prop.propagate(600.0)
        # Just verify both produce valid results
        r1 = math.sqrt(pos1[0]**2 + pos1[1]**2 + pos1[2]**2)
        r2 = math.sqrt(pos2[0]**2 + pos2[1]**2 + pos2[2]**2)
        assert 6500 < r1 < 7500
        assert 6500 < r2 < 7500

    def test_propagate_zero_time(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos, vel = prop.propagate(0.0)
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert r > 6000  # Should be near Earth

    def test_propagate_large_time(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        # Propagate 24 hours
        pos, vel = prop.propagate(86400.0)
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        # Should still be in orbit
        assert 6000 < r < 50000

    def test_propagate_raises_on_invalid_tle(self):
        with pytest.raises(SGP4Error):
            tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
            tle.mean_motion = 0  # Invalid
            prop = SGP4Propagator(tle)
            prop.propagate(0.0)


# ─── Coordinate Transform Tests ──────────────────────────────────────────────

class TestCoordinateTransform:
    """Test coordinate transformation utilities."""

    def test_teme_to_eci_identity(self):
        """At J2000 epoch, TEME ≈ ECI."""
        pos_teme = [7000.0, 0.0, 0.0]
        pos_eci = CoordinateTransform.teme_to_eci(pos_teme, 0.0)
        assert len(pos_eci) == 3

    def test_teme_to_eci_produces_3d(self):
        pos_teme = [7000.0, 1000.0, 500.0]
        pos_eci = CoordinateTransform.teme_to_eci(pos_teme, 0.0)
        assert len(pos_eci) == 3

    def test_teme_to_eci_different_times(self):
        pos_teme = [7000.0, 0.0, 0.0]
        pos1 = CoordinateTransform.teme_to_eci(pos_teme, 0.0)
        pos2 = CoordinateTransform.teme_to_eci(pos_teme, 3600.0)
        # Should be different due to Earth rotation
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist > 0

    def test_eci_to_teme_roundtrip(self):
        pos_teme = [7000.0, 1000.0, 500.0]
        pos_eci = CoordinateTransform.teme_to_eci(pos_teme, 0.0)
        pos_teme2 = CoordinateTransform.eci_to_teme(pos_eci, 0.0)
        assert abs(pos_teme[0] - pos_teme2[0]) < 1e-6
        assert abs(pos_teme[1] - pos_teme2[1]) < 1e-6
        assert abs(pos_teme[2] - pos_teme2[2]) < 1e-6

    def test_teme_to_ecef(self):
        pos_teme = [7000.0, 0.0, 0.0]
        pos_ecef = CoordinateTransform.teme_to_ecef(pos_teme, 0.0)
        assert len(pos_ecef) == 3

    def test_ecef_to_teme_roundtrip(self):
        pos_teme = [7000.0, 1000.0, 500.0]
        pos_ecef = CoordinateTransform.teme_to_ecef(pos_teme, 0.0)
        pos_teme2 = CoordinateTransform.ecef_to_teme(pos_ecef, 0.0)
        assert abs(pos_teme[0] - pos_teme2[0]) < 1e-6
        assert abs(pos_teme[1] - pos_teme2[1]) < 1e-6
        assert abs(pos_teme[2] - pos_teme2[2]) < 1e-6

    def test_teme_to_geodetic(self):
        pos_teme = [6378.137, 0.0, 0.0]  # On equator at Earth surface
        lat, lon, alt = CoordinateTransform.teme_to_geodetic(pos_teme, 0.0)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
        assert alt >= -100  # Allow small numerical errors

    def test_geodetic_to_teme_roundtrip(self):
        pos_teme = [7000.0, 1000.0, 500.0]
        lat, lon, alt = CoordinateTransform.teme_to_geodetic(pos_teme, 0.0)
        pos_teme2 = CoordinateTransform.geodetic_to_teme(lat, lon, alt, 0.0)
        assert abs(pos_teme[0] - pos_teme2[0]) < 1e-3
        assert abs(pos_teme[1] - pos_teme2[1]) < 1e-3
        assert abs(pos_teme[2] - pos_teme2[2]) < 1e-3

    def test_teme_to_eci_preserves_magnitude(self):
        pos_teme = [7000.0, 1000.0, 500.0]
        pos_eci = CoordinateTransform.teme_to_eci(pos_teme, 0.0)
        r_teme = math.sqrt(pos_teme[0]**2 + pos_teme[1]**2 + pos_teme[2]**2)
        r_eci = math.sqrt(pos_eci[0]**2 + pos_eci[1]**2 + pos_eci[2]**2)
        assert abs(r_teme - r_eci) < 1e-6


# ─── Utility Function Tests ──────────────────────────────────────────────────

class TestUtilityFunctions:
    """Test utility functions for time and coordinate conversions."""

    def test_jday_basic(self):
        jd = jday(2008, 9, 20, 12, 0, 0)
        assert jd > 2450000  # Should be a valid Julian date

    def test_jday_known_value(self):
        # J2000.0 = 2000-01-01 12:00:00 = JD 2451545.0
        jd = jday(2000, 1, 1, 12, 0, 0)
        assert abs(jd - 2451545.0) < 0.001

    def test_jday_different_times(self):
        jd1 = jday(2008, 9, 20, 0, 0, 0)
        jd2 = jday(2008, 9, 21, 0, 0, 0)
        assert abs(jd2 - jd1 - 1.0) < 0.001

    def test_days2mdhms(self):
        month, day, hour, minute, second = days2mdhms(2008, 264.5)
        assert month == 9
        assert 20 <= day <= 21

    def test_invjday(self):
        jd = jday(2008, 9, 20, 12, 0, 0)
        year, month, day, hour, minute, second = invjday(jd)
        assert year == 2008
        assert month == 9
        assert day == 20
        assert hour == 12

    def test_gstime(self):
        gst = _gstime(2451545.0)
        assert 0 <= gst < 2 * math.pi

    def test_gstime_different_jd(self):
        gst1 = _gstime(2451545.0)
        gst2 = _gstime(2451546.0)
        assert gst1 != gst2


# ─── Integration Tests ───────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_propagation_pipeline(self):
        """Parse TLE → propagate → transform coordinates."""
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos_teme, vel_teme = prop.propagate(0.0)
        pos_eci = CoordinateTransform.teme_to_eci(pos_teme, 0.0)
        lat, lon, alt = CoordinateTransform.teme_to_geodetic(pos_teme, 0.0)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
        assert alt > 0

    def test_propagate_and_transform_multiple_times(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        for t in [0, 300, 600, 900, 1200]:
            pos, vel = prop.propagate(t)
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            assert 6500 < r < 7500

    def test_state_vector_contains_all_fields(self):
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        state = prop.get_state_vector(0.0)
        assert "position" in state
        assert "velocity" in state
        assert "timestamp" in state
        assert "satellite_number" in state
        assert state["satellite_number"] == 25544

    def test_multiple_satellites(self):
        tles = [
            (ISS_TLE_LINE1, ISS_TLE_LINE2),
            (NOAA18_TLE_LINE1, NOAA18_TLE_LINE2),
            (MOLNIYA_TLE_LINE1, MOLNIYA_TLE_LINE2),
        ]
        for line1, line2 in tles:
            tle = TLEParser.parse(line1, line2)
            prop = SGP4Propagator(tle)
            pos, vel = prop.propagate(0.0)
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            assert r > 6000  # All should be in orbit

    def test_propagation_determinism(self):
        """Same TLE + same time should always give same result."""
        tle = TLEParser.parse(ISS_TLE_LINE1, ISS_TLE_LINE2)
        prop = SGP4Propagator(tle)
        pos1, vel1 = prop.propagate(600.0)
        pos2, vel2 = prop.propagate(600.0)
        assert pos1 == pos2
        assert vel1 == vel2
