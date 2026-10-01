"""Tests for TLE file parser — 2-line and 3-line formats, checksum validation, field extraction."""

import pytest

from src.fusion.tle_parser import (
    TLE,
    TLEChecksumError,
    TLEParseError,
    TLEParser,
    parse_tle,
    parse_tle_file,
    validate_tle,
)


# ─── Sample Data ──────────────────────────────────────────────────────────────

SAMPLE_TLE_3LINE = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""

SAMPLE_TLE_2LINE = """1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""

SAMPLE_TLE_MULTI = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
NOAA 18
1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2926
2 28654  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""


# ─── 3-Line TLE Parsing Tests ─────────────────────────────────────────────────

class TestTLE3LineParsing:
    """Tests for 3-line TLE parsing (name + 2 data lines)."""

    def test_parse_3line_returns_tle_with_name(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.name == "ISS (ZARYA)"

    def test_parse_3line_satellite_number(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.satellite_number == 25544

    def test_parse_3line_classification(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.classification == "U"

    def test_parse_3line_international_designator(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.international_designator == "98067A"

    def test_parse_3line_epoch_year(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.epoch_year == 8

    def test_parse_3line_epoch_day(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.epoch_day == pytest.approx(264.51782528)

    def test_parse_3line_mean_motion_derivative(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.mean_motion_derivative == pytest.approx(-0.00002182)

    def test_parse_3line_mean_motion_second_derivative(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.mean_motion_second_derivative == pytest.approx(0.0)

    def test_parse_3line_bstar(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.bstar == pytest.approx(-0.11606e-4)

    def test_parse_3line_ephemeris_type(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.ephemeris_type == 0

    def test_parse_3line_element_set_number(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.element_set_number == 292

    def test_parse_3line_line1_checksum(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.line1_checksum == 7

    def test_parse_3line_inclination(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.inclination == pytest.approx(51.6416)

    def test_parse_3line_raan(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.raan == pytest.approx(247.4627)

    def test_parse_3line_eccentricity(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.eccentricity == pytest.approx(0.0006703)

    def test_parse_3line_arg_of_perigee(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.arg_of_perigee == pytest.approx(130.5360)

    def test_parse_3line_mean_anomaly(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.mean_anomaly == pytest.approx(325.0288)

    def test_parse_3line_mean_motion(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.mean_motion == pytest.approx(15.72125391)

    def test_parse_3line_revolution_number(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.revolution_number == 56353

    def test_parse_3line_line2_checksum(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.line2_checksum == 7


# ─── 2-Line TLE Parsing Tests ─────────────────────────────────────────────────

class TestTLE2LineParsing:
    """Tests for 2-line TLE parsing (no name line)."""

    def test_parse_2line_name_is_none(self):
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.name is None

    def test_parse_2line_satellite_number(self):
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.satellite_number == 25544

    def test_parse_2line_inclination(self):
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.inclination == pytest.approx(51.6416)

    def test_parse_2line_eccentricity(self):
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.eccentricity == pytest.approx(0.0006703)

    def test_parse_2line_mean_motion(self):
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.mean_motion == pytest.approx(15.72125391)


# ─── File Parsing Tests ───────────────────────────────────────────────────────

class TestTLEFileParsing:
    """Tests for TLE file parsing."""

    def test_parse_tle_file(self, tmp_path):
        tle_file = tmp_path / "test.tle"
        tle_file.write_text(SAMPLE_TLE_3LINE)
        tle = parse_tle_file(str(tle_file))
        assert tle.name == "ISS (ZARYA)"
        assert tle.satellite_number == 25544

    def test_parse_tle_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_tle_file("/nonexistent/path/file.tle")


# ─── Multiple TLE Parsing Tests ───────────────────────────────────────────────

class TestTLEMultipleParsing:
    """Tests for parsing multiple TLEs from text."""

    def test_parse_multiple_tles(self):
        tles = TLEParser.parse_multiple(SAMPLE_TLE_MULTI)
        assert len(tles) == 2
        assert tles[0].name == "ISS (ZARYA)"
        assert tles[0].satellite_number == 25544
        assert tles[1].name == "NOAA 18"
        assert tles[1].satellite_number == 28654


# ─── Checksum Validation Tests ────────────────────────────────────────────────

class TestTLEChecksumValidation:
    """Tests for TLE checksum validation."""

    def test_valid_checksum_line1(self):
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        assert TLEParser._verify_checksum(line1) is True

    def test_valid_checksum_line2(self):
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        assert TLEParser._verify_checksum(line2) is True

    def test_invalid_checksum_line1_raises(self):
        bad_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        with pytest.raises(TLEChecksumError):
            TLEParser.parse(bad_line1, line2)

    def test_invalid_checksum_line2_raises(self):
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        bad_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563538"
        with pytest.raises(TLEChecksumError):
            TLEParser.parse(line1, bad_line2)

    def test_checksum_too_short_line(self):
        assert TLEParser._verify_checksum("1 25544") is False


# ─── Error Handling Tests ─────────────────────────────────────────────────────

class TestTLEErrorHandling:
    """Tests for TLE parsing error handling."""

    def test_empty_string_raises(self):
        with pytest.raises(TLEParseError):
            parse_tle("")

    def test_single_line_raises(self):
        with pytest.raises(TLEParseError):
            parse_tle("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927")

    def test_wrong_line1_prefix_raises(self):
        line1 = "3 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        with pytest.raises(TLEParseError):
            TLEParser.parse(line1, line2)

    def test_wrong_line2_prefix_raises(self):
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "3 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        with pytest.raises(TLEParseError):
            TLEParser.parse(line1, line2)

    def test_satellite_number_mismatch_raises(self):
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 99999  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        with pytest.raises(TLEParseError):
            TLEParser.parse(line1, line2)

    def test_line_too_short_raises(self):
        with pytest.raises(TLEParseError):
            TLEParser.parse("1 25544", "2 25544")


# ─── Scientific Notation Tests ─────────────────────────────────────────────────

class TestTLEScientificNotation:
    """Tests for TLE scientific notation field parsing."""

    def test_parse_scientific_positive(self):
        assert TLEParser._parse_scientific("12345-5") == pytest.approx(0.12345e-5)

    def test_parse_scientific_negative(self):
        assert TLEParser._parse_scientific("-11606-4") == pytest.approx(-0.11606e-4)

    def test_parse_scientific_zero(self):
        assert TLEParser._parse_scientific(" 00000-0") == pytest.approx(0.0)

    def test_parse_scientific_empty(self):
        assert TLEParser._parse_scientific("") == 0.0

    def test_parse_scientific_short_field(self):
        assert TLEParser._parse_scientific("5") == 0.0


# ─── TLE Validation Tests ─────────────────────────────────────────────────────

class TestTLEValidation:
    """Tests for TLE data validation."""

    def test_validate_valid_tle(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        validate_tle(tle)  # Should not raise

    def test_validate_inclination_out_of_range(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.inclination = 200.0
        with pytest.raises(TLEParseError):
            validate_tle(tle)

    def test_validate_eccentricity_out_of_range(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.eccentricity = 1.5
        with pytest.raises(TLEParseError):
            validate_tle(tle)

    def test_validate_mean_motion_out_of_range(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.mean_motion = 0.0
        with pytest.raises(TLEParseError):
            validate_tle(tle)

    def test_validate_raan_out_of_range(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.raan = 400.0
        with pytest.raises(TLEParseError):
            validate_tle(tle)

    def test_validate_arg_of_perigee_out_of_range(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.arg_of_perigee = 400.0
        with pytest.raises(TLEParseError):
            validate_tle(tle)

    def test_validate_mean_anomaly_out_of_range(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.mean_anomaly = 400.0
        with pytest.raises(TLEParseError):
            validate_tle(tle)


# ─── TLE Object Tests ─────────────────────────────────────────────────────────

class TestTLEObject:
    """Tests for TLE dataclass behavior."""

    def test_tle_is_valid(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.is_valid() is True

    def test_tle_str_representation(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        s = str(tle)
        assert "25544" in s
        assert "51.6416" in s

    def test_tle_name_none_for_2line(self):
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.name is None

    def test_tle_name_set_for_3line(self):
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.name == "ISS (ZARYA)"


# ─── Whitespace Handling Tests ────────────────────────────────────────────────

class TestTLEWhitespaceHandling:
    """Tests for TLE parsing with extra whitespace."""

    def test_parse_with_leading_trailing_whitespace(self):
        tle = parse_tle("  " + SAMPLE_TLE_3LINE + "  \n")
        assert tle.name == "ISS (ZARYA)"
        assert tle.satellite_number == 25544

    def test_parse_with_blank_lines(self):
        text = "\n\n" + SAMPLE_TLE_3LINE + "\n\n"
        tle = parse_tle(text)
        assert tle.satellite_number == 25544
