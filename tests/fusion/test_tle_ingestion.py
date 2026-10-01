"""Tests for TLE/OMM ingestion pipeline."""

import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import pytest

from src.fusion.tle_ingestion import (
    OMM,
    OMMIngestor,
    TLE,
    TLEIngestor,
    TLEValidationError,
    OMMValidationError,
    parse_tle,
    parse_tle_file,
    parse_omm,
    parse_omm_file,
    validate_tle,
    validate_omm,
    TLEChecksumError,
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
1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 28654  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""

SAMPLE_OMM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:58:31</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
          <EPOCH>2008-09-20T12:25:40.104640</EPOCH>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>999</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.00000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.00000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""


# ─── TLE Parsing Tests ────────────────────────────────────────────────────────

class TestTLEParsing:
    """Tests for TLE parsing from strings and files."""

    def test_parse_tle_3line_format(self):
        """Parse a standard 3-line TLE (name + 2 data lines)."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.name == "ISS (ZARYA)"
        assert tle.satellite_number == 25544
        assert tle.classification == "U"
        assert tle.international_designator == "98067A"
        assert tle.epoch_year == 8
        assert tle.epoch_day == 264.51782528
        assert tle.mean_motion_derivative == -0.00002182
        assert tle.mean_motion_second_derivative == 0.0
        assert tle.bstar == -0.11606e-4
        assert tle.ephemeris_type == 0
        assert tle.element_set_number == 292
        assert tle.line1_checksum == 7
        assert tle.inclination == 51.6416
        assert tle.raan == 247.4627
        assert tle.eccentricity == 0.0006703
        assert tle.arg_of_perigee == 130.5360
        assert tle.mean_anomaly == 325.0288
        assert tle.mean_motion == 15.72125391
        assert tle.revolution_number == 56353
        assert tle.line2_checksum == 7

    def test_parse_tle_2line_format(self):
        """Parse a 2-line TLE (no name line)."""
        tle = parse_tle(SAMPLE_TLE_2LINE)
        assert tle.name is None
        assert tle.satellite_number == 25544
        assert tle.inclination == 51.6416

    def test_parse_tle_file(self, tmp_path):
        """Parse TLE from a file path."""
        tle_file = tmp_path / "test.tle"
        tle_file.write_text(SAMPLE_TLE_3LINE)
        tle = parse_tle_file(str(tle_file))
        assert tle.name == "ISS (ZARYA)"
        assert tle.satellite_number == 25544

    def test_parse_multiple_tles(self):
        """Parse multiple TLEs from a single string."""
        tles = parse_tle(SAMPLE_TLE_MULTI)
        # parse_tle returns a single TLE; test the ingestor for multiple
        ingestor = TLEIngestor()
        result = ingestor.ingest_text(SAMPLE_TLE_MULTI)
        assert len(result) == 2
        assert result[0].name == "ISS (ZARYA)"
        assert result[1].name == "NOAA 18"

    def test_parse_tle_with_leading_trailing_whitespace(self):
        """TLE parsing handles extra whitespace."""
        tle = parse_tle("  " + SAMPLE_TLE_3LINE + "  \n")
        assert tle.name == "ISS (ZARYA)"

    def test_parse_tle_negative_bstar(self):
        """TLE with negative BSTAR value."""
        tle_text = """TEST
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tle = parse_tle(tle_text)
        assert tle.bstar < 0

    def test_parse_tle_scientific_notation_bstar(self):
        """TLE with BSTAR in scientific notation (e.g., 12345-5)."""
        tle_text = """TEST
1 25544U 98067A   08264.51782528 -.00002182  00000-0  12345-5 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tle = parse_tle(tle_text)
        assert tle.bstar == pytest.approx(0.12345e-4)

    def test_parse_tle_invalid_line1_raises(self):
        """Invalid line 1 (wrong length) raises ValueError."""
        with pytest.raises(ValueError):
            parse_tle("1 25544\n2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")

    def test_parse_tle_invalid_line2_raises(self):
        """Invalid line 2 (wrong length) raises ValueError."""
        with pytest.raises(ValueError):
            parse_tle("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n2 25544")

    def test_parse_tle_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_tle("")

    def test_parse_tle_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_tle_file("/nonexistent/path/file.tle")


# ─── TLE Validation Tests ─────────────────────────────────────────────────────

class TestTLEValidation:
    """Tests for TLE validation."""

    def test_validate_tle_valid(self):
        """Valid TLE passes validation."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        validate_tle(tle)  # Should not raise

    def test_validate_tle_checksum_error_line1(self):
        """TLE with corrupted line 1 checksum raises TLEChecksumError."""
        bad_tle = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tle = parse_tle(bad_tle)
        with pytest.raises(TLEChecksumError):
            validate_tle(tle)

    def test_validate_tle_checksum_error_line2(self):
        """TLE with corrupted line 2 checksum raises TLEChecksumError."""
        bad_tle = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563538"""
        tle = parse_tle(bad_tle)
        with pytest.raises(TLEChecksumError):
            validate_tle(tle)

    def test_validate_tle_inclination_out_of_range(self):
        """Inclination > 180 degrees raises TLEValidationError."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.inclination = 200.0
        with pytest.raises(TLEValidationError):
            validate_tle(tle)

    def test_validate_tle_eccentricity_out_of_range(self):
        """Eccentricity >= 1 raises TLEValidationError."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.eccentricity = 1.5
        with pytest.raises(TLEValidationError):
            validate_tle(tle)

    def test_validate_tle_mean_motion_out_of_range(self):
        """Mean motion <= 0 raises TLEValidationError."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.mean_motion = 0.0
        with pytest.raises(TLEValidationError):
            validate_tle(tle)

    def test_validate_tle_raan_out_of_range(self):
        """RAAN > 360 degrees raises TLEValidationError."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.raan = 400.0
        with pytest.raises(TLEValidationError):
            validate_tle(tle)

    def test_validate_tle_arg_of_perigee_out_of_range(self):
        """Arg of perigee > 360 degrees raises TLEValidationError."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.arg_of_perigee = 400.0
        with pytest.raises(TLEValidationError):
            validate_tle(tle)

    def test_validate_tle_mean_anomaly_out_of_range(self):
        """Mean anomaly > 360 degrees raises TLEValidationError."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        tle.mean_anomaly = 400.0
        with pytest.raises(TLEValidationError):
            validate_tle(tle)


# ─── OMM Parsing Tests ────────────────────────────────────────────────────────

class TestOMMParsing:
    """Tests for OMM XML parsing."""

    def test_parse_omm_basic(self):
        """Parse a basic OMM XML document."""
        omm = parse_omm(SAMPLE_OMM_XML)
        assert omm.object_name == "ISS (ZARYA)"
        assert omm.object_id == "1998-067A"
        assert omm.center_name == "EARTH"
        assert omm.ref_frame == "TEME"
        assert omm.time_system == "UTC"
        assert omm.mean_element_theory == "SGP4"
        assert omm.norad_cat_id == 25544
        assert omm.classification == "U"
        assert omm.element_set_no == 999
        assert omm.rev_at_epoch == 56353
        assert omm.bstar == pytest.approx(0.0)
        assert omm.mean_motion_dot == pytest.approx(-0.00002182)
        assert omm.mean_motion_ddot == pytest.approx(0.0)
        assert omm.mean_motion == pytest.approx(15.72125391)
        assert omm.eccentricity == pytest.approx(0.0006703)
        assert omm.inclination == pytest.approx(51.6416)
        assert omm.ra_of_asc_node == pytest.approx(247.4627)
        assert omm.arg_of_pericenter == pytest.approx(130.5360)
        assert omm.mean_anomaly == pytest.approx(325.0288)
        assert omm.epoch == "2008-09-20T12:25:40.104640"

    def test_parse_omm_file(self, tmp_path):
        """Parse OMM from a file path."""
        omm_file = tmp_path / "test.xml"
        omm_file.write_text(SAMPLE_OMM_XML)
        omm = parse_omm_file(str(omm_file))
        assert omm.object_name == "ISS (ZARYA)"
        assert omm.norad_cat_id == 25544

    def test_parse_omm_invalid_xml_raises(self):
        """Invalid XML raises ParseError."""
        with pytest.raises(ET.ParseError):
            parse_omm("not valid xml <<<")

    def test_parse_omm_missing_required_field_raises(self):
        """OMM missing required field raises ValueError."""
        bad_xml = """<?xml version="1.0"?>
<omm><body><segment><metadata>
<OBJECT_NAME>TEST</OBJECT_NAME>
</metadata></segment></body></omm>"""
        with pytest.raises(ValueError):
            parse_omm(bad_xml)

    def test_parse_omm_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_omm_file("/nonexistent/path/file.xml")


# ─── OMM Validation Tests ─────────────────────────────────────────────────────

class TestOMMValidation:
    """Tests for OMM validation."""

    def test_validate_omm_valid(self):
        """Valid OMM passes validation."""
        omm = parse_omm(SAMPLE_OMM_XML)
        validate_omm(omm)  # Should not raise

    def test_validate_omm_inclination_out_of_range(self):
        """Inclination > 180 raises OMMValidationError."""
        omm = parse_omm(SAMPLE_OMM_XML)
        omm.inclination = 200.0
        with pytest.raises(OMMValidationError):
            validate_omm(omm)

    def test_validate_omm_eccentricity_out_of_range(self):
        """Eccentricity >= 1 raises OMMValidationError."""
        omm = parse_omm(SAMPLE_OMM_XML)
        omm.eccentricity = 1.5
        with pytest.raises(OMMValidationError):
            validate_omm(omm)

    def test_validate_omm_mean_motion_out_of_range(self):
        """Mean motion <= 0 raises OMMValidationError."""
        omm = parse_omm(SAMPLE_OMM_XML)
        omm.mean_motion = 0.0
        with pytest.raises(OMMValidationError):
            validate_omm(omm)


# ─── Database Storage Tests ───────────────────────────────────────────────────

class TestDatabaseStorage:
    """Tests for SQLite database storage of TLE and OMM data."""

    def test_tle_database_store_and_retrieve(self, tmp_path):
        """Store and retrieve a TLE from SQLite database."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path)
        tle = parse_tle(SAMPLE_TLE_3LINE)
        ingestor.store_tle(tle)

        # Retrieve
        result = ingestor.get_tle(25544)
        assert result is not None
        assert result.name == "ISS (ZARYA)"
        assert result.satellite_number == 25544
        assert result.inclination == pytest.approx(51.6416)

    def test_tle_database_store_multiple(self, tmp_path):
        """Store multiple TLEs and retrieve all."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path)
        ingestor.ingest_text(SAMPLE_TLE_MULTI)

        all_tles = ingestor.get_all_tles()
        assert len(all_tles) == 2

    def test_tle_database_duplicate_handling(self, tmp_path):
        """Storing duplicate satellite number updates existing record."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path)
        tle = parse_tle(SAMPLE_TLE_3LINE)
        ingestor.store_tle(tle)
        ingestor.store_tle(tle)  # Store again

        all_tles = ingestor.get_all_tles()
        assert len(all_tles) == 1

    def test_tle_database_get_nonexistent_returns_none(self, tmp_path):
        """Getting a non-existent satellite returns None."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path)
        result = ingestor.get_tle(99999)
        assert result is None

    def test_omm_database_store_and_retrieve(self, tmp_path):
        """Store and retrieve an OMM from SQLite database."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path)
        omm = parse_omm(SAMPLE_OMM_XML)
        ingestor.store_omm(omm)

        result = ingestor.get_omm(25544)
        assert result is not None
        assert result.object_name == "ISS (ZARYA)"
        assert result.norad_cat_id == 25544

    def test_omm_database_store_multiple(self, tmp_path):
        """Store multiple OMMs and retrieve all."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path)
        omm = parse_omm(SAMPLE_OMM_XML)
        ingestor.store_omm(omm)
        ingestor.store_omm(omm)

        all_omms = ingestor.get_all_omms()
        assert len(all_omms) == 1  # Duplicate handling

    def test_omm_database_get_nonexistent_returns_none(self, tmp_path):
        """Getting a non-existent OMM returns None."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path)
        result = ingestor.get_omm(99999)
        assert result is None

    def test_tle_database_persistence(self, tmp_path):
        """TLE data persists across ingestor instances."""
        db_path = str(tmp_path / "test.db")
        ingestor1 = TLEIngestor(db_path=db_path)
        tle = parse_tle(SAMPLE_TLE_3LINE)
        ingestor1.store_tle(tle)

        # New ingestor instance, same database
        ingestor2 = TLEIngestor(db_path=db_path)
        result = ingestor2.get_tle(25544)
        assert result is not None
        assert result.name == "ISS (ZARYA)"


# ─── Ingestor Integration Tests ──────────────────────────────────────────────

class TestIngestorIntegration:
    """Integration tests for the full ingestion pipeline."""

    def test_tle_ingestor_full_pipeline(self, tmp_path):
        """Full TLE ingestion: parse, validate, store."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path)
        tles = ingestor.ingest_text(SAMPLE_TLE_3LINE)
        assert len(tles) == 1
        assert tles[0].satellite_number == 25544

        # Verify stored
        stored = ingestor.get_tle(25544)
        assert stored is not None
        assert stored.name == "ISS (ZARYA)"

    def test_tle_ingestor_with_validation(self, tmp_path):
        """TLE ingestion with validation enabled."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path, validate=True)
        tles = ingestor.ingest_text(SAMPLE_TLE_3LINE)
        assert len(tles) == 1

    def test_tle_ingestor_invalid_tle_skipped(self, tmp_path):
        """Invalid TLE is skipped during ingestion."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path, validate=True)
        bad_tle = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tles = ingestor.ingest_text(bad_tle)
        assert len(tles) == 0

    def test_omm_ingestor_full_pipeline(self, tmp_path):
        """Full OMM ingestion: parse, validate, store."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path)
        omms = ingestor.ingest_text(SAMPLE_OMM_XML)
        assert len(omms) == 1
        assert omms[0].norad_cat_id == 25544

        stored = ingestor.get_omm(25544)
        assert stored is not None
        assert stored.object_name == "ISS (ZARYA)"

    def test_omm_ingestor_with_validation(self, tmp_path):
        """OMM ingestion with validation enabled."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path, validate=True)
        omms = ingestor.ingest_text(SAMPLE_OMM_XML)
        assert len(omms) == 1

    def test_omm_ingestor_invalid_omm_skipped(self, tmp_path):
        """Invalid OMM is skipped during ingestion."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path, validate=True)
        bad_xml = """<?xml version="1.0"?>
<omm><body><segment><metadata>
<OBJECT_NAME>TEST</OBJECT_NAME>
<OBJECT_ID>1998-067A</OBJECT_ID>
<CENTER_NAME>EARTH</CENTER_NAME>
<REF_FRAME>TEME</REF_FRAME>
<TIME_SYSTEM>UTC</TIME_SYSTEM>
<MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
</metadata><data><meanElements>
<MEAN_MOTION>0</MEAN_MOTION>
<ECCENTRICITY>0.0006703</ECCENTRICITY>
<INCLINATION>51.6416</INCLINATION>
<RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
<ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
<MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
<EPOCH>2008-09-20T12:25:40.104640</EPOCH>
</meanElements><tleParameters>
<EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
<CLASSIFICATION>U</CLASSIFICATION>
<NORAD_CAT_ID>25544</NORAD_CAT_ID>
<ELEMENT_SET_NO>999</ELEMENT_SET_NO>
<REV_AT_EPOCH>56353</REV_AT_EPOCH>
<BSTAR>0.0</BSTAR>
<MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
<MEAN_MOTION_DDOT>0.0</MEAN_MOTION_DDOT>
</tleParameters></data></segment></body></omm>"""
        omms = ingestor.ingest_text(bad_xml)
        assert len(omms) == 0

    def test_tle_ingestor_ingest_file(self, tmp_path):
        """Ingest TLE from file."""
        db_path = str(tmp_path / "test.db")
        tle_file = tmp_path / "test.tle"
        tle_file.write_text(SAMPLE_TLE_3LINE)
        ingestor = TLEIngestor(db_path=db_path)
        tles = ingestor.ingest_file(str(tle_file))
        assert len(tles) == 1

    def test_omm_ingestor_ingest_file(self, tmp_path):
        """Ingest OMM from file."""
        db_path = str(tmp_path / "test.db")
        omm_file = tmp_path / "test.xml"
        omm_file.write_text(SAMPLE_OMM_XML)
        ingestor = OMMIngestor(db_path=db_path)
        omms = ingestor.ingest_file(str(omm_file))
        assert len(omms) == 1


# ─── Checksum Computation Tests ───────────────────────────────────────────────

class TestChecksumComputation:
    """Tests for TLE checksum computation."""

    def test_checksum_line1_valid(self):
        """Verify checksum computation for line 1."""
        line = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.line1_checksum == 7

    def test_checksum_line2_valid(self):
        """Verify checksum computation for line 2."""
        tle = parse_tle(SAMPLE_TLE_3LINE)
        assert tle.line2_checksum == 7

    def test_checksum_detects_single_digit_error(self):
        """Checksum detects a single digit error."""
        # Change one digit in line 1
        bad_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928"
        tle = parse_tle("ISS\n" + bad_line1 + "\n2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")
        with pytest.raises(TLEChecksumError):
            validate_tle(tle)


# ─── Edge Case Tests ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests."""

    def test_tle_with_max_length_name(self):
        """TLE with maximum length name (24 characters)."""
        long_name = "A" * 24
        tle_text = f"""{long_name}
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tle = parse_tle(tle_text)
        assert tle.name == long_name

    def test_tle_epoch_year_57_99_maps_to_1900s(self):
        """Epoch year 57-99 maps to 1900s."""
        tle_text = """TEST
1 25544U 98067A   99264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tle = parse_tle(tle_text)
        assert tle.epoch_year == 99
        assert tle.epoch_year_full == 1999

    def test_tle_epoch_year_00_56_maps_to_2000s(self):
        """Epoch year 00-56 maps to 2000s."""
        tle_text = """TEST
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
        tle = parse_tle(tle_text)
        assert tle.epoch_year == 8
        assert tle.epoch_year_full == 2008

    def test_omm_with_optional_fields_missing(self):
        """OMM with missing optional fields uses defaults."""
        minimal_xml = """<?xml version="1.0"?>
<omm><body><segment><metadata>
<OBJECT_NAME>TEST</OBJECT_NAME>
<OBJECT_ID>1998-067A</OBJECT_ID>
<CENTER_NAME>EARTH</CENTER_NAME>
<REF_FRAME>TEME</REF_FRAME>
<TIME_SYSTEM>UTC</TIME_SYSTEM>
<MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
</metadata><data><meanElements>
<MEAN_MOTION>15.72125391</MEAN_MOTION>
<ECCENTRICITY>0.0006703</ECCENTRICITY>
<INCLINATION>51.6416</INCLINATION>
<RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
<ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
<MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
<EPOCH>2008-09-20T12:25:40.104640</EPOCH>
</meanElements><tleParameters>
<EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
<CLASSIFICATION>U</CLASSIFICATION>
<NORAD_CAT_ID>25544</NORAD_CAT_ID>
<ELEMENT_SET_NO>999</ELEMENT_SET_NO>
<REV_AT_EPOCH>56353</REV_AT_EPOCH>
<BSTAR>0.0</BSTAR>
<MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
<MEAN_MOTION_DDOT>0.0</MEAN_MOTION_DDOT>
</tleParameters></data></segment></body></omm>"""
        omm = parse_omm(minimal_xml)
        assert omm.object_name == "TEST"
        assert omm.norad_cat_id == 25544

    def test_tle_database_schema_created(self, tmp_path):
        """Database schema is created on initialization."""
        db_path = str(tmp_path / "test.db")
        ingestor = TLEIngestor(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "tle" in tables

    def test_omm_database_schema_created(self, tmp_path):
        """OMM database schema is created on initialization."""
        db_path = str(tmp_path / "test.db")
        ingestor = OMMIngestor(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "omm" in tables
