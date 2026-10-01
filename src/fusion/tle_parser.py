"""TLE file parser — 2-line and 3-line TLE parsing, checksum validation, field extraction.

This module provides parsing and validation for Two-Line Element sets (TLEs)
in both 2-line (data only) and 3-line (name + data) formats. It does NOT
implement SGP4 propagation — only parsing and validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


# ─── Exceptions ───────────────────────────────────────────────────────────────


class TLEParseError(Exception):
    """Exception raised for TLE parsing errors."""
    pass


class TLEChecksumError(TLEParseError):
    """Exception raised when TLE checksum verification fails."""
    pass


# ─── Data Class ───────────────────────────────────────────────────────────────


@dataclass
class TLE:
    """Two-Line Element set data.

    Attributes:
        satellite_number: NORAD catalog number
        classification: Classification (U=unclassified, C=classified, S=secret)
        international_designator: International designator (YYLLLPPP)
        epoch_year: Epoch year (2-digit)
        epoch_day: Epoch day of year (with fractional day)
        mean_motion_derivative: First derivative of mean motion (rev/day^2)
        mean_motion_second_derivative: Second derivative of mean motion (rev/day^3)
        bstar: B* drag term (1/Earth radii)
        ephemeris_type: Ephemeris type
        element_set_number: Element set number
        line1_checksum: Line 1 checksum
        inclination: Inclination (degrees)
        raan: Right ascension of ascending node (degrees)
        eccentricity: Eccentricity
        arg_of_perigee: Argument of perigee (degrees)
        mean_anomaly: Mean anomaly (degrees)
        mean_motion: Mean motion (rev/day)
        revolution_number: Revolution number at epoch
        line2_checksum: Line 2 checksum
        name: Satellite name (None for 2-line TLEs)
    """

    satellite_number: int
    classification: str
    international_designator: str
    epoch_year: int
    epoch_day: float
    mean_motion_derivative: float
    mean_motion_second_derivative: float
    bstar: float
    ephemeris_type: int
    element_set_number: int
    line1_checksum: int
    inclination: float
    raan: float
    eccentricity: float
    arg_of_perigee: float
    mean_anomaly: float
    mean_motion: float
    revolution_number: int
    line2_checksum: int
    name: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if TLE data is valid."""
        return (
            self.satellite_number > 0
            and 0 <= self.inclination <= 180
            and 0 <= self.eccentricity < 1
            and self.mean_motion > 0
            and self.line1_checksum >= 0
            and self.line2_checksum >= 0
        )

    def __str__(self) -> str:
        return (
            f"TLE(SAT={self.satellite_number}, "
            f"INC={self.inclination:.4f}°, "
            f"ECC={self.eccentricity:.7f}, "
            f"MM={self.mean_motion:.8f} rev/day)"
        )

    def __repr__(self) -> str:
        return (
            f"TLE(satellite_number={self.satellite_number}, "
            f"inclination={self.inclination}, "
            f"eccentricity={self.eccentricity}, "
            f"mean_motion={self.mean_motion})"
        )


# ─── TLE Parser ───────────────────────────────────────────────────────────────


class TLEParser:
    """Parser for Two-Line Element sets."""

    @staticmethod
    def parse(line1: str, line2: str) -> TLE:
        """Parse a two-line element set.

        Args:
            line1: First line of TLE (69 characters)
            line2: Second line of TLE (69 characters)

        Returns:
            TLE object with parsed orbital elements

        Raises:
            TLEParseError: If lines are invalid
            TLEChecksumError: If checksums don't match
        """
        if not line1 or not line2:
            raise TLEParseError("TLE lines cannot be empty")

        if len(line1) < 69 or len(line2) < 69:
            raise TLEParseError("TLE lines must be at least 69 characters")

        if not line1.startswith("1 "):
            raise TLEParseError("Line 1 must start with '1 '")

        if not line2.startswith("2 "):
            raise TLEParseError("Line 2 must start with '2 '")

        # Verify checksums
        if not TLEParser._verify_checksum(line1):
            raise TLEChecksumError("Line 1 checksum verification failed")

        if not TLEParser._verify_checksum(line2):
            raise TLEChecksumError("Line 2 checksum verification failed")

        # Parse satellite number from both lines and verify match
        sat_num1 = int(line1[2:7])
        sat_num2 = int(line2[2:7])
        if sat_num1 != sat_num2:
            raise TLEParseError(
                f"Satellite number mismatch: {sat_num1} vs {sat_num2}"
            )

        # Parse Line 1 fields
        classification = line1[7]
        intl_desig = line1[9:17].strip()
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        mm_deriv = float(line1[33:43])
        mm_sec_deriv = TLEParser._parse_scientific(line1[44:52])
        bstar = TLEParser._parse_scientific(line1[53:61])
        ephemeris_type = int(line1[62])
        element_number = int(line1[64:68])
        line1_checksum = int(line1[68])

        # Parse Line 2 fields
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33].strip())
        arg_of_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        rev_number = int(line2[63:68])
        line2_checksum = int(line2[68])

        return TLE(
            satellite_number=sat_num1,
            classification=classification,
            international_designator=intl_desig,
            epoch_year=epoch_year,
            epoch_day=epoch_day,
            mean_motion_derivative=mm_deriv,
            mean_motion_second_derivative=mm_sec_deriv,
            bstar=bstar,
            ephemeris_type=ephemeris_type,
            element_set_number=element_number,
            line1_checksum=line1_checksum,
            inclination=inclination,
            raan=raan,
            eccentricity=eccentricity,
            arg_of_perigee=arg_of_perigee,
            mean_anomaly=mean_anomaly,
            mean_motion=mean_motion,
            revolution_number=rev_number,
            line2_checksum=line2_checksum,
        )

    @staticmethod
    def parse_multiple(text: str) -> List[TLE]:
        """Parse multiple TLEs from a text block.

        Args:
            text: Text containing one or more TLEs (2-line or 3-line format)

        Returns:
            List of TLE objects
        """
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        tles = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
                tle = TLEParser.parse(lines[i], lines[i + 1])
                # Check if there's a name line before this TLE
                if i > 0 and not lines[i - 1].startswith(("1 ", "2 ")):
                    tle.name = lines[i - 1]
                tles.append(tle)
                i += 2
            else:
                i += 1
        return tles

    @staticmethod
    def _verify_checksum(line: str) -> bool:
        """Verify TLE line checksum.

        The checksum is the sum of all digits in the line (0-9),
        with minus signs counting as 1, modulo 10.
        """
        if len(line) < 69:
            return False

        checksum = 0
        for i in range(68):
            c = line[i]
            if c.isdigit():
                checksum += int(c)
            elif c == "-":
                checksum += 1

        return (checksum % 10) == int(line[68])

    @staticmethod
    def _parse_scientific(field: str) -> float:
        """Parse scientific notation field with implied decimal point.

        TLE format uses fields like '-11606-4' meaning -0.11606e-4.
        """
        field = field.strip()
        if not field:
            return 0.0

        if len(field) < 2:
            return 0.0

        exp_sign = field[-2]
        exp_digit = field[-1]
        mantissa = field[:-2]

        try:
            mantissa_val = float(mantissa) / 1e5
            exp_val = int(exp_digit)
            if exp_sign == "-":
                exp_val = -exp_val
            return mantissa_val * (10 ** exp_val)
        except (ValueError, IndexError):
            return 0.0


# ─── Convenience Functions ────────────────────────────────────────────────────


def parse_tle(text: str) -> TLE:
    """Parse a TLE from text (2-line or 3-line format).

    Args:
        text: TLE text with optional name line

    Returns:
        TLE object

    Raises:
        TLEParseError: If text is invalid
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if len(lines) < 2:
        raise TLEParseError("TLE must have at least 2 lines")

    # Find the data lines
    line1_idx = None
    for i, line in enumerate(lines):
        if line.startswith("1 "):
            line1_idx = i
            break

    if line1_idx is None or line1_idx + 1 >= len(lines):
        raise TLEParseError("Could not find TLE data lines")

    line1 = lines[line1_idx]
    line2 = lines[line1_idx + 1]

    if not line2.startswith("2 "):
        raise TLEParseError("Line 2 must start with '2 '")

    tle = TLEParser.parse(line1, line2)

    # Set name if present
    if line1_idx > 0:
        tle.name = lines[line1_idx - 1]

    return tle


def parse_tle_file(filepath: str) -> TLE:
    """Parse a TLE from a file.

    Args:
        filepath: Path to TLE file

    Returns:
        TLE object

    Raises:
        FileNotFoundError: If file doesn't exist
        TLEParseError: If file content is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TLE file not found: {filepath}")

    with open(filepath, "r") as f:
        text = f.read()

    return parse_tle(text)


def validate_tle(tle: TLE) -> None:
    """Validate TLE data ranges.

    Args:
        tle: TLE object to validate

    Raises:
        TLEParseError: If any field is out of valid range
    """
    if tle.satellite_number <= 0:
        raise TLEParseError(f"Invalid satellite number: {tle.satellite_number}")

    if not 0 <= tle.inclination <= 180:
        raise TLEParseError(f"Invalid inclination: {tle.inclination}")

    if not 0 <= tle.eccentricity < 1:
        raise TLEParseError(f"Invalid eccentricity: {tle.eccentricity}")

    if tle.mean_motion <= 0:
        raise TLEParseError(f"Invalid mean motion: {tle.mean_motion}")

    if not 0 <= tle.raan <= 360:
        raise TLEParseError(f"Invalid RAAN: {tle.raan}")

    if not 0 <= tle.arg_of_perigee <= 360:
        raise TLEParseError(f"Invalid arg of perigee: {tle.arg_of_perigee}")

    if not 0 <= tle.mean_anomaly <= 360:
        raise TLEParseError(f"Invalid mean anomaly: {tle.mean_anomaly}")
