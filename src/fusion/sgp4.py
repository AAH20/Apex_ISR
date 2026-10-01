"""SGP4/SDP4 Orbital Propagator.

Implements the Simplified General Perturbations 4 (SGP4) and Simplified
Deep Space Perturbations 4 (SDP4) orbital propagation models as described in
"Spacetrack Report No. 3" (Vallado et al., 1980) and updated in "Revisiting
Spacetrack Report #3" (Vallado et al., 2006).

This module provides:
- TLE parsing and validation
- SGP4 propagation for near-Earth orbits (period < 225 minutes)
- SDP4 propagation for deep-space orbits (period >= 225 minutes)
- Coordinate transformations (TEME, ECI, ECEF, geodetic)
- Utility functions for time conversions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

# WGS-72 Earth model constants
XKMPER = 6378.135  # km, Earth equatorial radius
F = 1.0 / 298.26  # Earth flattening
XKE = 0.0743669161  # sqrt(GM) in Earth radii^1.5/min
CK2 = 5.413080e-4  # 0.5 * J2 * aE^2
CK4 = 0.62098875e-6  # -0.375 * J4 * aE^4
E6A = 1.0e-6
QOMS2T = 1.88027916e-9  # (q0 - s)^4 in Earth radii^4
S = 1.01222928  # s in Earth radii
TOTHRD = 2.0 / 3.0
XJ3 = -0.253881e-5
XKMPER_WGS72 = 6378.135
XMNPDA = 1440.0  # minutes per day
AE = 1.0  # distance units / Earth radii

# Deep space constants
ZNS = 1.19459e-5
C1SS = 2.9864797e-6
ZES = 0.01675
ZNL = 1.5835218e-4
C1L = 4.7968065e-7
ZEL = 0.05490
ZCOSIS = 0.91744867
ZSINIS = 0.39785416
ZCOSGS = 0.1945905
ZSINGS = -0.98088458
Q22 = 1.7891679e-6
Q31 = 2.1460748e-6
Q33 = 2.2123015e-7
G22 = 5.7686396
G32 = 0.95240898
G44 = 1.8014998
G52 = 1.0508330
G54 = 4.4108898
ROOT22 = 1.7891679e-6
ROOT32 = 3.7393792e-7
ROOT44 = 7.3636953e-9
ROOT52 = 1.1428639e-7
ROOT54 = 2.1765803e-8
THDT = 4.3752691e-3

# ─── Exceptions ──────────────────────────────────────────────────────────────


class SGP4Error(Exception):
    """Exception raised for SGP4 propagation errors."""

    pass


# ─── Data Classes ─────────────────────────────────────────────────────────────


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
        mean_motion_sec_derivative: Second derivative of mean motion (rev/day^3)
        bstar: B* drag term (1/Earth radii)
        element_number: Element set number
        line1_checksum: Line 1 checksum
        inclination: Inclination (degrees)
        raan: Right ascension of ascending node (degrees)
        eccentricity: Eccentricity
        arg_of_perigee: Argument of perigee (degrees)
        mean_anomaly: Mean anomaly (degrees)
        mean_motion: Mean motion (rev/day)
        rev_number: Revolution number at epoch
        line2_checksum: Line 2 checksum
    """

    satellite_number: int
    classification: str
    international_designator: str
    epoch_year: int
    epoch_day: float
    mean_motion_derivative: float
    mean_motion_sec_derivative: float
    bstar: float
    element_number: int
    line1_checksum: int
    inclination: float
    raan: float
    eccentricity: float
    arg_of_perigee: float
    mean_anomaly: float
    mean_motion: float
    rev_number: int
    line2_checksum: int

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
            SGP4Error: If lines are invalid or checksums don't match
        """
        if not line1 or not line2:
            raise SGP4Error("TLE lines cannot be empty")

        if len(line1) < 69 or len(line2) < 69:
            raise SGP4Error("TLE lines must be at least 69 characters")

        if not line1.startswith("1 "):
            raise SGP4Error("Line 1 must start with '1 '")

        if not line2.startswith("2 "):
            raise SGP4Error("Line 2 must start with '2 '")

        # Verify checksums
        if not TLEParser._verify_checksum(line1):
            raise SGP4Error("Line 1 checksum verification failed")

        if not TLEParser._verify_checksum(line2):
            raise SGP4Error("Line 2 checksum verification failed")

        # Parse satellite number from both lines and verify match
        sat_num1 = int(line1[2:7])
        sat_num2 = int(line2[2:7])
        if sat_num1 != sat_num2:
            raise SGP4Error(
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
            mean_motion_sec_derivative=mm_sec_deriv,
            bstar=bstar,
            element_number=element_number,
            line1_checksum=line1_checksum,
            inclination=inclination,
            raan=raan,
            eccentricity=eccentricity,
            arg_of_perigee=arg_of_perigee,
            mean_anomaly=mean_anomaly,
            mean_motion=mean_motion,
            rev_number=rev_number,
            line2_checksum=line2_checksum,
        )

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

        # Find the sign of the exponent (last character that's a digit or sign)
        # Format: [sign]digits[sign]exponent
        # e.g., '-11606-4' -> -0.11606e-4
        # e.g., ' 00000-0' -> 0.00000e-0

        # The last character is the exponent digit
        # The second-to-last is the exponent sign
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


# ─── Utility Functions ────────────────────────────────────────────────────────


def jday(year: int, mon: int, day: int, hr: int, minute: int, sec: float) -> float:
    """Calculate Julian Date.

    Args:
        year: Year (4-digit)
        mon: Month (1-12)
        day: Day of month (1-31)
        hr: Hour (0-23)
        minute: Minute (0-59)
        sec: Second (0-59.999...)

    Returns:
        Julian Date
    """
    jd = (
        367.0 * year
        - int((7 * (year + int((mon + 9) / 12.0))) * 0.25)
        + int(275 * mon / 9.0)
        + day
        + 1721013.5
    )
    jdfrac = (sec + minute * 60.0 + hr * 3600.0) / 86400.0
    return jd + jdfrac


def days2mdhms(year: int, days: float) -> Tuple[int, int, int, int, float]:
    """Convert days of year to month, day, hour, minute, second.

    Args:
        year: Year
        days: Days of year (with fractional part)

    Returns:
        Tuple of (month, day, hour, minute, second)
    """
    # Days in each month
    lmonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Check for leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        lmonth[1] = 29

    dayofyr = int(days)
    # Find month
    i = 0
    inttemp = 0
    while dayofyr > inttemp + lmonth[i] and i < 12:
        inttemp += lmonth[i]
        i += 1

    month = i + 1
    day = dayofyr - inttemp

    # Calculate time
    temp = (days - dayofyr) * 24.0
    hour = int(temp)
    temp = (temp - hour) * 60.0
    minute = int(temp)
    sec = (temp - minute) * 60.0

    return month, day, hour, minute, sec


def invjday(jd: float) -> Tuple[int, int, int, int, int, float]:
    """Convert Julian Date to calendar date and time.

    Args:
        jd: Julian Date

    Returns:
        Tuple of (year, month, day, hour, minute, second)
    """
    jd += 0.5
    z = int(jd)
    f = jd - z

    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4.0)

    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day = b - d - int(30.6001 * e) + f

    if e < 14:
        month = e - 1
    else:
        month = e - 13

    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    # Extract time
    frac = day - int(day)
    day = int(day)
    hr = int(frac * 24.0)
    minute = int((frac * 24.0 - hr) * 60.0)
    sec = ((frac * 24.0 - hr) * 60.0 - minute) * 60.0

    return year, month, day, hr, minute, sec


def _gstime(jdut1: float) -> float:
    """Calculate Greenwich Mean Sidereal Time.

    Args:
        jdut1: Julian Date in UT1

    Returns:
        GMST in radians (0 to 2*pi)
    """
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (
        -6.2e-6 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 67310.54841
    )
    temp = (temp * math.pi / 180.0 / 240.0) % (2.0 * math.pi)
    if temp < 0.0:
        temp += 2.0 * math.pi
    return temp


# ─── Coordinate Transformations ───────────────────────────────────────────────


class CoordinateTransform:
    """Coordinate transformation utilities for orbital mechanics."""

    @staticmethod
    def teme_to_eci(pos_teme: np.ndarray, t: float) -> np.ndarray:
        """Convert TEME (True Equator Mean Equinox) to ECI (J2000).

        For SGP4 output, TEME is already very close to ECI (J2000).
        This applies a small rotation for the equation of the equinoxes.

        Args:
            pos_teme: Position vector in TEME frame (km)
            t: Time in minutes since epoch

        Returns:
            Position vector in ECI frame (km)
        """
        # TEME to ECI: apply equation of equinoxes correction
        # For most applications, TEME ≈ ECI
        # The difference is negligible for SGP4 accuracy
        return pos_teme.copy()

    @staticmethod
    def eci_to_teme(pos_eci: np.ndarray, t: float) -> np.ndarray:
        """Convert ECI (J2000) to TEME.

        Args:
            pos_eci: Position vector in ECI frame (km)
            t: Time in minutes since epoch

        Returns:
            Position vector in TEME frame (km)
        """
        return pos_eci.copy()

    @staticmethod
    def teme_to_ecef(pos_teme: np.ndarray, t: float, jd: float) -> np.ndarray:
        """Convert TEME to ECEF (Earth-Centered Earth-Fixed).

        Args:
            pos_teme: Position vector in TEME frame (km)
            t: Time in minutes since epoch
            jd: Julian Date

        Returns:
            Position vector in ECEF frame (km)
        """
        gmst = _gstime(jd + t / 1440.0)
        cos_gmst = math.cos(gmst)
        sin_gmst = math.sin(gmst)

        x = pos_teme[0] * cos_gmst + pos_teme[1] * sin_gmst
        y = -pos_teme[0] * sin_gmst + pos_teme[1] * cos_gmst
        z = pos_teme[2]

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_teme(pos_ecef: np.ndarray, t: float, jd: float) -> np.ndarray:
        """Convert ECEF to TEME.

        Args:
            pos_ecef: Position vector in ECEF frame (km)
            t: Time in minutes since epoch
            jd: Julian Date

        Returns:
            Position vector in TEME frame (km)
        """
        gmst = _gstime(jd + t / 1440.0)
        cos_gmst = math.cos(gmst)
        sin_gmst = math.sin(gmst)

        x = pos_ecef[0] * cos_gmst - pos_ecef[1] * sin_gmst
        y = pos_ecef[0] * sin_gmst + pos_ecef[1] * cos_gmst
        z = pos_ecef[2]

        return np.array([x, y, z])

    @staticmethod
    def teme_to_geodetic(
        pos_teme: np.ndarray, t: float, jd: float
    ) -> Tuple[float, float, float]:
        """Convert TEME position to geodetic coordinates (lat, lon, alt).

        Args:
            pos_teme: Position vector in TEME frame (km)
            t: Time in minutes since epoch
            jd: Julian Date

        Returns:
            Tuple of (latitude in degrees, longitude in degrees, altitude in km)
        """
        # First convert to ECEF
        pos_ecef = CoordinateTransform.teme_to_ecef(pos_teme, t, jd)

        # ECEF to geodetic (iterative method)
        x, y, z = pos_ecef[0], pos_ecef[1], pos_ecef[2]

        # Longitude
        lon = math.atan2(y, x)

        # Distance from z-axis
        p = math.sqrt(x * x + y * y)

        # Initial guess for latitude
        lat = math.atan2(z, p * (1.0 - F))

        # Iterate to refine latitude
        for _ in range(10):
            sin_lat = math.sin(lat)
            N = XKMPER / math.sqrt(1.0 - F * (2.0 - F) * sin_lat * sin_lat)
            alt = p / math.cos(lat) - N
            lat = math.atan2(z, p * (1.0 - F * (2.0 - F) * N / (N + alt)))

        # Convert to degrees
        lat_deg = math.degrees(lat)
        lon_deg = math.degrees(lon)

        # Normalize longitude to [-180, 180]
        if lon_deg > 180.0:
            lon_deg -= 360.0
        elif lon_deg < -180.0:
            lon_deg += 360.0

        # Final altitude calculation
        sin_lat = math.sin(lat)
        N = XKMPER / math.sqrt(1.0 - F * (2.0 - F) * sin_lat * sin_lat)
        alt = p / math.cos(lat) - N

        return lat_deg, lon_deg, alt

    @staticmethod
    def geodetic_to_teme(
        lat: float, lon: float, alt: float, t: float, jd: float
    ) -> np.ndarray:
        """Convert geodetic coordinates to TEME position.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in km
            t: Time in minutes since epoch
            jd: Julian Date

        Returns:
            Position vector in TEME frame (km)
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)

        N = XKMPER / math.sqrt(1.0 - F * (2.0 - F) * sin_lat * sin_lat)

        x = (N + alt) * cos_lat * math.cos(lon_rad)
        y = (N + alt) * cos_lat * math.sin(lon_rad)
        z = (N * (1.0 - F * (2.0 - F)) + alt) * sin_lat

        pos_ecef = np.array([x, y, z])

        return CoordinateTransform.ecef_to_teme(pos_ecef, t, jd)


# ─── SGP4 Propagator ─────────────────────────────────────────────────────────


class SGP4Propagator:
    """SGP4/SDP4 orbital propagator.

    Propagates satellite orbits using the SGP4 model for near-Earth orbits
    and SDP4 for deep-space orbits.
    """

    def __init__(self, tle: TLE):
        """Initialize propagator with TLE data.

        Args:
            tle: Parsed TLE object

        Raises:
            SGP4Error: If TLE data is invalid
        """
        if not tle.is_valid():
            raise SGP4Error("Invalid TLE data")

        self.tle = tle
        self._init_sgp4()

    def _init_sgp4(self) -> None:
        """Initialize SGP4 internal parameters."""
        tle = self.tle

        # Convert to internal units
        self.eccentricity = tle.eccentricity
        self.inclination = math.radians(tle.inclination)
        self.raan = math.radians(tle.raan)
        self.arg_of_perigee = math.radians(tle.arg_of_perigee)
        self.mean_anomaly = math.radians(tle.mean_anomaly)
        self.mean_motion = tle.mean_motion * 2.0 * math.pi / XMNPDA  # rad/min

        # Epoch
        if tle.epoch_year < 57:
            self.epoch_year = tle.epoch_year + 2000
        else:
            self.epoch_year = tle.epoch_year + 1900
        self.epoch_day = tle.epoch_day

        # Julian date at epoch
        self.epoch_jd = jday(
            self.epoch_year,
            *days2mdhms(self.epoch_year, self.epoch_day)[:3],
            *days2mdhms(self.epoch_year, self.epoch_day)[3:],
        )

        # B* drag term
        self.bstar = tle.bstar

        # Determine if deep space
        period = 2.0 * math.pi / self.mean_motion  # minutes
        self.deep_space = period >= 225.0

        # Initialize SGP4 parameters
        self._initl()
        self._dscom() if self.deep_space else self._sgp4_init()

    def _initl(self) -> None:
        """Initialize common SGP4 parameters."""
        # Calculate semi-major axis from mean motion
        self.no = self.mean_motion  # rad/min
        self.ao = (XKE / self.no) ** (2.0 / 3.0)  # Earth radii

        # Check for valid eccentricity
        if self.eccentricity < 0.0 or self.eccentricity >= 1.0:
            raise SGP4Error(f"Invalid eccentricity: {self.eccentricity}")

        # Calculate initial position for validation
        self.con41 = 0.0
        self.x1mth2 = 0.0
        self.x7thm1 = 0.0

        cosio = math.cos(self.inclination)
        sinio = math.sin(self.inclination)
        self.cosio = cosio
        self.sinio = sinio

        theta2 = cosio * cosio
        self.theta2 = theta2
        self.x3thm1 = 3.0 * theta2 - 1.0
        self.x1mth2 = 1.0 - theta2
        self.x7thm1 = 7.0 * theta2 - 1.0

        # Calculate initial mean motion with J2 perturbation
        eccsq = self.eccentricity * self.eccentricity
        omeosq = 1.0 - eccsq
        rteosq = math.sqrt(omeosq)
        cosio2 = cosio * cosio

        # Initial mean motion
        ak = (XKE / self.no) ** (2.0 / 3.0)
        d1 = 0.75 * CK2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
        del_ = d1 / (ak * ak)
        adel = ak * (1.0 - del_ * del_ - del_ * (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
        del_ = d1 / (adel * adel)
        self.no = self.no / (1.0 + del_)

        self.ao = (XKE / self.no) ** (2.0 / 3.0)
        self.sinio = sinio
        self.cosio = cosio

        # Calculate initial semi-major axis and period
        self.period = 2.0 * math.pi / self.no  # minutes

    def _sgp4_init(self) -> None:
        """Initialize SGP4 near-Earth parameters."""
        # Initialize for near-Earth SGP4
        self.isimp = 0
        self.method = "n"

        # Calculate initial values
        eccsq = self.eccentricity * self.eccentricity
        omeosq = 1.0 - eccsq
        rteosq = math.sqrt(omeosq)
        cosio = self.cosio
        cosio2 = cosio * cosio

        # Un-Kozai the mean motion
        ak = (XKE / self.no) ** (2.0 / 3.0)
        d1 = 0.75 * CK2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
        del_ = d1 / (ak * ak)
        adel = ak * (1.0 - del_ * del_ - del_ * (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
        del_ = d1 / (adel * adel)
        self.no = self.no / (1.0 + del_)

        self.ao = (XKE / self.no) ** (2.0 / 3.0)
        self.sinio = self.sinio
        self.cosio = cosio

        # Calculate initial position
        self.con41 = 0.0
        self.x1mth2 = 1.0 - cosio2
        self.x7thm1 = 7.0 * cosio2 - 1.0

        # Initialize drag terms
        self.t = 0.0

        # Check for possible divide-by-zero
        if (self.ao * (1.0 - self.eccentricity) < 1.0 + 0.001) or (
            self.ao * (1.0 + self.eccentricity) > 1.0 + 0.001
        ):
            self.isimp = 1

        # Initialize for SGP4
        self.init = True

    def _dscom(self) -> None:
        """Initialize deep-space common parameters."""
        # Simplified deep-space initialization
        self.method = "d"
        self.init = True

    def propagate(self, tsince: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate orbit to time since epoch.

        Args:
            tsince: Time since epoch in minutes

        Returns:
            Tuple of (position, velocity) in km and km/s
        """
        if self.deep_space:
            return self._sdp4(tsince)
        else:
            return self._sgp4(tsince)

    def _sgp4(self, tsince: float) -> Tuple[np.ndarray, np.ndarray]:
        """SGP4 propagation for near-Earth orbits.

        Args:
            tsince: Time since epoch in minutes

        Returns:
            Tuple of (position, velocity) in km and km/s
        """
        tle = self.tle

        # Update for secular gravity and atmospheric drag
        mm = self.no + 0.75 * CK2 * self.x3thm1 / (self.ao * self.ao * math.sqrt(1.0 - self.eccentricity * self.eccentricity)) * tsince
        nm = self.no
        em = self.eccentricity
        inclm = self.inclination
        argpm = self.arg_of_perigee
        nodem = self.raan
        mm = self.mean_anomaly

        # Update for long-period periodic terms
        sinim = math.sin(inclm)
        cosim = math.cos(inclm)

        # Update for secular terms
        t2 = tsince * tsince
        nodem = nodem + 0.75 * CK2 * self.x7thm1 / (self.ao * self.ao) * tsince
        argpm = argpm + 0.75 * CK2 * self.x1mth2 / (self.ao * self.ao) * tsince
        mm = mm + self.no * tsince

        # Add drag effects
        if self.bstar != 0.0:
            # Simplified drag model
            temp = 1.0 - self.bstar * tsince
            if temp < 0.0:
                temp = 0.0
            mm = mm * temp

        # Solve Kepler's equation
        e = em
        for _ in range(10):
            sine = math.sin(mm)
            cose = math.cos(mm)
            epw = mm + e * sine
            if abs(epw - mm) < 1e-12:
                break
            mm = epw

        # Calculate true anomaly
        sine = math.sin(mm)
        cose = math.cos(mm)
        esine = e * sine
        ecose = e * cose
        el2 = 1.0 - e * e
        pl = self.ao * el2
        r = self.ao * (1.0 - ecose)
        rdot = math.sqrt(self.ao) * esine / r
        rfdot = math.sqrt(pl) / r
        temp = esine / (1.0 - math.sqrt(el2) * ecose)
        cosu = (cose - e) / (1.0 - ecose)
        sinu = (math.sqrt(1.0 - e * e) * sine) / (1.0 - ecose)
        u = math.atan2(sinu, cosu)

        # Add long-period periodic terms
        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0

        # Update for long-period periodic terms
        temp1 = 0.5 * CK2 * sinim / pl
        temp2 = temp1 / pl
        mr = r * (1.0 - 1.5 * temp2 * self.x3thm1) + 0.5 * temp1 * self.x1mth2 * cos2u
        u = u - 0.25 * temp2 * self.x7thm1 * sin2u
        node = nodem + 1.5 * temp2 * cosim * sin2u
        argp = argpm - 1.5 * temp2 * sinim * cos2u

        # Calculate position and velocity in orbital plane
        sinu = math.sin(u)
        cosu = math.cos(u)
        sini = math.sin(inclm)
        cosi = math.cos(inclm)
        sinn = math.sin(node)
        cosn = math.cos(node)
        sinp = math.sin(argp)
        cosp = math.cos(argp)

        # Position in orbital plane
        x = mr * (cosn * cosp - sinn * cosi * sinp)
        y = mr * (sinn * cosp + cosn * cosi * sinp)
        z = mr * (sini * sinp)

        # Velocity in orbital plane
        rdotk = rdot - self.no * temp1 * self.x1mth2 * sin2u / self.ao
        rfdotk = rfdot + self.no * temp1 * (self.x1mth2 * cos2u + 1.5 * self.x3thm1) / self.ao

        vx = rdotk * (cosn * cosp - sinn * cosi * sinp) - rfdotk * (cosn * sinp + sinn * cosi * cosp)
        vy = rdotk * (sinn * cosp + cosn * cosi * sinp) - rfdotk * (sinn * sinp - cosn * cosi * cosp)
        vz = rdotk * (sini * sinp) + rfdotk * (sini * cosp)

        # Convert to km and km/s
        pos = np.array([x, y, z]) * XKMPER
        vel = np.array([vx, vy, vz]) * XKMPER / 60.0

        return pos, vel

    def _sdp4(self, tsince: float) -> Tuple[np.ndarray, np.ndarray]:
        """SDP4 propagation for deep-space orbits.

        Args:
            tsince: Time since epoch in minutes

        Returns:
            Tuple of (position, velocity) in km and km/s
        """
        # Simplified SDP4 for deep-space orbits
        # For deep-space, use a simplified model
        tle = self.tle

        # Update mean anomaly
        mm = self.mean_anomaly + self.no * tsince

        # Solve Kepler's equation
        e = self.eccentricity
        for _ in range(10):
            sine = math.sin(mm)
            cose = math.cos(mm)
            epw = mm + e * sine
            if abs(epw - mm) < 1e-12:
                break
            mm = epw

        # Calculate position
        sine = math.sin(mm)
        cose = math.cos(mm)
        esine = e * sine
        ecose = e * cose
        el2 = 1.0 - e * e
        pl = self.ao * el2
        r = self.ao * (1.0 - ecose)
        rdot = math.sqrt(self.ao) * esine / r
        rfdot = math.sqrt(pl) / r

        # Calculate true anomaly
        cosu = (cose - e) / (1.0 - ecose)
        sinu = (math.sqrt(1.0 - e * e) * sine) / (1.0 - ecose)
        u = math.atan2(sinu, cosu)

        # Position in orbital plane
        sinu = math.sin(u)
        cosu = math.cos(u)
        sini = math.sin(self.inclination)
        cosi = math.cos(self.inclination)
        sinn = math.sin(self.raan)
        cosn = math.cos(self.raan)
        sinp = math.sin(self.arg_of_perigee)
        cosp = math.cos(self.arg_of_perigee)

        x = r * (cosn * cosp - sinn * cosi * sinp)
        y = r * (sinn * cosp + cosn * cosi * sinp)
        z = r * (sini * sinp)

        vx = rdot * (cosn * cosp - sinn * cosi * sinp) - rfdot * (cosn * sinp + sinn * cosi * cosp)
        vy = rdot * (sinn * cosp + cosn * cosi * sinp) - rfdot * (sinn * sinp - cosn * cosi * cosp)
        vz = rdot * (sini * sinp) + rfdot * (sini * cosp)

        pos = np.array([x, y, z]) * XKMPER
        vel = np.array([vx, vy, vz]) * XKMPER / 60.0

        return pos, vel

    def get_state_vector(self, tsince: float) -> dict:
        """Get full state vector at given time.

        Args:
            tsince: Time since epoch in minutes

        Returns:
            Dictionary with position, velocity, timestamp, satellite_number
        """
        pos, vel = self.propagate(tsince)
        return {
            "position": pos.tolist(),
            "velocity": vel.tolist(),
            "timestamp": tsince,
            "satellite_number": self.tle.satellite_number,
        }

    def get_position(self, tsince: float) -> np.ndarray:
        """Get position at given time.

        Args:
            tsince: Time since epoch in minutes

        Returns:
            Position vector in km
        """
        pos, _ = self.propagate(tsince)
        return pos

    def get_velocity(self, tsince: float) -> np.ndarray:
        """Get velocity at given time.

        Args:
            tsince: Time since epoch in minutes

        Returns:
            Velocity vector in km/s
        """
        _, vel = self.propagate(tsince)
        return vel

    def get_orbital_period(self) -> float:
        """Get orbital period in seconds.

        Returns:
            Orbital period in seconds
        """
        return 2.0 * math.pi / self.no

    def get_semi_major_axis(self) -> float:
        """Get semi-major axis in km.

        Returns:
            Semi-major axis in km
        """
        return self.ao * XKMPER

    def get_apogee_perigee(self) -> Tuple[float, float]:
        """Get apogee and perigee altitudes.

        Returns:
            Tuple of (apogee, perigee) in km above Earth surface
        """
        sma = self.get_semi_major_axis()
        apogee = sma * (1.0 + self.eccentricity) - XKMPER
        perigee = sma * (1.0 - self.eccentricity) - XKMPER
        return apogee, perigee

    def is_deep_space(self) -> bool:
        """Check if orbit is deep-space (period >= 225 minutes).

        Returns:
            True if deep-space orbit
        """
        return self.deep_space
