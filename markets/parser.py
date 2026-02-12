"""Parse market titles → city, date, bucket ranges (temperature + precipitation)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, date

from dateutil import parser as dateutil_parser


# ---------------------------------------------------------------------------
# Temperature markets
# ---------------------------------------------------------------------------


@dataclass
class MarketInfo:
    """Parsed event-level info from a Polymarket temperature event."""

    city: str
    market_date: date
    raw_title: str


@dataclass
class BucketInfo:
    """Parsed bucket-level info from a market question."""

    bucket_type: str  # "exact", "geq", "leq"
    bucket_value: int  # value in the market's native unit
    unit: str  # "C" or "F"
    raw_question: str


def parse_event_title(title: str) -> MarketInfo | None:
    """Parse event title like 'Highest temperature in Seoul on February 10?'

    Returns MarketInfo or None if unparseable.
    """
    # Pattern: "Highest temperature in {city} on {date}?"
    match = re.search(
        r'(?:highest|high)\s+temperature\s+in\s+(.+?)\s+on\s+(.+?)[\?]?$',
        title,
        re.IGNORECASE,
    )
    if not match:
        return None

    city = match.group(1).strip()
    date_str = match.group(2).strip().rstrip("?")

    try:
        parsed_date = dateutil_parser.parse(date_str, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None

    return MarketInfo(city=city, market_date=parsed_date, raw_title=title)


def parse_bucket_question(question: str) -> BucketInfo | None:
    """Parse bucket question from market outcomes.

    Handles both full-sentence and short formats, in °C and °F:
        "Will the highest temperature in New York City be 44°F or higher on February 11?"
        "be 8C or below"     → BucketInfo(type="leq", value=8, unit="C")
        "be 9C"              → BucketInfo(type="exact", value=9, unit="C")
        "be 14C or higher"   → BucketInfo(type="geq", value=14, unit="C")
        "44°F or higher"     → BucketInfo(type="geq", value=44, unit="F")
    """
    q = question.strip()

    # "X°C/F or higher/above/more" (full-sentence or short form)
    match = re.search(r'(-?\d+)\s*°?([CF])\s+or\s+(?:higher|above|more)', q, re.IGNORECASE)
    if match:
        return BucketInfo(
            bucket_type="geq",
            bucket_value=int(match.group(1)),
            unit=match.group(2).upper(),
            raw_question=q,
        )

    # "X°C/F or below/lower/less"
    match = re.search(r'(-?\d+)\s*°?([CF])\s+or\s+(?:below|lower|less)', q, re.IGNORECASE)
    if match:
        return BucketInfo(
            bucket_type="leq",
            bucket_value=int(match.group(1)),
            unit=match.group(2).upper(),
            raw_question=q,
        )

    # Exact: "be X°C/F on ..." (full-sentence) or "be X°C/F" or just "X°C/F"
    match = re.search(r'(?:^|be\s+)(-?\d+)\s*°?([CF])\s+on\s', q, re.IGNORECASE)
    if match:
        return BucketInfo(
            bucket_type="exact",
            bucket_value=int(match.group(1)),
            unit=match.group(2).upper(),
            raw_question=q,
        )

    # Exact short form: "be X°C/F" or "X°C/F" at end of string
    match = re.search(r'(?:^|be\s+)(-?\d+)\s*°?([CF])\s*$', q, re.IGNORECASE)
    if match:
        return BucketInfo(
            bucket_type="exact",
            bucket_value=int(match.group(1)),
            unit=match.group(2).upper(),
            raw_question=q,
        )

    return None


# ---------------------------------------------------------------------------
# Precipitation markets
# ---------------------------------------------------------------------------


@dataclass
class PrecipMarketInfo:
    """Parsed event-level info from a Polymarket precipitation event."""

    city: str
    month: str  # "February"
    year: int  # 2026
    raw_title: str


@dataclass
class PrecipBucketInfo:
    """Parsed bucket-level info from a precipitation market question."""

    bucket_type: str  # "lt", "range", "gt"
    low_inches: float | None
    high_inches: float | None
    raw_question: str


def parse_precip_event_title(title: str) -> PrecipMarketInfo | None:
    """Parse precipitation event title.

    Examples:
        "Precipitation in NYC in February?" → PrecipMarketInfo(city="NYC", month="February", year=2026)
        "Precipitation in Seattle in February 2026?" → same with explicit year
    """
    match = re.search(
        r'precipitation\s+in\s+(.+?)\s+in\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        r'(?:\s+(\d{4}))?',
        title,
        re.IGNORECASE,
    )
    if not match:
        return None

    city = match.group(1).strip()
    month = match.group(2).capitalize()
    year = int(match.group(3)) if match.group(3) else datetime.utcnow().year

    return PrecipMarketInfo(city=city, month=month, year=year, raw_title=title)


def parse_precip_bucket_question(question: str) -> PrecipBucketInfo | None:
    """Parse precipitation bucket question.

    Examples:
        "Less than 2 inches"         → {type: "lt", low: None, high: 2.0}
        "2 to 3 inches"              → {type: "range", low: 2.0, high: 3.0}
        "More than 6 inches"         → {type: "gt", low: 6.0, high: None}
        "6 inches or more"           → {type: "gt", low: 6.0, high: None}
        "Under 2 inches"             → {type: "lt", low: None, high: 2.0}
    """
    q = question.strip()

    # "Less than X inches" / "Under X inches" / "Below X inches"
    match = re.search(r'(?:less\s+than|under|below)\s+([\d.]+)\s+inch', q, re.IGNORECASE)
    if match:
        return PrecipBucketInfo(
            bucket_type="lt",
            low_inches=None,
            high_inches=float(match.group(1)),
            raw_question=q,
        )

    # "More than X inches" / "Over X inches" / "Above X inches" / "X inches or more"
    match = re.search(r'(?:more\s+than|over|above)\s+([\d.]+)\s+inch', q, re.IGNORECASE)
    if match:
        return PrecipBucketInfo(
            bucket_type="gt",
            low_inches=float(match.group(1)),
            high_inches=None,
            raw_question=q,
        )
    match = re.search(r'([\d.]+)\s+inch(?:es)?\s+or\s+more', q, re.IGNORECASE)
    if match:
        return PrecipBucketInfo(
            bucket_type="gt",
            low_inches=float(match.group(1)),
            high_inches=None,
            raw_question=q,
        )

    # "X to Y inches"
    match = re.search(r'([\d.]+)\s+to\s+([\d.]+)\s+inch', q, re.IGNORECASE)
    if match:
        return PrecipBucketInfo(
            bucket_type="range",
            low_inches=float(match.group(1)),
            high_inches=float(match.group(2)),
            raw_question=q,
        )

    return None
