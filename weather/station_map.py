"""City → ICAO code + WU URL path + timezone mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationInfo:
    icao: str
    wu_path: str  # e.g. "kr/incheon" for WU URL
    timezone: str
    display_unit: str = "C"  # "C" or "F" — what Polymarket uses for this city
    city_aliases: tuple[str, ...] = ()
    ghcnd_id: str | None = None  # GHCND station ID for NOAA precip data
    latitude: float | None = None
    longitude: float | None = None


# Static mapping: city name (lowercase) → StationInfo
# Extend as new Polymarket temperature markets appear.
STATION_MAP: dict[str, StationInfo] = {
    "seoul": StationInfo(
        icao="RKSI",
        wu_path="kr/incheon",
        timezone="Asia/Seoul",
        city_aliases=("incheon", "seoul incheon"),
        latitude=37.4692,
        longitude=126.4505,
    ),
    "new york": StationInfo(
        icao="KLGA",
        wu_path="us/ny/new-york-city",
        timezone="America/New_York",
        display_unit="F",
        city_aliases=("nyc", "new york city", "laguardia"),
        ghcnd_id="USW00014732",
        latitude=40.7794,
        longitude=-73.8803,
    ),
    "chicago": StationInfo(
        icao="KORD",
        wu_path="us/il/chicago",
        timezone="America/Chicago",
        display_unit="F",
        city_aliases=("ord", "o'hare", "chicago o'hare"),
        ghcnd_id="USW00094846",
        latitude=41.9742,
        longitude=-87.9073,
    ),
    "london": StationInfo(
        icao="EGLC",
        wu_path="gb/london",
        timezone="Europe/London",
        city_aliases=("london city",),
        latitude=51.5053,
        longitude=0.0553,
    ),
    "dallas": StationInfo(
        icao="KDAL",
        wu_path="us/tx/dallas",
        timezone="America/Chicago",
        display_unit="F",
        city_aliases=("dfw", "dallas-fort worth", "dallas fort worth", "dallas love field"),
        ghcnd_id="USW00013960",
        latitude=32.8471,
        longitude=-96.8517,
    ),
    "toronto": StationInfo(
        icao="CYYZ",
        wu_path="ca/on/toronto",
        timezone="America/Toronto",
        city_aliases=("yyz",),
        latitude=43.6772,
        longitude=-79.6306,
    ),
    "atlanta": StationInfo(
        icao="KATL",
        wu_path="us/ga/atlanta",
        timezone="America/New_York",
        display_unit="F",
        city_aliases=("atl",),
        ghcnd_id="USW00013874",
        latitude=33.6407,
        longitude=-84.4277,
    ),
    "miami": StationInfo(
        icao="KMIA",
        wu_path="us/fl/miami",
        timezone="America/New_York",
        display_unit="F",
        city_aliases=("mia",),
        ghcnd_id="USW00012839",
        latitude=25.7959,
        longitude=-80.2870,
    ),
    "seattle": StationInfo(
        icao="KSEA",
        wu_path="us/wa/seattle",
        timezone="America/Los_Angeles",
        display_unit="F",
        city_aliases=("sea",),
        ghcnd_id="USW00024233",
        latitude=47.4502,
        longitude=-122.3088,
    ),
    "ankara": StationInfo(
        icao="LTAC",
        wu_path="tr/ankara",
        timezone="Europe/Istanbul",
        city_aliases=("ankara turkey",),
        latitude=40.1281,
        longitude=32.9951,
    ),
    "buenos aires": StationInfo(
        icao="SAEZ",
        wu_path="ar/buenos-aires",
        timezone="America/Argentina/Buenos_Aires",
        city_aliases=("buenos aires",),
        latitude=-34.8222,
        longitude=-58.5358,
    ),
    "wellington": StationInfo(
        icao="NZWN",
        wu_path="nz/wellington",
        timezone="Pacific/Auckland",
        city_aliases=("wellington nz",),
        latitude=-41.3272,
        longitude=174.8053,
    ),
}

# Build reverse lookups
_ALIAS_MAP: dict[str, str] = {}
_ICAO_MAP: dict[str, str] = {}
for _city, _info in STATION_MAP.items():
    _ICAO_MAP[_info.icao] = _city
    for _alias in _info.city_aliases:
        _ALIAS_MAP[_alias.lower()] = _city


def lookup_city(name: str) -> StationInfo | None:
    """Look up station info by city name or alias (case-insensitive)."""
    key = name.lower().strip()
    if key in STATION_MAP:
        return STATION_MAP[key]
    if key in _ALIAS_MAP:
        return STATION_MAP[_ALIAS_MAP[key]]
    return None


def lookup_icao(icao: str) -> StationInfo | None:
    """Look up station info by ICAO code."""
    city = _ICAO_MAP.get(icao.upper())
    if city:
        return STATION_MAP[city]
    return None


def all_icao_codes() -> list[str]:
    """Return all tracked ICAO codes."""
    return [info.icao for info in STATION_MAP.values()]


def lookup_ghcnd(city_name: str) -> str | None:
    """Look up GHCND station ID by city name (for NOAA precip data)."""
    info = lookup_city(city_name)
    if info is not None:
        return info.ghcnd_id
    return None
