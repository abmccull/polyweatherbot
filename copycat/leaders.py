"""Seed wallet universe and policy defaults for copycat strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedLeader:
    name: str
    wallet: str
    tier: str
    base_status: str  # core/probation/excluded
    risk_multiplier: float = 1.0


SEED_LEADERS: list[SeedLeader] = [
    # Core
    SeedLeader(
        name="4gibg4i3o",
        wallet="0x0d16999d6ba24c2c2c8bff93f91014cc43fad24d",
        tier="S-TIER",
        base_status="core",
    ),
    SeedLeader(
        name="SappySeal67",
        wallet="0x067f7f6e3d5363f1cf962499d1be2ca6de8be041",
        tier="A-TIER",
        base_status="core",
    ),
    # Probation
    SeedLeader(
        name="GamblingIsAllYouNeed",
        wallet="0x507e52ef684ca2dd91f90a9d26d149dd3288beae",
        tier="B-TIER",
        base_status="probation",
    ),
    SeedLeader(
        name="swisstony",
        wallet="0x204f72f35326db932158cba6adff0b9a1da95e14",
        tier="B-TIER",
        base_status="probation",
    ),
    SeedLeader(
        name="gmanas",
        wallet="0xe90bec87d9ef430f27f9dcfe72c34b76967d5da2",
        tier="A-TIER",
        base_status="probation",
        risk_multiplier=0.40,
    ),
    SeedLeader(
        name="GetaLife",
        wallet="0xada56a3f1a9d7d3fe7ec8cbcaa0865109c57db70",
        tier="C-TIER",
        base_status="probation",
    ),
    # Excluded
    SeedLeader(
        name="eanvanezygv",
        wallet="0x9d94f602535e518ee1cb6aade0ca9569f1b1017d",
        tier="A-TIER",
        base_status="excluded",
    ),
    SeedLeader(
        name="LuckyNFT444",
        wallet="0x4bac379da2f29d87c01ff737843e396a2cec02b1",
        tier="B-TIER",
        base_status="excluded",
    ),
]
