"""All settings loaded from env vars with tunable parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class TunableParam:
    """A parameter that the learning module can adjust within safe bounds."""

    value: float
    min_val: float
    max_val: float
    consecutive_adjustments: int = 0
    max_consecutive: int = 3

    def adjust(self, delta: float) -> bool:
        """Adjust value by delta, respecting bounds and consecutive limits."""
        direction = 1 if delta > 0 else -1
        if self.consecutive_adjustments != 0:
            prev_direction = 1 if self.consecutive_adjustments > 0 else -1
            if prev_direction == direction and abs(self.consecutive_adjustments) >= self.max_consecutive:
                return False

        new_val = max(self.min_val, min(self.max_val, self.value + delta))
        if new_val == self.value:
            return False

        self.value = round(new_val, 4)
        if direction == (1 if self.consecutive_adjustments > 0 else -1) or self.consecutive_adjustments == 0:
            self.consecutive_adjustments += direction
        else:
            self.consecutive_adjustments = direction
        return True


@dataclass
class Config:
    # Polymarket
    poly_private_key: str = ""
    poly_funder_address: str = ""
    poly_api_key: str = ""
    poly_api_secret: str = ""
    poly_api_passphrase: str = ""
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137  # Polygon

    # Synoptic
    synoptic_api_token: str = ""

    # Database
    db_path: str = "station_sniper.db"

    # Mode
    dry_run: bool = True

    # Portfolio
    initial_bankroll: float = 200.0

    # Timing (seconds)
    market_scan_interval: int = 900  # 15 min
    observation_interval: int = 60  # 1 min
    learning_interval: int = 1800  # 30 min
    heartbeat_interval: int = 300  # 5 min
    redemption_interval: int = 3600  # 60 min

    # Tunable parameters
    min_confidence: TunableParam = field(
        default_factory=lambda: TunableParam(value=0.85, min_val=0.70, max_val=0.95)
    )
    max_price: TunableParam = field(
        default_factory=lambda: TunableParam(value=0.65, min_val=0.30, max_val=0.85)
    )
    bet_pct: TunableParam = field(
        default_factory=lambda: TunableParam(value=0.05, min_val=0.02, max_val=0.10)
    )

    # Fees
    maker_fee_rate: float = 0.0     # Fee on GTC limit orders (maker)
    taker_fee_rate: float = 0.02    # Fee on market/immediate orders (taker)
    kelly_fee_rate: float = 0.02    # Conservative fee for Kelly odds calculation
    sell_fee_rate: float = 0.01     # Blended fee for sell orders

    # Kelly sizing
    kelly_mode: bool = True
    kelly_base_fraction: float = 0.60
    kelly_min_fraction: float = 0.40
    kelly_max_fraction: float = 0.75
    kelly_confidence_floor: float = 0.80
    kelly_probability_haircut: float = 0.03
    kelly_min_edge: float = 0.05
    kelly_ramp_trades: int = 20
    drawdown_throttle_start: float = 0.15
    drawdown_throttle_full: float = 0.30

    # Sizing
    min_bet: float = 5.0
    max_bet: float = 150.0
    max_single_market_exposure: float = 0.35  # 35% of portfolio
    max_total_deployed: float = 0.75  # 75% of portfolio
    liquidity_cap: float = 0.80  # never take >80% of visible depth

    # Signal detection gates
    min_price: float = 0.30               # Don't buy YES tokens below this (market too uncertain)
    metar_max_age_minutes: float = 30.0   # METAR must be fresher than this
    geq_min_hour: int = 12                # Earliest local hour for geq signals (noon)
    leq_min_hour: int = 17                # Earliest local hour for leq signals (after peak heating)

    # Circuit breaker
    consecutive_loss_limit: int = 4
    circuit_breaker_hours: float = 1.5
    halt_portfolio_pct: float = 0.35  # halt if portfolio drops below 35% of initial

    # Profit lock + trailing stop
    profit_lock_enabled: bool = True
    profit_lock_trigger_ratio: float = 3.0    # Trigger when price >= 3x entry
    profit_lock_recoup_multiple: float = 2.0  # Sell enough to recoup 2x cost
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.35           # Sell if price drops 35% from peak
    position_check_interval: int = 60         # Check positions every 60s

    # DB cleanup
    archive_days: int = 90

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        cfg.poly_private_key = os.getenv("POLY_PRIVATE_KEY", "")
        cfg.poly_funder_address = os.getenv("POLY_FUNDER_ADDRESS", "")
        cfg.poly_api_key = os.getenv("POLY_API_KEY", "")
        cfg.poly_api_secret = os.getenv("POLY_API_SECRET", "")
        cfg.poly_api_passphrase = os.getenv("POLY_API_PASSPHRASE", "")
        cfg.synoptic_api_token = os.getenv("SYNOPTIC_API_TOKEN", "")
        cfg.db_path = os.getenv("DB_PATH", "station_sniper.db")
        cfg.dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
        cfg.initial_bankroll = float(os.getenv("INITIAL_BANKROLL", "200.0"))
        return cfg

    @property
    def db_url(self) -> str:
        return f"sqlite:///{Path(self.db_path).resolve()}"
