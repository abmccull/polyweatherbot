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
    clob_market_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    clob_user_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
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

    # Realtime transport
    enable_market_ws: bool = True
    enable_user_ws: bool = True
    ws_reconnect_seconds: int = 5
    ws_ping_interval: int = 20
    market_price_max_age: int = 5

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
    dynamic_max_bet_enabled: bool = True
    dynamic_max_bet_pct: float = 0.08       # Base max bet as % of current bankroll
    dynamic_max_bet_floor: float = 150.0    # Never below this max-bet floor
    dynamic_max_bet_cap: float = 5000.0     # Hard cap for dynamic max-bet growth
    max_single_market_exposure: float = 0.35  # 35% of portfolio
    max_total_deployed: float = 0.75  # 75% of portfolio
    liquidity_cap: float = 0.80  # never take >80% of visible depth

    # Adaptive aggression (only scales up when statistical edge is persistent)
    aggression_enabled: bool = True
    aggression_min_samples: int = 40
    aggression_target_win_rate: float = 0.90
    aggression_confidence_z: float = 1.64   # ~90% one-sided confidence
    aggression_max_boost: float = 0.75      # at most +75% max-bet expansion
    aggression_drawdown_guard: float = 0.12 # disable boost above this drawdown

    # EV gate (post-cost edge filter)
    enable_ev_gate: bool = True
    min_expected_edge: float = 0.02      # minimum net edge per share after fees/slippage
    min_expected_profit: float = 0.50    # minimum expected PnL in USDC per trade
    ev_base_slippage_bps: float = 20.0   # baseline execution slippage assumption
    ev_depth_slippage_max: float = 0.02  # depth-penalty cap as per-share price impact
    ev_spread_weight: float = 0.50       # penalize half the visible spread
    ev_dynamic_slippage: bool = True     # adapt baseline slippage from recent live fills

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
    hard_stop_loss_enabled: bool = True
    hard_stop_loss_pct: float = 0.65          # Exit if current price drops 65% from avg entry
    position_check_interval: int = 60         # Check positions every 60s

    # DB cleanup
    archive_days: int = 90

    @classmethod
    def from_env(cls) -> Config:
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.lower() in ("true", "1", "yes")

        cfg = cls()
        cfg.poly_private_key = os.getenv("POLY_PRIVATE_KEY", "")
        cfg.poly_funder_address = os.getenv("POLY_FUNDER_ADDRESS", "")
        cfg.poly_api_key = os.getenv("POLY_API_KEY", "")
        cfg.poly_api_secret = os.getenv("POLY_API_SECRET", "")
        cfg.poly_api_passphrase = os.getenv("POLY_API_PASSPHRASE", "")
        cfg.clob_market_ws_url = os.getenv("CLOB_MARKET_WS_URL", cfg.clob_market_ws_url)
        cfg.clob_user_ws_url = os.getenv("CLOB_USER_WS_URL", cfg.clob_user_ws_url)
        cfg.synoptic_api_token = os.getenv("SYNOPTIC_API_TOKEN", "")
        cfg.db_path = os.getenv("DB_PATH", "station_sniper.db")
        cfg.dry_run = _env_bool("DRY_RUN", True)
        cfg.enable_market_ws = _env_bool("ENABLE_MARKET_WS", cfg.enable_market_ws)
        cfg.enable_user_ws = _env_bool("ENABLE_USER_WS", cfg.enable_user_ws)
        cfg.ws_reconnect_seconds = int(os.getenv("WS_RECONNECT_SECONDS", str(cfg.ws_reconnect_seconds)))
        cfg.ws_ping_interval = int(os.getenv("WS_PING_INTERVAL", str(cfg.ws_ping_interval)))
        cfg.market_price_max_age = int(os.getenv("MARKET_PRICE_MAX_AGE", str(cfg.market_price_max_age)))
        cfg.dynamic_max_bet_enabled = _env_bool("DYNAMIC_MAX_BET_ENABLED", cfg.dynamic_max_bet_enabled)
        cfg.dynamic_max_bet_pct = float(os.getenv("DYNAMIC_MAX_BET_PCT", str(cfg.dynamic_max_bet_pct)))
        cfg.dynamic_max_bet_floor = float(os.getenv("DYNAMIC_MAX_BET_FLOOR", str(cfg.dynamic_max_bet_floor)))
        cfg.dynamic_max_bet_cap = float(os.getenv("DYNAMIC_MAX_BET_CAP", str(cfg.dynamic_max_bet_cap)))
        cfg.aggression_enabled = _env_bool("AGGRESSION_ENABLED", cfg.aggression_enabled)
        cfg.aggression_min_samples = int(os.getenv("AGGRESSION_MIN_SAMPLES", str(cfg.aggression_min_samples)))
        cfg.aggression_target_win_rate = float(
            os.getenv("AGGRESSION_TARGET_WIN_RATE", str(cfg.aggression_target_win_rate))
        )
        cfg.aggression_confidence_z = float(
            os.getenv("AGGRESSION_CONFIDENCE_Z", str(cfg.aggression_confidence_z))
        )
        cfg.aggression_max_boost = float(os.getenv("AGGRESSION_MAX_BOOST", str(cfg.aggression_max_boost)))
        cfg.aggression_drawdown_guard = float(
            os.getenv("AGGRESSION_DRAWDOWN_GUARD", str(cfg.aggression_drawdown_guard))
        )
        cfg.enable_ev_gate = _env_bool("ENABLE_EV_GATE", cfg.enable_ev_gate)
        cfg.min_expected_edge = float(os.getenv("MIN_EXPECTED_EDGE", str(cfg.min_expected_edge)))
        cfg.min_expected_profit = float(os.getenv("MIN_EXPECTED_PROFIT", str(cfg.min_expected_profit)))
        cfg.ev_base_slippage_bps = float(os.getenv("EV_BASE_SLIPPAGE_BPS", str(cfg.ev_base_slippage_bps)))
        cfg.ev_depth_slippage_max = float(os.getenv("EV_DEPTH_SLIPPAGE_MAX", str(cfg.ev_depth_slippage_max)))
        cfg.ev_spread_weight = float(os.getenv("EV_SPREAD_WEIGHT", str(cfg.ev_spread_weight)))
        cfg.ev_dynamic_slippage = _env_bool("EV_DYNAMIC_SLIPPAGE", cfg.ev_dynamic_slippage)
        cfg.hard_stop_loss_enabled = _env_bool("HARD_STOP_LOSS_ENABLED", cfg.hard_stop_loss_enabled)
        cfg.hard_stop_loss_pct = float(os.getenv("HARD_STOP_LOSS_PCT", str(cfg.hard_stop_loss_pct)))
        cfg.initial_bankroll = float(os.getenv("INITIAL_BANKROLL", "200.0"))
        return cfg

    @property
    def db_url(self) -> str:
        return f"sqlite:///{Path(self.db_path).resolve()}"
