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
    builder_api_key: str = ""
    builder_api_secret: str = ""
    builder_api_passphrase: str = ""
    relayer_url: str = "https://relayer-v2.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    clob_market_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    clob_user_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    gamma_host: str = "https://gamma-api.polymarket.com"
    polygon_rpc_urls: str = (
        "https://polygon-rpc.com,"
        "https://polygon.llamarpc.com,"
        "https://rpc.ankr.com/polygon,"
        "https://polygon-bor-rpc.publicnode.com"
    )
    polygon_rpc_timeout_seconds: int = 20
    chain_id: int = 137  # Polygon

    # Synoptic
    synoptic_api_token: str = ""

    # Database
    db_path: str = "station_sniper.db"

    # Mode
    dry_run: bool = True
    strategy_mode: str = "weather"  # "weather" or "copycat"

    # Portfolio
    initial_bankroll: float = 200.0

    # Copycat strategy
    copy_enabled: bool = True
    copy_poll_seconds: int = 10
    copy_followup_poll_seconds: int = 3
    copy_settle_max_hours: int = 72
    copy_settle_grace_seconds: int = 120
    copy_base_ticket_usd: float = 5.0
    copy_probation_multiplier: float = 0.60
    copy_gmanas_multiplier: float = 0.40
    copy_probation_max_open_risk_share: float = 0.60
    copy_probation_max_open_locks: int = 8
    copy_probation_min_seconds_between_entries: int = 20
    copy_slippage_abs_cap: float = 0.03
    copy_slippage_rel_cap: float = 0.05
    copy_min_leader_price: float = 0.05
    copy_max_leader_price: float = 0.95
    copy_skip_inactive_events: bool = True
    copy_core_poll_priority: bool = True
    copy_max_new_trades_per_poll: int = 25
    copy_max_match_exposure_usd: float = 10.0
    copy_max_open_risk_pct: float = 0.35
    copy_min_cash_buffer_usd: float = 20.0
    copy_min_matic_reserve: float = 0.30
    copy_est_matic_per_redemption: float = 0.04
    redeem_proxy_via_relayer: bool = True
    redeem_relayer_gas_limit: int = 1500000
    redeem_relayer_max_polls: int = 30
    redeem_relayer_poll_ms: int = 2000
    redeem_relayer_max_submits_per_cycle: int = 3
    redeem_relayer_submit_spacing_seconds: float = 1.25
    redeem_relayer_429_backoff_initial_seconds: int = 120
    redeem_relayer_429_backoff_max_seconds: int = 1800
    copy_dedup_enabled: bool = True
    copy_opposite_policy: str = "ignore_until_exit"
    copy_follow_buy_only: bool = True
    copy_max_wallet_trades_fetch: int = 100
    copy_trade_max_age_seconds: int = 300
    copy_activity_window_days: int = 60
    copy_min_trades_60d: int = 250
    copy_min_active_days_60d: int = 12
    leader_refresh_hours: int = 24
    leader_probation_days: int = 14
    copy_stuck_redeemable_hours: int = 6

    # Timing (seconds)
    market_scan_interval: int = 900  # 15 min
    observation_interval: int = 60  # 1 min
    learning_interval: int = 1800  # 30 min
    heartbeat_interval: int = 300  # 5 min
    redemption_interval: int = 3600  # 60 min
    redemption_quiet_hours_enabled: bool = True
    redemption_quiet_timezone: str = "America/Denver"
    redemption_quiet_start_hour: int = 23
    redemption_quiet_end_hour: int = 6

    # Realtime transport
    enable_market_ws: bool = True
    enable_user_ws: bool = True
    ws_reconnect_seconds: int = 5
    ws_ping_interval: int = 20
    market_price_max_age: int = 5
    orderbook_404_cooldown_seconds: int = 900

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

    # Self-learning (contextual Bayesian probability adjustments)
    self_learning_enabled: bool = True
    self_learning_lookback_days: int = 180
    self_learning_min_samples: int = 30
    self_learning_min_segment_samples: int = 5
    self_learning_prior_alpha: float = 2.0
    self_learning_prior_beta: float = 2.0
    self_learning_city_prior: float = 30.0
    self_learning_hour_prior: float = 20.0
    self_learning_bucket_prior: float = 16.0
    self_learning_precision_prior: float = 24.0
    self_learning_blend: float = 0.55
    self_learning_reliability_samples: int = 80
    self_learning_confidence_scale: float = 0.30
    self_learning_confidence_cap: float = 0.08

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

    # Shadow expansion mode (increase opportunity flow with strict caps)
    shadow_expansion_enabled: bool = False
    shadow_min_confidence: float = 0.78
    shadow_min_price: float = 0.20
    shadow_max_price: float = 0.75
    shadow_metar_max_age_minutes: float = 45.0
    shadow_geq_min_hour: int = 9
    shadow_leq_min_hour: int = 15
    shadow_max_bet_usd: float = 25.0
    shadow_max_bankroll_pct: float = 0.03
    shadow_min_expected_edge: float = 0.03
    shadow_min_expected_profit: float = 0.25
    shadow_exact_enabled: bool = False
    shadow_exact_min_hour: int = 15
    shadow_exact_min_confidence: float = 0.90
    shadow_exact_max_bet_usd: float = 10.0

    # Adaptive opportunity flow (relax gates when funnel is starved)
    adaptive_time_gates_enabled: bool = True
    adaptive_time_gate_no_emit_cycles: int = 90
    adaptive_time_gate_step_hours: int = 1
    adaptive_time_gate_min_geq_hour: int = 8
    adaptive_time_gate_min_leq_hour: int = 14
    adaptive_time_gate_reset_on_emit: bool = True

    # Liquidity quality gate
    liquidity_gate_enabled: bool = True
    liquidity_min_ask_depth_usd: float = 40.0
    liquidity_max_spread_abs: float = 0.08
    liquidity_max_spread_pct_of_ask: float = 0.22

    # Passive entry logic (reduce adverse selection on wide spreads)
    passive_entry_enabled: bool = True
    passive_entry_min_spread_abs: float = 0.02
    passive_entry_join_ticks: int = 1
    passive_entry_tick_size: float = 0.01

    # Signal funnel KPI watchdog (auto-fallback if opportunity flow is too low)
    signal_kpi_enabled: bool = True
    signal_kpi_window_hours: int = 6
    signal_kpi_min_candidates: int = 200
    signal_kpi_min_emitted: int = 1
    signal_kpi_max_time_gate_ratio: float = 0.65
    signal_kpi_max_exact_ratio: float = 0.65
    signal_kpi_auto_shadow_expand: bool = True
    signal_kpi_auto_exact_pilot: bool = True
    signal_kpi_auto_adaptive_gates: bool = True

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
        cfg.builder_api_key = os.getenv("BUILDER_API_KEY", "").strip() or cfg.poly_api_key
        cfg.builder_api_secret = os.getenv("BUILDER_API_SECRET", "").strip() or cfg.poly_api_secret
        cfg.builder_api_passphrase = (
            os.getenv("BUILDER_API_PASSPHRASE", "").strip() or cfg.poly_api_passphrase
        )
        cfg.relayer_url = os.getenv("RELAYER_URL", cfg.relayer_url).strip() or cfg.relayer_url
        cfg.polygon_rpc_urls = os.getenv("POLYGON_RPC_URLS", cfg.polygon_rpc_urls).strip() or cfg.polygon_rpc_urls
        cfg.polygon_rpc_timeout_seconds = int(
            os.getenv("POLYGON_RPC_TIMEOUT_SECONDS", str(cfg.polygon_rpc_timeout_seconds))
        )
        # Backward-compatible single-endpoint override.
        single_rpc = os.getenv("POLYGON_RPC", "").strip()
        if single_rpc:
            existing = [u.strip() for u in cfg.polygon_rpc_urls.split(",") if u.strip()]
            combined = [single_rpc] + [u for u in existing if u != single_rpc]
            cfg.polygon_rpc_urls = ",".join(combined)
        cfg.clob_market_ws_url = os.getenv("CLOB_MARKET_WS_URL", cfg.clob_market_ws_url)
        cfg.clob_user_ws_url = os.getenv("CLOB_USER_WS_URL", cfg.clob_user_ws_url)
        cfg.synoptic_api_token = os.getenv("SYNOPTIC_API_TOKEN", "")
        cfg.db_path = os.getenv("DB_PATH", "station_sniper.db")
        cfg.dry_run = _env_bool("DRY_RUN", True)
        cfg.strategy_mode = os.getenv("STRATEGY_MODE", cfg.strategy_mode).strip().lower() or cfg.strategy_mode
        if cfg.strategy_mode not in ("weather", "copycat"):
            cfg.strategy_mode = "weather"
        cfg.enable_market_ws = _env_bool("ENABLE_MARKET_WS", cfg.enable_market_ws)
        cfg.enable_user_ws = _env_bool("ENABLE_USER_WS", cfg.enable_user_ws)
        cfg.ws_reconnect_seconds = int(os.getenv("WS_RECONNECT_SECONDS", str(cfg.ws_reconnect_seconds)))
        cfg.ws_ping_interval = int(os.getenv("WS_PING_INTERVAL", str(cfg.ws_ping_interval)))
        cfg.market_price_max_age = int(os.getenv("MARKET_PRICE_MAX_AGE", str(cfg.market_price_max_age)))
        cfg.market_scan_interval = int(os.getenv("MARKET_SCAN_INTERVAL", str(cfg.market_scan_interval)))
        cfg.observation_interval = int(os.getenv("OBSERVATION_INTERVAL", str(cfg.observation_interval)))
        cfg.learning_interval = int(os.getenv("LEARNING_INTERVAL", str(cfg.learning_interval)))
        cfg.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", str(cfg.heartbeat_interval)))
        cfg.redemption_interval = max(60, int(os.getenv("REDEMPTION_INTERVAL", str(cfg.redemption_interval))))
        cfg.redemption_quiet_hours_enabled = _env_bool(
            "REDEMPTION_QUIET_HOURS_ENABLED",
            cfg.redemption_quiet_hours_enabled,
        )
        cfg.redemption_quiet_timezone = (
            os.getenv("REDEMPTION_QUIET_TIMEZONE", cfg.redemption_quiet_timezone).strip()
            or cfg.redemption_quiet_timezone
        )
        cfg.redemption_quiet_start_hour = int(
            os.getenv("REDEMPTION_QUIET_START_HOUR", str(cfg.redemption_quiet_start_hour))
        ) % 24
        cfg.redemption_quiet_end_hour = int(
            os.getenv("REDEMPTION_QUIET_END_HOUR", str(cfg.redemption_quiet_end_hour))
        ) % 24
        cfg.orderbook_404_cooldown_seconds = int(
            os.getenv("ORDERBOOK_404_COOLDOWN_SECONDS", str(cfg.orderbook_404_cooldown_seconds))
        )
        cfg.copy_enabled = _env_bool("COPY_ENABLED", cfg.copy_enabled)
        cfg.copy_poll_seconds = int(os.getenv("COPY_POLL_SECONDS", str(cfg.copy_poll_seconds)))
        cfg.copy_followup_poll_seconds = int(
            os.getenv("COPY_FOLLOWUP_POLL_SECONDS", str(cfg.copy_followup_poll_seconds))
        )
        cfg.copy_settle_max_hours = int(os.getenv("COPY_SETTLE_MAX_HOURS", str(cfg.copy_settle_max_hours)))
        cfg.copy_settle_grace_seconds = int(
            os.getenv("COPY_SETTLE_GRACE_SECONDS", str(cfg.copy_settle_grace_seconds))
        )
        cfg.copy_base_ticket_usd = float(os.getenv("COPY_BASE_TICKET_USD", str(cfg.copy_base_ticket_usd)))
        cfg.copy_probation_multiplier = float(
            os.getenv("COPY_PROBATION_MULTIPLIER", str(cfg.copy_probation_multiplier))
        )
        cfg.copy_gmanas_multiplier = float(
            os.getenv("COPY_GMANAS_MULTIPLIER", str(cfg.copy_gmanas_multiplier))
        )
        cfg.copy_probation_max_open_risk_share = float(
            os.getenv(
                "COPY_PROBATION_MAX_OPEN_RISK_SHARE",
                str(cfg.copy_probation_max_open_risk_share),
            )
        )
        cfg.copy_probation_max_open_locks = int(
            os.getenv("COPY_PROBATION_MAX_OPEN_LOCKS", str(cfg.copy_probation_max_open_locks))
        )
        cfg.copy_probation_min_seconds_between_entries = int(
            os.getenv(
                "COPY_PROBATION_MIN_SECONDS_BETWEEN_ENTRIES",
                str(cfg.copy_probation_min_seconds_between_entries),
            )
        )
        cfg.copy_slippage_abs_cap = float(
            os.getenv("COPY_SLIPPAGE_ABS_CAP", str(cfg.copy_slippage_abs_cap))
        )
        cfg.copy_slippage_rel_cap = float(
            os.getenv("COPY_SLIPPAGE_REL_CAP", str(cfg.copy_slippage_rel_cap))
        )
        cfg.copy_min_leader_price = float(
            os.getenv("COPY_MIN_LEADER_PRICE", str(cfg.copy_min_leader_price))
        )
        cfg.copy_max_leader_price = float(
            os.getenv("COPY_MAX_LEADER_PRICE", str(cfg.copy_max_leader_price))
        )
        cfg.copy_skip_inactive_events = _env_bool(
            "COPY_SKIP_INACTIVE_EVENTS", cfg.copy_skip_inactive_events,
        )
        cfg.copy_core_poll_priority = _env_bool(
            "COPY_CORE_POLL_PRIORITY", cfg.copy_core_poll_priority,
        )
        cfg.copy_max_new_trades_per_poll = int(
            os.getenv("COPY_MAX_NEW_TRADES_PER_POLL", str(cfg.copy_max_new_trades_per_poll))
        )
        cfg.copy_max_match_exposure_usd = float(
            os.getenv("COPY_MAX_MATCH_EXPOSURE_USD", str(cfg.copy_max_match_exposure_usd))
        )
        cfg.copy_max_open_risk_pct = float(
            os.getenv("COPY_MAX_OPEN_RISK_PCT", str(cfg.copy_max_open_risk_pct))
        )
        cfg.copy_min_cash_buffer_usd = float(
            os.getenv("COPY_MIN_CASH_BUFFER_USD", str(cfg.copy_min_cash_buffer_usd))
        )
        cfg.copy_min_matic_reserve = float(
            os.getenv("COPY_MIN_MATIC_RESERVE", str(cfg.copy_min_matic_reserve))
        )
        cfg.copy_est_matic_per_redemption = float(
            os.getenv("COPY_EST_MATIC_PER_REDEMPTION", str(cfg.copy_est_matic_per_redemption))
        )
        cfg.redeem_proxy_via_relayer = _env_bool("REDEEM_PROXY_VIA_RELAYER", cfg.redeem_proxy_via_relayer)
        cfg.redeem_relayer_gas_limit = int(
            os.getenv("REDEEM_RELAYER_GAS_LIMIT", str(cfg.redeem_relayer_gas_limit))
        )
        cfg.redeem_relayer_max_polls = int(
            os.getenv("REDEEM_RELAYER_MAX_POLLS", str(cfg.redeem_relayer_max_polls))
        )
        cfg.redeem_relayer_poll_ms = int(
            os.getenv("REDEEM_RELAYER_POLL_MS", str(cfg.redeem_relayer_poll_ms))
        )
        cfg.redeem_relayer_max_submits_per_cycle = int(
            os.getenv("REDEEM_RELAYER_MAX_SUBMITS_PER_CYCLE", str(cfg.redeem_relayer_max_submits_per_cycle))
        )
        cfg.redeem_relayer_submit_spacing_seconds = float(
            os.getenv("REDEEM_RELAYER_SUBMIT_SPACING_SECONDS", str(cfg.redeem_relayer_submit_spacing_seconds))
        )
        cfg.redeem_relayer_429_backoff_initial_seconds = int(
            os.getenv(
                "REDEEM_RELAYER_429_BACKOFF_INITIAL_SECONDS",
                str(cfg.redeem_relayer_429_backoff_initial_seconds),
            )
        )
        cfg.redeem_relayer_429_backoff_max_seconds = int(
            os.getenv(
                "REDEEM_RELAYER_429_BACKOFF_MAX_SECONDS",
                str(cfg.redeem_relayer_429_backoff_max_seconds),
            )
        )
        cfg.copy_dedup_enabled = _env_bool("COPY_DEDUP_ENABLED", cfg.copy_dedup_enabled)
        cfg.copy_opposite_policy = os.getenv("COPY_OPPOSITE_POLICY", cfg.copy_opposite_policy)
        cfg.copy_follow_buy_only = _env_bool("COPY_FOLLOW_BUY_ONLY", cfg.copy_follow_buy_only)
        cfg.copy_max_wallet_trades_fetch = int(
            os.getenv("COPY_MAX_WALLET_TRADES_FETCH", str(cfg.copy_max_wallet_trades_fetch))
        )
        cfg.copy_trade_max_age_seconds = int(
            os.getenv("COPY_TRADE_MAX_AGE_SECONDS", str(cfg.copy_trade_max_age_seconds))
        )
        cfg.copy_activity_window_days = int(
            os.getenv("COPY_ACTIVITY_WINDOW_DAYS", str(cfg.copy_activity_window_days))
        )
        cfg.copy_min_trades_60d = int(os.getenv("COPY_MIN_TRADES_60D", str(cfg.copy_min_trades_60d)))
        cfg.copy_min_active_days_60d = int(
            os.getenv("COPY_MIN_ACTIVE_DAYS_60D", str(cfg.copy_min_active_days_60d))
        )
        cfg.leader_refresh_hours = int(os.getenv("LEADER_REFRESH_HOURS", str(cfg.leader_refresh_hours)))
        cfg.leader_probation_days = int(os.getenv("LEADER_PROBATION_DAYS", str(cfg.leader_probation_days)))
        cfg.copy_stuck_redeemable_hours = int(
            os.getenv("COPY_STUCK_REDEEMABLE_HOURS", str(cfg.copy_stuck_redeemable_hours))
        )
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
        cfg.self_learning_enabled = _env_bool("SELF_LEARNING_ENABLED", cfg.self_learning_enabled)
        cfg.self_learning_lookback_days = int(
            os.getenv("SELF_LEARNING_LOOKBACK_DAYS", str(cfg.self_learning_lookback_days))
        )
        cfg.self_learning_min_samples = int(
            os.getenv("SELF_LEARNING_MIN_SAMPLES", str(cfg.self_learning_min_samples))
        )
        cfg.self_learning_min_segment_samples = int(
            os.getenv("SELF_LEARNING_MIN_SEGMENT_SAMPLES", str(cfg.self_learning_min_segment_samples))
        )
        cfg.self_learning_prior_alpha = float(
            os.getenv("SELF_LEARNING_PRIOR_ALPHA", str(cfg.self_learning_prior_alpha))
        )
        cfg.self_learning_prior_beta = float(
            os.getenv("SELF_LEARNING_PRIOR_BETA", str(cfg.self_learning_prior_beta))
        )
        cfg.self_learning_city_prior = float(
            os.getenv("SELF_LEARNING_CITY_PRIOR", str(cfg.self_learning_city_prior))
        )
        cfg.self_learning_hour_prior = float(
            os.getenv("SELF_LEARNING_HOUR_PRIOR", str(cfg.self_learning_hour_prior))
        )
        cfg.self_learning_bucket_prior = float(
            os.getenv("SELF_LEARNING_BUCKET_PRIOR", str(cfg.self_learning_bucket_prior))
        )
        cfg.self_learning_precision_prior = float(
            os.getenv("SELF_LEARNING_PRECISION_PRIOR", str(cfg.self_learning_precision_prior))
        )
        cfg.self_learning_blend = float(os.getenv("SELF_LEARNING_BLEND", str(cfg.self_learning_blend)))
        cfg.self_learning_reliability_samples = int(
            os.getenv("SELF_LEARNING_RELIABILITY_SAMPLES", str(cfg.self_learning_reliability_samples))
        )
        cfg.self_learning_confidence_scale = float(
            os.getenv("SELF_LEARNING_CONFIDENCE_SCALE", str(cfg.self_learning_confidence_scale))
        )
        cfg.self_learning_confidence_cap = float(
            os.getenv("SELF_LEARNING_CONFIDENCE_CAP", str(cfg.self_learning_confidence_cap))
        )
        cfg.enable_ev_gate = _env_bool("ENABLE_EV_GATE", cfg.enable_ev_gate)
        cfg.min_expected_edge = float(os.getenv("MIN_EXPECTED_EDGE", str(cfg.min_expected_edge)))
        cfg.min_expected_profit = float(os.getenv("MIN_EXPECTED_PROFIT", str(cfg.min_expected_profit)))
        cfg.ev_base_slippage_bps = float(os.getenv("EV_BASE_SLIPPAGE_BPS", str(cfg.ev_base_slippage_bps)))
        cfg.ev_depth_slippage_max = float(os.getenv("EV_DEPTH_SLIPPAGE_MAX", str(cfg.ev_depth_slippage_max)))
        cfg.ev_spread_weight = float(os.getenv("EV_SPREAD_WEIGHT", str(cfg.ev_spread_weight)))
        cfg.ev_dynamic_slippage = _env_bool("EV_DYNAMIC_SLIPPAGE", cfg.ev_dynamic_slippage)
        cfg.shadow_expansion_enabled = _env_bool("SHADOW_EXPANSION_ENABLED", cfg.shadow_expansion_enabled)
        cfg.shadow_min_confidence = float(os.getenv("SHADOW_MIN_CONFIDENCE", str(cfg.shadow_min_confidence)))
        cfg.shadow_min_price = float(os.getenv("SHADOW_MIN_PRICE", str(cfg.shadow_min_price)))
        cfg.shadow_max_price = float(os.getenv("SHADOW_MAX_PRICE", str(cfg.shadow_max_price)))
        cfg.shadow_metar_max_age_minutes = float(
            os.getenv("SHADOW_METAR_MAX_AGE_MINUTES", str(cfg.shadow_metar_max_age_minutes))
        )
        cfg.shadow_geq_min_hour = int(os.getenv("SHADOW_GEQ_MIN_HOUR", str(cfg.shadow_geq_min_hour)))
        cfg.shadow_leq_min_hour = int(os.getenv("SHADOW_LEQ_MIN_HOUR", str(cfg.shadow_leq_min_hour)))
        cfg.shadow_max_bet_usd = float(os.getenv("SHADOW_MAX_BET_USD", str(cfg.shadow_max_bet_usd)))
        cfg.shadow_max_bankroll_pct = float(
            os.getenv("SHADOW_MAX_BANKROLL_PCT", str(cfg.shadow_max_bankroll_pct))
        )
        cfg.shadow_min_expected_edge = float(
            os.getenv("SHADOW_MIN_EXPECTED_EDGE", str(cfg.shadow_min_expected_edge))
        )
        cfg.shadow_min_expected_profit = float(
            os.getenv("SHADOW_MIN_EXPECTED_PROFIT", str(cfg.shadow_min_expected_profit))
        )
        cfg.shadow_exact_enabled = _env_bool("SHADOW_EXACT_ENABLED", cfg.shadow_exact_enabled)
        cfg.shadow_exact_min_hour = int(os.getenv("SHADOW_EXACT_MIN_HOUR", str(cfg.shadow_exact_min_hour)))
        cfg.shadow_exact_min_confidence = float(
            os.getenv("SHADOW_EXACT_MIN_CONFIDENCE", str(cfg.shadow_exact_min_confidence))
        )
        cfg.shadow_exact_max_bet_usd = float(
            os.getenv("SHADOW_EXACT_MAX_BET_USD", str(cfg.shadow_exact_max_bet_usd))
        )
        cfg.adaptive_time_gates_enabled = _env_bool(
            "ADAPTIVE_TIME_GATES_ENABLED", cfg.adaptive_time_gates_enabled,
        )
        cfg.adaptive_time_gate_no_emit_cycles = int(
            os.getenv("ADAPTIVE_TIME_GATE_NO_EMIT_CYCLES", str(cfg.adaptive_time_gate_no_emit_cycles))
        )
        cfg.adaptive_time_gate_step_hours = int(
            os.getenv("ADAPTIVE_TIME_GATE_STEP_HOURS", str(cfg.adaptive_time_gate_step_hours))
        )
        cfg.adaptive_time_gate_min_geq_hour = int(
            os.getenv("ADAPTIVE_TIME_GATE_MIN_GEQ_HOUR", str(cfg.adaptive_time_gate_min_geq_hour))
        )
        cfg.adaptive_time_gate_min_leq_hour = int(
            os.getenv("ADAPTIVE_TIME_GATE_MIN_LEQ_HOUR", str(cfg.adaptive_time_gate_min_leq_hour))
        )
        cfg.adaptive_time_gate_reset_on_emit = _env_bool(
            "ADAPTIVE_TIME_GATE_RESET_ON_EMIT", cfg.adaptive_time_gate_reset_on_emit,
        )
        cfg.liquidity_gate_enabled = _env_bool("LIQUIDITY_GATE_ENABLED", cfg.liquidity_gate_enabled)
        cfg.liquidity_min_ask_depth_usd = float(
            os.getenv("LIQUIDITY_MIN_ASK_DEPTH_USD", str(cfg.liquidity_min_ask_depth_usd))
        )
        cfg.liquidity_max_spread_abs = float(
            os.getenv("LIQUIDITY_MAX_SPREAD_ABS", str(cfg.liquidity_max_spread_abs))
        )
        cfg.liquidity_max_spread_pct_of_ask = float(
            os.getenv("LIQUIDITY_MAX_SPREAD_PCT_OF_ASK", str(cfg.liquidity_max_spread_pct_of_ask))
        )
        cfg.passive_entry_enabled = _env_bool("PASSIVE_ENTRY_ENABLED", cfg.passive_entry_enabled)
        cfg.passive_entry_min_spread_abs = float(
            os.getenv("PASSIVE_ENTRY_MIN_SPREAD_ABS", str(cfg.passive_entry_min_spread_abs))
        )
        cfg.passive_entry_join_ticks = int(os.getenv("PASSIVE_ENTRY_JOIN_TICKS", str(cfg.passive_entry_join_ticks)))
        cfg.passive_entry_tick_size = float(os.getenv("PASSIVE_ENTRY_TICK_SIZE", str(cfg.passive_entry_tick_size)))
        cfg.signal_kpi_enabled = _env_bool("SIGNAL_KPI_ENABLED", cfg.signal_kpi_enabled)
        cfg.signal_kpi_window_hours = int(os.getenv("SIGNAL_KPI_WINDOW_HOURS", str(cfg.signal_kpi_window_hours)))
        cfg.signal_kpi_min_candidates = int(
            os.getenv("SIGNAL_KPI_MIN_CANDIDATES", str(cfg.signal_kpi_min_candidates))
        )
        cfg.signal_kpi_min_emitted = int(os.getenv("SIGNAL_KPI_MIN_EMITTED", str(cfg.signal_kpi_min_emitted)))
        cfg.signal_kpi_max_time_gate_ratio = float(
            os.getenv("SIGNAL_KPI_MAX_TIME_GATE_RATIO", str(cfg.signal_kpi_max_time_gate_ratio))
        )
        cfg.signal_kpi_max_exact_ratio = float(
            os.getenv("SIGNAL_KPI_MAX_EXACT_RATIO", str(cfg.signal_kpi_max_exact_ratio))
        )
        cfg.signal_kpi_auto_shadow_expand = _env_bool(
            "SIGNAL_KPI_AUTO_SHADOW_EXPAND", cfg.signal_kpi_auto_shadow_expand,
        )
        cfg.signal_kpi_auto_exact_pilot = _env_bool(
            "SIGNAL_KPI_AUTO_EXACT_PILOT", cfg.signal_kpi_auto_exact_pilot,
        )
        cfg.signal_kpi_auto_adaptive_gates = _env_bool(
            "SIGNAL_KPI_AUTO_ADAPTIVE_GATES", cfg.signal_kpi_auto_adaptive_gates,
        )
        cfg.hard_stop_loss_enabled = _env_bool("HARD_STOP_LOSS_ENABLED", cfg.hard_stop_loss_enabled)
        cfg.hard_stop_loss_pct = float(os.getenv("HARD_STOP_LOSS_PCT", str(cfg.hard_stop_loss_pct)))
        cfg.initial_bankroll = float(os.getenv("INITIAL_BANKROLL", "200.0"))
        return cfg

    @property
    def db_url(self) -> str:
        return f"sqlite:///{Path(self.db_path).resolve()}"
