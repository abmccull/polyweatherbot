"""SQLAlchemy models for all persistent data."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Market context
    event_id = Column(String, nullable=False, index=True)
    condition_id = Column(String, nullable=True)
    city = Column(String, nullable=False)
    market_date = Column(String, nullable=False)  # YYYY-MM-DD
    icao_station = Column(String, nullable=False)

    # Bucket
    bucket_value = Column(Integer, nullable=False)
    bucket_type = Column(String, nullable=False)  # "exact", "geq", "leq"
    bucket_unit = Column(String, nullable=True, default="C")  # "C", "F", or "inches"
    token_id = Column(String, nullable=False, index=True)

    # Precipitation bucket boundaries (NULL for temperature trades)
    bucket_low_inches = Column(Float, nullable=True)
    bucket_high_inches = Column(Float, nullable=True)

    # Market type
    market_type = Column(String, nullable=True, default="temperature")  # "temperature" or "precipitation"

    # Trade execution
    action = Column(String, nullable=False, default="BUY", index=True)
    requested_size = Column(Float, nullable=True)
    requested_cost = Column(Float, nullable=True)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    order_id = Column(String, nullable=True)
    order_status = Column(String, nullable=True)
    fill_price = Column(Float, nullable=True)
    expected_edge = Column(Float, nullable=True)
    expected_profit = Column(Float, nullable=True)
    expected_slippage = Column(Float, nullable=True)
    calibrated_probability = Column(Float, nullable=True)

    # Fee tracking
    fee_paid = Column(Float, nullable=True, default=0.0)

    # Signal context
    confidence = Column(Float, nullable=False)
    metar_temp_c = Column(Float, nullable=True)
    metar_precision = Column(String, nullable=True)  # "tenths" or "whole"
    wu_displayed_high = Column(Float, nullable=True)
    margin_from_boundary = Column(Float, nullable=True)
    local_hour = Column(Integer, nullable=True)

    # Exit management
    exit_reason = Column(String, nullable=True)      # "PROFIT_LOCK", "TRAILING_STOP", "STOP_LOSS"
    parent_trade_id = Column(Integer, nullable=True)  # Links SELL to original BUY token

    # Resolution (filled after market resolves)
    resolution_value = Column(Float, nullable=True)
    resolved_correct = Column(Boolean, nullable=True)
    pnl = Column(Float, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station = Column(String, nullable=False, index=True)
    raw_metar = Column(Text, nullable=True)
    temp_c = Column(Float, nullable=True)
    precision = Column(String, nullable=True)  # "tenths" or "whole"
    obs_time = Column(DateTime, nullable=False)
    source = Column(String, nullable=False)  # "metar_cache", "synoptic", "nws"
    precip_1h = Column(Float, nullable=True)
    precip_24h = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, nullable=False, index=True)
    city = Column(String, nullable=False)
    market_date = Column(String, nullable=False)
    bucket_value = Column(Integer, nullable=False)
    bucket_type = Column(String, nullable=False)
    token_id = Column(String, nullable=False)
    best_bid = Column(Float, nullable=True)
    best_ask = Column(Float, nullable=True)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SignalCandidate(Base):
    __tablename__ = "signal_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, nullable=False, index=True)
    token_id = Column(String, nullable=False, index=True)
    city = Column(String, nullable=False)
    market_date = Column(String, nullable=False)
    market_type = Column(String, nullable=True, default="temperature")
    bucket_type = Column(String, nullable=False)
    bucket_value = Column(Integer, nullable=False)
    bucket_unit = Column(String, nullable=True, default="C")

    metar_temp_c = Column(Float, nullable=True)
    metar_precision = Column(String, nullable=True)
    metar_age_minutes = Column(Float, nullable=True)
    local_hour = Column(Integer, nullable=True)
    matched_bucket = Column(Boolean, nullable=True)

    best_bid = Column(Float, nullable=True)
    best_ask = Column(Float, nullable=True)
    spread = Column(Float, nullable=True)
    bid_depth = Column(Float, nullable=True)
    ask_depth = Column(Float, nullable=True)

    confidence_total = Column(Float, nullable=True)
    calibrated_probability = Column(Float, nullable=True)
    confidence_base = Column(Float, nullable=True)
    confidence_precision_bonus = Column(Float, nullable=True)
    confidence_peak_bonus = Column(Float, nullable=True)
    confidence_recency_bonus = Column(Float, nullable=True)
    confidence_historical_blend = Column(Float, nullable=True)
    confidence_calibration_adj = Column(Float, nullable=True)

    status = Column(String, nullable=False)  # "EMITTED", "NO_PRICE", etc.
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ParameterAdjustment(Base):
    __tablename__ = "parameter_adjustments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_name = Column(String, nullable=False)
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)
    delta = Column(Float, nullable=False)
    reason = Column(String, nullable=True)
    win_rate = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class BotState(Base):
    __tablename__ = "bot_state"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyPnL(Base):
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False, unique=True)  # YYYY-MM-DD
    trades_count = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    losses = Column(Integer, nullable=False, default=0)
    gross_pnl = Column(Float, nullable=False, default=0.0)
    fees = Column(Float, nullable=False, default=0.0)
    net_pnl = Column(Float, nullable=False, default=0.0)
    portfolio_value = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class LeaderProfile(Base):
    __tablename__ = "leader_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    tier = Column(String, nullable=True)
    base_status = Column(String, nullable=False, default="core")  # core/probation/excluded
    risk_multiplier = Column(Float, nullable=False, default=1.0)
    enabled = Column(Boolean, nullable=False, default=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class LeaderMetricsDaily(Base):
    __tablename__ = "leader_metrics_daily"
    __table_args__ = (
        UniqueConstraint("wallet", "date", name="uq_leader_metrics_wallet_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False)  # YYYY-MM-DD UTC
    trades_60d = Column(Integer, nullable=False, default=0)
    active_days_60d = Column(Integer, nullable=False, default=0)
    activity_truncated = Column(Boolean, nullable=False, default=False)
    recent_active_ok = Column(Boolean, nullable=False, default=False)
    week_sports_pnl = Column(Float, nullable=True)
    month_sports_pnl = Column(Float, nullable=True)
    week_global_pnl = Column(Float, nullable=True)
    month_global_pnl = Column(Float, nullable=True)
    recent_success_ok = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class LeaderEligibilityDecision(Base):
    __tablename__ = "leader_eligibility_decisions"
    __table_args__ = (
        UniqueConstraint("wallet", "date", name="uq_leader_eligibility_wallet_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False)  # YYYY-MM-DD UTC
    status = Column(String, nullable=False)  # core/probation/excluded
    eligible = Column(Boolean, nullable=False, default=False)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class CopySignalEvent(Base):
    __tablename__ = "copy_signal_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    leader_wallet = Column(String, nullable=False, index=True)
    leader_name = Column(String, nullable=True)
    transaction_hash = Column(String, nullable=False, unique=True, index=True)
    side = Column(String, nullable=False)
    outcome = Column(String, nullable=True)
    token_id = Column(String, nullable=False, index=True)
    condition_id = Column(String, nullable=True, index=True)
    event_slug = Column(String, nullable=True, index=True)
    match_key = Column(String, nullable=False, index=True)
    leader_price = Column(Float, nullable=False)
    leader_size = Column(Float, nullable=False, default=0.0)
    event_title = Column(String, nullable=True)
    event_end = Column(DateTime, nullable=True)
    status = Column(String, nullable=False)  # accepted/skipped/error
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class CopyOrderIntent(Base):
    __tablename__ = "copy_order_intents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_event_id = Column(Integer, nullable=True, index=True)
    leader_wallet = Column(String, nullable=False, index=True)
    match_key = Column(String, nullable=False, index=True)
    condition_id = Column(String, nullable=True, index=True)
    event_slug = Column(String, nullable=True)
    token_id = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False, default="BUY")
    outcome = Column(String, nullable=True)
    requested_price = Column(Float, nullable=True)
    max_copy_price = Column(Float, nullable=True)
    selected_price = Column(Float, nullable=True)
    size_usd = Column(Float, nullable=False, default=0.0)
    shares = Column(Float, nullable=False, default=0.0)
    order_id = Column(String, nullable=True, index=True)
    order_status = Column(String, nullable=True)
    status = Column(String, nullable=False)  # accepted/skipped/placed/failed/dry_run
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class CopyPositionLock(Base):
    __tablename__ = "copy_position_locks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_key = Column(String, nullable=False, unique=True, index=True)
    condition_id = Column(String, nullable=True, index=True)
    event_slug = Column(String, nullable=True)
    token_id = Column(String, nullable=False)
    side = Column(String, nullable=False)
    outcome = Column(String, nullable=True)
    status = Column(String, nullable=False, default="OPEN")  # OPEN/CLOSED
    opened_by_wallet = Column(String, nullable=True, index=True)
    last_signal_tx = Column(String, nullable=True)
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class RedemptionEvent(Base):
    __tablename__ = "redemption_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    outcome = Column(String, nullable=True)
    size = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False)  # REDEEMED/FAILED
    tx_hash = Column(String, nullable=True, index=True)
    usdc_balance_after = Column(Float, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
