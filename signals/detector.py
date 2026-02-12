"""Core signal detection: compare METAR high vs market price, emit signals.

Latency arbitrage thesis:
  METAR updates faster than consumer weather sources (Weather Underground).
  When a METAR observation shows the daily high just crossed a threshold,
  there's a 5-30 minute window before markets react. We buy YES tokens
  on "geq" buckets during that window.

Required conditions for a valid signal:
  1. Market date == daily high date (no stale/future data)
  2. METAR observation is fresh (< 30 min)
  3. Time-of-day gate: geq needs hour >= 12, leq needs hour >= 17
  4. Temperature hits the bucket (with margin from boundary)
  5. Price is in the "mispriced" range (0.30 - 0.65)
  6. Confidence score >= min_confidence (0.85)
  7. "exact" buckets are skipped (unreliable for latency arb)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pytz

from config import Config
from learning.tracker import TradeTracker
from markets.registry import ActiveMarket, MarketRegistry
from signals.confidence import ConfidenceFactors, compute_confidence
from utils.logging import get_logger
from utils.time_utils import is_peak_heating, local_hour, local_date
from weather.metar_feed import MetarFeed
from weather.temperature import temp_hits_bucket, Precision

log = get_logger("detector")


@dataclass
class TradeSignal:
    """A detected trading opportunity."""

    event_id: str
    condition_id: str
    city: str
    market_date: date
    icao_station: str
    token_id: str
    bucket_type: str
    bucket_value: int
    unit: str  # "C" or "F"
    market_type: str  # "temperature" or "precipitation"
    confidence: ConfidenceFactors
    metar_temp_c: float
    metar_precision: str
    wu_displayed_high: int | None
    margin_from_boundary: float
    local_hour_val: int
    best_ask: float | None
    ask_depth: float


class SignalDetector:
    """Detects trade signals by comparing METAR daily highs with market prices."""

    def __init__(
        self,
        config: Config,
        metar_feed: MetarFeed,
        registry: MarketRegistry,
        tracker: TradeTracker | None = None,
        calibrator=None,
    ) -> None:
        self._config = config
        self._metar = metar_feed
        self._registry = registry
        self._tracker = tracker
        self._calibrator = calibrator
        # Track which signals we've already emitted to avoid duplicates
        self._emitted: set[str] = set()

    async def detect(self) -> list[TradeSignal]:
        """Scan all active markets and detect trade signals."""
        signals: list[TradeSignal] = []
        max_price = self._config.max_price.value
        min_price = self._config.min_price
        min_confidence = self._config.min_confidence.value

        for active in self._registry.get_all_active():
            market_signals = await self._check_market(
                active, min_price, max_price, min_confidence,
            )
            signals.extend(market_signals)

        if signals:
            log.info("signals_detected", count=len(signals))
        return signals

    async def _check_market(
        self,
        active: ActiveMarket,
        min_price: float,
        max_price: float,
        min_confidence: float,
    ) -> list[TradeSignal]:
        """Check a single market for signals with all 7 gates."""
        market = active.market
        signals: list[TradeSignal] = []

        # ── Gate 1: Get METAR daily high ──────────────────────────────
        daily_high = self._metar.get_daily_high(market.icao)
        if daily_high is None or daily_high.high is None:
            return signals

        # ── Gate 2: Date validation ───────────────────────────────────
        # Daily high must be for the same date as the market.
        # Prevents trading on stale data or future market dates.
        if daily_high.date != market.info.market_date:
            log.debug(
                "date_mismatch",
                station=market.icao,
                city=market.info.city,
                daily_high_date=str(daily_high.date),
                market_date=str(market.info.market_date),
            )
            return signals

        # ── Gate 3: METAR freshness ───────────────────────────────────
        # Stale METAR = no latency edge; market already caught up.
        if daily_high.last_obs_time is None:
            return signals

        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        metar_age_seconds = (now_utc - daily_high.last_obs_time).total_seconds()
        metar_age_minutes = metar_age_seconds / 60.0

        if metar_age_minutes > self._config.metar_max_age_minutes:
            log.debug(
                "metar_stale",
                station=market.icao,
                age_min=round(metar_age_minutes, 1),
                max_age=self._config.metar_max_age_minutes,
            )
            return signals

        # ── Gate 4: Time-of-day ───────────────────────────────────────
        hour = local_hour(market.timezone)
        temp = daily_high.high

        for bucket_data in market.buckets:
            bucket = bucket_data.bucket
            signal_key = f"{market.event_id}:{bucket_data.token_id}:{daily_high.date}"

            # Skip already-emitted signals
            if signal_key in self._emitted:
                continue

            # ── Gate 4a: Skip "exact" buckets ─────────────────────────
            # Exact buckets are unreliable for latency arb (narrow window,
            # rounding uncertainty makes them near-random).
            if bucket.bucket_type == "exact":
                continue

            # ── Gate 4b: Directional time-of-day gating ──────────────
            if bucket.bucket_type == "geq":
                # geq = "daily high >= X"
                # Edge: METAR shows temp just crossed UP through X.
                # Only valid once peak heating has started (noon+).
                # Before noon, temps are still rising and daily high is
                # just the morning reading — meaningless.
                if hour < self._config.geq_min_hour:
                    continue

            elif bucket.bucket_type == "leq":
                # leq = "daily high <= X"
                # Edge: peak heating has ENDED and temp stayed below X.
                # Only valid after peak heating passes (5 PM+).
                # Before that, temp could still rise above X.
                if hour < self._config.leq_min_hour:
                    continue

            # ── Gate 5: Bucket match ──────────────────────────────────
            match = temp_hits_bucket(
                temp, bucket.bucket_type, bucket.bucket_value, unit=bucket.unit,
            )
            if not match.hit:
                continue

            # ── Gate 6: Price range ───────────────────────────────────
            # Price must be in the "mispriced" range where we have edge.
            # Too low (< min_price): market sees it as very unlikely — why?
            # Too high (> max_price): market already knows, minimal profit.
            price_info = active.bucket_prices.get(bucket_data.token_id)
            best_ask = price_info.best_ask if price_info else None
            ask_depth = price_info.ask_depth if price_info else 0.0

            if best_ask is None:
                continue

            if best_ask < min_price:
                log.debug(
                    "price_below_floor",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    best_ask=best_ask,
                    min_price=min_price,
                )
                continue

            if best_ask > max_price:
                log.debug(
                    "price_above_ceiling",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    best_ask=best_ask,
                    max_price=max_price,
                )
                continue

            # ── Gate 7: Confidence scoring ────────────────────────────
            peak = is_peak_heating(market.timezone)

            historical_accuracy = None
            if self._tracker is not None:
                historical_accuracy = self._tracker.get_historical_accuracy(
                    city=market.info.city, hour=hour,
                )

            calibration_adj = 0.0
            if self._calibrator is not None:
                calibration_adj = self._calibrator.get_adjustment_for_confidence(
                    match.confidence,
                )

            confidence = compute_confidence(
                bucket_match=match,
                precision=temp.precision,
                wu_lag_confirmed=False,
                is_peak_hours=peak,
                historical_accuracy=historical_accuracy,
                calibration_adjustment=calibration_adj,
                metar_age_minutes=metar_age_minutes,
            )

            if confidence.total < min_confidence:
                log.debug(
                    "signal_below_threshold",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    confidence=round(confidence.total, 3),
                    threshold=min_confidence,
                    breakdown={
                        "base": round(confidence.base, 3),
                        "precision": round(confidence.precision_bonus, 3),
                        "peak": round(confidence.peak_hours_bonus, 3),
                        "recency": round(confidence.recency_bonus, 3),
                    },
                )
                continue

            # ── All gates passed — emit signal ────────────────────────
            signal = TradeSignal(
                event_id=market.event_id,
                condition_id=bucket_data.condition_id,
                city=market.info.city,
                market_date=market.info.market_date,
                icao_station=market.icao,
                token_id=bucket_data.token_id,
                bucket_type=bucket.bucket_type,
                bucket_value=bucket.bucket_value,
                unit=bucket.unit,
                market_type=market.market_type,
                confidence=confidence,
                metar_temp_c=temp.celsius,
                metar_precision=temp.precision.value,
                wu_displayed_high=None,
                margin_from_boundary=match.margin,
                local_hour_val=hour,
                best_ask=best_ask,
                ask_depth=ask_depth,
            )

            self._emitted.add(signal_key)
            signals.append(signal)

            log.info(
                "signal_emitted",
                city=market.info.city,
                bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                confidence=round(confidence.total, 3),
                metar_c=temp.celsius,
                metar_age_min=round(metar_age_minutes, 1),
                best_ask=best_ask,
                margin=round(match.margin, 2),
                local_hour=hour,
                is_peak=peak,
            )

        return signals

    def reset_emitted(self) -> None:
        """Clear emitted signals cache (e.g., at start of new day)."""
        self._emitted.clear()
