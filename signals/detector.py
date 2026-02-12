"""Core signal detection: compare METAR high vs market price, emit signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pytz

from config import Config
from learning.tracker import TradeTracker
from markets.registry import ActiveMarket, MarketRegistry
from signals.confidence import ConfidenceFactors, compute_confidence
from utils.logging import get_logger
from utils.time_utils import is_peak_heating, local_hour
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
    """Detects trade signals by comparing METAR daily highs with WU data."""

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
        """Scan all active markets and detect trade signals.

        Algorithm:
        For each active market:
          1. Get METAR daily high for the market's station
          2. For each bucket in the market:
             a. Does the daily high "hit" this bucket?
             b. Is the YES price below max_price?
             c. Is WU showing a different (lower) high? (lag confirmed)
             d. Compute confidence score
             e. If confidence >= min_confidence AND price is cheap → EMIT
        """
        signals: list[TradeSignal] = []
        max_price = self._config.max_price.value
        min_confidence = self._config.min_confidence.value

        for active in self._registry.get_all_active():
            market = active.market
            market_signals = await self._check_market(active, max_price, min_confidence)
            signals.extend(market_signals)

        if signals:
            log.info("signals_detected", count=len(signals))
        return signals

    async def _check_market(
        self, active: ActiveMarket, max_price: float, min_confidence: float,
    ) -> list[TradeSignal]:
        """Check a single market for signals."""
        market = active.market
        signals: list[TradeSignal] = []

        # 1. Get METAR daily high
        daily_high = self._metar.get_daily_high(market.icao)
        if daily_high is None or daily_high.high is None:
            return signals

        temp = daily_high.high
        display_unit = market.display_unit

        # Check each bucket
        for bucket_data in market.buckets:
            bucket = bucket_data.bucket
            signal_key = f"{market.event_id}:{bucket_data.token_id}:{daily_high.date}"

            # Skip already-emitted signals
            if signal_key in self._emitted:
                continue

            # a. Does daily high hit this bucket? (unit-aware)
            match = temp_hits_bucket(temp, bucket.bucket_type, bucket.bucket_value, unit=bucket.unit)
            if not match.hit:
                continue

            # b. Check price — is YES cheap enough?
            price_info = active.bucket_prices.get(bucket_data.token_id)
            best_ask = price_info.best_ask if price_info else None
            ask_depth = price_info.ask_depth if price_info else 0.0

            if best_ask is not None and best_ask > max_price:
                continue

            # c. Compute METAR age (proxy for WU lag — fresh obs = WU hasn't caught up)
            metar_age = None
            if daily_high.last_obs_time is not None:
                delta = datetime.utcnow().replace(tzinfo=pytz.utc) - daily_high.last_obs_time
                metar_age = delta.total_seconds() / 60.0

            # d. Compute confidence
            peak = is_peak_heating(market.timezone)

            # Historical accuracy from tracker (activates historical_blend)
            historical_accuracy = None
            if self._tracker is not None:
                local_h = local_hour(market.timezone)
                historical_accuracy = self._tracker.get_historical_accuracy(
                    city=market.info.city, hour=local_h,
                )

            # Calibration adjustment from calibrator
            calibration_adj = 0.0
            if self._calibrator is not None:
                calibration_adj = self._calibrator.get_adjustment_for_confidence(match.confidence)

            confidence = compute_confidence(
                bucket_match=match,
                precision=temp.precision,
                wu_lag_confirmed=False,
                is_peak_hours=peak,
                historical_accuracy=historical_accuracy,
                calibration_adjustment=calibration_adj,
                metar_age_minutes=metar_age,
            )

            # e. Emit if above threshold
            if confidence.total < min_confidence:
                log.debug(
                    "signal_below_threshold",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    confidence=round(confidence.total, 3),
                    threshold=min_confidence,
                )
                continue

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
                local_hour_val=local_hour(market.timezone),
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
                metar_age_min=round(metar_age, 1) if metar_age is not None else None,
                best_ask=best_ask,
                margin=round(match.margin, 2),
            )

        return signals

    def reset_emitted(self) -> None:
        """Clear emitted signals cache (e.g., at start of new day)."""
        self._emitted.clear()
