"""Precipitation signal detection: running METAR totals vs market buckets."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime

from config import Config
from markets.discovery import DiscoveredPrecipMarket, DiscoveredPrecipBucket
from markets.parser import PrecipBucketInfo
from signals.confidence import ConfidenceFactors
from utils.logging import get_logger
from weather.metar_feed import MetarFeed
from weather.precipitation import MonthlyPrecipAccumulator

log = get_logger("precip_detector")

# Historical average daily precip rates (inches/day) for reference cities.
# Used to estimate max remaining precip in a month.
DEFAULT_AVG_DAILY_PRECIP = 0.15  # conservative US average
MAX_DAILY_PRECIP = 1.5  # extreme upper bound per day for plausibility


@dataclass
class PrecipTradeSignal:
    """A detected precipitation trading opportunity."""

    event_id: str
    condition_id: str
    city: str
    month: str
    token_id: str
    bucket_type: str  # "lt", "range", "gt"
    low_inches: float | None
    high_inches: float | None
    side: str  # "YES" or "NO"
    confidence: float
    running_total: float
    days_remaining: int
    raw_question: str
    best_ask: float | None
    ask_depth: float


class PrecipDetector:
    """Detects precipitation trade signals by comparing running METAR totals to market buckets."""

    def __init__(self, config: Config, metar_feed: MetarFeed) -> None:
        self._config = config
        self._metar = metar_feed
        self._emitted: set[str] = set()

    def detect(
        self,
        precip_markets: list[DiscoveredPrecipMarket],
        bucket_prices: dict[str, tuple[float | None, float]],
    ) -> list[PrecipTradeSignal]:
        """Scan all active precipitation markets and detect signals.

        Args:
            precip_markets: Active precipitation markets from discovery.
            bucket_prices: token_id → (best_ask, ask_depth) mapping.

        Returns:
            List of trade signals.
        """
        signals: list[PrecipTradeSignal] = []
        max_price = self._config.max_price.value

        for market in precip_markets:
            market_signals = self._check_market(market, bucket_prices, max_price)
            signals.extend(market_signals)

        if signals:
            log.info("precip_signals_detected", count=len(signals))
        return signals

    def _check_market(
        self,
        market: DiscoveredPrecipMarket,
        bucket_prices: dict[str, tuple[float | None, float]],
        max_price: float,
    ) -> list[PrecipTradeSignal]:
        """Check a single precipitation market for signals."""
        signals: list[PrecipTradeSignal] = []

        # Get running monthly precip total
        acc = self._metar.get_monthly_precip(market.icao)
        if acc is None:
            return signals

        # Verify the accumulator month matches the market month
        now = datetime.utcnow()
        month_num = _month_name_to_num(market.info.month)
        if month_num is None:
            return signals

        expected_month = f"{market.info.year}-{month_num:02d}"
        if acc.month != expected_month:
            return signals

        total = acc.total_inches
        days_in_month = calendar.monthrange(market.info.year, month_num)[1]
        current_day = now.day
        days_remaining = max(0, days_in_month - current_day)

        # Estimate plausible remaining precip
        max_remaining = days_remaining * MAX_DAILY_PRECIP

        for bucket_data in market.buckets:
            bucket = bucket_data.bucket
            signal_key = f"precip:{market.event_id}:{bucket_data.token_id}"

            if signal_key in self._emitted:
                continue

            price_info = bucket_prices.get(bucket_data.token_id)
            best_ask = price_info[0] if price_info else None
            ask_depth = price_info[1] if price_info else 0.0

            # Evaluate bucket
            result = self._evaluate_bucket(bucket, total, days_remaining, max_remaining)
            if result is None:
                continue

            side, confidence = result

            # Check price threshold
            if best_ask is not None and best_ask > max_price:
                continue

            # Check confidence threshold
            if confidence < self._config.min_confidence.value:
                log.debug(
                    "precip_signal_below_threshold",
                    city=market.info.city,
                    bucket=bucket.raw_question,
                    confidence=round(confidence, 3),
                )
                continue

            signal = PrecipTradeSignal(
                event_id=market.event_id,
                condition_id=bucket_data.condition_id,
                city=market.info.city,
                month=market.info.month,
                token_id=bucket_data.token_id,
                bucket_type=bucket.bucket_type,
                low_inches=bucket.low_inches,
                high_inches=bucket.high_inches,
                side=side,
                confidence=confidence,
                running_total=total,
                days_remaining=days_remaining,
                raw_question=bucket.raw_question,
                best_ask=best_ask,
                ask_depth=ask_depth,
            )

            self._emitted.add(signal_key)
            signals.append(signal)

            log.info(
                "precip_signal_emitted",
                city=market.info.city,
                month=market.info.month,
                bucket=bucket.raw_question,
                side=side,
                confidence=round(confidence, 3),
                running_total=round(total, 2),
                days_remaining=days_remaining,
                best_ask=best_ask,
            )

        return signals

    def _evaluate_bucket(
        self,
        bucket: PrecipBucketInfo,
        total: float,
        days_remaining: int,
        max_remaining: float,
    ) -> tuple[str, float] | None:
        """Evaluate whether we have a signal for a bucket.

        Returns (side, confidence) or None if no signal.
        side is "YES" (buy YES) or "NO" (buy NO / sell YES).
        """
        if bucket.bucket_type == "lt":
            # "Less than X inches"
            upper = bucket.high_inches
            if upper is None:
                return None

            if total >= upper:
                # Already exceeded → bucket is dead → NO
                return ("NO", _days_confidence(days_remaining, margin=total - upper))

            if total + max_remaining < upper and days_remaining <= 5:
                # Total + max possible remaining still under limit → YES
                margin = upper - (total + max_remaining)
                return ("YES", _days_confidence(days_remaining, margin=margin))

            return None

        elif bucket.bucket_type == "gt":
            # "More than X inches"
            lower = bucket.low_inches
            if lower is None:
                return None

            if total > lower:
                # Already exceeded → bucket hit → YES
                return ("YES", _days_confidence(days_remaining, margin=total - lower))

            if total + max_remaining <= lower and days_remaining <= 5:
                # Can't possibly reach → NO
                margin = lower - (total + max_remaining)
                return ("NO", _days_confidence(days_remaining, margin=margin))

            return None

        elif bucket.bucket_type == "range":
            # "X to Y inches"
            lower = bucket.low_inches
            upper = bucket.high_inches
            if lower is None or upper is None:
                return None

            if total > upper:
                # Already above range → NO
                return ("NO", _days_confidence(days_remaining, margin=total - upper))

            if total + max_remaining < lower:
                # Can't reach range → NO
                margin = lower - (total + max_remaining)
                return ("NO", _days_confidence(days_remaining, margin=margin))

            # Inside the range with few days left
            if lower <= total <= upper and days_remaining <= 5:
                # Check if remaining precip could push us out
                headroom = upper - total
                avg_remaining = days_remaining * DEFAULT_AVG_DAILY_PRECIP
                if avg_remaining < headroom:
                    margin = headroom - avg_remaining
                    return ("YES", _days_confidence(days_remaining, margin=margin))

            return None

        return None

    def reset_emitted(self) -> None:
        """Clear emitted signals (e.g., at start of new month)."""
        self._emitted.clear()


def _days_confidence(days_remaining: int, margin: float) -> float:
    """Compute confidence based on days remaining and margin from bucket edge.

    The fewer days remaining, the more confident we are.
    Margin is in inches — how far we are from the critical boundary.
    """
    # Days factor: 0 days → 0.95, 1 day → 0.90, 3 days → 0.80, 5 days → 0.70
    if days_remaining == 0:
        days_factor = 0.95
    elif days_remaining <= 2:
        days_factor = 0.90
    elif days_remaining <= 5:
        days_factor = 0.80 - (days_remaining - 3) * 0.05
    else:
        days_factor = 0.60

    # Margin factor: larger margin → higher confidence
    if margin > 2.0:
        margin_factor = 1.0
    elif margin > 1.0:
        margin_factor = 0.95
    elif margin > 0.5:
        margin_factor = 0.85
    elif margin > 0.1:
        margin_factor = 0.70
    else:
        margin_factor = 0.55

    return max(0.0, min(1.0, days_factor * margin_factor))


def _month_name_to_num(name: str) -> int | None:
    """Convert month name to number."""
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    return months.get(name.lower())
