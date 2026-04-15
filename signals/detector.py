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

from dataclasses import dataclass
from datetime import date, datetime

import pytz

from config import Config
from db.engine import get_session
from db.models import SignalCandidate
from learning.tracker import TradeTracker
from markets.registry import ActiveMarket, MarketRegistry
from signals.confidence import ConfidenceFactors, compute_confidence
from utils.logging import get_logger
from utils.time_utils import is_peak_heating, local_hour
from weather.metar_feed import MetarFeed
from weather.temperature import temp_hits_bucket

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
    best_bid: float | None = None
    win_probability: float | None = None


class SignalDetector:
    """Detects trade signals by comparing METAR daily highs with market prices."""

    def __init__(
        self,
        config: Config,
        metar_feed: MetarFeed,
        registry: MarketRegistry,
        tracker: TradeTracker | None = None,
        calibrator=None,
        self_learner=None,
    ) -> None:
        self._config = config
        self._metar = metar_feed
        self._registry = registry
        self._tracker = tracker
        self._calibrator = calibrator
        self._self_learner = self_learner
        # Track which signals we've already emitted to avoid duplicates
        self._emitted: set[str] = set()
        # Adaptive gate relaxation state (in-memory, resets on restart).
        self._no_emit_cycles = 0
        self._adaptive_geq_relax_hours = 0
        self._adaptive_leq_relax_hours = 0

    async def detect(self) -> list[TradeSignal]:
        """Scan all active markets and detect trade signals."""
        signals: list[TradeSignal] = []
        candidates: list[SignalCandidate] = []
        (
            min_price,
            max_price,
            min_confidence,
            metar_max_age_minutes,
            geq_min_hour,
            leq_min_hour,
        ) = self._effective_signal_gates()

        for active in self._registry.get_all_active():
            market_signals, market_candidates = await self._check_market(
                active,
                min_price,
                max_price,
                min_confidence,
                metar_max_age_minutes,
                geq_min_hour,
                leq_min_hour,
            )
            signals.extend(market_signals)
            candidates.extend(market_candidates)

        if candidates:
            self._persist_candidates(candidates)

        self._update_adaptive_time_gates(emitted_count=len(signals))

        if signals:
            log.info("signals_detected", count=len(signals))
        return signals

    async def _check_market(
        self,
        active: ActiveMarket,
        min_price: float,
        max_price: float,
        min_confidence: float,
        metar_max_age_minutes: float,
        geq_min_hour: int,
        leq_min_hour: int,
    ) -> tuple[list[TradeSignal], list[SignalCandidate]]:
        """Check a single market for signals with all 7 gates."""
        market = active.market
        signals: list[TradeSignal] = []
        candidates: list[SignalCandidate] = []

        # ── Gate 1: Get METAR daily high ──────────────────────────────
        daily_high = self._metar.get_daily_high(market.icao)
        if daily_high is None or daily_high.high is None:
            return signals, candidates

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
            return signals, candidates

        # ── Gate 3: METAR freshness ───────────────────────────────────
        # Stale METAR = no latency edge; market already caught up.
        if daily_high.last_obs_time is None:
            return signals, candidates

        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        metar_age_seconds = (now_utc - daily_high.last_obs_time).total_seconds()
        metar_age_minutes = metar_age_seconds / 60.0

        if metar_age_minutes > metar_max_age_minutes:
            log.debug(
                "metar_stale",
                station=market.icao,
                age_min=round(metar_age_minutes, 1),
                max_age=metar_max_age_minutes,
            )
            return signals, candidates

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
            if bucket.bucket_type == "exact":
                exact_allowed, exact_reason = self._exact_bucket_allowed(
                    local_hour_val=hour,
                    metar_precision=temp.precision.value,
                )
                if not exact_allowed:
                    status = "SKIP_EXACT" if exact_reason == "exact_disabled" else "EXACT_BLOCKED"
                    candidates.append(self._build_candidate(
                        market=market,
                        bucket_type=bucket.bucket_type,
                        bucket_value=bucket.bucket_value,
                        unit=bucket.unit,
                        token_id=bucket_data.token_id,
                        metar_temp_c=temp.celsius,
                        metar_precision=temp.precision.value,
                        metar_age_minutes=metar_age_minutes,
                        local_hour_val=hour,
                        status=status,
                        reason=exact_reason,
                    ))
                    continue

            # ── Gate 4b: Directional time-of-day gating ──────────────
            if bucket.bucket_type == "geq":
                # geq = "daily high >= X"
                # Edge: METAR shows temp just crossed UP through X.
                # Only valid once peak heating has started (noon+).
                # Before noon, temps are still rising and daily high is
                # just the morning reading — meaningless.
                if hour < geq_min_hour:
                    candidates.append(self._build_candidate(
                        market=market,
                        bucket_type=bucket.bucket_type,
                        bucket_value=bucket.bucket_value,
                        unit=bucket.unit,
                        token_id=bucket_data.token_id,
                        metar_temp_c=temp.celsius,
                        metar_precision=temp.precision.value,
                        metar_age_minutes=metar_age_minutes,
                        local_hour_val=hour,
                        status="TIME_GATE_BLOCKED",
                        reason=f"geq_before_{geq_min_hour}",
                    ))
                    continue

            elif bucket.bucket_type == "leq":
                # leq = "daily high <= X"
                # Edge: peak heating has ENDED and temp stayed below X.
                # Only valid after peak heating passes (5 PM+).
                # Before that, temp could still rise above X.
                if hour < leq_min_hour:
                    candidates.append(self._build_candidate(
                        market=market,
                        bucket_type=bucket.bucket_type,
                        bucket_value=bucket.bucket_value,
                        unit=bucket.unit,
                        token_id=bucket_data.token_id,
                        metar_temp_c=temp.celsius,
                        metar_precision=temp.precision.value,
                        metar_age_minutes=metar_age_minutes,
                        local_hour_val=hour,
                        status="TIME_GATE_BLOCKED",
                        reason=f"leq_before_{leq_min_hour}",
                    ))
                    continue

            # ── Gate 5: Bucket match ──────────────────────────────────
            match = temp_hits_bucket(
                temp, bucket.bucket_type, bucket.bucket_value, unit=bucket.unit,
            )
            if not match.hit:
                candidates.append(self._build_candidate(
                    market=market,
                    bucket_type=bucket.bucket_type,
                    bucket_value=bucket.bucket_value,
                    unit=bucket.unit,
                    token_id=bucket_data.token_id,
                    metar_temp_c=temp.celsius,
                    metar_precision=temp.precision.value,
                    metar_age_minutes=metar_age_minutes,
                    local_hour_val=hour,
                    matched_bucket=False,
                    status="BUCKET_MISS",
                    reason="metar_not_in_bucket",
                ))
                continue

            # ── Gate 6: Price range ───────────────────────────────────
            # Price must be in the "mispriced" range where we have edge.
            # Too low (< min_price): market sees it as very unlikely — why?
            # Too high (> max_price): market already knows, minimal profit.
            price_info = active.bucket_prices.get(bucket_data.token_id)
            best_bid = price_info.best_bid if price_info else None
            best_ask = price_info.best_ask if price_info else None
            bid_depth = price_info.bid_depth if price_info else 0.0
            ask_depth = price_info.ask_depth if price_info else 0.0

            if best_ask is None:
                candidates.append(self._build_candidate(
                    market=market,
                    bucket_type=bucket.bucket_type,
                    bucket_value=bucket.bucket_value,
                    unit=bucket.unit,
                    token_id=bucket_data.token_id,
                    metar_temp_c=temp.celsius,
                    metar_precision=temp.precision.value,
                    metar_age_minutes=metar_age_minutes,
                    local_hour_val=hour,
                    matched_bucket=True,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    status="NO_PRICE",
                    reason="missing_best_ask",
                ))
                continue

            if best_ask < min_price:
                log.debug(
                    "price_below_floor",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    best_ask=best_ask,
                    min_price=min_price,
                )
                candidates.append(self._build_candidate(
                    market=market,
                    bucket_type=bucket.bucket_type,
                    bucket_value=bucket.bucket_value,
                    unit=bucket.unit,
                    token_id=bucket_data.token_id,
                    metar_temp_c=temp.celsius,
                    metar_precision=temp.precision.value,
                    metar_age_minutes=metar_age_minutes,
                    local_hour_val=hour,
                    matched_bucket=True,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    status="PRICE_GATE_BLOCKED",
                    reason="below_min_price",
                ))
                continue

            if best_ask > max_price:
                log.debug(
                    "price_above_ceiling",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    best_ask=best_ask,
                    max_price=max_price,
                )
                candidates.append(self._build_candidate(
                    market=market,
                    bucket_type=bucket.bucket_type,
                    bucket_value=bucket.bucket_value,
                    unit=bucket.unit,
                    token_id=bucket_data.token_id,
                    metar_temp_c=temp.celsius,
                    metar_precision=temp.precision.value,
                    metar_age_minutes=metar_age_minutes,
                    local_hour_val=hour,
                    matched_bucket=True,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    status="PRICE_GATE_BLOCKED",
                    reason="above_max_price",
                ))
                continue

            # ── Gate 6b: Liquidity quality ────────────────────────────
            liquidity_ok, liquidity_reason = self._passes_liquidity_gate(
                best_bid=best_bid,
                best_ask=best_ask,
                ask_depth=ask_depth,
            )
            if not liquidity_ok:
                candidates.append(self._build_candidate(
                    market=market,
                    bucket_type=bucket.bucket_type,
                    bucket_value=bucket.bucket_value,
                    unit=bucket.unit,
                    token_id=bucket_data.token_id,
                    metar_temp_c=temp.celsius,
                    metar_precision=temp.precision.value,
                    metar_age_minutes=metar_age_minutes,
                    local_hour_val=hour,
                    matched_bucket=True,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    status="LIQUIDITY_BLOCKED",
                    reason=liquidity_reason,
                ))
                continue

            # ── Gate 7: Confidence scoring ────────────────────────────
            peak = is_peak_heating(market.timezone)

            historical_accuracy = None
            context_signal = None
            if self._self_learner is not None:
                context_signal = self._self_learner.get_context_signal(
                    city=market.info.city,
                    hour=hour,
                    bucket_type=bucket.bucket_type,
                    precision=temp.precision.value,
                )
                if context_signal is not None:
                    historical_accuracy = context_signal.context_probability

            if historical_accuracy is None and self._tracker is not None:
                historical_accuracy = self._tracker.get_historical_accuracy(
                    city=market.info.city,
                    hour=hour,
                )

            calibration_adj = 0.0
            if self._calibrator is not None:
                calibration_adj = self._calibrator.get_adjustment_for_confidence(
                    match.confidence,
                )
            if context_signal is not None:
                calibration_adj += context_signal.confidence_adjustment
                calibration_adj = max(-0.20, min(0.20, calibration_adj))

            confidence = compute_confidence(
                bucket_match=match,
                precision=temp.precision,
                wu_lag_confirmed=False,
                is_peak_hours=peak,
                historical_accuracy=historical_accuracy,
                calibration_adjustment=calibration_adj,
                metar_age_minutes=metar_age_minutes,
            )
            win_probability = confidence.total
            if self._calibrator is not None:
                calibrated = self._calibrator.get_calibrated_probability(confidence.total)
                if calibrated is not None:
                    win_probability = calibrated
            if context_signal is not None and self._self_learner is not None:
                win_probability = self._self_learner.blend_probability(
                    base_probability=win_probability,
                    context=context_signal,
                )

            bucket_min_confidence = self._bucket_min_confidence(
                bucket_type=bucket.bucket_type,
                base_min_confidence=min_confidence,
            )
            if confidence.total < bucket_min_confidence:
                log.debug(
                    "signal_below_threshold",
                    city=market.info.city,
                    bucket=f"{bucket.bucket_type}:{bucket.bucket_value}{bucket.unit}",
                    confidence=round(confidence.total, 3),
                    threshold=bucket_min_confidence,
                    breakdown={
                        "base": round(confidence.base, 3),
                        "precision": round(confidence.precision_bonus, 3),
                        "peak": round(confidence.peak_hours_bonus, 3),
                        "recency": round(confidence.recency_bonus, 3),
                    },
                )
                candidates.append(self._build_candidate(
                    market=market,
                    bucket_type=bucket.bucket_type,
                    bucket_value=bucket.bucket_value,
                    unit=bucket.unit,
                    token_id=bucket_data.token_id,
                    metar_temp_c=temp.celsius,
                    metar_precision=temp.precision.value,
                    metar_age_minutes=metar_age_minutes,
                    local_hour_val=hour,
                    matched_bucket=True,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    confidence=confidence,
                    calibrated_probability=win_probability,
                    status="CONFIDENCE_BLOCKED",
                    reason=(
                        "below_exact_min_confidence"
                        if bucket.bucket_type == "exact"
                        else "below_min_confidence"
                    ),
                ))
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
                best_bid=best_bid,
                win_probability=win_probability,
            )

            self._emitted.add(signal_key)
            signals.append(signal)
            candidates.append(self._build_candidate(
                market=market,
                bucket_type=bucket.bucket_type,
                bucket_value=bucket.bucket_value,
                unit=bucket.unit,
                token_id=bucket_data.token_id,
                metar_temp_c=temp.celsius,
                metar_precision=temp.precision.value,
                metar_age_minutes=metar_age_minutes,
                local_hour_val=hour,
                matched_bucket=True,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                confidence=confidence,
                calibrated_probability=win_probability,
                status="EMITTED",
                reason="all_gates_passed",
            ))

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

        return signals, candidates

    def _build_candidate(
        self,
        market,
        bucket_type: str,
        bucket_value: int,
        unit: str,
        token_id: str,
        metar_temp_c: float | None,
        metar_precision: str | None,
        metar_age_minutes: float | None,
        local_hour_val: int | None,
        status: str,
        reason: str,
        matched_bucket: bool | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        bid_depth: float | None = None,
        ask_depth: float | None = None,
        confidence: ConfidenceFactors | None = None,
        calibrated_probability: float | None = None,
    ) -> SignalCandidate:
        spread = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        return SignalCandidate(
            event_id=market.event_id,
            token_id=token_id,
            city=market.info.city,
            market_date=str(market.info.market_date),
            market_type=market.market_type,
            bucket_type=bucket_type,
            bucket_value=bucket_value,
            bucket_unit=unit,
            metar_temp_c=metar_temp_c,
            metar_precision=metar_precision,
            metar_age_minutes=metar_age_minutes,
            local_hour=local_hour_val,
            matched_bucket=matched_bucket,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            confidence_total=confidence.total if confidence else None,
            calibrated_probability=calibrated_probability,
            confidence_base=confidence.base if confidence else None,
            confidence_precision_bonus=confidence.precision_bonus if confidence else None,
            confidence_peak_bonus=confidence.peak_hours_bonus if confidence else None,
            confidence_recency_bonus=confidence.recency_bonus if confidence else None,
            confidence_historical_blend=confidence.historical_blend if confidence else None,
            confidence_calibration_adj=confidence.calibration_adjustment if confidence else None,
            status=status,
            reason=reason,
        )

    def _persist_candidates(self, candidates: list[SignalCandidate]) -> None:
        session = get_session()
        try:
            session.bulk_save_objects(candidates)
            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("signal_candidate_persist_failed", error=str(e))
        finally:
            session.close()

    def reset_emitted(self) -> None:
        """Clear emitted signals cache (e.g., at start of new day)."""
        self._emitted.clear()

    def _effective_signal_gates(self) -> tuple[float, float, float, float, int, int]:
        """Return active signal gate values, optionally relaxed in shadow mode."""
        min_price = self._config.min_price
        max_price = self._config.max_price.value
        min_confidence = self._config.min_confidence.value
        metar_max_age_minutes = self._config.metar_max_age_minutes
        geq_min_hour = self._config.geq_min_hour
        leq_min_hour = self._config.leq_min_hour

        if self._config.shadow_expansion_enabled:
            min_price = min(min_price, self._config.shadow_min_price)
            max_price = max(max_price, self._config.shadow_max_price)
            min_confidence = min(min_confidence, self._config.shadow_min_confidence)
            metar_max_age_minutes = max(metar_max_age_minutes, self._config.shadow_metar_max_age_minutes)
            geq_min_hour = min(geq_min_hour, self._config.shadow_geq_min_hour)
            leq_min_hour = min(leq_min_hour, self._config.shadow_leq_min_hour)

        if self._config.adaptive_time_gates_enabled:
            geq_min_hour = max(
                self._config.adaptive_time_gate_min_geq_hour,
                geq_min_hour - max(0, self._adaptive_geq_relax_hours),
            )
            leq_min_hour = max(
                self._config.adaptive_time_gate_min_leq_hour,
                leq_min_hour - max(0, self._adaptive_leq_relax_hours),
            )

        return (
            min_price,
            max_price,
            min_confidence,
            metar_max_age_minutes,
            geq_min_hour,
            leq_min_hour,
        )

    def _exact_bucket_allowed(self, local_hour_val: int, metar_precision: str | None) -> tuple[bool, str]:
        """Gate exact buckets to controlled shadow pilot conditions."""
        if not self._config.shadow_expansion_enabled or not self._config.shadow_exact_enabled:
            return False, "exact_disabled"
        if local_hour_val < self._config.shadow_exact_min_hour:
            return False, f"exact_before_{self._config.shadow_exact_min_hour}"
        if metar_precision != "tenths":
            return False, "exact_requires_tenths_precision"
        return True, "exact_shadow_enabled"

    def _bucket_min_confidence(self, bucket_type: str, base_min_confidence: float) -> float:
        """Use stricter confidence threshold for exact-bucket shadow pilot."""
        if bucket_type == "exact" and self._config.shadow_exact_enabled:
            return max(base_min_confidence, self._config.shadow_exact_min_confidence)
        return base_min_confidence

    def _passes_liquidity_gate(
        self,
        best_bid: float | None,
        best_ask: float,
        ask_depth: float,
    ) -> tuple[bool, str]:
        """Reject low-quality books with shallow asks or very wide spreads."""
        if not self._config.liquidity_gate_enabled:
            return True, "liquidity_gate_disabled"

        if ask_depth < self._config.liquidity_min_ask_depth_usd:
            return False, "ask_depth_below_min"
        if best_bid is None:
            return False, "missing_best_bid"

        spread = best_ask - best_bid
        if spread < 0:
            return False, "inverted_spread"
        if spread > self._config.liquidity_max_spread_abs:
            return False, "spread_above_abs_limit"
        if best_ask > 0 and (spread / best_ask) > self._config.liquidity_max_spread_pct_of_ask:
            return False, "spread_above_pct_limit"
        return True, "liquidity_ok"

    def _update_adaptive_time_gates(self, emitted_count: int) -> None:
        """Adaptively relax directional time gates when opportunity flow is starved."""
        if not self._config.adaptive_time_gates_enabled:
            return

        if emitted_count > 0:
            self._no_emit_cycles = 0
            if (
                self._config.adaptive_time_gate_reset_on_emit
                and (self._adaptive_geq_relax_hours > 0 or self._adaptive_leq_relax_hours > 0)
            ):
                self._adaptive_geq_relax_hours = 0
                self._adaptive_leq_relax_hours = 0
                log.info("adaptive_time_gates_reset")
            return

        self._no_emit_cycles += 1
        threshold = max(1, int(self._config.adaptive_time_gate_no_emit_cycles))
        if self._no_emit_cycles < threshold:
            return

        self._no_emit_cycles = 0
        step = max(1, int(self._config.adaptive_time_gate_step_hours))

        base_geq = self._config.geq_min_hour
        base_leq = self._config.leq_min_hour
        if self._config.shadow_expansion_enabled:
            base_geq = min(base_geq, self._config.shadow_geq_min_hour)
            base_leq = min(base_leq, self._config.shadow_leq_min_hour)

        max_geq_relax = max(0, base_geq - self._config.adaptive_time_gate_min_geq_hour)
        max_leq_relax = max(0, base_leq - self._config.adaptive_time_gate_min_leq_hour)

        prev_geq = self._adaptive_geq_relax_hours
        prev_leq = self._adaptive_leq_relax_hours
        self._adaptive_geq_relax_hours = min(max_geq_relax, prev_geq + step)
        self._adaptive_leq_relax_hours = min(max_leq_relax, prev_leq + step)

        if self._adaptive_geq_relax_hours != prev_geq or self._adaptive_leq_relax_hours != prev_leq:
            log.info(
                "adaptive_time_gates_relaxed",
                geq_relax_hours=self._adaptive_geq_relax_hours,
                leq_relax_hours=self._adaptive_leq_relax_hours,
                geq_active_hour=max(base_geq - self._adaptive_geq_relax_hours, self._config.adaptive_time_gate_min_geq_hour),
                leq_active_hour=max(base_leq - self._adaptive_leq_relax_hours, self._config.adaptive_time_gate_min_leq_hour),
            )
