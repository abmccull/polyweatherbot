"""Context-aware self-learning model for adaptive probability estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_

from config import Config
from db.engine import get_session
from db.models import Trade
from db.state import load_state, save_state
from utils.logging import get_logger

log = get_logger("self_learner")

NON_EXECUTED_STATUSES = ("FAILED", "CANCELED", "REJECTED")
STATE_KEY = "learning.self_learner.v1"
STATE_VERSION = 1

SEGMENT_WEIGHTS: dict[str, float] = {
    "city": 0.45,
    "hour": 0.25,
    "bucket_type": 0.20,
    "precision": 0.10,
}


def _is_executed_order() -> object:
    return or_(
        Trade.order_status.is_(None),
        Trade.order_status.notin_(NON_EXECUTED_STATUSES),
    )


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class SegmentModel:
    """Posterior stats for one context segment key."""

    wins: int
    trades: int
    posterior: float
    lift: float
    shrinkage: float


@dataclass(frozen=True)
class ContextSignal:
    """Self-learning output for a specific signal context."""

    context_probability: float
    reliability: float
    confidence_adjustment: float
    matched_segments: dict[str, int]


class SelfLearner:
    """Bayesian context learner with shrinkage and reliability-weighted blending."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._global_probability = 0.5
        self._trained_samples = 0
        self._updated_at: str | None = None
        self._segments: dict[str, dict[str, SegmentModel]] = {
            "city": {},
            "hour": {},
            "bucket_type": {},
            "precision": {},
        }

    def restore(self) -> bool:
        """Restore the persisted model state from bot_state."""
        state = load_state(STATE_KEY, None)
        if not isinstance(state, dict):
            return False

        try:
            if int(state.get("version", 0)) != STATE_VERSION:
                return False

            self._global_probability = _clip01(float(state.get("global_probability", 0.5)))
            self._trained_samples = max(0, int(state.get("trained_samples", 0)))
            updated_at = state.get("updated_at")
            self._updated_at = str(updated_at) if updated_at else None

            raw_segments = state.get("segments", {})
            restored: dict[str, dict[str, SegmentModel]] = {
                "city": {},
                "hour": {},
                "bucket_type": {},
                "precision": {},
            }
            if isinstance(raw_segments, dict):
                for segment_type, values in raw_segments.items():
                    if segment_type not in restored or not isinstance(values, dict):
                        continue
                    for key, payload in values.items():
                        if not isinstance(payload, dict):
                            continue
                        restored[segment_type][str(key)] = SegmentModel(
                            wins=max(0, int(payload.get("wins", 0))),
                            trades=max(0, int(payload.get("trades", 0))),
                            posterior=_clip01(float(payload.get("posterior", 0.5))),
                            lift=float(payload.get("lift", 0.0)),
                            shrinkage=_clip01(float(payload.get("shrinkage", 0.0))),
                        )
            self._segments = restored
            return True
        except Exception as e:
            log.warning("self_learning_restore_failed", error=str(e))
            return False

    def retrain(self) -> dict[str, Any]:
        """Rebuild model from resolved historical trades and persist it."""
        alpha = max(0.1, float(self._config.self_learning_prior_alpha))
        beta = max(0.1, float(self._config.self_learning_prior_beta))
        lookback_days = max(1, int(self._config.self_learning_lookback_days))
        min_segment_samples = max(1, int(self._config.self_learning_min_segment_samples))
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        session = get_session()
        try:
            rows = (
                session.query(
                    Trade.city,
                    Trade.local_hour,
                    Trade.bucket_type,
                    Trade.metar_precision,
                    Trade.resolved_correct,
                )
                .filter(
                    Trade.action == "BUY",
                    Trade.resolved_correct.isnot(None),
                    Trade.resolved_at.isnot(None),
                    Trade.resolved_at >= cutoff,
                    Trade.size > 0.0,
                    _is_executed_order(),
                )
                .all()
            )
        finally:
            session.close()

        wins = 0
        total = 0
        counters: dict[str, dict[str, tuple[int, int]]] = {
            "city": {},
            "hour": {},
            "bucket_type": {},
            "precision": {},
        }

        for city, hour, bucket_type, precision, resolved_correct in rows:
            won = bool(resolved_correct)
            wins += 1 if won else 0
            total += 1

            self._increment_counter(counters["city"], self._normalize_city(city), won)
            self._increment_counter(counters["hour"], self._normalize_hour(hour), won)
            self._increment_counter(counters["bucket_type"], self._normalize_bucket_type(bucket_type), won)
            self._increment_counter(counters["precision"], self._normalize_precision(precision), won)

        if total > 0:
            self._global_probability = _clip01((wins + alpha) / (total + alpha + beta))
        else:
            self._global_probability = _clip01(alpha / (alpha + beta))
        self._trained_samples = total
        self._updated_at = datetime.utcnow().isoformat()

        self._segments = {
            "city": self._build_segment_models(
                counters["city"], self._config.self_learning_city_prior, alpha, beta, min_segment_samples,
            ),
            "hour": self._build_segment_models(
                counters["hour"], self._config.self_learning_hour_prior, alpha, beta, min_segment_samples,
            ),
            "bucket_type": self._build_segment_models(
                counters["bucket_type"], self._config.self_learning_bucket_prior, alpha, beta, min_segment_samples,
            ),
            "precision": self._build_segment_models(
                counters["precision"], self._config.self_learning_precision_prior, alpha, beta, min_segment_samples,
            ),
        }

        self._persist()
        diagnostics = self.get_diagnostics()
        log.info("self_learning_retrained", **diagnostics)
        return diagnostics

    def get_context_signal(
        self,
        city: str,
        hour: int | None,
        bucket_type: str,
        precision: str,
    ) -> ContextSignal | None:
        """Estimate context probability and confidence adjustment for a new signal."""
        if not self._config.self_learning_enabled:
            return None
        if self._trained_samples < self._config.self_learning_min_samples:
            return None

        matched_segments: dict[str, int] = {}
        effective_samples = 0
        context_logit = _logit(self._global_probability)

        keys = {
            "city": self._normalize_city(city),
            "hour": self._normalize_hour(hour),
            "bucket_type": self._normalize_bucket_type(bucket_type),
            "precision": self._normalize_precision(precision),
        }

        for segment_type, key in keys.items():
            if key is None:
                continue
            model = self._segments.get(segment_type, {}).get(key)
            if model is None:
                continue

            weight = SEGMENT_WEIGHTS.get(segment_type, 0.0)
            context_logit += model.lift * model.shrinkage * weight
            matched_segments[segment_type] = model.trades
            effective_samples += model.trades

        if not matched_segments:
            return None

        context_probability = _clip01(_sigmoid(context_logit))
        reliability_den = max(1, int(self._config.self_learning_reliability_samples))
        reliability = min(1.0, effective_samples / reliability_den)

        conf_scale = float(self._config.self_learning_confidence_scale)
        conf_cap = abs(float(self._config.self_learning_confidence_cap))
        confidence_adjustment = (context_probability - self._global_probability) * conf_scale * reliability
        confidence_adjustment = max(-conf_cap, min(conf_cap, confidence_adjustment))

        return ContextSignal(
            context_probability=context_probability,
            reliability=reliability,
            confidence_adjustment=confidence_adjustment,
            matched_segments=matched_segments,
        )

    def blend_probability(self, base_probability: float, context: ContextSignal | None) -> float:
        """Blend base probability with self-learning context probability."""
        base = _clip01(base_probability)
        if not self._config.self_learning_enabled or context is None:
            return base

        blend_strength = _clip01(float(self._config.self_learning_blend))
        blend = blend_strength * _clip01(context.reliability)
        blended = base + (context.context_probability - base) * blend
        return _clip01(blended)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return compact model diagnostics for logs/metrics."""
        return {
            "enabled": self._config.self_learning_enabled,
            "trained_samples": int(self._trained_samples),
            "global_probability": round(self._global_probability, 5),
            "city_segments": len(self._segments.get("city", {})),
            "hour_segments": len(self._segments.get("hour", {})),
            "bucket_segments": len(self._segments.get("bucket_type", {})),
            "precision_segments": len(self._segments.get("precision", {})),
            "updated_at": self._updated_at,
        }

    def _persist(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "global_probability": self._global_probability,
            "trained_samples": self._trained_samples,
            "updated_at": self._updated_at,
            "segments": {
                name: {
                    key: {
                        "wins": model.wins,
                        "trades": model.trades,
                        "posterior": model.posterior,
                        "lift": model.lift,
                        "shrinkage": model.shrinkage,
                    }
                    for key, model in values.items()
                }
                for name, values in self._segments.items()
            },
        }
        save_state(STATE_KEY, payload)

    @staticmethod
    def _increment_counter(store: dict[str, tuple[int, int]], key: str | None, won: bool) -> None:
        if key is None:
            return
        wins, total = store.get(key, (0, 0))
        wins += 1 if won else 0
        total += 1
        store[key] = (wins, total)

    def _build_segment_models(
        self,
        counter: dict[str, tuple[int, int]],
        prior_strength: float,
        alpha: float,
        beta: float,
        min_samples: int,
    ) -> dict[str, SegmentModel]:
        models: dict[str, SegmentModel] = {}
        prior_strength = max(0.0, float(prior_strength))
        global_logit = _logit(self._global_probability)

        for key, (wins, trades) in counter.items():
            if trades < min_samples:
                continue
            posterior = _clip01((wins + alpha) / (trades + alpha + beta))
            lift = _logit(posterior) - global_logit
            shrinkage = trades / (trades + prior_strength) if prior_strength > 0 else 1.0
            models[key] = SegmentModel(
                wins=wins,
                trades=trades,
                posterior=posterior,
                lift=lift,
                shrinkage=_clip01(shrinkage),
            )
        return models

    @staticmethod
    def _normalize_city(city: str | None) -> str | None:
        if not city:
            return None
        key = city.strip().lower()
        return key or None

    @staticmethod
    def _normalize_hour(hour: int | None) -> str | None:
        if hour is None:
            return None
        h = int(hour)
        if h < 0 or h > 23:
            return None
        return str(h)

    @staticmethod
    def _normalize_bucket_type(bucket_type: str | None) -> str | None:
        if not bucket_type:
            return None
        key = bucket_type.strip().lower()
        return key or None

    @staticmethod
    def _normalize_precision(precision: str | None) -> str | None:
        if not precision:
            return None
        key = precision.strip().lower()
        return key or None
