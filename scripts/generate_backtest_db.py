#!/usr/bin/env python3
"""Generate a large, diverse synthetic DB for Station Sniper backtesting."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from db.engine import get_session, init_db
from db.models import SignalCandidate, Trade


@dataclass(frozen=True)
class CityProfile:
    city: str
    station: str
    climate: str


CITY_PROFILES = [
    CityProfile("Chicago", "KORD", "continental"),
    CityProfile("New York", "KJFK", "coastal"),
    CityProfile("Dallas", "KDFW", "hot"),
    CityProfile("Phoenix", "KPHX", "desert"),
    CityProfile("Atlanta", "KATL", "humid"),
    CityProfile("Denver", "KDEN", "mountain"),
    CityProfile("Miami", "KMIA", "tropical"),
    CityProfile("Seattle", "KSEA", "marine"),
    CityProfile("Los Angeles", "KLAX", "coastal"),
    CityProfile("San Francisco", "KSFO", "marine"),
    CityProfile("Boston", "KBOS", "coastal"),
    CityProfile("Detroit", "KDTW", "continental"),
    CityProfile("Minneapolis", "KMSP", "continental"),
    CityProfile("Houston", "KIAH", "humid"),
    CityProfile("Austin", "KAUS", "hot"),
    CityProfile("Las Vegas", "KLAS", "desert"),
    CityProfile("Salt Lake City", "KSLC", "mountain"),
    CityProfile("Portland", "KPDX", "marine"),
    CityProfile("San Diego", "KSAN", "coastal"),
    CityProfile("Philadelphia", "KPHL", "coastal"),
    CityProfile("Washington", "KDCA", "humid"),
    CityProfile("Charlotte", "KCLT", "humid"),
    CityProfile("Nashville", "KBNA", "humid"),
    CityProfile("St. Louis", "KSTL", "continental"),
    CityProfile("Kansas City", "KMCI", "continental"),
    CityProfile("New Orleans", "KMSY", "tropical"),
    CityProfile("Tampa", "KTPA", "tropical"),
    CityProfile("Orlando", "KMCO", "tropical"),
    CityProfile("Cleveland", "KCLE", "continental"),
    CityProfile("Pittsburgh", "KPIT", "continental"),
    CityProfile("Buffalo", "KBUF", "continental"),
    CityProfile("Cincinnati", "KCVG", "continental"),
    CityProfile("Indianapolis", "KIND", "continental"),
    CityProfile("Raleigh", "KRDU", "humid"),
    CityProfile("Baltimore", "KBWI", "humid"),
    CityProfile("Anchorage", "PANC", "cold"),
    CityProfile("Honolulu", "PHNL", "tropical"),
    CityProfile("London", "EGLL", "marine"),
    CityProfile("Toronto", "CYYZ", "continental"),
    CityProfile("Montreal", "CYUL", "continental"),
]

TEMP_BUCKETS_F = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
PRECIP_BUCKETS = [
    ("lt", None, 1.0),
    ("lt", None, 2.0),
    ("range", 0.5, 1.5),
    ("range", 1.0, 2.0),
    ("range", 2.0, 4.0),
    ("gt", 3.0, None),
    ("gt", 5.0, None),
]

EMITTED_RATIO = 0.55


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wipe_db_files(db_path: Path) -> None:
    for candidate in [db_path, Path(str(db_path) + "-shm"), Path(str(db_path) + "-wal")]:
        if candidate.exists():
            candidate.unlink()


def _regime_adjustment(i: int, total: int) -> float:
    """Simulate cyclic market quality over the full dataset."""
    if total <= 0:
        return 0.0
    frac = i / total
    if frac < 0.20:
        return 0.07
    if frac < 0.45:
        return 0.02
    if frac < 0.70:
        return -0.01
    if frac < 0.90:
        return -0.05
    return 0.00


def _seasonal_conf_adj(month: int, climate: str) -> float:
    if climate in ("tropical", "humid") and month in (6, 7, 8, 9):
        return -0.03
    if climate in ("continental", "cold") and month in (12, 1, 2):
        return -0.02
    if climate in ("desert", "hot") and month in (4, 5, 10):
        return 0.02
    return 0.0


def _pick_market_type(precip_ratio: float) -> str:
    return "precipitation" if random.random() < precip_ratio else "temperature"


def _build_temperature_bucket() -> tuple[str, int, str, float | None, float | None]:
    bucket_type = random.choice(["geq", "leq"])
    bucket_value = random.choice(TEMP_BUCKETS_F)
    return bucket_type, bucket_value, "F", None, None


def _build_precip_bucket() -> tuple[str, int, str, float | None, float | None]:
    bucket_type, low, high = random.choice(PRECIP_BUCKETS)
    return bucket_type, 0, "inches", low, high


def _status_for_executed() -> str:
    r = random.random()
    if r < 0.78:
        return "FILLED"
    if r < 0.90:
        return "MATCHED"
    return "DRY_RUN"


def generate(
    db_path: Path,
    trades_count: int,
    candidates_count: int,
    noise_trades_count: int,
    years: int,
    precip_ratio: float,
    seed: int,
) -> None:
    random.seed(seed)
    _wipe_db_files(db_path)

    cfg = Config(db_path=str(db_path), dry_run=True, initial_bankroll=1000.0)
    init_db(cfg)

    session = get_session()
    try:
        start = datetime.utcnow() - timedelta(days=365 * max(1, years))
        total_minutes = max(1, 365 * max(1, years) * 24 * 60)
        trades: list[Trade] = []
        candidates: list[SignalCandidate] = []

        wins = 0
        total_pnl = 0.0
        temp_count = 0
        precip_count = 0

        for i in range(trades_count):
            profile = random.choice(CITY_PROFILES)
            base_minutes = int((i / max(1, trades_count - 1)) * total_minutes)
            jitter = random.randint(-180, 180)
            created_at = start + timedelta(minutes=_clip(base_minutes + jitter, 0, total_minutes))
            market_day = created_at.date()

            # Temperature resolves quickly (D+1), precip monthly (D+35-ish).
            market_type = _pick_market_type(precip_ratio)
            if market_type == "temperature":
                resolved_at = created_at + timedelta(days=1, hours=random.randint(2, 20))
                temp_count += 1
                bucket_type, bucket_value, bucket_unit, low_inches, high_inches = _build_temperature_bucket()
                resolution_value = round(random.uniform(-10.0, 46.0), 2)
                local_hour = random.randint(10, 22)
            else:
                resolved_at = created_at + timedelta(days=35 + random.randint(0, 8))
                precip_count += 1
                bucket_type, bucket_value, bucket_unit, low_inches, high_inches = _build_precip_bucket()
                resolution_value = round(random.uniform(0.0, 12.0), 3)
                local_hour = random.randint(8, 23)

            raw_conf = _clip(random.gauss(0.84, 0.10), 0.62, 0.992)
            regime_adj = _regime_adjustment(i, trades_count)
            season_adj = _seasonal_conf_adj(created_at.month, profile.climate)
            calibrated_probability = _clip(
                raw_conf + random.uniform(-0.08, 0.08) + regime_adj + season_adj,
                0.30,
                0.995,
            )

            # Wider price/edge/liquidity diversity than the initial generator.
            target_edge = random.uniform(-0.04, 0.14) + regime_adj
            price_floor = 0.15 if market_type == "temperature" else 0.08
            price = _clip(calibrated_probability - target_edge + random.uniform(-0.05, 0.05), price_floor, 0.92)

            shares = round(random.uniform(8.0, 520.0), 4)

            liquidity_tier = random.random()
            if liquidity_tier < 0.20:
                slip_bps = random.uniform(30.0, 220.0)
                bid_ask_spread = random.uniform(0.010, 0.045)
                depth = random.uniform(10.0, 150.0)
            elif liquidity_tier < 0.70:
                slip_bps = random.uniform(8.0, 80.0)
                bid_ask_spread = random.uniform(0.004, 0.020)
                depth = random.uniform(100.0, 1200.0)
            else:
                slip_bps = random.uniform(2.0, 25.0)
                bid_ask_spread = random.uniform(0.001, 0.010)
                depth = random.uniform(1200.0, 9000.0)

            # Mostly adverse slippage, sometimes price improvement.
            fill_slip_sign = -1.0 if random.random() < 0.10 else 1.0
            fill_slip = fill_slip_sign * price * (slip_bps / 10_000.0)
            fill_price = _clip(price + fill_slip, 0.01, 0.99)
            cost = shares * fill_price

            won = random.random() < calibrated_probability
            if won:
                pnl = (1.0 - fill_price) * shares
                wins += 1
            else:
                pnl = -fill_price * shares
            total_pnl += pnl

            expected_slippage = _clip(abs(fill_slip) * random.uniform(0.70, 1.35), 0.0, 0.12)
            fee_drag = calibrated_probability * 0.02 * (1.0 - price)
            expected_edge = calibrated_probability - price - fee_drag - expected_slippage
            expected_profit = expected_edge * shares

            token_id = (
                f"tok_{profile.city.lower().replace(' ', '_')}_{market_day}_{market_type}_"
                f"{bucket_type}_{i}"
            )
            event_id = f"evt_{profile.city.lower().replace(' ', '_')}_{market_day}_{market_type}"

            trade = Trade(
                event_id=event_id,
                condition_id=f"cond_{i}",
                city=profile.city,
                market_date=str(market_day),
                icao_station=profile.station,
                bucket_value=bucket_value,
                bucket_type=bucket_type,
                bucket_unit=bucket_unit,
                token_id=token_id,
                bucket_low_inches=low_inches,
                bucket_high_inches=high_inches,
                market_type=market_type,
                action="BUY",
                requested_size=shares,
                requested_cost=shares * price,
                price=price,
                size=shares,
                cost=cost,
                order_id=f"order_exec_{i:07d}",
                order_status=_status_for_executed(),
                fill_price=fill_price,
                expected_edge=expected_edge,
                expected_profit=expected_profit,
                expected_slippage=expected_slippage,
                calibrated_probability=calibrated_probability,
                confidence=raw_conf,
                metar_temp_c=round(random.uniform(-12.0, 45.0), 2) if market_type == "temperature" else None,
                metar_precision=random.choice(["whole", "tenths"]) if market_type == "temperature" else None,
                margin_from_boundary=round(random.uniform(0.01, 5.0), 3),
                local_hour=local_hour,
                resolution_value=resolution_value,
                resolved_correct=won,
                pnl=pnl,
                resolved_at=resolved_at,
                created_at=created_at,
                updated_at=resolved_at,
            )
            trades.append(trade)

            # Emitted candidate for each executed trade
            best_bid = _clip(price - bid_ask_spread, 0.01, 0.98)
            candidates.append(
                SignalCandidate(
                    event_id=event_id,
                    token_id=token_id,
                    city=profile.city,
                    market_date=str(market_day),
                    market_type=market_type,
                    bucket_type=bucket_type,
                    bucket_value=bucket_value,
                    bucket_unit=bucket_unit,
                    metar_temp_c=trade.metar_temp_c,
                    metar_precision=trade.metar_precision,
                    metar_age_minutes=round(random.uniform(1.0, 35.0), 2),
                    local_hour=local_hour,
                    matched_bucket=True,
                    best_bid=best_bid,
                    best_ask=price,
                    spread=max(0.0, price - best_bid),
                    bid_depth=depth * random.uniform(0.7, 1.4),
                    ask_depth=depth,
                    confidence_total=raw_conf,
                    calibrated_probability=calibrated_probability,
                    confidence_base=_clip(raw_conf - 0.30, 0.1, 0.8),
                    confidence_precision_bonus=0.10 if trade.metar_precision == "tenths" else 0.0,
                    confidence_peak_bonus=random.choice([-0.10, 0.10]),
                    confidence_recency_bonus=random.choice([0.0, 0.05, 0.10, 0.15]),
                    confidence_historical_blend=random.uniform(-0.07, 0.07),
                    confidence_calibration_adj=calibrated_probability - raw_conf,
                    status="EMITTED",
                    reason="all_gates_passed",
                    created_at=created_at,
                )
            )

        blocked_statuses = [
            ("PRICE_GATE_BLOCKED", "above_max_price"),
            ("PRICE_GATE_BLOCKED", "below_min_price"),
            ("TIME_GATE_BLOCKED", "geq_before_12"),
            ("TIME_GATE_BLOCKED", "leq_before_17"),
            ("CONFIDENCE_BLOCKED", "below_min_confidence"),
            ("BUCKET_MISS", "metar_not_in_bucket"),
            ("NO_PRICE", "missing_best_ask"),
            ("SKIP_EXACT", "exact_bucket"),
        ]

        for i in range(candidates_count):
            profile = random.choice(CITY_PROFILES)
            created_at = start + timedelta(minutes=random.randint(0, total_minutes))
            market_day = created_at.date()
            market_type = _pick_market_type(precip_ratio)
            if market_type == "temperature":
                bucket_type, bucket_value, bucket_unit, _, _ = _build_temperature_bucket()
                metar_temp_c = round(random.uniform(-16.0, 47.0), 2)
                metar_precision = random.choice(["whole", "tenths"])
            else:
                bucket_type, bucket_value, bucket_unit, _, _ = _build_precip_bucket()
                metar_temp_c = None
                metar_precision = None

            # Force some exact buckets for skip-exact diversity
            status, reason = random.choice(blocked_statuses)
            if status == "SKIP_EXACT":
                bucket_type = "exact"

            conf = _clip(random.gauss(0.81, 0.12), 0.35, 0.98)
            calibrated_probability = _clip(conf + random.uniform(-0.12, 0.12), 0.01, 0.99)
            matched_bucket = random.random() < 0.65

            best_bid = _clip(random.uniform(0.02, 0.90), 0.01, 0.98)
            spread = random.uniform(0.001, 0.06)
            best_ask = _clip(best_bid + spread, 0.02, 0.99)

            candidates.append(
                SignalCandidate(
                    event_id=f"evt_candidate_{i % 500}",
                    token_id=f"tok_candidate_{i:07d}",
                    city=profile.city,
                    market_date=str(market_day),
                    market_type=market_type,
                    bucket_type=bucket_type,
                    bucket_value=bucket_value,
                    bucket_unit=bucket_unit,
                    metar_temp_c=metar_temp_c,
                    metar_precision=metar_precision,
                    metar_age_minutes=round(random.uniform(1.0, 60.0), 2),
                    local_hour=random.randint(0, 23),
                    matched_bucket=matched_bucket,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=best_ask - best_bid,
                    bid_depth=random.uniform(5.0, 4000.0),
                    ask_depth=random.uniform(5.0, 4000.0),
                    confidence_total=conf,
                    calibrated_probability=calibrated_probability,
                    confidence_base=_clip(conf - 0.35, 0.0, 0.7),
                    confidence_precision_bonus=random.choice([0.0, 0.10]),
                    confidence_peak_bonus=random.choice([-0.10, 0.10]),
                    confidence_recency_bonus=random.choice([0.0, 0.05, 0.10, 0.15]),
                    confidence_historical_blend=random.uniform(-0.08, 0.08),
                    confidence_calibration_adj=calibrated_probability - conf,
                    status=status if random.random() > EMITTED_RATIO else "EMITTED",
                    reason=reason if random.random() > EMITTED_RATIO else "all_gates_passed",
                    created_at=created_at,
                )
            )

        # Add unresolved/failed/noise records to mimic production DB clutter.
        for i in range(noise_trades_count):
            profile = random.choice(CITY_PROFILES)
            created_at = start + timedelta(minutes=random.randint(0, total_minutes))
            market_day = created_at.date()

            is_sell = random.random() < 0.35
            action = "SELL" if is_sell else "BUY"
            order_status = random.choice(["FAILED", "CANCELED", "REJECTED", "OPEN", "PENDING"])
            price = random.uniform(0.05, 0.95)
            size = 0.0 if order_status in ("FAILED", "CANCELED", "REJECTED") else random.uniform(1.0, 60.0)

            noise_trade = Trade(
                event_id=f"evt_noise_{i % 150}",
                condition_id=f"noise_cond_{i}",
                city=profile.city,
                market_date=str(market_day),
                icao_station=profile.station,
                bucket_value=random.choice(TEMP_BUCKETS_F),
                bucket_type=random.choice(["geq", "leq"]),
                bucket_unit="F",
                token_id=f"tok_noise_{i:07d}",
                market_type="temperature",
                action=action,
                requested_size=size if size > 0 else random.uniform(1.0, 60.0),
                requested_cost=(size if size > 0 else random.uniform(1.0, 60.0)) * price,
                price=price,
                size=size,
                cost=size * price,
                order_id=f"order_noise_{i:07d}",
                order_status=order_status,
                fill_price=(price + random.uniform(-0.01, 0.01)) if size > 0 else None,
                confidence=_clip(random.gauss(0.78, 0.12), 0.2, 0.98),
                resolved_correct=None,
                pnl=None,
                resolved_at=None,
                created_at=created_at,
                updated_at=created_at,
            )
            trades.append(noise_trade)

        session.bulk_save_objects(trades)
        session.bulk_save_objects(candidates)
        session.commit()

        executed_win_rate = wins / trades_count if trades_count > 0 else 0.0
        print(
            "generated",
            f"db={db_path}",
            f"resolved_trades={trades_count}",
            f"noise_trades={noise_trades_count}",
            f"candidates={len(candidates)}",
            f"temp_trades={temp_count}",
            f"precip_trades={precip_count}",
            f"win_rate={executed_win_rate:.3f}",
            f"total_pnl={total_pnl:.2f}",
        )
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Station Sniper backtest DB")
    parser.add_argument("--db", default="station_sniper_backtest_large.db", help="Output SQLite DB path")
    parser.add_argument("--trades", type=int, default=8000, help="Number of resolved BUY trades to generate")
    parser.add_argument(
        "--blocked-candidates",
        type=int,
        default=25000,
        help="Additional signal candidates (blocked + emitted mix)",
    )
    parser.add_argument(
        "--noise-trades",
        type=int,
        default=3000,
        help="Additional unresolved/failed/noise trade rows",
    )
    parser.add_argument("--years", type=int, default=4, help="Historical span in years")
    parser.add_argument("--precip-ratio", type=float, default=0.20, help="Share of precipitation trades [0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    generate(
        db_path=db_path,
        trades_count=max(1, args.trades),
        candidates_count=max(0, args.blocked_candidates),
        noise_trades_count=max(0, args.noise_trades),
        years=max(1, args.years),
        precip_ratio=_clip(args.precip_ratio, 0.0, 1.0),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
