"""Tests for backtest replay realism controls."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backtest import run_backtest
from db.models import Base, Trade


def _seed_db(db_path: str, trades: list[Trade]) -> None:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.add_all(trades)
        session.commit()
    finally:
        session.close()


def _trade(
    *,
    token_id: str,
    confidence: float,
    price: float,
    resolved_correct: bool,
    calibrated_probability: float | None = None,
    fill_price: float | None = None,
) -> Trade:
    now = datetime.utcnow()
    return Trade(
        event_id="evt1",
        condition_id="cond1",
        city="Chicago",
        market_date="2026-02-23",
        icao_station="KORD",
        bucket_value=70,
        bucket_type="geq",
        bucket_unit="F",
        token_id=token_id,
        market_type="temperature",
        action="BUY",
        requested_size=100.0,
        requested_cost=40.0,
        price=price,
        size=100.0,
        cost=price * 100.0,
        order_status="DRY_RUN",
        fill_price=fill_price,
        confidence=confidence,
        calibrated_probability=calibrated_probability,
        resolved_correct=resolved_correct,
        pnl=10.0 if resolved_correct else -10.0,
        resolved_at=now - timedelta(hours=1),
        created_at=now - timedelta(hours=2),
    )


def test_backtest_can_use_calibrated_probability(tmp_path):
    db_path = tmp_path / "bt_calibrated.db"
    _seed_db(
        str(db_path),
        [
            _trade(
                token_id="tok1",
                confidence=0.81,
                calibrated_probability=0.93,
                price=0.80,
                resolved_correct=True,
            )
        ],
    )

    without_calibrated = run_backtest(
        db_path=str(db_path),
        use_calibrated_prob=False,
        use_fill_price=False,
    )
    with_calibrated = run_backtest(
        db_path=str(db_path),
        use_calibrated_prob=True,
        use_fill_price=False,
    )

    assert without_calibrated["results"]["total_trades"] == 0
    assert with_calibrated["results"]["total_trades"] == 1


def test_backtest_modeled_slippage_reduces_pnl(tmp_path):
    db_path = tmp_path / "bt_slippage.db"
    _seed_db(
        str(db_path),
        [
            _trade(
                token_id="tok2",
                confidence=0.95,
                calibrated_probability=0.95,
                price=0.30,
                resolved_correct=True,
                fill_price=None,
            )
        ],
    )

    no_slip = run_backtest(
        db_path=str(db_path),
        use_fill_price=False,
        entry_slippage_bps=0.0,
    )
    high_slip = run_backtest(
        db_path=str(db_path),
        use_fill_price=False,
        entry_slippage_bps=300.0,
    )

    assert no_slip["results"]["total_trades"] == 1
    assert high_slip["results"]["total_trades"] == 1
    assert high_slip["results"]["final_bankroll"] < no_slip["results"]["final_bankroll"]


def test_backtest_dynamic_max_bet_scales_with_bankroll(tmp_path):
    db_path = tmp_path / "bt_dynamic_max.db"
    trades = [
        _trade(
            token_id=f"tok_dyn_{i}",
            confidence=0.96,
            calibrated_probability=0.96,
            price=0.30,
            resolved_correct=True,
            fill_price=0.30,
        )
        for i in range(6)
    ]
    _seed_db(str(db_path), trades)

    fixed = run_backtest(
        db_path=str(db_path),
        max_bet=100.0,
        dynamic_max_bet_enabled=False,
    )
    dynamic = run_backtest(
        db_path=str(db_path),
        max_bet=100.0,
        dynamic_max_bet_enabled=True,
        dynamic_max_bet_pct=0.20,
        dynamic_max_bet_floor=100.0,
        dynamic_max_bet_cap=1000.0,
        aggression_enabled=False,
    )

    assert fixed["results"]["total_trades"] == dynamic["results"]["total_trades"] == 6
    assert dynamic["results"]["final_bankroll"] > fixed["results"]["final_bankroll"]
