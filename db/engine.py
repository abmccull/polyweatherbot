"""SQLite connection and session factory."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config import Config
from db.models import Base


_session_factory: sessionmaker | None = None
_engine = None


def init_db(config: Config) -> sessionmaker:
    """Create SQLite file, tables, and return a session factory."""
    global _session_factory, _engine

    _engine = create_engine(config.db_url, echo=False)

    # Enable WAL mode and relaxed sync for better concurrent performance
    with _engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.commit()

    Base.metadata.create_all(_engine)
    _migrate_add_precip_columns(_engine)
    _migrate_add_fee_column(_engine)
    _migrate_add_exit_columns(_engine)
    _migrate_add_requested_columns(_engine)
    _migrate_add_expected_columns(_engine)
    _migrate_add_calibration_columns(_engine)
    _migrate_fix_sell_trades(_engine)
    _migrate_fix_failed_order_rows(_engine)
    _migrate_backfill_sell_parent_links(_engine)
    _session_factory = sessionmaker(bind=_engine)
    return _session_factory


def _migrate_add_precip_columns(engine) -> None:
    """Add bucket_low_inches / bucket_high_inches to trades if missing.

    Handles existing DBs without requiring a fresh database.
    """
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(trades)"))
        columns = {row[1] for row in result}

        if "bucket_low_inches" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN bucket_low_inches REAL"))
        if "bucket_high_inches" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN bucket_high_inches REAL"))
        conn.commit()


def _migrate_add_fee_column(engine) -> None:
    """Add fee_paid column to trades if missing."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(trades)"))
        columns = {row[1] for row in result}

        if "fee_paid" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN fee_paid REAL DEFAULT 0.0"))
        conn.commit()


def _migrate_add_exit_columns(engine) -> None:
    """Add exit_reason and parent_trade_id columns to trades if missing."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(trades)"))
        columns = {row[1] for row in result}

        if "exit_reason" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN exit_reason TEXT"))
        if "parent_trade_id" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN parent_trade_id INTEGER"))
        conn.commit()


def _migrate_add_requested_columns(engine) -> None:
    """Add requested_size/requested_cost columns to trades if missing."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(trades)"))
        columns = {row[1] for row in result}

        if "requested_size" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN requested_size REAL"))
        if "requested_cost" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN requested_cost REAL"))
        conn.commit()


def _migrate_add_expected_columns(engine) -> None:
    """Add expected_edge/expected_profit/expected_slippage to trades if missing."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(trades)"))
        columns = {row[1] for row in result}

        if "expected_edge" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN expected_edge REAL"))
        if "expected_profit" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN expected_profit REAL"))
        if "expected_slippage" not in columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN expected_slippage REAL"))
        conn.commit()


def _migrate_add_calibration_columns(engine) -> None:
    """Add calibrated_probability columns to trades/signal_candidates if missing."""
    with engine.connect() as conn:
        trade_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(trades)"))}
        if "calibrated_probability" not in trade_columns:
            conn.execute(text("ALTER TABLE trades ADD COLUMN calibrated_probability REAL"))

        signal_exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='signal_candidates'")
        ).first()
        if signal_exists:
            signal_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(signal_candidates)"))}
            if "calibrated_probability" not in signal_columns:
                conn.execute(text("ALTER TABLE signal_candidates ADD COLUMN calibrated_probability REAL"))
        conn.commit()


def _migrate_fix_sell_trades(engine) -> None:
    """Fix existing SELL trades that were incorrectly marked as resolved.

    SELL trades should be inert execution records — all P&L is computed
    at BUY resolution time.
    """
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE trades SET resolved_correct=NULL, pnl=NULL, resolved_at=NULL "
            "WHERE action='SELL' AND resolved_correct IS NOT NULL"
        ))
        conn.commit()


def _migrate_fix_failed_order_rows(engine) -> None:
    """Neutralize legacy FAILED/CANCELED/REJECTED rows so they are audit-only."""
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE trades SET size=0.0, cost=0.0, fee_paid=0.0 "
            "WHERE order_status IN ('FAILED', 'CANCELED', 'REJECTED') "
            "AND (size != 0.0 OR cost != 0.0 OR COALESCE(fee_paid, 0.0) != 0.0)"
        ))
        conn.execute(text(
            "UPDATE trades SET resolved_correct=NULL, pnl=NULL, resolved_at=NULL "
            "WHERE action='BUY' AND order_status IN ('FAILED', 'CANCELED', 'REJECTED')"
        ))
        conn.commit()


def _migrate_backfill_sell_parent_links(engine) -> None:
    """Backfill SELL.parent_trade_id when a token has exactly one executable BUY."""
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE trades "
            "SET parent_trade_id = ("
            "  SELECT MIN(b.id) FROM trades b "
            "  WHERE b.action='BUY' "
            "    AND b.token_id=trades.token_id "
            "    AND (b.order_status IS NULL OR b.order_status NOT IN ('FAILED', 'CANCELED', 'REJECTED'))"
            ") "
            "WHERE trades.action='SELL' "
            "  AND trades.parent_trade_id IS NULL "
            "  AND (trades.order_status IS NULL OR trades.order_status NOT IN ('FAILED', 'CANCELED', 'REJECTED')) "
            "  AND ("
            "    SELECT COUNT(*) FROM trades b "
            "    WHERE b.action='BUY' "
            "      AND b.token_id=trades.token_id "
            "      AND (b.order_status IS NULL OR b.order_status NOT IN ('FAILED', 'CANCELED', 'REJECTED'))"
            "  ) = 1"
        ))
        conn.commit()


def get_session() -> Session:
    """Return a new session from the factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


def cleanup_old_data(archive_days: int = 90) -> int:
    """Archive observations older than archive_days. Returns rows deleted."""
    if _engine is None:
        return 0

    cutoff = datetime.utcnow() - timedelta(days=archive_days)
    with _engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM observations WHERE created_at < :cutoff"),
            {"cutoff": cutoff},
        )
        deleted = result.rowcount or 0

    # VACUUM must run outside an explicit transaction in SQLite.
    with _engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql("VACUUM")

    return deleted
