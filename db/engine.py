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
    _migrate_fix_sell_trades(_engine)
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
        deleted = result.rowcount
        # Vacuum to reclaim space
        conn.execute(text("VACUUM"))
    return deleted
