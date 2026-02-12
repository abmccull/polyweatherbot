"""Key-value state persistence using the bot_state table."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from db.engine import get_session
from db.models import BotState


def save_state(key: str, value: Any) -> None:
    """JSON-serialize and upsert a value into bot_state."""
    session = get_session()
    try:
        existing = session.query(BotState).filter(BotState.key == key).first()
        serialized = json.dumps(value)
        if existing:
            existing.value = serialized
            existing.updated_at = datetime.utcnow()
        else:
            session.add(BotState(key=key, value=serialized, updated_at=datetime.utcnow()))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def load_state(key: str, default: Any = None) -> Any:
    """JSON-deserialize a value from bot_state, returning default if missing."""
    session = get_session()
    try:
        row = session.query(BotState).filter(BotState.key == key).first()
        if row is None:
            return default
        return json.loads(row.value)
    finally:
        session.close()
