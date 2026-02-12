"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config
from db import engine as db_engine
from db.models import Base


@pytest.fixture
def config():
    """Minimal config for testing."""
    return Config(dry_run=True, initial_bankroll=1000.0)


@pytest.fixture
def db_session(monkeypatch):
    """In-memory SQLite DB with all tables created.

    Monkey-patches db_engine so get_session() returns sessions
    bound to this in-memory DB.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    monkeypatch.setattr(db_engine, "_session_factory", factory)
    monkeypatch.setattr(db_engine, "_engine", engine)

    session = factory()
    yield session
    session.close()
