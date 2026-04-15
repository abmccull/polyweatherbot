import asyncio
import time

import pytest

from config import Config
from copycat.engine import CopycatEngine, LeaderRuntime
from copycat.rules import build_match_key, classify_duplicate_reason, compute_max_copy_price


def test_max_copy_price_uses_abs_and_relative_cap():
    assert compute_max_copy_price(0.50, 0.03, 0.05) == 0.53
    # Relative cap wins when absolute cap is looser.
    assert round(compute_max_copy_price(0.90, 0.10, 0.05), 4) == 0.945


def test_match_key_prefers_condition_id_then_slug():
    assert build_match_key("0xABC", "some-slug", "123") == "cond:0xabc"
    assert build_match_key("", "some-slug", "123") == "slug:some-slug"
    assert build_match_key(None, None, "123") == "token:123"


def test_duplicate_reason_same_or_opposite_side():
    assert classify_duplicate_reason("BUY", "BUY") == "same_match_scalein_ignored"
    assert classify_duplicate_reason("BUY", "SELL") == "opposite_side_ignored_locked"


def test_find_event_market_and_active_flags():
    event = {
        "active": True,
        "closed": False,
        "archived": False,
        "markets": [
            {"conditionId": "0xaaa", "active": True, "closed": False, "archived": False},
            {"conditionId": "0xbbb", "active": False, "closed": False, "archived": False},
        ],
    }
    market = CopycatEngine._find_event_market(event, "0xbbb")
    assert market is not None
    assert market["conditionId"] == "0xbbb"
    assert CopycatEngine._is_event_market_active(event, market) is False


def test_compute_ticket_size_applies_status_and_leader_multiplier():
    cfg = Config(copy_base_ticket_usd=10.0, copy_probation_multiplier=0.60)
    engine = CopycatEngine(cfg, executor=object(), redeemer=object())  # type: ignore[arg-type]
    core = LeaderRuntime(
        wallet="0xcore",
        name="core",
        tier="core",
        status="core",
        multiplier=1.20,
    )
    probation = LeaderRuntime(
        wallet="0xprob",
        name="prob",
        tier="prob",
        status="probation",
        multiplier=1.50,
    )
    assert engine._compute_ticket_size(core) == 12.0
    assert engine._compute_ticket_size(probation) == 9.0


def test_is_sports_event_by_tags_or_title():
    by_tag = {"tags": [{"label": "Sports"}], "title": "Random"}
    by_title = {"tags": [], "title": "Will this NBA game go to OT?"}
    not_sports = {"tags": [{"label": "Politics"}], "title": "Election odds"}

    assert CopycatEngine._is_sports_event(by_tag) is True
    assert CopycatEngine._is_sports_event(by_title) is True
    assert CopycatEngine._is_sports_event(not_sports) is False


def test_trade_timestamp_normalization_supports_seconds_and_milliseconds():
    sec = 1_706_000_000
    ms = sec * 1000
    assert CopycatEngine._to_trade_ts_seconds(sec) == sec
    assert CopycatEngine._to_trade_ts_seconds(ms) == sec
    assert CopycatEngine._to_trade_ts_seconds("1706000000") == sec
    assert CopycatEngine._to_trade_ts_seconds("1706000000000") == sec
    assert CopycatEngine._to_trade_ts_seconds(None) == 0


@pytest.mark.asyncio
async def test_poll_wallet_filters_stale_trades_before_processing():
    now = int(time.time())

    class _Client:
        async def fetch_trades(self, wallet: str, limit: int = 100, offset: int = 0) -> list[dict]:
            return [
                {"transactionHash": "0xold", "timestamp": (now - 120) * 1000},
                {"transactionHash": "0xfresh", "timestamp": (now - 10) * 1000},
            ]

    cfg = Config(copy_trade_max_age_seconds=60, copy_max_wallet_trades_fetch=100)
    engine = CopycatEngine(cfg, executor=object(), redeemer=object())  # type: ignore[arg-type]
    engine._client = _Client()  # type: ignore[assignment]

    processed: list[str] = []

    async def _process_trade(_leader: LeaderRuntime, trade: dict) -> None:
        processed.append(str(trade.get("transactionHash")))

    engine._process_trade = _process_trade  # type: ignore[method-assign]

    leader = LeaderRuntime(
        wallet="0xabc",
        name="test",
        tier="core",
        status="active",
        multiplier=1.0,
        last_seen_ts=now - 180,
    )
    detected = await engine._poll_wallet(leader)

    assert detected == 1
    assert processed == ["0xfresh"]


@pytest.mark.asyncio
async def test_poll_wallet_caps_processed_new_trades():
    now = int(time.time())

    class _Client:
        async def fetch_trades(self, wallet: str, limit: int = 100, offset: int = 0) -> list[dict]:
            return [
                {
                    "transactionHash": f"0x{idx}",
                    "timestamp": (now - 10 + idx) * 1000,
                }
                for idx in range(10)
            ]

    cfg = Config(
        copy_trade_max_age_seconds=60,
        copy_max_wallet_trades_fetch=100,
        copy_max_new_trades_per_poll=3,
    )
    engine = CopycatEngine(cfg, executor=object(), redeemer=object())  # type: ignore[arg-type]
    engine._client = _Client()  # type: ignore[assignment]

    processed: list[str] = []

    async def _process_trade(_leader: LeaderRuntime, trade: dict) -> None:
        processed.append(str(trade.get("transactionHash")))

    engine._process_trade = _process_trade  # type: ignore[method-assign]

    leader = LeaderRuntime(
        wallet="0xabc",
        name="test",
        tier="core",
        status="core",
        multiplier=1.0,
        last_seen_ts=now - 180,
    )
    detected = await engine._poll_wallet(leader)

    assert detected == 3
    assert processed == ["0x7", "0x8", "0x9"]


@pytest.mark.asyncio
async def test_poll_once_prioritizes_core_wallets(monkeypatch):
    cfg = Config(copy_enabled=True, copy_poll_seconds=2, copy_core_poll_priority=True)
    engine = CopycatEngine(cfg, executor=object(), redeemer=object())  # type: ignore[arg-type]
    engine._leaders = {
        "0xprob": LeaderRuntime(
            wallet="0xprob",
            name="prob",
            tier="B",
            status="probation",
            multiplier=1.0,
            last_seen_ts=0,
        ),
        "0xcore": LeaderRuntime(
            wallet="0xcore",
            name="core",
            tier="A",
            status="core",
            multiplier=1.0,
            last_seen_ts=0,
        ),
    }
    calls: list[str] = []

    async def _poll_wallet(leader: LeaderRuntime) -> int:
        calls.append(leader.status)
        return 0

    async def _no_sleep(_: float) -> None:
        return None

    engine._poll_wallet = _poll_wallet  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    await engine.poll_once()

    assert calls == ["core", "probation"]
