"""FastAPI read API for copycat monitoring dashboard."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import (
    Base,
    CopyOrderIntent,
    CopyPositionLock,
    CopySignalEvent,
    LeaderEligibilityDecision,
    LeaderMetricsDaily,
    LeaderProfile,
    RedemptionEvent,
    Trade,
)


DB_PATH = os.getenv("DB_PATH", "station_sniper.db")
DB_URL = f"sqlite:///{Path(DB_PATH).resolve()}"
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "")
METRICS_FILE = Path(os.getenv("METRICS_FILE", "metrics.json"))
RATE_LIMIT_PER_MIN = int(os.getenv("API_RATE_LIMIT_PER_MIN", "120"))

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
app = FastAPI(title="Station Sniper Monitoring API", version="1.0.0")
Base.metadata.create_all(engine)

_rate_window: dict[str, deque[float]] = defaultdict(deque)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _check_auth(request: Request) -> None:
    if not API_BEARER_TOKEN:
        return
    raw = request.headers.get("Authorization", "")
    token = raw.replace("Bearer ", "", 1).strip() if raw.startswith("Bearer ") else ""
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    dq = _rate_window[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="rate_limited")
    dq.append(now)


@app.middleware("http")
async def _auth_rate_middleware(request: Request, call_next):
    try:
        _check_auth(request)
        _check_rate_limit(request)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    return await call_next(request)


@app.get("/api/v1/health")
def health():
    return {
        "ok": True,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "db_path": str(Path(DB_PATH).resolve()),
    }


@app.get("/api/v1/summary")
def summary():
    metrics = {}
    if METRICS_FILE.exists():
        try:
            metrics = json.loads(METRICS_FILE.read_text())
        except Exception:
            metrics = {}

    session = SessionLocal()
    try:
        open_locks = session.query(CopyPositionLock).filter(CopyPositionLock.status == "OPEN").count()
        signals_accepted = session.query(CopySignalEvent).filter(CopySignalEvent.status == "accepted").count()
        signals_skipped = session.query(CopySignalEvent).filter(CopySignalEvent.status == "skipped").count()
        intents = session.query(CopyOrderIntent).count()
        redemptions = session.query(RedemptionEvent).count()
        resolved_copy_rows = (
            session.query(Trade)
            .filter(
                Trade.action == "BUY",
                Trade.market_type == "copycat",
                Trade.resolved_correct.isnot(None),
            )
            .all()
        )
        wins = sum(1 for t in resolved_copy_rows if t.resolved_correct)
        losses = len(resolved_copy_rows) - wins
        net_pnl = sum(float(t.pnl or 0.0) for t in resolved_copy_rows)
    finally:
        session.close()

    return {
        "metrics": metrics,
        "counts": {
            "open_locks": open_locks,
            "signals_accepted": signals_accepted,
            "signals_skipped": signals_skipped,
            "order_intents": intents,
            "redemption_events": redemptions,
        },
        "performance": {
            "resolved": len(resolved_copy_rows),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(resolved_copy_rows)) if resolved_copy_rows else None,
            "net_pnl": net_pnl,
        },
    }


@app.get("/api/v1/positions")
def positions():
    session = SessionLocal()
    try:
        rows = (
            session.query(CopyPositionLock)
            .filter(CopyPositionLock.status == "OPEN")
            .order_by(CopyPositionLock.opened_at.desc())
            .all()
        )
        return [
            {
                "match_key": r.match_key,
                "condition_id": r.condition_id,
                "event_slug": r.event_slug,
                "token_id": r.token_id,
                "side": r.side,
                "outcome": r.outcome,
                "opened_by_wallet": r.opened_by_wallet,
                "opened_at": _iso(r.opened_at),
                "updated_at": _iso(r.updated_at),
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/api/v1/trades")
def trades(window: str = Query(default="7d", pattern="^(1d|7d|30d)$")):
    day_map = {"1d": 1, "7d": 7, "30d": 30}
    since = datetime.utcnow() - timedelta(days=day_map[window])
    session = SessionLocal()
    try:
        rows = (
            session.query(CopyOrderIntent)
            .filter(CopyOrderIntent.created_at >= since)
            .order_by(CopyOrderIntent.created_at.desc())
            .limit(1000)
            .all()
        )
        return [
            {
                "id": r.id,
                "leader_wallet": r.leader_wallet,
                "match_key": r.match_key,
                "token_id": r.token_id,
                "status": r.status,
                "reason": r.reason,
                "size_usd": r.size_usd,
                "requested_price": r.requested_price,
                "selected_price": r.selected_price,
                "max_copy_price": r.max_copy_price,
                "order_id": r.order_id,
                "order_status": r.order_status,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/api/v1/leaders")
def leaders():
    session = SessionLocal()
    try:
        profiles = session.query(LeaderProfile).order_by(LeaderProfile.name.asc()).all()
        out = []
        for p in profiles:
            dec = (
                session.query(LeaderEligibilityDecision)
                .filter(LeaderEligibilityDecision.wallet == p.wallet)
                .order_by(LeaderEligibilityDecision.date.desc())
                .first()
            )
            met = (
                session.query(LeaderMetricsDaily)
                .filter(LeaderMetricsDaily.wallet == p.wallet)
                .order_by(LeaderMetricsDaily.date.desc())
                .first()
            )
            out.append(
                {
                    "wallet": p.wallet,
                    "name": p.name,
                    "tier": p.tier,
                    "base_status": p.base_status,
                    "risk_multiplier": p.risk_multiplier,
                    "enabled": p.enabled,
                    "eligibility": {
                        "date": dec.date if dec else None,
                        "eligible": dec.eligible if dec else None,
                        "status": dec.status if dec else None,
                        "reason": dec.reason if dec else None,
                    },
                    "metrics": {
                        "date": met.date if met else None,
                        "trades_60d": met.trades_60d if met else None,
                        "active_days_60d": met.active_days_60d if met else None,
                        "activity_truncated": met.activity_truncated if met else None,
                        "recent_active_ok": met.recent_active_ok if met else None,
                        "recent_success_ok": met.recent_success_ok if met else None,
                        "week_sports_pnl": met.week_sports_pnl if met else None,
                        "month_sports_pnl": met.month_sports_pnl if met else None,
                        "week_global_pnl": met.week_global_pnl if met else None,
                        "month_global_pnl": met.month_global_pnl if met else None,
                    },
                }
            )
        return out
    finally:
        session.close()


@app.get("/api/v1/copy/locks")
def copy_locks():
    session = SessionLocal()
    try:
        rows = (
            session.query(CopyPositionLock)
            .order_by(CopyPositionLock.updated_at.desc())
            .limit(500)
            .all()
        )
        return [
            {
                "match_key": r.match_key,
                "condition_id": r.condition_id,
                "event_slug": r.event_slug,
                "token_id": r.token_id,
                "status": r.status,
                "side": r.side,
                "outcome": r.outcome,
                "opened_by_wallet": r.opened_by_wallet,
                "opened_at": _iso(r.opened_at),
                "closed_at": _iso(r.closed_at),
                "updated_at": _iso(r.updated_at),
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/api/v1/copy/signals")
def copy_signals(status: str | None = Query(default=None)):
    session = SessionLocal()
    try:
        q = session.query(CopySignalEvent)
        if status:
            q = q.filter(CopySignalEvent.status == status)
        rows = q.order_by(CopySignalEvent.created_at.desc()).limit(1000).all()
        return [
            {
                "id": r.id,
                "leader_wallet": r.leader_wallet,
                "leader_name": r.leader_name,
                "transaction_hash": r.transaction_hash,
                "side": r.side,
                "outcome": r.outcome,
                "token_id": r.token_id,
                "condition_id": r.condition_id,
                "event_slug": r.event_slug,
                "match_key": r.match_key,
                "leader_price": r.leader_price,
                "leader_size": r.leader_size,
                "status": r.status,
                "reason": r.reason,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/api/v1/redemptions")
def redemptions():
    session = SessionLocal()
    try:
        rows = session.query(RedemptionEvent).order_by(RedemptionEvent.created_at.desc()).limit(500).all()
        return [
            {
                "id": r.id,
                "condition_id": r.condition_id,
                "title": r.title,
                "outcome": r.outcome,
                "size": r.size,
                "status": r.status,
                "tx_hash": r.tx_hash,
                "usdc_balance_after": r.usdc_balance_after,
                "error": r.error,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ]
    finally:
        session.close()
