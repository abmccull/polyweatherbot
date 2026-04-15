"""Sports copycat execution engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from config import Config
from copycat.leaders import SEED_LEADERS, SeedLeader
from copycat.polymarket_client import PolymarketDataClient
from copycat.rules import build_match_key, classify_duplicate_reason, compute_max_copy_price
from db.engine import get_session
from db.models import (
    CopyOrderIntent,
    CopyPositionLock,
    CopySignalEvent,
    LeaderEligibilityDecision,
    LeaderMetricsDaily,
    LeaderProfile,
    RedemptionEvent,
)
from trading.executor import TradeExecutor
from trading.redeemer import RedemptionResult, PositionRedeemer
from utils.logging import get_logger

log = get_logger("copycat")


@dataclass
class LeaderRuntime:
    wallet: str
    name: str
    tier: str
    status: str
    multiplier: float
    last_seen_ts: int | None = None


class CopycatEngine:
    """Polls curated wallets, enforces dedup/risk rules, and places copy trades."""

    def __init__(
        self,
        config: Config,
        executor: TradeExecutor,
        redeemer: PositionRedeemer,
    ) -> None:
        self._config = config
        self._executor = executor
        self._redeemer = redeemer
        self._client = PolymarketDataClient()
        self._leaders: dict[str, LeaderRuntime] = {}
        self._next_leader_refresh_at: datetime | None = None

        self._signals_seen = 0
        self._signals_accepted = 0
        self._signals_skipped = 0
        self._orders_placed = 0
        self._last_accept_ts_by_wallet: dict[str, int] = {}

    async def close(self) -> None:
        await self._client.close()

    async def startup(self) -> None:
        """Seed DB state and load active leader set."""
        self._upsert_seed_profiles()
        await self.refresh_leaders(force=True)
        await self.sync_locks_with_positions()

    async def refresh_leaders(self, force: bool = False) -> None:
        """Recompute 60-day eligibility and active/probation roster."""
        now = datetime.now(timezone.utc)
        if not force and self._next_leader_refresh_at and now < self._next_leader_refresh_at:
            return
        self._next_leader_refresh_at = now + timedelta(hours=max(1, self._config.leader_refresh_hours))

        self._upsert_seed_profiles()
        wallets = [s.wallet.lower() for s in SEED_LEADERS]
        today = now.date().isoformat()
        cutoff_ts = int((now - timedelta(days=max(1, self._config.copy_activity_window_days))).timestamp())

        week_sports = await self._load_leaderboard_map(period="WEEK", category="SPORTS")
        month_sports = await self._load_leaderboard_map(period="MONTH", category="SPORTS")
        week_global = await self._load_leaderboard_map(period="WEEK", category=None)
        month_global = await self._load_leaderboard_map(period="MONTH", category=None)

        session = get_session()
        try:
            for seed in SEED_LEADERS:
                wallet = seed.wallet.lower()
                trades_60d, active_days, truncated = await self._activity_stats_60d(wallet, cutoff_ts)
                ws = week_sports.get(wallet)
                ms = month_sports.get(wallet)
                wg = week_global.get(wallet)
                mg = month_global.get(wallet)

                recent_active_ok = (
                    (trades_60d >= self._config.copy_min_trades_60d and active_days >= self._config.copy_min_active_days_60d)
                    or truncated
                )
                recent_success_ok = any(
                    (x is not None and x > 0.0) for x in (ms, ws, mg, wg)
                )

                if seed.base_status == "excluded":
                    eligible = False
                    status = "excluded"
                    reason = "seed_excluded"
                elif recent_active_ok and recent_success_ok:
                    status = seed.base_status
                    eligible = True
                    reason = "ok"
                else:
                    status = "excluded"
                    eligible = False
                    reason = "recent_gate_failed"

                lm = (
                    session.query(LeaderMetricsDaily)
                    .filter(LeaderMetricsDaily.wallet == wallet, LeaderMetricsDaily.date == today)
                    .one_or_none()
                )
                if lm is None:
                    lm = LeaderMetricsDaily(wallet=wallet, date=today)
                    session.add(lm)
                lm.trades_60d = trades_60d
                lm.active_days_60d = active_days
                lm.activity_truncated = truncated
                lm.recent_active_ok = recent_active_ok
                lm.week_sports_pnl = ws
                lm.month_sports_pnl = ms
                lm.week_global_pnl = wg
                lm.month_global_pnl = mg
                lm.recent_success_ok = recent_success_ok

                dec = (
                    session.query(LeaderEligibilityDecision)
                    .filter(LeaderEligibilityDecision.wallet == wallet, LeaderEligibilityDecision.date == today)
                    .one_or_none()
                )
                if dec is None:
                    dec = LeaderEligibilityDecision(wallet=wallet, date=today)
                    session.add(dec)
                dec.status = status
                dec.eligible = eligible
                dec.reason = reason

            session.commit()
        except Exception as e:
            session.rollback()
            log.error("leader_refresh_failed", error=str(e))
        finally:
            session.close()

        self._load_runtime_leaders()
        log.info("leader_refresh_complete", active_leaders=len(self._leaders))

    async def poll_once(self) -> None:
        """Poll each eligible wallet once and process any new trades."""
        if not self._config.copy_enabled:
            return
        if not self._leaders:
            await self.refresh_leaders(force=True)
            if not self._leaders:
                return

        leaders = list(self._leaders.values())
        if not leaders:
            return
        if self._config.copy_core_poll_priority:
            leaders.sort(
                key=lambda l: (
                    0 if (l.status or "").lower() == "core" else 1,
                    (l.name or "").lower(),
                )
            )
        per_wallet_delay = max(1.0, float(self._config.copy_poll_seconds) / float(len(leaders)))
        for idx, leader in enumerate(leaders):
            detected = await self._poll_wallet(leader)
            if detected > 0 and self._config.copy_followup_poll_seconds > 0:
                await asyncio.sleep(self._config.copy_followup_poll_seconds)
                await self._poll_wallet(leader)
            if idx < len(leaders) - 1:
                await asyncio.sleep(per_wallet_delay)

    async def sync_locks_with_positions(self) -> int:
        """Close match locks when position no longer exists on-chain."""
        funder = self._config.poly_funder_address
        if not funder:
            return 0
        try:
            positions = await self._client.fetch_positions(funder)
        except Exception as e:
            log.warning("copy_lock_sync_failed_fetch_positions", error=str(e))
            return 0

        open_conditions = {
            (p.get("conditionId") or "").lower()
            for p in positions
            if float(p.get("size", 0) or 0) > 0 and p.get("conditionId")
        }

        session = get_session()
        closed = 0
        try:
            rows = session.query(CopyPositionLock).filter(CopyPositionLock.status == "OPEN").all()
            for row in rows:
                cid = (row.condition_id or "").lower()
                if cid and cid not in open_conditions:
                    row.status = "CLOSED"
                    row.closed_at = datetime.utcnow()
                    closed += 1
            if closed:
                session.commit()
        except Exception as e:
            session.rollback()
            log.warning("copy_lock_sync_failed", error=str(e))
        finally:
            session.close()
        return closed

    def record_redemption_cycle(self, result: RedemptionResult) -> None:
        """Persist redemption attempts and log stuck-redeemable alerts."""
        if not result.details:
            return
        session = get_session()
        try:
            for item in result.details:
                evt = RedemptionEvent(
                    condition_id=item.get("condition_id", ""),
                    title=item.get("title"),
                    outcome=item.get("outcome"),
                    size=float(item.get("size", 0.0) or 0.0),
                    status=item.get("status", "FAILED"),
                    tx_hash=item.get("tx_hash"),
                    usdc_balance_after=result.usdc_balance,
                    error=item.get("error"),
                )
                session.add(evt)
            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("redemption_event_persist_failed", error=str(e))
        finally:
            session.close()

    def get_metrics(self) -> dict:
        """Return compact runtime + DB metrics for observability."""
        session = get_session()
        try:
            open_locks = session.query(func.count(CopyPositionLock.id)).filter(
                CopyPositionLock.status == "OPEN"
            ).scalar() or 0
            accepted = session.query(func.count(CopySignalEvent.id)).filter(
                CopySignalEvent.status == "accepted"
            ).scalar() or 0
            skipped = session.query(func.count(CopySignalEvent.id)).filter(
                CopySignalEvent.status == "skipped"
            ).scalar() or 0
            intents = session.query(func.count(CopyOrderIntent.id)).scalar() or 0
            leaders = len(self._leaders)
        finally:
            session.close()

        return {
            "signals_seen": self._signals_seen,
            "signals_accepted": self._signals_accepted,
            "signals_skipped": self._signals_skipped,
            "orders_placed": self._orders_placed,
            "db_signals_accepted": accepted,
            "db_signals_skipped": skipped,
            "db_order_intents": intents,
            "open_match_locks": open_locks,
            "active_leaders": leaders,
        }

    async def _poll_wallet(self, leader: LeaderRuntime) -> int:
        try:
            trades = await self._client.fetch_trades(
                leader.wallet,
                limit=max(20, min(500, self._config.copy_max_wallet_trades_fetch)),
                offset=0,
            )
        except Exception as e:
            log.warning("copy_wallet_poll_failed", leader=leader.name, wallet=leader.wallet, error=str(e))
            return 0

        if not trades:
            return 0

        now_ts = int(datetime.now(timezone.utc).timestamp())
        max_age = max(1, int(self._config.copy_trade_max_age_seconds))
        fresh_cutoff = now_ts - max_age

        normalized: list[tuple[int, dict]] = []
        for t in trades:
            ts = self._to_trade_ts_seconds(t.get("timestamp", 0))
            if ts <= 0:
                continue
            normalized.append((ts, t))
        if not normalized:
            return 0

        normalized.sort(key=lambda x: x[0])
        max_ts = normalized[-1][0]
        if leader.last_seen_ts is None:
            # Warm start: jump watermark to the recent window to avoid replaying stale history.
            leader.last_seen_ts = max(max_ts, fresh_cutoff)
            return 0

        new_candidates = [(ts, t) for ts, t in normalized if ts > leader.last_seen_ts]
        if not new_candidates:
            return 0
        leader.last_seen_ts = max_ts

        fresh_new = [t for ts, t in new_candidates if ts >= fresh_cutoff]
        stale_filtered = len(new_candidates) - len(fresh_new)
        if stale_filtered > 0:
            log.info(
                "copy_stale_trades_filtered",
                leader=leader.name,
                wallet=leader.wallet,
                stale=stale_filtered,
                processed=len(fresh_new),
                max_age_seconds=max_age,
            )
        max_new = max(1, int(self._config.copy_max_new_trades_per_poll))
        if len(fresh_new) > max_new:
            dropped = len(fresh_new) - max_new
            fresh_new = fresh_new[-max_new:]
            log.info(
                "copy_wallet_new_trades_capped",
                leader=leader.name,
                wallet=leader.wallet,
                dropped=dropped,
                processed=max_new,
            )

        detected = 0
        for t in fresh_new:
            detected += 1
            await self._process_trade(leader, t)
        return detected

    async def _process_trade(self, leader: LeaderRuntime, trade: dict) -> None:
        self._signals_seen += 1
        tx_hash = (trade.get("transactionHash") or "").lower()
        side = (trade.get("side") or "").upper()
        outcome = trade.get("outcome")
        token_id = str(trade.get("asset") or "").strip()
        condition_id = (trade.get("conditionId") or "").lower() or None
        event_slug = (trade.get("eventSlug") or "").strip().lower() or None
        event_title = trade.get("title")
        leader_price = float(trade.get("price", 0) or 0)
        leader_size = float(trade.get("size", 0) or 0)
        trade_ts = self._to_trade_ts_seconds(trade.get("timestamp", 0))

        if not tx_hash or not token_id:
            self._record_signal(
                leader=leader,
                tx_hash=tx_hash or f"missing-{datetime.utcnow().timestamp()}",
                side=side or "UNKNOWN",
                outcome=outcome,
                token_id=token_id or "missing",
                condition_id=condition_id,
                event_slug=event_slug,
                leader_price=leader_price,
                leader_size=leader_size,
                event_title=event_title,
                event_end=None,
                status="skipped",
                reason="invalid_trade_payload",
            )
            self._signals_skipped += 1
            return

        if self._signal_exists(tx_hash):
            return
        if self._config.copy_follow_buy_only and side != "BUY":
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, None, "skipped", "side_not_buy",
            )
            self._signals_skipped += 1
            return

        event = await self._client.fetch_event_by_slug(event_slug) if event_slug else None
        if event is None:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, None, "skipped", "missing_event_metadata",
            )
            self._signals_skipped += 1
            return
        event_market = self._find_event_market(event, condition_id)
        if self._config.copy_skip_inactive_events and not self._is_event_market_active(event, event_market):
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, None, "skipped", "inactive_or_closed_event",
            )
            self._signals_skipped += 1
            return

        if not self._is_sports_event(event):
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, None, "skipped", "not_sports",
            )
            self._signals_skipped += 1
            return

        price_floor = min(self._config.copy_min_leader_price, self._config.copy_max_leader_price)
        price_ceiling = max(self._config.copy_min_leader_price, self._config.copy_max_leader_price)
        if leader_price < price_floor or leader_price > price_ceiling:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, None, "skipped", "leader_price_out_of_band",
            )
            self._signals_skipped += 1
            return

        end_dt = self._parse_iso_datetime(
            (event_market or {}).get("endDate") or event.get("endDate")
        )
        if end_dt is None:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, None, "skipped", "missing_end_date",
            )
            self._signals_skipped += 1
            return

        hours_to_settle = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0
        if hours_to_settle <= 0:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", "already_settled",
            )
            self._signals_skipped += 1
            return
        if trade_ts > 0:
            settle_cutoff = int(end_dt.timestamp()) + max(0, int(self._config.copy_settle_grace_seconds))
            if trade_ts > settle_cutoff:
                self._record_signal(
                    leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                    leader_price, leader_size, event_title, end_dt, "skipped", "post_settlement_trade",
                )
                self._signals_skipped += 1
                return
        if hours_to_settle > self._config.copy_settle_max_hours:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", "settle_horizon_exceeded",
            )
            self._signals_skipped += 1
            return

        match_key = build_match_key(condition_id, event_slug, token_id)
        existing_lock = self._get_open_lock(match_key)
        if existing_lock is not None and self._config.copy_dedup_enabled:
            reason = classify_duplicate_reason(existing_lock.side, side)
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", reason, match_key=match_key,
            )
            self._signals_skipped += 1
            return

        max_copy_price = compute_max_copy_price(
            leader_price,
            self._config.copy_slippage_abs_cap,
            self._config.copy_slippage_rel_cap,
        )
        if max_copy_price <= 0:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", "invalid_leader_price", match_key=match_key,
            )
            self._signals_skipped += 1
            return

        best_bid, best_ask, _, _ = self._executor.refresh_prices(token_id)
        if best_ask is None:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", "no_orderbook_ask", match_key=match_key,
            )
            self._signals_skipped += 1
            return
        if float(best_ask) > max_copy_price:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", "slippage_cap_exceeded", match_key=match_key,
            )
            self._signals_skipped += 1
            return
        if (
            leader.status == "probation"
            and self._config.copy_probation_min_seconds_between_entries > 0
        ):
            last_accept = self._last_accept_ts_by_wallet.get(leader.wallet, 0)
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if last_accept > 0 and (now_ts - last_accept) < self._config.copy_probation_min_seconds_between_entries:
                self._record_signal(
                    leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                    leader_price, leader_size, event_title, end_dt, "skipped", "probation_wallet_cooldown",
                    match_key=match_key,
                )
                self._signals_skipped += 1
                return

        size_usd = self._compute_ticket_size(leader)
        allowed, reason = await self._check_risk(match_key, size_usd, leader)
        if not allowed:
            self._record_signal(
                leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
                leader_price, leader_size, event_title, end_dt, "skipped", reason, match_key=match_key,
            )
            self._signals_skipped += 1
            return

        signal_id = self._record_signal(
            leader, tx_hash, side, outcome, token_id, condition_id, event_slug,
            leader_price, leader_size, event_title, end_dt, "accepted", "ok", match_key=match_key,
        )
        self._signals_accepted += 1
        self._last_accept_ts_by_wallet[leader.wallet] = int(datetime.now(timezone.utc).timestamp())

        intent_id = self._create_intent(
            signal_id=signal_id,
            leader_wallet=leader.wallet,
            match_key=match_key,
            condition_id=condition_id,
            event_slug=event_slug,
            token_id=token_id,
            side="BUY",
            outcome=outcome,
            requested_price=float(best_ask),
            max_copy_price=max_copy_price,
            size_usd=size_usd,
            status="accepted",
            reason="ok",
        )

        market_date = end_dt.date().isoformat()
        exec_trade = await self._executor.execute_copy_order(
            token_id=token_id,
            condition_id=condition_id,
            event_id=event_slug or condition_id or token_id,
            event_slug=event_slug,
            event_title=event_title,
            market_date=market_date,
            leader_wallet=leader.wallet,
            match_key=match_key,
            outcome=outcome,
            leader_price=leader_price,
            max_copy_price=max_copy_price,
            size_usd=size_usd,
        )

        if exec_trade is None:
            self._update_intent(intent_id, status="failed", reason="execution_failed")
            return

        order_status = (exec_trade.order_status or "").upper()
        if order_status == "DRY_RUN":
            status = "dry_run"
        elif order_status in ("PLACED", "LIVE", "OPEN", "MATCHED", "FILLED", "UNKNOWN"):
            status = "placed"
        else:
            status = "failed"
        self._update_intent(
            intent_id,
            order_id=exec_trade.order_id,
            order_status=exec_trade.order_status,
            selected_price=exec_trade.price,
            shares=exec_trade.requested_size or exec_trade.size,
            status=status,
            reason="ok" if status != "failed" else "order_status_failed",
        )
        if status != "failed":
            self._orders_placed += 1
            self._ensure_open_lock(
                match_key=match_key,
                condition_id=condition_id,
                event_slug=event_slug,
                token_id=token_id,
                side="BUY",
                outcome=outcome,
                opened_by_wallet=leader.wallet,
                last_signal_tx=tx_hash,
            )

    async def _check_risk(
        self,
        match_key: str,
        size_usd: float,
        leader: LeaderRuntime,
    ) -> tuple[bool, str]:
        session = get_session()
        probation_open_risk = 0.0
        probation_open_locks = 0
        try:
            match_exposure = session.query(func.sum(CopyOrderIntent.size_usd)).filter(
                CopyOrderIntent.match_key == match_key,
                CopyOrderIntent.status.in_(("accepted", "placed", "dry_run")),
            ).scalar() or 0.0
            if match_exposure + size_usd > self._config.copy_max_match_exposure_usd:
                return False, "match_exposure_cap"

            open_locks = session.query(
                CopyPositionLock.match_key,
                CopyPositionLock.opened_by_wallet,
            ).filter(
                CopyPositionLock.status == "OPEN"
            ).all()
            open_match_keys = [k[0] for k in open_locks]
            open_risk = 0.0
            if open_match_keys:
                open_risk = session.query(func.sum(CopyOrderIntent.size_usd)).filter(
                    CopyOrderIntent.match_key.in_(open_match_keys),
                    CopyOrderIntent.status.in_(("accepted", "placed", "dry_run")),
                ).scalar() or 0.0
            probation_wallets = {
                wallet for wallet, rt in self._leaders.items()
                if (rt.status or "").lower() == "probation"
            }
            probation_match_keys = [
                mk for mk, opened_by in open_locks
                if (opened_by or "").lower() in probation_wallets
            ]
            probation_open_locks = len(probation_match_keys)
            if probation_match_keys:
                probation_open_risk = session.query(func.sum(CopyOrderIntent.size_usd)).filter(
                    CopyOrderIntent.match_key.in_(probation_match_keys),
                    CopyOrderIntent.status.in_(("accepted", "placed", "dry_run")),
                ).scalar() or 0.0
        finally:
            session.close()

        balance = await self._redeemer.get_balance()
        tradable_cash = balance if balance is not None else self._config.initial_bankroll
        # Use an equity-like basis to avoid deadlocking when capital is currently deployed.
        bankroll = max(self._config.initial_bankroll, tradable_cash + open_risk)
        needs_matic_reserve = await self._redeemer.needs_native_gas_reserve()
        if needs_matic_reserve:
            native_balance = await self._redeemer.get_native_balance()
            if native_balance is None:
                return False, "matic_balance_unavailable"
            if native_balance < self._config.copy_min_matic_reserve:
                return False, "matic_reserve_low"
        max_open = bankroll * self._config.copy_max_open_risk_pct
        if open_risk + size_usd > max_open:
            return False, "open_risk_cap"
        if bankroll - open_risk - size_usd < self._config.copy_min_cash_buffer_usd:
            return False, "cash_buffer_cap"
        if (leader.status or "").lower() == "probation":
            max_probation_open = max_open * min(
                1.0,
                max(0.0, self._config.copy_probation_max_open_risk_share),
            )
            if probation_open_risk + size_usd > max_probation_open:
                return False, "probation_risk_share_cap"
            if (
                self._config.copy_probation_max_open_locks > 0
                and probation_open_locks >= self._config.copy_probation_max_open_locks
            ):
                return False, "probation_lock_cap"
        return True, "ok"

    def _compute_ticket_size(self, leader: LeaderRuntime) -> float:
        base = max(0.0, self._config.copy_base_ticket_usd)
        status_mult = self._config.copy_probation_multiplier if leader.status == "probation" else 1.0
        leader_mult = leader.multiplier if leader.multiplier and leader.multiplier > 0 else 1.0
        if (
            leader.wallet == "0xe90bec87d9ef430f27f9dcfe72c34b76967d5da2"
            and abs(leader_mult - 1.0) < 1e-9
        ):
            # Backward compatibility for env-only gmanas multiplier overrides.
            leader_mult = self._config.copy_gmanas_multiplier
        mult = status_mult * leader_mult
        return round(base * max(0.0, mult), 2)

    def _load_runtime_leaders(self) -> None:
        session = get_session()
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            rows = (
                session.query(LeaderProfile, LeaderEligibilityDecision)
                .join(
                    LeaderEligibilityDecision,
                    (LeaderEligibilityDecision.wallet == LeaderProfile.wallet)
                    & (LeaderEligibilityDecision.date == today),
                )
                .filter(LeaderEligibilityDecision.eligible.is_(True))
                .all()
            )
            leaders: dict[str, LeaderRuntime] = {}
            for prof, dec in rows:
                wallet = (prof.wallet or "").lower()
                if not wallet:
                    continue
                leaders[wallet] = LeaderRuntime(
                    wallet=wallet,
                    name=prof.name,
                    tier=prof.tier or "",
                    status=dec.status,
                    multiplier=prof.risk_multiplier or 1.0,
                )
            self._leaders = leaders
        finally:
            session.close()

    def _upsert_seed_profiles(self) -> None:
        session = get_session()
        try:
            for seed in SEED_LEADERS:
                wallet = seed.wallet.lower()
                row = session.query(LeaderProfile).filter(LeaderProfile.wallet == wallet).one_or_none()
                if row is None:
                    row = LeaderProfile(
                        wallet=wallet,
                        name=seed.name,
                        tier=seed.tier,
                        base_status=seed.base_status,
                        risk_multiplier=seed.risk_multiplier,
                        enabled=True,
                    )
                    session.add(row)
                else:
                    row.name = seed.name
                    row.tier = seed.tier
                    row.base_status = seed.base_status
                    row.risk_multiplier = seed.risk_multiplier
                    row.enabled = True
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def _activity_stats_60d(self, wallet: str, cutoff_ts: int) -> tuple[int, int, bool]:
        """Fetch recent trades stats with pagination caps."""
        trades = []
        oldest_ts: int | None = None
        truncated = False
        for offset in range(0, 3001, 500):
            batch = await self._client.fetch_trades(wallet, limit=500, offset=offset)
            if not batch:
                break
            for t in batch:
                ts = self._to_trade_ts_seconds(t.get("timestamp", 0))
                if ts <= 0:
                    continue
                oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
                if ts >= cutoff_ts:
                    trades.append(t)
            if len(batch) < 500:
                break
            if oldest_ts is not None and oldest_ts < cutoff_ts:
                break
            if offset == 3000:
                truncated = True
        days = {
            datetime.utcfromtimestamp(self._to_trade_ts_seconds(t.get("timestamp", 0))).strftime("%Y-%m-%d")
            for t in trades
            if self._to_trade_ts_seconds(t.get("timestamp", 0)) > 0
        }
        return len(trades), len(days), truncated

    @staticmethod
    def _to_trade_ts_seconds(raw_ts: object) -> int:
        """Normalize Polymarket trade timestamps to unix seconds (supports sec/ms)."""
        try:
            ts = int(float(raw_ts or 0))
        except (TypeError, ValueError):
            return 0
        # Data API may return milliseconds.
        if ts > 10_000_000_000:
            ts //= 1000
        return ts

    async def _load_leaderboard_map(self, period: str, category: str | None) -> dict[str, float]:
        wallets = {s.wallet.lower() for s in SEED_LEADERS}
        found: dict[str, float] = {}
        for offset in range(0, 5001, 500):
            try:
                rows = await self._client.fetch_leaderboard(
                    period,
                    category=category,
                    limit=500,
                    offset=offset,
                )
            except Exception:
                break
            if not rows:
                break
            for row in rows:
                wallet = (row.get("proxyWallet") or "").lower()
                if wallet in wallets:
                    found[wallet] = float(row.get("pnl", 0) or 0)
            if len(rows) < 500:
                break
        return found

    @staticmethod
    def _parse_iso_datetime(raw: str | None) -> datetime | None:
        if not raw:
            return None
        val = raw.strip()
        if not val:
            return None
        if val.endswith("Z"):
            val = val.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _is_sports_event(event: dict) -> bool:
        tags = event.get("tags") or []
        labels = []
        for t in tags:
            if isinstance(t, dict):
                labels.extend([str(t.get("label") or ""), str(t.get("slug") or "")])
            else:
                labels.append(str(t))
        joined = " ".join(labels).lower()
        if "sport" in joined:
            return True
        title = str(event.get("title") or "").lower()
        # Conservative fallback when tags are sparse.
        keywords = ("nba", "nfl", "mlb", "nhl", "soccer", "premier league", "uefa", "tennis")
        return any(k in title for k in keywords)

    @staticmethod
    def _find_event_market(event: dict, condition_id: str | None) -> dict | None:
        markets = event.get("markets") or []
        if not isinstance(markets, list) or not markets:
            return None
        target = (condition_id or "").lower()
        if target:
            for market in markets:
                if not isinstance(market, dict):
                    continue
                if (market.get("conditionId") or "").lower() == target:
                    return market
        for market in markets:
            if isinstance(market, dict):
                return market
        return None

    @staticmethod
    def _is_event_market_active(event: dict, market: dict | None) -> bool:
        for row in (event, market or {}):
            if not isinstance(row, dict):
                continue
            if row.get("archived") is True:
                return False
            if row.get("closed") is True:
                return False
            if row.get("active") is False:
                return False
        return True

    def _signal_exists(self, tx_hash: str) -> bool:
        session = get_session()
        try:
            row = (
                session.query(CopySignalEvent.id)
                .filter(CopySignalEvent.transaction_hash == tx_hash)
                .first()
            )
            return row is not None
        finally:
            session.close()

    def _record_signal(
        self,
        leader: LeaderRuntime,
        tx_hash: str,
        side: str,
        outcome: str | None,
        token_id: str,
        condition_id: str | None,
        event_slug: str | None,
        leader_price: float,
        leader_size: float,
        event_title: str | None,
        event_end: datetime | None,
        status: str,
        reason: str,
        match_key: str | None = None,
    ) -> int | None:
        session = get_session()
        try:
            row = CopySignalEvent(
                leader_wallet=leader.wallet,
                leader_name=leader.name,
                transaction_hash=tx_hash,
                side=side,
                outcome=outcome,
                token_id=token_id,
                condition_id=condition_id,
                event_slug=event_slug,
                match_key=match_key or build_match_key(condition_id, event_slug, token_id),
                leader_price=leader_price,
                leader_size=leader_size,
                event_title=event_title,
                event_end=event_end.replace(tzinfo=None) if event_end else None,
                status=status,
                reason=reason,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        except Exception as e:
            session.rollback()
            log.warning("copy_signal_persist_failed", error=str(e), tx_hash=tx_hash)
            return None
        finally:
            session.close()

    def _create_intent(
        self,
        *,
        signal_id: int | None,
        leader_wallet: str,
        match_key: str,
        condition_id: str | None,
        event_slug: str | None,
        token_id: str,
        side: str,
        outcome: str | None,
        requested_price: float,
        max_copy_price: float,
        size_usd: float,
        status: str,
        reason: str,
    ) -> int:
        session = get_session()
        try:
            row = CopyOrderIntent(
                signal_event_id=signal_id,
                leader_wallet=leader_wallet,
                match_key=match_key,
                condition_id=condition_id,
                event_slug=event_slug,
                token_id=token_id,
                side=side,
                outcome=outcome,
                requested_price=requested_price,
                max_copy_price=max_copy_price,
                size_usd=size_usd,
                status=status,
                reason=reason,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        finally:
            session.close()

    def _update_intent(
        self,
        intent_id: int,
        *,
        order_id: str | None = None,
        order_status: str | None = None,
        selected_price: float | None = None,
        shares: float | None = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
        session = get_session()
        try:
            row = session.query(CopyOrderIntent).filter(CopyOrderIntent.id == intent_id).one_or_none()
            if row is None:
                return
            if order_id is not None:
                row.order_id = order_id
            if order_status is not None:
                row.order_status = order_status
            if selected_price is not None:
                row.selected_price = selected_price
            if shares is not None:
                row.shares = shares
            if status is not None:
                row.status = status
            if reason is not None:
                row.reason = reason
            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("copy_intent_update_failed", error=str(e), intent_id=intent_id)
        finally:
            session.close()

    def _get_open_lock(self, match_key: str) -> CopyPositionLock | None:
        session = get_session()
        try:
            return (
                session.query(CopyPositionLock)
                .filter(CopyPositionLock.match_key == match_key, CopyPositionLock.status == "OPEN")
                .one_or_none()
            )
        finally:
            session.close()

    def _ensure_open_lock(
        self,
        *,
        match_key: str,
        condition_id: str | None,
        event_slug: str | None,
        token_id: str,
        side: str,
        outcome: str | None,
        opened_by_wallet: str,
        last_signal_tx: str,
    ) -> None:
        session = get_session()
        try:
            row = (
                session.query(CopyPositionLock)
                .filter(CopyPositionLock.match_key == match_key)
                .one_or_none()
            )
            if row is None:
                row = CopyPositionLock(
                    match_key=match_key,
                    condition_id=condition_id,
                    event_slug=event_slug,
                    token_id=token_id,
                    side=side,
                    outcome=outcome,
                    status="OPEN",
                    opened_by_wallet=opened_by_wallet,
                    last_signal_tx=last_signal_tx,
                )
                session.add(row)
            else:
                row.status = "OPEN"
                row.condition_id = condition_id or row.condition_id
                row.event_slug = event_slug or row.event_slug
                row.token_id = token_id
                row.side = side
                row.outcome = outcome
                row.last_signal_tx = last_signal_tx
            session.commit()
        except Exception as e:
            session.rollback()
            log.warning("copy_lock_upsert_failed", error=str(e), match_key=match_key)
        finally:
            session.close()
