"""Resolve trades, compute stats, log outcomes."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, date, timedelta

from sqlalchemy import func, and_

from db.engine import get_session
from db.models import Trade, DailyPnL
from settlement.noaa import NOAAClient
from settlement.open_meteo import OpenMeteoClient
from utils.logging import get_logger
from weather.station_map import lookup_city, lookup_ghcnd
from weather.temperature import temp_hits_bucket, PreciseTemp, Precision

log = get_logger("tracker")


@dataclass
class TradeStats:
    """Rolling statistics for trade performance."""

    total_trades: int = 0
    resolved_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_roi: float = 0.0


class TradeTracker:
    """Tracks trade outcomes and computes performance statistics."""

    def __init__(
        self,
        noaa_client: NOAAClient | None = None,
        open_meteo: OpenMeteoClient | None = None,
    ) -> None:
        self._noaa = noaa_client
        self._open_meteo = open_meteo

    async def resolve_trades(self) -> int:
        """Check for unresolved trades and attempt to resolve them.

        Returns number of newly resolved trades.
        """
        session = get_session()
        resolved_count = 0

        try:
            # Find unresolved BUY trades (SELL trades are inert execution records)
            unresolved = session.query(Trade).filter(
                Trade.action != "SELL",
                Trade.resolved_correct.is_(None),
                Trade.order_status != "FAILED",
            ).all()

            if not unresolved:
                return 0

            log.info("resolving_trades", count=len(unresolved))

            for trade in unresolved:
                try:
                    if trade.market_type == "precipitation":
                        resolved = await self._resolve_precip(trade)
                    else:
                        resolved = await self._resolve_single(trade)
                    if resolved:
                        resolved_count += 1
                except Exception as e:
                    log.warning("resolve_error", trade_id=trade.id, error=str(e))

            session.commit()
        except Exception as e:
            session.rollback()
            log.error("resolve_batch_error", error=str(e))
        finally:
            session.close()

        if resolved_count:
            log.info("trades_resolved", count=resolved_count)
            self._update_daily_pnl()

        return resolved_count

    async def _resolve_single(self, trade: Trade) -> bool:
        """Attempt to resolve a single temperature trade via NOAA or Open-Meteo."""
        station = lookup_city(trade.city)
        if station is None:
            return False

        market_date = date.fromisoformat(trade.market_date)

        # Only try to resolve if market date has passed
        now = datetime.utcnow().date()
        if market_date >= now:
            return False

        # Try NOAA first (US cities with GHCND IDs)
        if station.ghcnd_id and self._noaa:
            tmax_c = await self._noaa.get_daily_tmax(station.ghcnd_id, market_date)
            if tmax_c is not None:
                return self._apply_temp_resolution(trade, tmax_c, source="noaa")

        # Fallback: Open-Meteo (international cities)
        if station.latitude is not None and self._open_meteo:
            tmax_c = await self._open_meteo.get_daily_tmax(
                station.latitude, station.longitude, market_date,
            )
            if tmax_c is not None:
                return self._apply_temp_resolution(trade, tmax_c, source="open_meteo")

        return False

    @staticmethod
    def _get_sell_summary(trade: Trade, session) -> tuple[float, float, float]:
        """Get sell summary for the token of a BUY trade.

        Returns:
            (sold_shares, sell_proceeds, sell_fees) for SELL trades
            matching this BUY trade's token_id.
        """
        sells = session.query(Trade).filter(
            Trade.action == "SELL",
            Trade.token_id == trade.token_id,
        ).all()
        sold_shares = sum(s.size for s in sells)
        sell_proceeds = sum(s.cost for s in sells)  # cost stores proceeds for SELL trades
        sell_fees = sum(s.fee_paid or 0.0 for s in sells)
        return sold_shares, sell_proceeds, sell_fees

    def _apply_temp_resolution(self, trade: Trade, tmax_c: float, source: str = "") -> bool:
        """Apply temperature resolution to a BUY trade given the actual daily max.

        Accounts for partial exits: P&L is computed on remaining shares plus
        net profit from any SELL trades.
        """
        unit = trade.bucket_unit or "C"
        resolved_temp = PreciseTemp(celsius=tmax_c, precision=Precision.WHOLE)
        match = temp_hits_bucket(resolved_temp, trade.bucket_type, trade.bucket_value, unit=unit)

        trade.resolution_value = tmax_c
        trade.resolved_correct = match.hit
        trade.resolved_at = datetime.utcnow()

        fill = trade.fill_price or trade.price

        # Get sell summary to account for partial exits
        session = get_session()
        try:
            sold_shares, sell_proceeds, sell_fees = self._get_sell_summary(trade, session)
        finally:
            session.close()

        remaining_shares = trade.size - sold_shares

        if remaining_shares <= 0.01:
            # Fully exited via sells — P&L is sell proceeds minus total cost minus fees
            trade.pnl = sell_proceeds - sell_fees - trade.cost
        else:
            # Compute P&L on remaining (unsold) shares
            if match.hit:
                remaining_pnl = (1.0 - fill) * remaining_shares
            else:
                remaining_pnl = -fill * remaining_shares

            # Add net profit from sold shares
            sell_net = sell_proceeds - sell_fees - (fill * sold_shares)
            trade.pnl = remaining_pnl + sell_net

        trade.fee_paid = sell_fees  # total fees attributed to this position

        log.info(
            "trade_resolved",
            trade_id=trade.id,
            city=trade.city,
            bucket=f"{trade.bucket_type}:{trade.bucket_value}",
            tmax_c=tmax_c,
            source=source,
            correct=match.hit,
            pnl=round(trade.pnl, 2),
            sold_shares=round(sold_shares, 2),
            remaining=round(remaining_shares, 2),
        )
        return True

    async def _resolve_precip(self, trade: Trade) -> bool:
        """Resolve a precipitation trade using NOAA monthly data."""
        if self._noaa is None:
            return False

        # Parse market_date to get year/month (stored as "YYYY-MM-DD")
        try:
            market_date = date.fromisoformat(trade.market_date)
        except ValueError:
            return False
        year, month = market_date.year, market_date.month

        # Only resolve after month has fully ended + 3-day grace period
        now = datetime.utcnow().date()
        last_day = calendar.monthrange(year, month)[1]
        month_end = date(year, month, last_day)
        grace_deadline = month_end + timedelta(days=3)
        if now <= grace_deadline:
            return False

        # Look up GHCND station
        ghcnd_id = lookup_ghcnd(trade.city)
        if ghcnd_id is None:
            log.debug("no_ghcnd_for_city", city=trade.city)
            return False

        # Fetch NOAA data
        result = await self._noaa.get_monthly_precip(ghcnd_id, year, month)
        if result is None:
            return False

        # Require >= 25 days of data for reliability
        if result.days_with_data < 25:
            log.debug(
                "insufficient_noaa_days",
                station=ghcnd_id,
                days=result.days_with_data,
            )
            return False

        # Check bucket hit
        hit = self._precip_bucket_hit(
            trade.bucket_type, trade.bucket_low_inches,
            trade.bucket_high_inches, result.total_inches,
        )

        trade.resolution_value = round(result.total_inches, 3)
        trade.resolved_correct = hit
        trade.resolved_at = datetime.utcnow()

        # Calculate P&L (same logic as temperature trades)
        if trade.order_status == "DRY_RUN":
            if hit:
                trade.pnl = (1.0 - trade.price) * trade.size
            else:
                trade.pnl = -trade.price * trade.size
        else:
            fill = trade.fill_price or trade.price
            if hit:
                trade.pnl = (1.0 - fill) * trade.size
            else:
                trade.pnl = -fill * trade.size

        log.info(
            "precip_trade_resolved",
            trade_id=trade.id,
            city=trade.city,
            noaa_inches=round(result.total_inches, 3),
            bucket=f"{trade.bucket_type}:{trade.bucket_low_inches}-{trade.bucket_high_inches}",
            correct=hit,
            pnl=round(trade.pnl, 2),
        )
        return True

    @staticmethod
    def _precip_bucket_hit(
        bucket_type: str,
        low_inches: float | None,
        high_inches: float | None,
        actual_inches: float,
    ) -> bool:
        """Check whether actual precipitation falls within the bucket."""
        if bucket_type == "lt":
            return actual_inches < (high_inches or 0.0)
        elif bucket_type == "gt":
            return actual_inches > (low_inches or 0.0)
        elif bucket_type == "range":
            return (low_inches or 0.0) <= actual_inches <= (high_inches or 0.0)
        return False

    @staticmethod
    def _compute_stats(trades: list[Trade]) -> TradeStats:
        """Compute TradeStats from a list of resolved trades."""
        stats = TradeStats(total_trades=len(trades), resolved_trades=len(trades))
        if not trades:
            return stats
        stats.wins = sum(1 for t in trades if t.resolved_correct)
        stats.losses = stats.resolved_trades - stats.wins
        stats.win_rate = stats.wins / stats.resolved_trades if stats.resolved_trades > 0 else 0.0
        stats.total_pnl = sum(t.pnl or 0.0 for t in trades)
        total_cost = sum(t.cost for t in trades if t.cost > 0)
        stats.avg_roi = stats.total_pnl / total_cost if total_cost > 0 else 0.0
        return stats

    def _resolved_query(self, session, lookback_days: int | None = None):
        """Build base query for resolved BUY trades with optional lookback."""
        query = session.query(Trade).filter(
            Trade.action == "BUY",
            Trade.resolved_correct.isnot(None),
        )
        if lookback_days is not None:
            cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0)
            cutoff -= timedelta(days=lookback_days)
            query = query.filter(Trade.resolved_at >= cutoff)
        return query

    def get_stats(self, lookback_days: int | None = None) -> TradeStats:
        """Compute rolling trade statistics."""
        session = get_session()
        try:
            trades = self._resolved_query(session, lookback_days).all()
            return self._compute_stats(trades)
        finally:
            session.close()

    def get_stats_by_confidence_band(self) -> dict[str, TradeStats]:
        """Stats grouped by confidence bands."""
        session = get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.action == "BUY",
                Trade.resolved_correct.isnot(None),
            ).all()
            bands: dict[str, list[Trade]] = {
                "0.70-0.80": [],
                "0.80-0.85": [],
                "0.85-0.90": [],
                "0.90-0.95": [],
                "0.95-1.00": [],
            }
            for t in trades:
                c = t.confidence
                if c < 0.80:
                    bands["0.70-0.80"].append(t)
                elif c < 0.85:
                    bands["0.80-0.85"].append(t)
                elif c < 0.90:
                    bands["0.85-0.90"].append(t)
                elif c < 0.95:
                    bands["0.90-0.95"].append(t)
                else:
                    bands["0.95-1.00"].append(t)

            return {band: self._compute_stats(band_trades) for band, band_trades in bands.items()}
        finally:
            session.close()

    def get_stats_by_city(self, lookback_days: int | None = None) -> dict[str, TradeStats]:
        """Stats grouped by city."""
        session = get_session()
        try:
            trades = self._resolved_query(session, lookback_days).all()
            by_city: dict[str, list[Trade]] = {}
            for t in trades:
                by_city.setdefault(t.city, []).append(t)
            return {city: self._compute_stats(city_trades) for city, city_trades in by_city.items()}
        finally:
            session.close()

    def get_stats_by_hour(self, lookback_days: int | None = None) -> dict[int, TradeStats]:
        """Stats grouped by local hour of trade."""
        session = get_session()
        try:
            trades = self._resolved_query(session, lookback_days).all()
            by_hour: dict[int, list[Trade]] = {}
            for t in trades:
                h = t.local_hour
                if h is not None:
                    by_hour.setdefault(h, []).append(t)
            return {hour: self._compute_stats(hour_trades) for hour, hour_trades in by_hour.items()}
        finally:
            session.close()

    def get_historical_accuracy(
        self, city: str, hour: int | None = None, lookback_days: int = 30,
    ) -> float | None:
        """Historical win rate for a city (optionally filtered by hour).

        Returns wins/total as float, or None if fewer than 5 samples.
        """
        session = get_session()
        try:
            query = self._resolved_query(session, lookback_days)
            query = query.filter(Trade.city == city)
            if hour is not None:
                query = query.filter(Trade.local_hour == hour)
            trades = query.all()
            if len(trades) < 5:
                return None
            wins = sum(1 for t in trades if t.resolved_correct)
            return wins / len(trades)
        finally:
            session.close()

    def _update_daily_pnl(self) -> None:
        """Update daily P&L summary table."""
        session = get_session()
        try:
            today = datetime.utcnow().date().isoformat()
            trades = session.query(Trade).filter(
                Trade.resolved_correct.isnot(None),
                Trade.market_date == today,
            ).all()

            existing = session.query(DailyPnL).filter(DailyPnL.date == today).first()
            if existing is None:
                existing = DailyPnL(date=today)
                session.add(existing)

            existing.trades_count = len(trades)
            existing.wins = sum(1 for t in trades if t.resolved_correct)
            existing.losses = existing.trades_count - existing.wins
            existing.gross_pnl = sum(t.pnl or 0.0 for t in trades)
            existing.fees = sum(t.fee_paid or 0.0 for t in trades)
            existing.net_pnl = existing.gross_pnl - existing.fees

            session.commit()
        except Exception as e:
            session.rollback()
            log.error("daily_pnl_update_failed", error=str(e))
        finally:
            session.close()
