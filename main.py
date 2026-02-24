"""Async event loop orchestrator for Station Sniper."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime

from config import Config
from db.engine import init_db, cleanup_old_data
from observability import write_metrics_snapshot
from learning.calibrator import ConfidenceCalibrator
from learning.optimizer import ParameterOptimizer, restore_tunable_params
from learning.tracker import TradeTracker
from markets.discovery import MarketDiscovery
from markets.realtime import ClobMarketFeed
from markets.registry import MarketRegistry
from settlement.noaa import NOAAClient
from settlement.open_meteo import OpenMeteoClient
from signals.detector import SignalDetector
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from trading.position_manager import PositionManager
from trading.reconciliation import ExecutionReconciler
from trading.redeemer import PositionRedeemer
from utils.logging import setup_logging, get_logger
from weather.metar_feed import MetarFeed
from weather.nws_feed import NWSFeed
from weather.synoptic_feed import SynopticFeed

log = get_logger("main")


class StationSniper:
    """Main application: coordinates all subsystems."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._running = False

        # Subsystems
        self.metar_feed = MetarFeed()
        self.synoptic_feed = SynopticFeed(config.synoptic_api_token)
        self.nws_feed = NWSFeed()
        self.discovery = MarketDiscovery()
        self.registry = MarketRegistry(self.discovery)
        self.noaa_client = NOAAClient()
        self.open_meteo = OpenMeteoClient()
        self.portfolio = Portfolio(config)
        self.tracker = TradeTracker(noaa_client=self.noaa_client, open_meteo=self.open_meteo)
        self.executor = TradeExecutor(config, self.portfolio, tracker=self.tracker)
        self.calibrator = ConfidenceCalibrator(self.tracker)
        self.detector = SignalDetector(config, self.metar_feed, self.registry,
                                       tracker=self.tracker,
                                       calibrator=self.calibrator)
        self.position_manager = PositionManager(self.executor, config)
        self.redeemer = PositionRedeemer(config)
        self.optimizer = ParameterOptimizer(config, self.tracker)
        self.market_feed = ClobMarketFeed(config)
        self.reconciler = ExecutionReconciler(config)

    async def start(self) -> None:
        """Start all loops."""
        log.info(
            "starting",
            dry_run=self.config.dry_run,
            initial_bankroll=self.config.initial_bankroll,
        )

        # Init DB
        init_db(self.config)

        # Restore persisted state from DB
        self.portfolio.restore_state()
        self.position_manager.restore_state()

        # Restore tunable params from DB
        restored = restore_tunable_params(self.config)
        for r in restored:
            log.info("param_restored", detail=r)

        # Init CLOB client
        self.executor.init_client()

        # Initial market scan (temperature only)
        await self.registry.refresh()
        log.info(
            "initial_scan_complete",
            temp_markets=self.registry.count,
        )

        # Start realtime market/user streams (optional; falls back gracefully).
        self._sync_realtime_tokens()
        await self.market_feed.start()
        await self.reconciler.start()

        self._running = True

        # Run concurrent loops
        try:
            await asyncio.gather(
                self._market_scan_loop(),
                self._observation_loop(),
                self._redemption_loop(),
                self._learning_loop(),
                self._heartbeat_loop(),
            )
        except asyncio.CancelledError:
            log.info("loops_cancelled")
        finally:
            await self._shutdown()

    async def _market_scan_loop(self) -> None:
        """Refresh market registry every 15 minutes."""
        while self._running:
            try:
                await asyncio.sleep(self.config.market_scan_interval)
                await self.registry.refresh()
                self._sync_realtime_tokens()
                log.info(
                    "market_scan_complete",
                    temp_markets=self.registry.count,
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("market_scan_error", error=str(e))
                await asyncio.sleep(60)

    async def _observation_loop(self) -> None:
        """Poll METAR → update highs → refresh prices → detect signals → execute."""
        while self._running:
            try:
                # 1. Poll METAR cache (updates daily highs AND monthly precip)
                observations = await self.metar_feed.poll()

                # 2. Refresh order book prices for active markets
                await self._refresh_prices()

                # 3. Detect temperature signals
                signals = await self.detector.detect()

                # 4. Execute temperature trades
                for sig in signals:
                    trade = await self.executor.execute(sig)
                    if trade:
                        log.info(
                            "trade_executed",
                            trade_id=trade.id,
                            city=sig.city,
                            bucket=f"{sig.bucket_type}:{sig.bucket_value}{sig.unit}",
                            market_type="temperature",
                        )

                # 5. Check positions for profit lock / trailing stop
                exits = await self.position_manager.check_positions(
                    price_getter=lambda tid: self._get_best_bid(tid)
                )
                for exit_trade in exits:
                    log.info(
                        "position_exit",
                        trade_id=exit_trade.id,
                        reason=exit_trade.exit_reason,
                        city=exit_trade.city,
                        shares=round(exit_trade.size, 2),
                        price=exit_trade.price,
                    )

            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("observation_loop_error", error=str(e))

            await asyncio.sleep(self.config.observation_interval)

    def _get_best_bid(self, token_id: str) -> float | None:
        """Look up best bid price for a token from the registry."""
        bucket_price = self.registry.get_bucket_price(token_id)
        if bucket_price is not None:
            return bucket_price.best_bid
        return None

    async def _refresh_prices(self) -> None:
        """Refresh order book prices for all active bucket tokens."""
        market_feed = getattr(self, "market_feed", None)
        # Temperature markets
        for active in self.registry.get_all_active():
            for bucket in active.market.buckets:
                bid = ask = None
                bid_depth = ask_depth = 0.0

                # Prefer websocket snapshot when fresh; fall back to REST order book.
                if market_feed is not None and market_feed.is_fresh(
                    bucket.token_id, self.config.market_price_max_age,
                ):
                    ws_price = market_feed.get_price(bucket.token_id)
                    if ws_price is not None:
                        bid = ws_price.best_bid
                        ask = ws_price.best_ask
                        bid_depth = ws_price.bid_depth
                        ask_depth = ws_price.ask_depth

                if bid is None and ask is None:
                    bid, ask, bid_depth, ask_depth = self.executor.refresh_prices(bucket.token_id)

                self.registry.update_prices(bucket.token_id, bid, ask, bid_depth, ask_depth)

    def _sync_realtime_tokens(self) -> None:
        """Push current registry token universe into websocket subscription."""
        token_ids: list[str] = []
        for active in self.registry.get_all_active():
            for bucket in active.market.buckets:
                token_ids.append(bucket.token_id)
        self.market_feed.set_tokens(token_ids)

    async def _redemption_loop(self) -> None:
        """Check for and redeem winning positions every 5 minutes."""
        while self._running:
            try:
                await asyncio.sleep(self.config.redemption_interval)
                result = await self.redeemer.check_and_redeem()
                if result.redeemed or result.failed:
                    log.info(
                        "redemption_cycle",
                        redeemed=result.redeemed,
                        failed=result.failed,
                        total_value=round(result.total_value, 2),
                        usdc_balance=round(result.usdc_balance, 2) if result.usdc_balance else None,
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("redemption_loop_error", error=str(e))
                await asyncio.sleep(60)

    async def _learning_loop(self) -> None:
        """Resolve trades, compute stats, adjust parameters every 30 minutes."""
        cycles = 0
        while self._running:
            try:
                await asyncio.sleep(self.config.learning_interval)
                cycles += 1

                # Resolve completed trades
                resolved = await self.tracker.resolve_trades()

                # Run optimizer
                adjustments = self.optimizer.optimize()
                for adj in adjustments:
                    log.info("param_adjustment", detail=adj)

                # Run confidence calibration
                calibration = self.calibrator.calibrate()
                if calibration:
                    log.info("calibration_complete", bands=len(calibration))
                model_diag = self.calibrator.get_model_diagnostics()
                if model_diag is not None:
                    log.info("calibration_model", **model_diag)

                # Update portfolio outcomes for circuit breaker
                stats = self.tracker.get_stats(lookback_days=1)
                log.info(
                    "learning_cycle",
                    resolved=resolved,
                    adjustments=len(adjustments),
                    daily_win_rate=round(stats.win_rate, 3),
                    daily_pnl=round(stats.total_pnl, 2),
                )

                # DB cleanup every ~24 hours (48 cycles × 30 min)
                if cycles % 48 == 0:
                    deleted = cleanup_old_data(self.config.archive_days)
                    if deleted:
                        log.info("db_cleanup", rows_deleted=deleted)

            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("learning_loop_error", error=str(e))
                await asyncio.sleep(60)

    async def _heartbeat_loop(self) -> None:
        """Log health status every 5 minutes."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                portfolio_val = self.portfolio.get_value()
                deployed = self.portfolio.get_deployed()
                stats = self.tracker.get_stats()
                open_positions = self.position_manager.get_open_positions()

                log.info(
                    "heartbeat",
                    portfolio_value=round(portfolio_val, 2),
                    deployed=round(deployed, 2),
                    active_temp_markets=self.registry.count,
                    open_positions=len(open_positions),
                    total_trades=stats.total_trades,
                    win_rate=round(stats.win_rate, 3),
                    total_pnl=round(stats.total_pnl, 2),
                    min_confidence=self.config.min_confidence.value,
                    max_price=self.config.max_price.value,
                    bet_pct=self.config.bet_pct.value,
                )

                # Write metrics snapshot for external observability
                write_metrics_snapshot(
                    self.config,
                    self.portfolio,
                    self.tracker,
                    self.position_manager,
                    active_markets=self.registry.count,
                    calibrator=self.calibrator,
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("heartbeat_failed", error=str(e))

    async def _shutdown(self) -> None:
        """Graceful shutdown: close connections, flush DB."""
        log.info("shutting_down")
        self._running = False

        await self.noaa_client.close()
        await self.open_meteo.close()
        await self.redeemer.close()
        await self.market_feed.stop()
        await self.reconciler.stop()
        await self.metar_feed.close()
        await self.synoptic_feed.close()
        await self.nws_feed.close()
        await self.discovery.close()

        log.info("shutdown_complete")

    def stop(self) -> None:
        """Signal the bot to stop."""
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Station Sniper: Polymarket Weather Latency Arbitrage Bot")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="Run in dry-run mode (log signals without placing orders)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = Config.from_env()
    if args.dry_run is not None:
        config.dry_run = args.dry_run

    bot = StationSniper(config)

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def handle_signal(sig: int, frame) -> None:
        log.info("signal_received", signal=sig)
        bot.stop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
