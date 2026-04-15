"""Async event loop orchestrator for Station Sniper."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone

from config import Config
from copycat.engine import CopycatEngine
from db.engine import init_db, cleanup_old_data
from observability import write_metrics_snapshot, write_copy_metrics_snapshot
from learning.calibrator import ConfidenceCalibrator
from learning.optimizer import ParameterOptimizer, restore_tunable_params
from learning.self_learner import SelfLearner
from learning.tracker import TradeTracker
from markets.discovery import MarketDiscovery
from markets.realtime import ClobMarketFeed
from markets.registry import MarketRegistry
from settlement.noaa import NOAAClient
from settlement.open_meteo import OpenMeteoClient
from signals.detector import SignalDetector
from signals.watchdog import SignalWatchdog
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from trading.position_manager import PositionManager
from trading.reconciliation import ExecutionReconciler
from trading.redeemer import PositionRedeemer
from utils.logging import setup_logging, get_logger
from utils.redemption_schedule import is_within_quiet_hours, next_redemption_run_utc
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
        self.tracker = TradeTracker(
            noaa_client=self.noaa_client,
            open_meteo=self.open_meteo,
            funder_wallet=config.poly_funder_address,
        )
        self.executor = TradeExecutor(config, self.portfolio, tracker=self.tracker)
        self.calibrator = ConfidenceCalibrator(self.tracker)
        self.self_learner = SelfLearner(config) if config.self_learning_enabled else None
        self.detector = SignalDetector(config, self.metar_feed, self.registry,
                                       tracker=self.tracker,
                                       calibrator=self.calibrator,
                                       self_learner=self.self_learner)
        self.signal_watchdog = SignalWatchdog(config)
        self.position_manager = PositionManager(self.executor, config)
        self.redeemer = PositionRedeemer(config)
        self.optimizer = ParameterOptimizer(config, self.tracker)
        self.market_feed = ClobMarketFeed(config)
        self.reconciler = ExecutionReconciler(config)
        self.copy_engine = CopycatEngine(config, self.executor, self.redeemer)

    async def start(self) -> None:
        """Start all loops."""
        log.info(
            "starting",
            dry_run=self.config.dry_run,
            initial_bankroll=self.config.initial_bankroll,
            strategy_mode=self.config.strategy_mode,
        )

        # Init DB
        init_db(self.config)

        # Init CLOB client
        self.executor.init_client()

        # Copycat mode: skip weather loops and run wallet-copy workflow.
        if self.config.strategy_mode == "copycat":
            if self.self_learner is not None:
                if self.self_learner.restore():
                    log.info("self_learning_restored", **self.self_learner.get_diagnostics())
                self_learning_bootstrap = self.self_learner.retrain()
                log.info("self_learning_bootstrap", **self_learning_bootstrap)
            await self.copy_engine.startup()
            await self.reconciler.start()
            self._running = True
            try:
                await asyncio.gather(
                    self._copy_poll_loop(),
                    self._copy_leader_refresh_loop(),
                    self._redemption_loop(),
                    self._copy_learning_loop(),
                    self._copy_heartbeat_loop(),
                )
            except asyncio.CancelledError:
                log.info("copy_loops_cancelled")
            finally:
                await self._shutdown()
            return

        # Restore persisted state from DB
        self.portfolio.restore_state()
        self.position_manager.restore_state()

        # Restore tunable params from DB
        restored = restore_tunable_params(self.config)
        for r in restored:
            log.info("param_restored", detail=r)

        # Restore/retrain self-learning model
        if self.self_learner is not None:
            if self.self_learner.restore():
                log.info("self_learning_restored", **self.self_learner.get_diagnostics())
            self_learning_bootstrap = self.self_learner.retrain()
            log.info("self_learning_bootstrap", **self_learning_bootstrap)

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
        """Check for and redeem winning positions on configured cadence."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_run = next_redemption_run_utc(
                    now,
                    interval_seconds=self.config.redemption_interval,
                    quiet_hours_enabled=self.config.redemption_quiet_hours_enabled,
                    timezone_name=self.config.redemption_quiet_timezone,
                    quiet_start_hour=self.config.redemption_quiet_start_hour,
                    quiet_end_hour=self.config.redemption_quiet_end_hour,
                )
                sleep_seconds = max(1.0, (next_run - now).total_seconds())
                await asyncio.sleep(sleep_seconds)

                now = datetime.now(timezone.utc)
                if self.config.redemption_quiet_hours_enabled and is_within_quiet_hours(
                    now,
                    timezone_name=self.config.redemption_quiet_timezone,
                    quiet_start_hour=self.config.redemption_quiet_start_hour,
                    quiet_end_hour=self.config.redemption_quiet_end_hour,
                ):
                    log.info(
                        "redemption_cycle_skipped_quiet_hours",
                        timezone=self.config.redemption_quiet_timezone,
                        quiet_start_hour=self.config.redemption_quiet_start_hour,
                        quiet_end_hour=self.config.redemption_quiet_end_hour,
                    )
                    continue

                result = await self.redeemer.check_and_redeem()
                if self.config.strategy_mode == "copycat":
                    self.copy_engine.record_redemption_cycle(result)
                    closed = await self.copy_engine.sync_locks_with_positions()
                    if closed:
                        log.info("copy_locks_closed_after_redemption", closed=closed)
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

    async def _copy_poll_loop(self) -> None:
        """Poll copy-trade leaders and place copy intents."""
        while self._running:
            try:
                await self.copy_engine.poll_once()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("copy_poll_loop_error", error=str(e))
                await asyncio.sleep(2)
            await asyncio.sleep(1)

    async def _copy_leader_refresh_loop(self) -> None:
        """Refresh leader eligibility snapshots on configured cadence."""
        interval = max(1800, int(self.config.leader_refresh_hours * 3600))
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.copy_engine.refresh_leaders(force=False)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("copy_leader_refresh_loop_error", error=str(e))
                await asyncio.sleep(60)

    async def _copy_heartbeat_loop(self) -> None:
        """Emit copycat-mode health telemetry and metrics snapshots."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                copy_metrics = self.copy_engine.get_metrics()
                balance_summary = await self.redeemer.get_position_balance_summary()
                copy_stats_all = self.tracker.get_stats(market_type="copycat")
                copy_stats_30d = self.tracker.get_stats(lookback_days=30, market_type="copycat")
                self_learning_diag = (
                    self.self_learner.get_diagnostics() if self.self_learner is not None else None
                )

                log.info(
                    "copy_heartbeat",
                    active_leaders=copy_metrics.get("active_leaders"),
                    open_match_locks=copy_metrics.get("open_match_locks"),
                    signals_seen=copy_metrics.get("signals_seen"),
                    signals_accepted=copy_metrics.get("signals_accepted"),
                    signals_skipped=copy_metrics.get("signals_skipped"),
                    orders_placed=copy_metrics.get("orders_placed"),
                    tradable_usdc=round(balance_summary.get("tradable_usdc") or 0.0, 2),
                    redeemable_positions=balance_summary.get("redeemable_positions"),
                    redeemable_value=round(balance_summary.get("redeemable_value") or 0.0, 2),
                    stuck_redeemable_positions=balance_summary.get("stuck_redeemable_positions"),
                    open_positions=balance_summary.get("open_positions"),
                    resolved_copycat=copy_stats_all.resolved_trades,
                    win_rate_copycat=round(copy_stats_all.win_rate, 4),
                    pnl_copycat=round(copy_stats_all.total_pnl, 2),
                )

                write_copy_metrics_snapshot(
                    self.config,
                    {
                        "copycat": copy_metrics,
                        "balances": balance_summary,
                        "performance": {
                            "all_time": {
                                "resolved": copy_stats_all.resolved_trades,
                                "wins": copy_stats_all.wins,
                                "losses": copy_stats_all.losses,
                                "win_rate": round(copy_stats_all.win_rate, 4),
                                "net_pnl": round(copy_stats_all.total_pnl, 2),
                                "avg_roi": round(copy_stats_all.avg_roi, 4),
                            },
                            "d30": {
                                "resolved": copy_stats_30d.resolved_trades,
                                "wins": copy_stats_30d.wins,
                                "losses": copy_stats_30d.losses,
                                "win_rate": round(copy_stats_30d.win_rate, 4),
                                "net_pnl": round(copy_stats_30d.total_pnl, 2),
                                "avg_roi": round(copy_stats_30d.avg_roi, 4),
                            },
                        },
                        "self_learning": self_learning_diag,
                    },
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("copy_heartbeat_failed", error=str(e))

    async def _copy_learning_loop(self) -> None:
        """Resolve copycat trades and retrain learning model periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.config.learning_interval)
                resolved = await self.tracker.resolve_trades()

                if self.self_learner is not None:
                    learning_diag = self.self_learner.retrain()
                    log.info("self_learning_cycle", **learning_diag)

                stats = self.tracker.get_stats(lookback_days=30, market_type="copycat")
                log.info(
                    "copy_learning_cycle",
                    resolved=resolved,
                    resolved_30d=stats.resolved_trades,
                    win_rate_30d=round(stats.win_rate, 4),
                    pnl_30d=round(stats.total_pnl, 2),
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("copy_learning_loop_error", error=str(e))
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

                if self.self_learner is not None:
                    learning_diag = self.self_learner.retrain()
                    log.info("self_learning_cycle", **learning_diag)

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

                if self.config.signal_kpi_enabled:
                    funnel = self.tracker.get_signal_funnel_stats(
                        window_hours=self.config.signal_kpi_window_hours
                    )
                    wd = self.signal_watchdog.evaluate_and_apply(funnel)
                    if wd.triggered:
                        log.warning(
                            "signal_watchdog_triggered",
                            candidates=wd.candidates,
                            emitted=wd.emitted,
                            time_gate_ratio=round(wd.time_gate_ratio, 4),
                            exact_ratio=round(wd.exact_ratio, 4),
                            actions=wd.actions,
                        )

                # Write metrics snapshot for external observability
                write_metrics_snapshot(
                    self.config,
                    self.portfolio,
                    self.tracker,
                    self.position_manager,
                    active_markets=self.registry.count,
                    calibrator=self.calibrator,
                    self_learner=self.self_learner,
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
        await self.copy_engine.close()

        log.info("shutdown_complete")

    def stop(self) -> None:
        """Signal the bot to stop."""
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Station Sniper: Polymarket Weather Latency Arbitrage Bot")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="Run in dry-run mode (log signals without placing orders)")
    parser.add_argument("--strategy-mode", choices=["weather", "copycat"], default=None,
                        help="Select strategy runtime mode")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = Config.from_env()
    if args.dry_run is not None:
        config.dry_run = args.dry_run
    if args.strategy_mode is not None:
        config.strategy_mode = args.strategy_mode

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
