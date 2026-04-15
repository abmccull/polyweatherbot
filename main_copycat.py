"""Async event loop orchestrator for copycat-only runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import tempfile
from datetime import datetime, timezone

from config import Config
from copycat.engine import CopycatEngine
from db.engine import init_db
from learning.self_learner import SelfLearner
from learning.tracker import TradeTracker
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from trading.reconciliation import ExecutionReconciler
from trading.redeemer import PositionRedeemer
from utils.logging import get_logger, setup_logging
from utils.redemption_schedule import is_within_quiet_hours, next_redemption_run_utc

log = get_logger("main_copycat")
METRICS_FILE = os.getenv("METRICS_FILE", "metrics.json")


def _write_metrics_snapshot(payload: dict) -> None:
    try:
        dir_name = os.path.dirname(os.path.abspath(METRICS_FILE))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, METRICS_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        log.warning("copy_metrics_snapshot_failed", error=str(e))


class CopycatTrader:
    """Main copycat runtime loop."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._running = False
        self.portfolio = Portfolio(config)
        self.tracker = TradeTracker(funder_wallet=config.poly_funder_address)
        self.self_learner = SelfLearner(config) if config.self_learning_enabled else None
        self.executor = TradeExecutor(config, self.portfolio, tracker=self.tracker)
        self.redeemer = PositionRedeemer(config)
        self.reconciler = ExecutionReconciler(config)
        self.copy_engine = CopycatEngine(config, self.executor, self.redeemer)

    async def start(self) -> None:
        log.info(
            "starting",
            dry_run=self.config.dry_run,
            initial_bankroll=self.config.initial_bankroll,
            strategy_mode=self.config.strategy_mode,
        )
        init_db(self.config)
        self.executor.init_client()
        if self.self_learner is not None:
            if self.self_learner.restore():
                log.info("self_learning_restored", **self.self_learner.get_diagnostics())
            learning_bootstrap = self.self_learner.retrain()
            log.info("self_learning_bootstrap", **learning_bootstrap)
        await self.copy_engine.startup()
        await self.reconciler.start()
        self._running = True
        try:
            await asyncio.gather(
                self._copy_poll_loop(),
                self._copy_leader_refresh_loop(),
                self._redemption_loop(),
                self._learning_loop(),
                self._heartbeat_loop(),
            )
        except asyncio.CancelledError:
            log.info("copy_loops_cancelled")
        finally:
            await self._shutdown()

    async def _copy_poll_loop(self) -> None:
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

    async def _redemption_loop(self) -> None:
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

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                copy_metrics = self.copy_engine.get_metrics()
                balance_summary = await self.redeemer.get_position_balance_summary()
                portfolio_value = self.portfolio.get_value()
                deployed = self.portfolio.get_deployed()
                copy_stats_all = self.tracker.get_stats(market_type="copycat")
                copy_stats_30d = self.tracker.get_stats(lookback_days=30, market_type="copycat")
                self_learning_diag = (
                    self.self_learner.get_diagnostics() if self.self_learner is not None else None
                )
                payload = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "mode": "copycat",
                    "strategy_mode": self.config.strategy_mode,
                    "portfolio": {
                        "value": round(portfolio_value, 2),
                        "deployed": round(deployed, 2),
                        "initial_bankroll": self.config.initial_bankroll,
                    },
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
                }
                _write_metrics_snapshot(payload)
                log.info(
                    "copy_heartbeat",
                    portfolio_value=round(portfolio_value, 2),
                    deployed=round(deployed, 2),
                    active_leaders=copy_metrics.get("active_leaders"),
                    open_match_locks=copy_metrics.get("open_match_locks"),
                    signals_seen=copy_metrics.get("signals_seen"),
                    signals_accepted=copy_metrics.get("signals_accepted"),
                    signals_skipped=copy_metrics.get("signals_skipped"),
                    orders_placed=copy_metrics.get("orders_placed"),
                    tradable_usdc=round(balance_summary.get("tradable_usdc") or 0.0, 2),
                    matic_balance=round(balance_summary.get("matic_balance") or 0.0, 6),
                    redeemable_positions=balance_summary.get("redeemable_positions"),
                    redeemable_value=round(balance_summary.get("redeemable_value") or 0.0, 2),
                    redeem_gas_required_matic=round(balance_summary.get("redeem_gas_required_matic") or 0.0, 6),
                    redeem_gas_ok=balance_summary.get("redeem_gas_ok"),
                    stuck_redeemable_positions=balance_summary.get("stuck_redeemable_positions"),
                    open_positions=balance_summary.get("open_positions"),
                    resolved_copycat=copy_stats_all.resolved_trades,
                    win_rate_copycat=round(copy_stats_all.win_rate, 4),
                    pnl_copycat=round(copy_stats_all.total_pnl, 2),
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("copy_heartbeat_failed", error=str(e))

    async def _learning_loop(self) -> None:
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

    async def _shutdown(self) -> None:
        log.info("shutting_down")
        self._running = False
        await self.redeemer.close()
        await self.reconciler.stop()
        await self.copy_engine.close()
        log.info("shutdown_complete")

    def stop(self) -> None:
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Station Sniper: Polymarket Sports Copycat Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Run in dry-run mode (log signals without placing orders)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = Config.from_env()
    config.strategy_mode = "copycat"
    if args.dry_run is not None:
        config.dry_run = args.dry_run

    bot = CopycatTrader(config)
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
