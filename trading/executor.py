"""py-clob-client order placement (POLY_PROXY signature type 1)."""

from __future__ import annotations

from datetime import datetime

from config import Config
from db.engine import get_session
from db.models import Trade
from signals.detector import TradeSignal
from trading.portfolio import Portfolio
from trading.sizing import compute_size, compute_size_kelly
from utils.logging import get_logger

log = get_logger("executor")


class TradeExecutor:
    """Places orders on Polymarket CLOB using py-clob-client."""

    def __init__(self, config: Config, portfolio: Portfolio, tracker=None) -> None:
        self._config = config
        self._portfolio = portfolio
        self._tracker = tracker
        self._clob_client = None

    def init_client(self) -> None:
        """Initialize the CLOB client with POLY_PROXY credentials."""
        if self._config.dry_run:
            # Dry-run still needs market data, so initialize a read-only client when possible.
            try:
                from py_clob_client.client import ClobClient

                self._clob_client = ClobClient(
                    self._config.clob_host,
                    chain_id=self._config.chain_id,
                )
                log.info("dry_run_read_only_client_initialized")
            except Exception as e:
                self._clob_client = None
                log.warning(
                    "dry_run_client_init_failed",
                    error=str(e),
                    msg="Running without CLOB market data; signal flow may be limited",
                )
            return

        if not self._config.poly_private_key:
            log.warning("no_private_key", msg="POLY_PRIVATE_KEY not set, trading disabled")
            return

        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                self._config.clob_host,
                key=self._config.poly_private_key,
                chain_id=self._config.chain_id,
                signature_type=1,  # POLY_PROXY for email/Google login
                funder=self._config.poly_funder_address,
            )

            # Derive or retrieve API credentials
            if self._config.poly_api_key:
                self._clob_client.set_api_creds(
                    self._clob_client.create_or_derive_api_creds()
                )
            else:
                creds = self._clob_client.create_or_derive_api_creds()
                self._clob_client.set_api_creds(creds)

            log.info("clob_client_initialized")
        except Exception as e:
            log.error("clob_client_init_failed", error=str(e))
            self._clob_client = None

    async def execute(self, signal: TradeSignal) -> Trade | None:
        """Execute a trade signal.

        1. Get order book for the token
        2. Calculate position size
        3. Check portfolio risk limits
        4. Place GTC limit order
        5. Record trade in DB
        """
        # Determine price
        price = signal.best_ask
        if price is None:
            log.warning("no_ask_price", token_id=signal.token_id, city=signal.city)
            return None
        win_probability = signal.win_probability if signal.win_probability is not None else signal.confidence.total
        win_probability = max(0.0, min(1.0, win_probability))

        # Calculate size
        portfolio_value = self._portfolio.get_value()
        resolved_trades = 0
        performance_win_rate: float | None = None
        performance_samples = 0
        if self._tracker is not None:
            total_stats = self._tracker.get_stats()
            resolved_trades = total_stats.resolved_trades
            perf_stats = self._tracker.get_stats(lookback_days=30)
            if perf_stats.resolved_trades >= self._config.aggression_min_samples:
                performance_win_rate = perf_stats.win_rate
                performance_samples = perf_stats.resolved_trades
            elif total_stats.resolved_trades >= self._config.aggression_min_samples:
                performance_win_rate = total_stats.win_rate
                performance_samples = total_stats.resolved_trades

        if self._config.kelly_mode:
            size_usd = compute_size_kelly(
                self._config,
                portfolio_value,
                signal.confidence.total,
                signal.ask_depth,
                price,
                resolved_trades=resolved_trades,
                peak_value=self._portfolio.peak_value,
                win_probability=win_probability,
                performance_win_rate=performance_win_rate,
                performance_samples=performance_samples,
            )
            if size_usd <= 0:
                log.info(
                    "kelly_no_bet",
                    city=signal.city,
                    confidence=signal.confidence.total,
                    win_probability=round(win_probability, 4),
                    performance_win_rate=None if performance_win_rate is None else round(performance_win_rate, 4),
                    performance_samples=performance_samples,
                    price=price,
                )
                return None
        else:
            size_usd = compute_size(
                self._config,
                portfolio_value,
                signal.confidence.total,
                signal.ask_depth,
                price,
                performance_win_rate=performance_win_rate,
                performance_samples=performance_samples,
                peak_value=self._portfolio.peak_value,
            )

        # Expected value gate: require positive post-cost edge before risking capital.
        exp_edge, exp_profit, exp_slippage = self._estimate_trade_expectancy(
            signal=signal,
            price=price,
            size_usd=size_usd,
        )
        if self._config.enable_ev_gate:
            if exp_edge < self._config.min_expected_edge:
                log.info(
                    "trade_blocked_ev_edge",
                    city=signal.city,
                    token_id=signal.token_id,
                    expected_edge=round(exp_edge, 4),
                    win_probability=round(win_probability, 4),
                    min_required=self._config.min_expected_edge,
                )
                return None
            if exp_profit < self._config.min_expected_profit:
                log.info(
                    "trade_blocked_ev_profit",
                    city=signal.city,
                    token_id=signal.token_id,
                    expected_profit=round(exp_profit, 3),
                    win_probability=round(win_probability, 4),
                    min_required=self._config.min_expected_profit,
                )
                return None

        # Check risk limits
        allowed, reason = self._portfolio.can_trade(signal.event_id, size_usd)
        if not allowed:
            log.info("trade_blocked", reason=reason, city=signal.city)
            return None

        # Calculate shares
        shares = size_usd / price

        # Record trade
        trade = Trade(
            event_id=signal.event_id,
            condition_id=signal.condition_id,
            city=signal.city,
            market_date=str(signal.market_date),
            icao_station=signal.icao_station,
            bucket_value=signal.bucket_value,
            bucket_type=signal.bucket_type,
            bucket_unit=signal.unit,
            token_id=signal.token_id,
            market_type=signal.market_type,
            action="BUY",
            requested_size=shares,
            requested_cost=size_usd,
            price=price,
            size=shares,
            cost=size_usd,
            confidence=signal.confidence.total,
            metar_temp_c=signal.metar_temp_c,
            metar_precision=signal.metar_precision,
            expected_edge=exp_edge,
            expected_profit=exp_profit,
            expected_slippage=exp_slippage,
            calibrated_probability=win_probability,
            wu_displayed_high=signal.wu_displayed_high,
            margin_from_boundary=signal.margin_from_boundary,
            local_hour=signal.local_hour_val,
        )

        # In live mode, economic exposure is fill-truth (reconciler updates size/cost on fills).
        if not self._config.dry_run:
            trade.size = 0.0
            trade.cost = 0.0

        if self._config.dry_run:
            trade.order_status = "DRY_RUN"
            log.info(
                "dry_run_trade",
                city=signal.city,
                bucket=f"{signal.bucket_type}:{signal.bucket_value}",
                price=price,
                size_usd=round(size_usd, 2),
                shares=round(shares, 1),
                confidence=round(signal.confidence.total, 3),
            )
        else:
            # Place real order
            order_result = self._place_order(signal.token_id, price, shares)
            if order_result:
                trade.order_id = order_result.get("orderID", "")
                trade.order_status = order_result.get("status", "UNKNOWN")
                fill_price = self._to_float(order_result.get("fillPrice"))
                filled_size = self._to_float(
                    order_result.get("filledSize")
                    or order_result.get("sizeMatched")
                    or order_result.get("matchedSize")
                )
                if filled_size and filled_size > 0:
                    px = fill_price or price
                    trade.size = filled_size
                    trade.cost = filled_size * px
                    trade.fill_price = px
                elif fill_price is not None and (trade.order_status or "").upper() in ("FILLED", "MATCHED"):
                    trade.size = shares
                    trade.cost = shares * fill_price
                    trade.fill_price = fill_price
                log.info(
                    "order_placed",
                    order_id=trade.order_id,
                    status=trade.order_status,
                    city=signal.city,
                    price=price,
                    size_usd=round(size_usd, 2),
                )
            else:
                trade.order_status = "FAILED"
                # Keep a zero-impact audit row for failed attempts.
                trade.size = 0.0
                trade.cost = 0.0
                trade.fee_paid = 0.0
                log.error("order_failed", city=signal.city, token_id=signal.token_id)

        # Save to DB
        session = get_session()
        try:
            session.add(trade)
            session.commit()
            session.refresh(trade)
            log.info("trade_recorded", trade_id=trade.id, status=trade.order_status)
            if trade.order_status == "FAILED":
                return None
            return trade
        except Exception as e:
            session.rollback()
            log.error("trade_db_save_failed", error=str(e))
            return None
        finally:
            session.close()

    async def execute_sell(
        self,
        token_id: str,
        shares: float,
        price: float,
        reason: str,
        event_id: str,
        city: str,
        parent_trade_id: int | None = None,
    ) -> Trade | None:
        """Execute a sell order (profit lock or trailing stop).

        Args:
            token_id: Token to sell
            shares: Number of shares to sell
            price: Expected sell price (best bid)
            reason: "PROFIT_LOCK", "TRAILING_STOP", or "STOP_LOSS"
            event_id: Market event ID
            city: City name
            parent_trade_id: Original BUY trade ID for reference

        Returns:
            Trade record or None on failure.
        """
        proceeds = shares * price
        fee = proceeds * self._config.sell_fee_rate

        trade = Trade(
            event_id=event_id,
            condition_id="",
            city=city,
            market_date="",
            icao_station="",
            bucket_value=0,
            bucket_type="",
            bucket_unit="",
            token_id=token_id,
            market_type="temperature",
            action="SELL",
            requested_size=shares,
            requested_cost=proceeds,
            price=price,
            size=shares,
            cost=proceeds,
            confidence=0.0,
            fee_paid=fee,
            exit_reason=reason,
            parent_trade_id=parent_trade_id,
            # SELL trades are inert execution records — all P&L computed at BUY resolution
            pnl=None,
            resolved_correct=None,
            resolved_at=None,
        )

        # In live mode, economic exposure is fill-truth (reconciler updates size/cost on fills).
        if not self._config.dry_run:
            trade.size = 0.0
            trade.cost = 0.0
            trade.fee_paid = 0.0

        if self._config.dry_run:
            trade.order_status = "DRY_RUN"
            log.info(
                "dry_run_sell",
                reason=reason,
                city=city,
                shares=round(shares, 2),
                price=price,
                proceeds=round(proceeds, 2),
                fee=round(fee, 2),
            )
        else:
            order_result = self._place_order(token_id, price, shares, side="SELL")
            if order_result:
                trade.order_id = order_result.get("orderID", "")
                trade.order_status = order_result.get("status", "UNKNOWN")
                fill_price = self._to_float(order_result.get("fillPrice"))
                filled_size = self._to_float(
                    order_result.get("filledSize")
                    or order_result.get("sizeMatched")
                    or order_result.get("matchedSize")
                )
                if filled_size and filled_size > 0:
                    px = fill_price or price
                    trade.size = filled_size
                    trade.cost = filled_size * px  # proceeds for SELL
                    trade.fill_price = px
                    trade.fee_paid = trade.cost * self._config.sell_fee_rate
                elif fill_price is not None and (trade.order_status or "").upper() in ("FILLED", "MATCHED"):
                    trade.size = shares
                    trade.cost = shares * fill_price
                    trade.fill_price = fill_price
                    trade.fee_paid = trade.cost * self._config.sell_fee_rate
                log.info(
                    "sell_order_placed",
                    reason=reason,
                    order_id=trade.order_id,
                    city=city,
                    shares=round(shares, 2),
                    price=price,
                )
            else:
                trade.order_status = "FAILED"
                # Keep a zero-impact audit row for failed attempts.
                trade.size = 0.0
                trade.cost = 0.0
                trade.fee_paid = 0.0
                log.error("sell_order_failed", reason=reason, city=city, token_id=token_id)

        session = get_session()
        try:
            session.add(trade)
            session.commit()
            session.refresh(trade)
            log.info("sell_trade_recorded", trade_id=trade.id, reason=reason, status=trade.order_status)
            if trade.order_status == "FAILED":
                return None
            return trade
        except Exception as e:
            session.rollback()
            log.error("sell_trade_db_save_failed", error=str(e))
            return None
        finally:
            session.close()

    def _place_order(self, token_id: str, price: float, size: float, side: str = "BUY") -> dict | None:
        """Place a GTC limit order via py-clob-client."""
        if self._clob_client is None:
            log.error("clob_client_not_initialized")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
            )
            resp = self._clob_client.create_and_post_order(order_args)

            if isinstance(resp, dict):
                return resp
            # Handle response object
            return {"orderID": getattr(resp, "orderID", ""), "status": "PLACED"}
        except Exception as e:
            log.error("order_placement_error", error=str(e), token_id=token_id, side=side)
            return None

    def _estimate_trade_expectancy(
        self,
        signal: TradeSignal,
        price: float,
        size_usd: float,
    ) -> tuple[float, float, float]:
        """Return (net_edge_per_share, expected_profit_usd, slippage_per_share)."""
        if price <= 0 or size_usd <= 0:
            return 0.0, 0.0, 0.0

        q = signal.win_probability if signal.win_probability is not None else signal.confidence.total
        q = max(0.0, min(1.0, q))
        shares = size_usd / price

        slippage_per_share = self._estimate_slippage_per_share(
            price=price,
            size_usd=size_usd,
            ask_depth=signal.ask_depth,
            best_bid=signal.best_bid,
        )

        # Conservative fee drag on a winning share payout.
        fee_drag = q * self._config.kelly_fee_rate * (1.0 - price)
        net_edge = q - price - fee_drag - slippage_per_share
        expected_profit = net_edge * shares
        return net_edge, expected_profit, slippage_per_share

    def _estimate_slippage_per_share(
        self,
        price: float,
        size_usd: float,
        ask_depth: float,
        best_bid: float | None,
    ) -> float:
        """Estimate per-share slippage using dynamic bps + spread + depth impact."""
        base_bps = self._config.ev_base_slippage_bps
        if self._config.ev_dynamic_slippage and self._tracker is not None:
            observed_bps = self._tracker.get_recent_buy_slippage_bps()
            if observed_bps is not None:
                base_bps = max(base_bps, observed_bps)

        base_component = price * (base_bps / 10_000.0)

        spread_component = 0.0
        if best_bid is not None and price > best_bid:
            spread_component = (price - best_bid) * self._config.ev_spread_weight

        depth_component = 0.0
        if ask_depth > 0 and size_usd > 0:
            take_ratio = min(1.0, size_usd / ask_depth)
            depth_component = self._config.ev_depth_slippage_max * (take_ratio ** 2)

        return base_component + spread_component + depth_component

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_order_book(self, token_id: str) -> dict | None:
        """Fetch order book for a token. Returns {bids, asks} with depth."""
        if self._clob_client is None:
            return None

        try:
            book = self._clob_client.get_order_book(token_id)
            return book
        except Exception as e:
            log.warning("order_book_fetch_failed", error=str(e), token_id=token_id)
            return None

    def refresh_prices(self, token_id: str) -> tuple[float | None, float | None, float, float]:
        """Fetch current best bid/ask and depth for a token.

        Returns: (best_bid, best_ask, bid_depth, ask_depth)
        """
        book = self.get_order_book(token_id)
        if book is None:
            return None, None, 0.0, 0.0

        best_bid = None
        best_ask = None
        bid_depth = 0.0
        ask_depth = 0.0

        try:
            bids = book.get("bids", []) if isinstance(book, dict) else getattr(book, "bids", [])
            asks = book.get("asks", []) if isinstance(book, dict) else getattr(book, "asks", [])

            for bid in bids:
                p = float(bid.get("price", 0) if isinstance(bid, dict) else getattr(bid, "price", 0))
                s = float(bid.get("size", 0) if isinstance(bid, dict) else getattr(bid, "size", 0))
                bid_depth += p * s
                if best_bid is None or p > best_bid:
                    best_bid = p

            for ask in asks:
                p = float(ask.get("price", 0) if isinstance(ask, dict) else getattr(ask, "price", 0))
                s = float(ask.get("size", 0) if isinstance(ask, dict) else getattr(ask, "size", 0))
                ask_depth += p * s
                if best_ask is None or p < best_ask:
                    best_ask = p
        except Exception as e:
            log.warning("price_parse_error", error=str(e))

        return best_bid, best_ask, bid_depth, ask_depth
