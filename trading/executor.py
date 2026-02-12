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
            log.info("dry_run_mode", msg="CLOB client not initialized in dry-run mode")
            return

        if not self._config.poly_private_key:
            log.warning("no_private_key", msg="POLY_PRIVATE_KEY not set, trading disabled")
            return

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs

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

        # Calculate size
        portfolio_value = self._portfolio.get_value()
        if self._config.kelly_mode:
            resolved_trades = 0
            if self._tracker is not None:
                stats = self._tracker.get_stats()
                resolved_trades = stats.resolved_trades
            size_usd = compute_size_kelly(
                self._config,
                portfolio_value,
                signal.confidence.total,
                signal.ask_depth,
                price,
                resolved_trades=resolved_trades,
                peak_value=self._portfolio.peak_value,
            )
            if size_usd <= 0:
                log.info("kelly_no_bet", city=signal.city, confidence=signal.confidence.total, price=price)
                return None
        else:
            size_usd = compute_size(
                self._config,
                portfolio_value,
                signal.confidence.total,
                signal.ask_depth,
                price,
            )

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
            price=price,
            size=shares,
            cost=size_usd,
            confidence=signal.confidence.total,
            metar_temp_c=signal.metar_temp_c,
            metar_precision=signal.metar_precision,
            wu_displayed_high=signal.wu_displayed_high,
            margin_from_boundary=signal.margin_from_boundary,
            local_hour=signal.local_hour_val,
        )

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
                trade.fill_price = order_result.get("fillPrice")
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
                log.error("order_failed", city=signal.city, token_id=signal.token_id)

        # Save to DB
        session = get_session()
        try:
            session.add(trade)
            session.commit()
            session.refresh(trade)
            log.info("trade_recorded", trade_id=trade.id)
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
            reason: "PROFIT_LOCK" or "TRAILING_STOP"
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
                trade.fill_price = order_result.get("fillPrice")
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
                log.error("sell_order_failed", reason=reason, city=city, token_id=token_id)

        session = get_session()
        try:
            session.add(trade)
            session.commit()
            session.refresh(trade)
            log.info("sell_trade_recorded", trade_id=trade.id, reason=reason)
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
