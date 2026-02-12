"""Position sizing: flat % fallback + Kelly criterion for optimal bankroll growth."""

from __future__ import annotations

from config import Config
from utils.logging import get_logger

log = get_logger("sizing")


def compute_size(
    config: Config,
    portfolio_value: float,
    confidence: float,
    ask_depth: float,
    price: float,
) -> float:
    """Compute position size in USDC (flat % fallback).

    Base: max($10, bet_pct x portfolio_value)
    Confidence multiplier: linear from 0.85->1.0x to 1.0->1.2x
    Liquidity cap: never take >80% of visible ask depth
    Bounded: [min_bet, max_bet]
    """
    bet_pct = config.bet_pct.value

    # Base size
    base = max(10.0, bet_pct * portfolio_value)

    # Confidence multiplier: linear interpolation
    if confidence >= 1.0:
        conf_mult = 1.2
    elif confidence >= 0.85:
        conf_mult = 1.0 + (confidence - 0.85) / (1.0 - 0.85) * 0.2
    else:
        conf_mult = 0.8 + (confidence / 0.85) * 0.2

    size = base * conf_mult

    # Liquidity cap
    if ask_depth > 0:
        liquidity_limit = ask_depth * config.liquidity_cap
        if size > liquidity_limit:
            log.info(
                "size_liquidity_capped",
                original=round(size, 2),
                capped=round(liquidity_limit, 2),
                ask_depth=round(ask_depth, 2),
            )
            size = liquidity_limit

    # Absolute bounds
    size = max(config.min_bet, min(config.max_bet, size))

    # Convert to share count, then back to cost for clean numbers
    if price > 0:
        shares = size / price
        size = round(shares * price, 2)

    return size


# ---------------------------------------------------------------------------
# Kelly Criterion sizing
# ---------------------------------------------------------------------------

def kelly_fraction(q: float, price: float, fee_rate: float) -> float:
    """Raw full Kelly fraction for a binary market.

    Args:
        q: Estimated probability of winning (after haircut)
        price: Ask price per share (cost basis)
        fee_rate: Fee applied to profit on win

    Returns:
        Full Kelly fraction (can be negative if no edge).
    """
    effective_payout = 1.0 - fee_rate * (1.0 - price)
    b = (effective_payout - price) / price  # net odds ratio
    if b <= 0:
        return 0.0
    f = q - (1.0 - q) / b
    return f


def kelly_multiplier_for_confidence(
    confidence: float,
    min_frac: float,
    base_frac: float,
    max_frac: float,
) -> float:
    """Map confidence to fractional Kelly multiplier.

    | Confidence | Fraction     |
    |-----------|-------------|
    | 0.80-0.85 | min_frac    |
    | 0.85-0.90 | base_frac   |
    | 0.90-0.95 | linear base->max |
    | 0.95+     | max_frac    |
    """
    if confidence < 0.85:
        return min_frac
    elif confidence < 0.90:
        return base_frac
    elif confidence < 0.95:
        # Linear interpolation from base to max over [0.90, 0.95]
        t = (confidence - 0.90) / 0.05
        return base_frac + t * (max_frac - base_frac)
    else:
        return max_frac


def drawdown_throttle(
    portfolio_value: float,
    peak_value: float,
    start: float,
    full: float,
) -> float:
    """Drawdown-based multiplier on Kelly fraction.

    Returns a value in [0.5, 1.0]:
    - No drawdown or < start: 1.0
    - At `full` drawdown: 0.5
    - Linear between start and full
    """
    if peak_value <= 0:
        return 1.0
    dd = 1.0 - (portfolio_value / peak_value)
    if dd <= start:
        return 1.0
    if dd >= full:
        return 0.5
    # Linear interpolation
    t = (dd - start) / (full - start)
    return 1.0 - t * 0.5


def compute_size_kelly(
    config: Config,
    portfolio_value: float,
    confidence: float,
    ask_depth: float,
    price: float,
    resolved_trades: int = 0,
    peak_value: float = 0.0,
) -> float:
    """Kelly-optimal position sizing.

    Args:
        config: Global config with Kelly params
        portfolio_value: Current portfolio value in USDC
        confidence: Confidence score (0.0-1.0)
        ask_depth: Total visible ask depth in USDC
        price: Expected fill price per share
        resolved_trades: Number of resolved trades (for ramp-up)
        peak_value: All-time high portfolio value (for drawdown throttle)

    Returns:
        Position size in USDC (cost basis).
    """
    # Floor check
    if confidence < config.kelly_confidence_floor:
        log.debug("kelly_below_floor", confidence=confidence, floor=config.kelly_confidence_floor)
        return 0.0

    # Probability estimate with haircut
    q = confidence - config.kelly_probability_haircut

    # Minimum edge check
    if (q - price) < config.kelly_min_edge:
        log.debug("kelly_insufficient_edge", q=round(q, 3), price=price, min_edge=config.kelly_min_edge)
        return 0.0

    # Full Kelly fraction
    f_full = kelly_fraction(q, price, config.kelly_fee_rate)
    if f_full <= 0:
        log.debug("kelly_no_edge", q=round(q, 3), price=price, f_full=round(f_full, 4))
        return 0.0

    # Fractional Kelly multiplier based on confidence
    k_mult = kelly_multiplier_for_confidence(
        confidence, config.kelly_min_fraction, config.kelly_base_fraction, config.kelly_max_fraction,
    )

    # Drawdown throttle
    dd_mult = drawdown_throttle(
        portfolio_value, peak_value, config.drawdown_throttle_start, config.drawdown_throttle_full,
    )

    # Ramp-up: scale down early trades
    ramp_mult = 1.0
    if resolved_trades < config.kelly_ramp_trades and config.kelly_ramp_trades > 0:
        # Start at 20% of Kelly, linearly ramp to 100%
        ramp_mult = 0.2 + 0.8 * (resolved_trades / config.kelly_ramp_trades)

    f_adjusted = f_full * k_mult * dd_mult * ramp_mult
    size = f_adjusted * portfolio_value

    log.info(
        "kelly_sizing",
        confidence=round(confidence, 3),
        q=round(q, 3),
        price=price,
        f_full=round(f_full, 4),
        k_mult=round(k_mult, 2),
        dd_mult=round(dd_mult, 2),
        ramp_mult=round(ramp_mult, 2),
        raw_size=round(size, 2),
    )

    # Liquidity cap
    if ask_depth > 0:
        liquidity_limit = ask_depth * config.liquidity_cap
        if size > liquidity_limit:
            log.info(
                "kelly_liquidity_capped",
                original=round(size, 2),
                capped=round(liquidity_limit, 2),
            )
            size = liquidity_limit

    # Absolute bounds
    size = max(config.min_bet, min(config.max_bet, size))

    # Convert to share count, then back to cost for clean numbers
    if price > 0:
        shares = size / price
        size = round(shares * price, 2)

    return size
