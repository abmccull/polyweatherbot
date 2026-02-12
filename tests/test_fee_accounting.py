"""Fee calculation tests."""

from __future__ import annotations

import pytest

from config import Config
from trading.sizing import kelly_fraction


@pytest.fixture
def config():
    return Config(dry_run=True, initial_bankroll=1000.0)


class TestSellFee:
    """SELL fee = proceeds x sell_fee_rate."""

    def test_sell_fee_calculation(self, config):
        """Fee should be proceeds * sell_fee_rate."""
        shares = 100.0
        price = 0.50
        proceeds = shares * price  # 50.0
        fee = proceeds * config.sell_fee_rate  # 50.0 * 0.01 = 0.50
        assert fee == pytest.approx(0.50, rel=1e-3)

    def test_sell_fee_scales_with_proceeds(self, config):
        """Doubling proceeds doubles the fee."""
        fee1 = (50.0 * 0.40) * config.sell_fee_rate
        fee2 = (100.0 * 0.40) * config.sell_fee_rate
        assert fee2 == pytest.approx(fee1 * 2, rel=1e-3)


class TestKellyEffectivePayout:
    """Kelly fraction accounts for fees in effective payout."""

    def test_fee_reduces_fraction(self):
        """Higher fee rate -> lower Kelly fraction."""
        q, price = 0.90, 0.20
        f_low_fee = kelly_fraction(q, price, fee_rate=0.01)
        f_high_fee = kelly_fraction(q, price, fee_rate=0.05)
        assert f_low_fee > f_high_fee

    def test_zero_fee_baseline(self):
        """Zero fee gives maximum Kelly fraction."""
        q, price = 0.90, 0.20
        f_zero = kelly_fraction(q, price, fee_rate=0.0)
        f_with = kelly_fraction(q, price, fee_rate=0.02)
        assert f_zero > f_with

    def test_effective_payout_formula(self):
        """Verify the effective payout: 1 - fee_rate * (1 - price)."""
        price = 0.25
        fee_rate = 0.02
        effective_payout = 1.0 - fee_rate * (1.0 - price)
        assert effective_payout == pytest.approx(0.985, rel=1e-3)


class TestBuyMakerFee:
    """BUY maker fee should be 0%."""

    def test_maker_fee_zero(self, config):
        """Maker fee rate is 0 for GTC limit orders."""
        assert config.maker_fee_rate == 0.0

    def test_taker_fee_nonzero(self, config):
        """Taker fee rate is non-zero (for reference)."""
        assert config.taker_fee_rate > 0.0
        assert config.taker_fee_rate == 0.02
