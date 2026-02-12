"""Auto-redemption of winning positions on Polymarket CTF contract."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from config import Config
from utils.logging import get_logger

log = get_logger("redeemer")

DATA_API = "https://data-api.polymarket.com"
POLYGON_RPC = "https://polygon-rpc.com"
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

CTF_ABI = [
    {
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

USDC_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]


@dataclass
class RedemptionResult:
    redeemed: int
    failed: int
    total_value: float
    usdc_balance: float | None


class PositionRedeemer:
    """Monitors positions and auto-redeems winning ones via CTF contract."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._funder = config.poly_funder_address
        self._private_key = config.poly_private_key
        self._client = httpx.AsyncClient(timeout=30.0)
        self._redeemed: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redeemer")
        self._w3 = None
        self._account = None
        self._ctf = None

    def _init_web3(self) -> bool:
        """Lazy-init Web3 (only needed when we actually redeem)."""
        if self._w3 is not None:
            return True

        if not self._private_key:
            log.warning("no_private_key_for_redemption")
            return False

        try:
            from web3 import Web3

            self._w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
            self._account = self._w3.eth.account.from_key(self._private_key)
            self._ctf = self._w3.eth.contract(
                address=Web3.to_checksum_address(CTF_CONTRACT), abi=CTF_ABI
            )
            log.info("web3_initialized", account=self._account.address)
            return True
        except Exception as e:
            log.error("web3_init_failed", error=str(e))
            return False

    async def check_and_redeem(self) -> RedemptionResult:
        """Fetch positions, find winners, redeem them.

        Returns summary of what happened.
        """
        result = RedemptionResult(redeemed=0, failed=0, total_value=0.0, usdc_balance=None)

        positions = await self._fetch_positions()
        if not positions:
            return result

        # Find redeemable winning positions
        winners = []
        for pos in positions:
            size = float(pos.get("size", 0))
            if size <= 0:
                continue

            redeemable = pos.get("redeemable", False)
            cur_price = float(pos.get("curPrice", 0))
            condition_id = pos.get("conditionId", "")

            if redeemable and cur_price == 1 and condition_id:
                if condition_id not in self._redeemed:
                    winners.append(pos)

        if not winners:
            return result

        log.info("winners_found", count=len(winners))

        # Init Web3 if needed
        if not self._init_web3():
            log.error("cannot_redeem_no_web3")
            result.failed = len(winners)
            return result

        # Redeem each winner
        for pos in winners:
            condition_id = pos.get("conditionId", "")
            title = pos.get("title", "Unknown")[:60]
            outcome = pos.get("outcome", "")
            size = float(pos.get("size", 0))

            success = await self._redeem_position(condition_id, outcome)
            if success:
                self._redeemed.add(condition_id)
                result.redeemed += 1
                result.total_value += size
                log.info(
                    "position_redeemed",
                    title=title,
                    outcome=outcome,
                    size=round(size, 2),
                )
            else:
                result.failed += 1

        # Check USDC balance after redemptions
        result.usdc_balance = await self._get_usdc_balance()
        if result.usdc_balance is not None:
            log.info("usdc_balance", balance=round(result.usdc_balance, 2))

        return result

    async def get_balance(self) -> float | None:
        """Get current USDC balance."""
        return await self._get_usdc_balance()

    async def _fetch_positions(self) -> list[dict]:
        """Fetch all positions from Polymarket Data API."""
        if not self._funder:
            return []

        try:
            resp = await self._client.get(
                f"{DATA_API}/positions",
                params={"user": self._funder},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error("fetch_positions_failed", error=str(e))
            return []

    async def _redeem_position(self, condition_id: str, outcome: str) -> bool:
        """Execute on-chain redemption via CTF contract."""
        if self._config.dry_run:
            log.info("dry_run_redeem", condition_id=condition_id[:16])
            return True

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._redeem_sync, condition_id, outcome
        )

    def _redeem_sync(self, condition_id: str, outcome: str) -> bool:
        """Synchronous redemption call (runs in thread pool)."""
        try:
            from web3 import Web3

            # Determine index set: Yes/Up = index 0 (bit 1), No/Down = index 1 (bit 2)
            outcome_lower = outcome.lower()
            if outcome_lower in ("yes", "up"):
                index_sets = [1]
            else:
                index_sets = [2]

            # Convert condition_id to bytes32
            if condition_id.startswith("0x"):
                condition_bytes = bytes.fromhex(condition_id[2:])
            else:
                condition_bytes = bytes.fromhex(condition_id)

            nonce = self._w3.eth.get_transaction_count(self._account.address)
            gas_price = self._w3.eth.gas_price

            tx = self._ctf.functions.redeemPositions(
                condition_bytes, index_sets
            ).build_transaction(
                {
                    "from": self._account.address,
                    "nonce": nonce,
                    "gas": 200000,
                    "gasPrice": int(gas_price * 1.1),
                }
            )

            signed = self._account.sign_transaction(tx)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)

            log.info("redemption_tx_sent", tx_hash=tx_hash.hex())

            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if receipt["status"] == 1:
                log.info("redemption_confirmed", tx_hash=tx_hash.hex())
                return True
            else:
                log.error("redemption_tx_failed", tx_hash=tx_hash.hex())
                return False

        except Exception as e:
            log.error("redemption_error", error=str(e), condition_id=condition_id[:16])
            return False

    async def _get_usdc_balance(self) -> float | None:
        """Get USDC balance via Web3."""
        if not self._init_web3():
            return None

        try:
            from web3 import Web3

            usdc = self._w3.eth.contract(
                address=Web3.to_checksum_address(USDC_ADDRESS), abi=USDC_ABI
            )
            funder_addr = Web3.to_checksum_address(self._funder)
            balance = usdc.functions.balanceOf(funder_addr).call()
            return balance / 1e6  # USDC has 6 decimals
        except Exception as e:
            log.error("usdc_balance_error", error=str(e))
            return None

    async def close(self) -> None:
        await self._client.aclose()
        self._executor.shutdown(wait=False)
