"""Auto-redemption of winning positions on Polymarket CTF contract."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from typing import Any

import httpx

from config import Config
from utils.logging import get_logger

log = get_logger("redeemer")

DATA_API = "https://data-api.polymarket.com"
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER_CONTRACT = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
RELAYER_PROXY_CONFIG = {
    137: {
        "proxy_factory": "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052",
        "relay_hub": "0xD216153c06E857cD7f72665E0aF1d7D82172F494",
    }
}
RELAYER_ENDPOINTS = {
    "relay_payload": "/relay-payload",
    "submit": "/submit",
    "transaction": "/transaction",
}
RELAYER_SUCCESS_STATES = {"STATE_MINED", "STATE_CONFIRMED"}
RELAYER_FAIL_STATE = "STATE_FAILED"
RELAYER_DEFAULT_GAS_LIMIT = 500_000
DEFAULT_POLYGON_RPCS = (
    "https://polygon-rpc.com",
    "https://polygon.llamarpc.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon-bor-rpc.publicnode.com",
)

CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

CTF_PAYOUT_ABI = [
    {
        "inputs": [{"name": "", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "", "type": "bytes32"},
            {"name": "", "type": "uint256"},
        ],
        "name": "payoutNumerators",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

NEG_RISK_ADAPTER_ABI = [
    {
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "amounts", "type": "uint256[]"},
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

CTF_APPROVAL_ABI = [
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"},
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"},
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class RedemptionResult:
    redeemed: int
    failed: int
    total_value: float
    usdc_balance: float | None
    details: list[dict]


class PositionRedeemer:
    """Monitors positions and auto-redeems winning ones via CTF contract."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._funder = config.poly_funder_address
        self._private_key = config.poly_private_key
        self._client = httpx.AsyncClient(timeout=30.0)
        self._redeemed: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redeemer")
        self._w3: Any | None = None
        self._account: Any | None = None
        self._ctf: Any | None = None
        self._rpc_urls = self._parse_rpc_urls(config.polygon_rpc_urls)
        self._active_rpc_url: str | None = None
        self._rpc_index = -1
        self._native_balance_cache: float | None = None
        self._native_balance_cache_at: float = 0.0
        self._native_balance_cache_ttl_sec: float = 15.0
        self._warned_account_funder_mismatch = False
        self._funder_is_contract_cache: bool | None = None
        self._funder_is_contract_cache_at: float = 0.0
        self._funder_is_contract_ttl_sec: float = 300.0
        self._warned_proxy_redeem_unsupported = False
        self._warned_relayer_unavailable = False
        self._warned_builder_creds_incomplete = False
        self._builder_creds_checked = False
        self._relayer_auth_unauthorized = False
        self._neg_risk_adapter_approval_ok = False
        self._condition_payout_den_cache: dict[str, int] = {}
        self._condition_payout_num_cache: dict[tuple[str, int], int] = {}
        self._no_effect_cooldown_seconds = 6 * 3600
        self._no_effect_cooldown_until: dict[str, float] = {}
        self._relayer_backoff_initial_seconds = max(
            1.0, float(self._config.redeem_relayer_429_backoff_initial_seconds)
        )
        self._relayer_backoff_max_seconds = max(
            self._relayer_backoff_initial_seconds,
            float(self._config.redeem_relayer_429_backoff_max_seconds),
        )
        self._relayer_backoff_next_seconds = self._relayer_backoff_initial_seconds
        self._relayer_backoff_until_ts = 0.0
        self._relayer_backoff_log_at = 0.0

    def _init_web3(self) -> bool:
        """Lazy-init Web3 (only needed when we actually redeem)."""
        if self._w3 is not None:
            return True

        if not self._private_key:
            log.warning("no_private_key_for_redemption")
            return False

        return self._rotate_rpc(reason="init", start_from_next=False)

    def _parse_rpc_urls(self, raw: str | None) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        candidates = raw.split(",") if raw else []
        for item in candidates:
            url = item.strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        if not urls:
            urls = list(DEFAULT_POLYGON_RPCS)
        return urls

    def _activate_rpc(self, rpc_url: str) -> None:
        from web3 import Web3

        timeout = max(5, int(self._config.polygon_rpc_timeout_seconds))
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        chain_id = int(w3.eth.chain_id)
        if chain_id != int(self._config.chain_id):
            raise RuntimeError(f"unexpected_chain_id={chain_id}")
        account = w3.eth.account.from_key(self._private_key)
        ctf = w3.eth.contract(
            address=Web3.to_checksum_address(CTF_CONTRACT), abi=CTF_ABI
        )
        self._w3 = w3
        self._account = account
        self._ctf = ctf
        self._active_rpc_url = rpc_url
        if (
            self._funder
            and account.address.lower() != self._funder.lower()
            and not self._warned_account_funder_mismatch
        ):
            self._warned_account_funder_mismatch = True
            log.warning(
                "redeemer_account_funder_mismatch",
                account=account.address,
                funder=self._funder,
                msg="On-chain redemption uses signer wallet gas and position ownership",
            )

    def _rotate_rpc(self, reason: str, start_from_next: bool = True) -> bool:
        if not self._rpc_urls:
            log.error("web3_no_rpc_endpoints")
            return False

        total = len(self._rpc_urls)
        if self._rpc_index < 0 or not start_from_next:
            start = 0
        else:
            start = (self._rpc_index + 1) % total

        for offset in range(total):
            idx = (start + offset) % total
            rpc_url = self._rpc_urls[idx]
            try:
                self._activate_rpc(rpc_url)
                self._rpc_index = idx
                log.info(
                    "web3_rpc_selected",
                    rpc_url=rpc_url,
                    reason=reason,
                    candidate=f"{idx + 1}/{total}",
                    account=self._account.address,
                )
                return True
            except Exception as e:
                log.warning(
                    "web3_rpc_candidate_failed",
                    rpc_url=rpc_url,
                    reason=reason,
                    error=str(e),
                )

        self._w3 = None
        self._account = None
        self._ctf = None
        self._active_rpc_url = None
        self._rpc_index = -1
        log.error("web3_all_rpc_candidates_failed", reason=reason, candidates=total)
        return False

    def _failover_after_error(self, stage: str, error: Exception) -> bool:
        err = str(error).lower()
        likely_transport = any(
            token in err
            for token in (
                "401",
                "403",
                "429",
                "unauthorized",
                "forbidden",
                "timeout",
                "timed out",
                "connection",
                "bad gateway",
                "gateway timeout",
                "service unavailable",
                "reset by peer",
            )
        )
        if likely_transport:
            return self._rotate_rpc(reason=f"{stage}_rpc_error", start_from_next=True)
        return False

    def _rpc_attempts(self) -> int:
        return max(1, len(self._rpc_urls))

    def _rpc_label(self) -> str:
        return self._active_rpc_url or "unknown"

    def _required_redeem_matic(self, redeemable_count: int) -> float:
        per_pos = max(0.0, float(self._config.copy_est_matic_per_redemption))
        reserve = max(0.0, float(self._config.copy_min_matic_reserve))
        estimate = per_pos * max(0, int(redeemable_count))
        return max(reserve, estimate)

    def _has_builder_credentials(self) -> bool:
        if not (
            self._config.redeem_proxy_via_relayer
            and self._config.relayer_url
            and self._private_key
        ):
            return False
        if self._config.builder_api_key and (
            not self._config.builder_api_secret or not self._config.builder_api_passphrase
        ):
            if not self._warned_builder_creds_incomplete:
                self._warned_builder_creds_incomplete = True
                log.warning(
                    "builder_creds_incomplete",
                    msg="BUILDER_API_KEY is set but BUILDER_API_SECRET/PASSPHRASE are missing",
                )
            return False
        if (
            self._config.builder_api_key
            and self._config.builder_api_secret
            and self._config.builder_api_passphrase
        ):
            return True
        self._ensure_builder_credentials()
        return bool(
            self._config.builder_api_key
            and self._config.builder_api_secret
            and self._config.builder_api_passphrase
        )

    def _relayer_backoff_seconds_remaining(self) -> float:
        return max(0.0, self._relayer_backoff_until_ts - time.time())

    def _mark_relayer_rate_limited(self) -> None:
        delay = max(1.0, self._relayer_backoff_next_seconds)
        now = time.time()
        self._relayer_backoff_until_ts = max(self._relayer_backoff_until_ts, now + delay)
        self._relayer_backoff_next_seconds = min(
            self._relayer_backoff_max_seconds,
            max(1.0, self._relayer_backoff_next_seconds * 2.0),
        )

    def _reset_relayer_backoff(self) -> None:
        self._relayer_backoff_until_ts = 0.0
        self._relayer_backoff_next_seconds = self._relayer_backoff_initial_seconds

    def _ensure_builder_credentials(self) -> None:
        """Populate missing builder creds by deriving API creds from CLOB client."""
        if self._builder_creds_checked:
            return
        self._builder_creds_checked = True
        if self._config.builder_api_key:
            # Respect explicit Builder key from env; avoid replacing it with unrelated CLOB creds.
            return
        if not self._private_key:
            return
        try:
            from py_clob_client.client import ClobClient

            client = ClobClient(
                self._config.clob_host,
                key=self._private_key,
                chain_id=self._config.chain_id,
                signature_type=1,
                funder=self._funder,
            )
            creds = client.create_or_derive_api_creds()
            self._config.builder_api_key = getattr(creds, "api_key", "") or ""
            self._config.builder_api_secret = getattr(creds, "api_secret", "") or ""
            self._config.builder_api_passphrase = getattr(creds, "api_passphrase", "") or ""
            if (
                self._config.builder_api_key
                and self._config.builder_api_secret
                and self._config.builder_api_passphrase
            ):
                log.info("builder_creds_derived_from_clob")
            else:
                log.warning("builder_creds_derive_empty")
        except Exception as e:
            log.warning("builder_creds_derive_failed", error=str(e))

    async def get_redeem_mode(self) -> str:
        """Return active redemption mode: onchain_bot, relayer_proxy_wallet, or manual_proxy_wallet."""
        if not self._private_key:
            return "disabled_no_private_key"

        # Ensure account exists for signer/funder checks.
        if self._account is None and not self._init_web3():
            return "onchain_bot"

        signer = (self._account.address if self._account is not None else "").lower()
        funder = (self._funder or "").lower()
        if not (signer and funder and signer != funder):
            return "onchain_bot"

        chain_has_proxy_relayer = int(self._config.chain_id) in RELAYER_PROXY_CONFIG
        if chain_has_proxy_relayer and self._has_builder_credentials() and not self._relayer_auth_unauthorized:
            return "relayer_proxy_wallet"
        return "manual_proxy_wallet"

    async def needs_native_gas_reserve(self) -> bool:
        """True when copy risk checks should enforce MATIC reserve."""
        return (await self.get_redeem_mode()) == "onchain_bot"

    @staticmethod
    def _parse_outcome_index(raw: Any) -> int | None:
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _outcome_to_index_sets(outcome: str, outcome_index: int | None = None) -> list[int]:
        if outcome_index is not None and int(outcome_index) in (0, 1):
            return [1, 2]
        if outcome_index is not None and outcome_index >= 0:
            return [1 << int(outcome_index)]

        outcome_lower = (outcome or "").lower()
        if outcome_lower in ("yes", "up", "over", "above"):
            return [1, 2]
        if outcome_lower in ("no", "down", "under", "below"):
            return [1, 2]
        # Conservative fallback for unknown binary labels: include both partitions.
        return [1, 2]

    async def _is_funder_contract_wallet(self) -> bool:
        """Return True when POLY_FUNDER_ADDRESS is a smart contract wallet."""
        now = time.time()
        if (
            self._funder_is_contract_cache is not None
            and now - self._funder_is_contract_cache_at <= self._funder_is_contract_ttl_sec
        ):
            return self._funder_is_contract_cache

        from web3 import Web3

        attempts = self._rpc_attempts()
        for attempt in range(attempts):
            if not self._init_web3():
                return False
            try:
                if not self._funder:
                    return False
                code = self._w3.eth.get_code(Web3.to_checksum_address(self._funder))
                is_contract = bool(code and code != b"\x00" and code != b"")
                self._funder_is_contract_cache = is_contract
                self._funder_is_contract_cache_at = time.time()
                return is_contract
            except Exception as e:
                retried = False
                if attempt < attempts - 1:
                    retried = self._failover_after_error(stage="funder_code", error=e)
                log.warning(
                    "funder_contract_check_failed",
                    attempt=f"{attempt + 1}/{attempts}",
                    will_retry=retried,
                    error=str(e),
                    rpc_url=self._rpc_label(),
                )
                if retried:
                    continue
                break
        return False

    async def check_and_redeem(self) -> RedemptionResult:
        """Fetch positions, find winners, redeem them.

        Returns summary of what happened.
        """
        result = RedemptionResult(redeemed=0, failed=0, total_value=0.0, usdc_balance=None, details=[])

        positions = await self._fetch_positions()
        if not positions:
            return result

        # Find redeemable winning positions
        winners = []
        now_ts = time.time()
        for pos in positions:
            size = float(pos.get("size", 0))
            if size <= 0:
                continue

            redeemable = pos.get("redeemable", False)
            cur_price = float(pos.get("curPrice", 0))
            condition_id = pos.get("conditionId", "")
            outcome_index = self._parse_outcome_index(pos.get("outcomeIndex"))
            payout_num, payout_den = await self._get_position_payout_ratio(condition_id, outcome_index)
            has_positive_payout = (
                payout_num is not None and payout_den is not None and payout_den > 0 and payout_num > 0
            )

            if not redeemable or not condition_id:
                continue
            if not has_positive_payout and cur_price != 1:
                continue

            cooldown_until = self._no_effect_cooldown_until.get(condition_id, 0.0)
            if cooldown_until > now_ts:
                continue
            if condition_id not in self._redeemed:
                winners.append(pos)

        if not winners:
            return result

        log.info("winners_found", count=len(winners))

        redeem_mode = await self.get_redeem_mode()
        if redeem_mode == "manual_proxy_wallet":
            if not self._warned_proxy_redeem_unsupported:
                self._warned_proxy_redeem_unsupported = True
                reason = (
                    "Relayer credentials rejected (401 unauthorized)"
                    if self._relayer_auth_unauthorized
                    else "Proxy wallet detected but relayer builder credentials unavailable"
                )
                log.warning(
                    "redemption_proxy_wallet_unsupported",
                    signer=self._account.address if self._account is not None else None,
                    funder=self._funder,
                    msg=reason,
                )
            result.usdc_balance = await self._get_usdc_balance()
            return result

        if redeem_mode == "relayer_proxy_wallet":
            backoff_remaining = self._relayer_backoff_seconds_remaining()
            if backoff_remaining > 0:
                now = time.time()
                if now - self._relayer_backoff_log_at >= 30.0:
                    self._relayer_backoff_log_at = now
                    log.warning(
                        "redeem_relayer_backoff_active",
                        winners=len(winners),
                        remaining_seconds=round(backoff_remaining, 1),
                    )
                result.usdc_balance = await self._get_usdc_balance()
                return result

        if redeem_mode == "onchain_bot":
            required_matic = self._required_redeem_matic(len(winners))
            native_balance = await self._get_native_balance(cached=False)
            if native_balance is None or native_balance < required_matic:
                result.usdc_balance = await self._get_usdc_balance()
                log.warning(
                    "redemption_skipped_low_matic",
                    winners=len(winners),
                    matic_balance=None if native_balance is None else round(native_balance, 6),
                    required_matic=round(required_matic, 6),
                )
                return result

        # Init Web3 if needed
        if redeem_mode == "onchain_bot" and not self._init_web3():
            log.error("cannot_redeem_no_web3")
            result.failed = len(winners)
            return result

        max_relayer_submits = max(1, int(self._config.redeem_relayer_max_submits_per_cycle))
        relayer_spacing = max(0.0, float(self._config.redeem_relayer_submit_spacing_seconds))
        relayer_attempted = 0

        # Redeem each winner
        for pos in winners:
            condition_id = pos.get("conditionId", "")
            title = pos.get("title", "Unknown")[:60]
            outcome = pos.get("outcome", "")
            outcome_index = self._parse_outcome_index(pos.get("outcomeIndex"))
            size = float(pos.get("size", 0))
            negative_risk = bool(pos.get("negativeRisk", False))
            asset_id = pos.get("asset")
            pre_asset_balance = await self._get_position_asset_balance(asset_id)
            if pre_asset_balance is not None and pre_asset_balance <= 0:
                self._redeemed.add(condition_id)
                log.info(
                    "redemption_skipped_zero_onchain_balance",
                    condition_id=condition_id[:16],
                    asset=asset_id,
                    title=title,
                )
                continue

            if redeem_mode == "relayer_proxy_wallet":
                if relayer_attempted >= max_relayer_submits:
                    log.info(
                        "redeem_cycle_submit_cap_reached",
                        attempted=relayer_attempted,
                        max_per_cycle=max_relayer_submits,
                        remaining=max(0, len(winners) - relayer_attempted),
                    )
                    break
                if relayer_attempted > 0 and relayer_spacing > 0:
                    await asyncio.sleep(relayer_spacing)
                success, tx_hash, err = await self._redeem_position_via_relayer(
                    condition_id,
                    outcome,
                    outcome_index,
                    size=size,
                    negative_risk=negative_risk,
                )
                relayer_attempted += 1
            else:
                success, tx_hash, err = await self._redeem_position(
                    condition_id,
                    outcome,
                    outcome_index,
                    size=size,
                    negative_risk=negative_risk,
                )

            if success:
                post_asset_balance = await self._get_position_asset_balance(asset_id)
                if (
                    pre_asset_balance is not None
                    and post_asset_balance is not None
                    and post_asset_balance >= pre_asset_balance
                ):
                    success = False
                    err = "redeem_no_effect_balance_unchanged"
                    self._no_effect_cooldown_until[condition_id] = (
                        time.time() + float(self._no_effect_cooldown_seconds)
                    )
                    log.warning(
                        "redemption_no_effect_detected",
                        condition_id=condition_id[:16],
                        asset=asset_id,
                        pre_balance=pre_asset_balance,
                        post_balance=post_asset_balance,
                        cooldown_seconds=self._no_effect_cooldown_seconds,
                    )

            if success:
                self._redeemed.add(condition_id)
                result.redeemed += 1
                result.total_value += size
                result.details.append(
                    {
                        "condition_id": condition_id,
                        "title": title,
                        "outcome": outcome,
                        "size": size,
                        "status": "REDEEMED",
                        "tx_hash": tx_hash,
                    }
                )
                log.info(
                    "position_redeemed",
                    title=title,
                    outcome=outcome,
                    size=round(size, 2),
                )
            else:
                result.failed += 1
                result.details.append(
                    {
                        "condition_id": condition_id,
                        "title": title,
                        "outcome": outcome,
                        "size": size,
                        "status": "FAILED",
                        "tx_hash": tx_hash,
                        "error": err,
                    }
                )
                if redeem_mode == "relayer_proxy_wallet" and err == "relayer_rate_limited_429":
                    log.warning(
                        "redeem_cycle_paused_rate_limited",
                        attempted=relayer_attempted,
                        remaining=max(0, len(winners) - relayer_attempted),
                        backoff_seconds=round(self._relayer_backoff_seconds_remaining(), 1),
                    )
                    break

        # Check USDC balance after redemptions
        result.usdc_balance = await self._get_usdc_balance()
        if result.usdc_balance is not None:
            log.info("usdc_balance", balance=round(result.usdc_balance, 2))

        return result

    async def get_balance(self) -> float | None:
        """Get current USDC balance."""
        return await self._get_usdc_balance()

    async def get_native_balance(self) -> float | None:
        """Get current native MATIC balance for redemption signer."""
        return await self._get_native_balance(cached=True)

    async def get_positions(self) -> list[dict]:
        """Fetch open positions for funder wallet."""
        return await self._fetch_positions()

    async def get_position_balance_summary(self) -> dict:
        """Return tradable balance + redeemable/open position aggregates."""
        positions = await self._fetch_positions()
        tradable = await self._get_usdc_balance()
        matic = await self._get_native_balance(cached=True)
        funder_is_contract = await self._is_funder_contract_wallet()
        redeem_mode = await self.get_redeem_mode()
        redeemable_count = 0
        redeemable_value = 0.0
        stuck_redeemable = 0
        open_positions = 0
        now = datetime.now(timezone.utc)
        for pos in positions:
            size = float(pos.get("size", 0) or 0)
            if size <= 0:
                continue
            open_positions += 1
            if pos.get("redeemable", False):
                asset_balance = await self._get_position_asset_balance(pos.get("asset"))
                if asset_balance is not None and asset_balance <= 0:
                    continue
                outcome_index = self._parse_outcome_index(pos.get("outcomeIndex"))
                payout_num, payout_den = await self._get_position_payout_ratio(
                    pos.get("conditionId", ""),
                    outcome_index,
                )
                has_positive_payout = (
                    payout_num is not None and payout_den is not None and payout_den > 0 and payout_num > 0
                )
                fallback_price_one = float(pos.get("curPrice", 0) or 0) == 1.0
                if not has_positive_payout and not fallback_price_one:
                    continue
                redeemable_count += 1
                shares = (asset_balance / 1e6) if asset_balance is not None else size
                if has_positive_payout and payout_den:
                    redeemable_value += shares * (float(payout_num) / float(payout_den))
                else:
                    redeemable_value += shares
                end_raw = pos.get("endDate")
                if end_raw:
                    try:
                        end_dt = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
                        if end_dt.tzinfo is None:
                            end_dt = end_dt.replace(tzinfo=timezone.utc)
                        age_hours = (now - end_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
                        if age_hours >= max(1, self._config.copy_stuck_redeemable_hours):
                            stuck_redeemable += 1
                    except ValueError:
                        pass
        required_matic = self._required_redeem_matic(redeemable_count) if redeem_mode == "onchain_bot" else 0.0
        return {
            "tradable_usdc": tradable,
            "matic_balance": matic,
            "funder_is_contract_wallet": funder_is_contract,
            "redeemer_signer": self._account.address if self._account is not None else None,
            "redeemer_funder": self._funder,
            "open_positions": open_positions,
            "redeemable_positions": redeemable_count,
            "redeemable_value": redeemable_value,
            "stuck_redeemable_positions": stuck_redeemable,
            "redeem_gas_required_matic": required_matic,
            "redeem_gas_ok": (
                True
                if redeem_mode != "onchain_bot"
                else (matic is not None and matic >= required_matic)
            ),
            "relayer_backoff_seconds_remaining": (
                round(self._relayer_backoff_seconds_remaining(), 1)
                if redeem_mode == "relayer_proxy_wallet"
                else 0.0
            ),
            "redeem_mode": redeem_mode,
            "redeem_automation_ready": redeem_mode in ("onchain_bot", "relayer_proxy_wallet"),
            "redeem_automation_reason": (
                "ok"
                if redeem_mode in ("onchain_bot", "relayer_proxy_wallet")
                else (
                    (
                        "invalid_builder_credentials"
                        if self._relayer_auth_unauthorized
                        else "missing_builder_credentials"
                    )
                    if redeem_mode == "manual_proxy_wallet"
                    else "missing_private_key"
                )
            ),
        }

    async def _fetch_positions(self) -> list[dict]:
        """Fetch all positions from Polymarket Data API."""
        if not self._funder:
            return []

        try:
            all_rows: list[dict] = []
            seen_keys: set[str] = set()
            limit = 200
            offset = 0
            while True:
                resp = await self._client.get(
                    f"{DATA_API}/positions",
                    params={"user": self._funder, "limit": limit, "offset": offset},
                )
                resp.raise_for_status()
                batch = resp.json()
                if not isinstance(batch, list) or not batch:
                    break
                for row in batch:
                    if not isinstance(row, dict):
                        continue
                    key = str(
                        row.get("asset")
                        or f"{row.get('conditionId')}::{row.get('outcomeIndex')}::{row.get('proxyWallet')}"
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_rows.append(row)
                if len(batch) < limit:
                    break
                offset += limit
            return all_rows
        except Exception as e:
            log.error("fetch_positions_failed", error=str(e))
            return []

    async def _redeem_position(
        self,
        condition_id: str,
        outcome: str,
        outcome_index: int | None = None,
        *,
        size: float = 0.0,
        negative_risk: bool = False,
    ) -> tuple[bool, str | None, str | None]:
        """Execute on-chain redemption via CTF contract."""
        if self._config.dry_run:
            log.info("dry_run_redeem", condition_id=condition_id[:16])
            return True, None, None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._redeem_sync,
            condition_id,
            outcome,
            outcome_index,
            size,
            negative_risk,
        )

    async def _relay_proxy_call(
        self,
        *,
        to_address: str,
        call_data: str,
        metadata: str,
    ) -> tuple[bool, str | None, str | None]:
        relay_payload = await self._get_relay_payload(self._account.address)
        nonce = relay_payload.get("nonce")
        relay_address = relay_payload.get("address")
        if nonce is None or not relay_address:
            return False, None, "invalid_relay_payload"

        batch_data = self._encode_proxy_batch_data(
            to_address=to_address,
            call_data=call_data,
        )
        gas_limit = max(50_000, int(self._config.redeem_relayer_gas_limit or RELAYER_DEFAULT_GAS_LIMIT))
        chain_cfg = RELAYER_PROXY_CONFIG.get(int(self._config.chain_id))
        if not chain_cfg:
            return False, None, f"unsupported_chain_id={self._config.chain_id}"

        request_body = self._build_relayer_proxy_request(
            from_address=self._account.address,
            proxy_wallet=self._funder,
            proxy_factory=chain_cfg["proxy_factory"],
            relay_hub=chain_cfg["relay_hub"],
            relay_address=relay_address,
            nonce=str(nonce),
            encoded_proxy_data=batch_data,
            gas_limit=str(gas_limit),
            metadata=metadata,
        )
        submit_path = RELAYER_ENDPOINTS["submit"]
        body_json = self._serialize_json_body(request_body)
        headers = self._build_builder_headers("POST", submit_path, body_json)
        headers["Content-Type"] = "application/json"
        resp = await self._client.post(
            self._relayer_url(submit_path),
            headers=headers,
            content=body_json,
        )
        resp.raise_for_status()
        payload = resp.json()
        tx_id = payload.get("transactionID")
        tx_hash = payload.get("transactionHash")
        if not tx_id:
            return False, tx_hash, "missing_transaction_id"

        final_tx = await self._wait_relayer_transaction(tx_id)
        if final_tx is None:
            return False, tx_hash, "relayer_timeout"

        state = str(final_tx.get("state") or "")
        chain_tx_hash = final_tx.get("transactionHash") or tx_hash
        if state in RELAYER_SUCCESS_STATES:
            self._reset_relayer_backoff()
            return True, chain_tx_hash, None
        if state == RELAYER_FAIL_STATE:
            relayer_error = str(final_tx.get("error") or "").strip()
            relayer_error_msg = str(final_tx.get("errorMsg") or "").strip()
            err = relayer_error_msg or relayer_error or "relayer_failed"
            return False, chain_tx_hash, err
        return False, chain_tx_hash, f"relayer_state={state or 'unknown'}"

    async def _is_operator_approved(self, owner: str, operator: str) -> bool | None:
        from web3 import Web3

        if not self._init_web3():
            return None
        try:
            ctf = self._w3.eth.contract(
                address=Web3.to_checksum_address(CTF_CONTRACT),
                abi=CTF_APPROVAL_ABI,
            )
            return bool(
                ctf.functions.isApprovedForAll(
                    Web3.to_checksum_address(owner),
                    Web3.to_checksum_address(operator),
                ).call()
            )
        except Exception as e:
            log.warning("approval_check_failed", error=str(e))
            return None

    async def _get_position_asset_balance(self, asset_id: str | int | None) -> int | None:
        if not asset_id or not self._funder:
            return None
        from web3 import Web3

        if not self._init_web3():
            return None
        try:
            token_id = int(asset_id)
            ctf = self._w3.eth.contract(
                address=Web3.to_checksum_address(CTF_CONTRACT),
                abi=[
                    {
                        "inputs": [
                            {"name": "account", "type": "address"},
                            {"name": "id", "type": "uint256"},
                        ],
                        "name": "balanceOf",
                        "outputs": [{"name": "", "type": "uint256"}],
                        "stateMutability": "view",
                        "type": "function",
                    }
                ],
            )
            bal = ctf.functions.balanceOf(
                Web3.to_checksum_address(self._funder),
                token_id,
            ).call()
            return int(bal)
        except Exception as e:
            log.warning("position_asset_balance_failed", error=str(e))
            return None

    async def _get_position_payout_ratio(
        self,
        condition_id: str,
        outcome_index: int | None,
    ) -> tuple[int | None, int | None]:
        if not condition_id or outcome_index is None or int(outcome_index) < 0:
            return None, None
        if not self._init_web3():
            return None, None

        from web3 import Web3

        cid = self._normalize_hex(condition_id)
        cid_key = cid.lower()
        cid_bytes = bytes.fromhex(cid[2:])
        try:
            denom = self._condition_payout_den_cache.get(cid_key)
            if denom is None:
                ctf = self._w3.eth.contract(
                    address=Web3.to_checksum_address(CTF_CONTRACT),
                    abi=CTF_PAYOUT_ABI,
                )
                denom = int(ctf.functions.payoutDenominator(cid_bytes).call())
                self._condition_payout_den_cache[cid_key] = denom
            num_key = (cid_key, int(outcome_index))
            num = self._condition_payout_num_cache.get(num_key)
            if num is None and denom > 0:
                ctf = self._w3.eth.contract(
                    address=Web3.to_checksum_address(CTF_CONTRACT),
                    abi=CTF_PAYOUT_ABI,
                )
                num = int(ctf.functions.payoutNumerators(cid_bytes, int(outcome_index)).call())
                self._condition_payout_num_cache[num_key] = num
            return num, denom
        except Exception as e:
            log.warning(
                "position_payout_ratio_failed",
                condition_id=cid_key[:16],
                outcome_index=outcome_index,
                error=str(e),
            )
            return None, None

    def _encode_set_approval_for_all_calldata(self, operator: str, approved: bool = True) -> str:
        from eth_abi import encode
        from web3 import Web3

        selector = Web3.keccak(text="setApprovalForAll(address,bool)")[:4]
        encoded_args = encode(
            ["address", "bool"],
            [Web3.to_checksum_address(operator), bool(approved)],
        )
        return "0x" + (selector + encoded_args).hex()

    async def _ensure_neg_risk_adapter_approval(self) -> bool:
        if self._neg_risk_adapter_approval_ok:
            return True
        if not self._funder:
            return False

        approved = await self._is_operator_approved(self._funder, NEG_RISK_ADAPTER_CONTRACT)
        if approved is True:
            self._neg_risk_adapter_approval_ok = True
            return True
        if approved is None:
            return False

        log.info("neg_risk_adapter_approval_missing", operator=NEG_RISK_ADAPTER_CONTRACT)
        approval_data = self._encode_set_approval_for_all_calldata(NEG_RISK_ADAPTER_CONTRACT, True)
        success, tx_hash, err = await self._relay_proxy_call(
            to_address=CTF_CONTRACT,
            call_data=approval_data,
            metadata="auto-approve-neg-risk-adapter",
        )
        if success:
            self._neg_risk_adapter_approval_ok = True
            log.info("neg_risk_adapter_approved", tx_hash=tx_hash)
            return True
        log.warning("neg_risk_adapter_approval_failed", tx_hash=tx_hash, error=err)
        return False

    async def _redeem_position_via_relayer(
        self,
        condition_id: str,
        outcome: str,
        outcome_index: int | None = None,
        *,
        size: float = 0.0,
        negative_risk: bool = False,
    ) -> tuple[bool, str | None, str | None]:
        """Execute proxy-wallet redemption through Polymarket relayer API."""
        if self._config.dry_run:
            log.info("dry_run_redeem_relayer", condition_id=condition_id[:16])
            return True, None, None

        if not self._has_builder_credentials():
            if not self._warned_relayer_unavailable:
                self._warned_relayer_unavailable = True
                log.warning("redeem_relayer_unavailable", reason="missing_builder_credentials")
            return False, None, "missing_builder_credentials"

        if not self._funder:
            return False, None, "missing_funder_address"

        if self._account is None and not self._init_web3():
            return False, None, "web3_init_failed"
        if self._account is None:
            return False, None, "missing_signer_account"

        try:
            use_neg_risk_adapter = self._should_use_neg_risk_adapter(
                negative_risk=negative_risk,
                outcome=outcome,
                outcome_index=outcome_index,
            )
            if use_neg_risk_adapter:
                approval_ok = await self._ensure_neg_risk_adapter_approval()
                if not approval_ok:
                    log.warning("neg_risk_adapter_approval_not_granted_fallback")
                redeem_data = self._encode_neg_risk_redeem_calldata(
                    condition_id=condition_id,
                    size=size,
                    outcome=outcome,
                    outcome_index=outcome_index,
                )
                target_contract = NEG_RISK_ADAPTER_CONTRACT
            else:
                redeem_data = self._encode_redeem_calldata(condition_id, outcome, outcome_index)
                target_contract = CTF_CONTRACT

            success, tx_hash, err = await self._relay_proxy_call(
                to_address=target_contract,
                call_data=redeem_data,
                metadata=f"auto-redeem:{condition_id[:12]}",
            )
            if success:
                log.info("redemption_relayer_confirmed", tx_hash=tx_hash)
                return True, tx_hash, None
            log.warning("redemption_relayer_failed", tx_hash=tx_hash, error=err)
            return False, tx_hash, err
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429:
                self._mark_relayer_rate_limited()
                body = ""
                if e.response is not None:
                    body = e.response.text[:300]
                log.warning(
                    "redeem_relayer_rate_limited",
                    status=status,
                    response=body,
                    backoff_seconds=round(self._relayer_backoff_seconds_remaining(), 1),
                )
                return False, None, "relayer_rate_limited_429"
            if status == 401:
                self._relayer_auth_unauthorized = True
                body = ""
                if e.response is not None:
                    body = e.response.text[:300]
                log.error(
                    "redeem_relayer_unauthorized",
                    status=status,
                    msg="Builder credentials rejected by relayer",
                    response=body,
                )
                return False, None, "relayer_unauthorized_builder_credentials"
            return False, None, str(e)
        except Exception as e:
            return False, None, str(e)

    def _redeem_sync(
        self,
        condition_id: str,
        outcome: str,
        outcome_index: int | None = None,
        size: float = 0.0,
        negative_risk: bool = False,
    ) -> tuple[bool, str | None, str | None]:
        """Synchronous redemption call (runs in thread pool)."""
        from web3 import Web3

        index_sets = self._outcome_to_index_sets(outcome, outcome_index)

        # Convert condition_id to bytes32
        if condition_id.startswith("0x"):
            condition_bytes = bytes.fromhex(condition_id[2:])
        else:
            condition_bytes = bytes.fromhex(condition_id)

        last_err = "unknown"
        attempts = self._rpc_attempts()

        for attempt in range(attempts):
            tx_hash_hex: str | None = None
            if not self._init_web3():
                return False, None, "web3_init_failed"
            try:
                nonce = self._w3.eth.get_transaction_count(self._account.address)
                gas_price = self._w3.eth.gas_price

                use_neg_risk_adapter = self._should_use_neg_risk_adapter(
                    negative_risk=negative_risk,
                    outcome=outcome,
                    outcome_index=outcome_index,
                )
                if use_neg_risk_adapter:
                    amounts = self._neg_risk_amounts(
                        size=size,
                        outcome=outcome,
                        outcome_index=outcome_index,
                    )
                    adapter = self._w3.eth.contract(
                        address=Web3.to_checksum_address(NEG_RISK_ADAPTER_CONTRACT),
                        abi=NEG_RISK_ADAPTER_ABI,
                    )
                    tx = adapter.functions.redeemPositions(
                        condition_bytes,
                        amounts,
                    ).build_transaction(
                        {
                            "from": self._account.address,
                            "nonce": nonce,
                            "gas": 240000,
                            "gasPrice": int(gas_price * 1.1),
                        }
                    )
                else:
                    tx = self._ctf.functions.redeemPositions(
                        Web3.to_checksum_address(USDC_ADDRESS),
                        b"\x00" * 32,
                        condition_bytes,
                        index_sets,
                    ).build_transaction(
                        {
                            "from": self._account.address,
                            "nonce": nonce,
                            "gas": 240000,
                            "gasPrice": int(gas_price * 1.1),
                        }
                    )

                signed = self._account.sign_transaction(tx)
                tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
                tx_hash_hex = tx_hash.hex()

                log.info("redemption_tx_sent", tx_hash=tx_hash_hex, rpc_url=self._rpc_label())

                receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

                if receipt["status"] == 1:
                    log.info("redemption_confirmed", tx_hash=tx_hash_hex, rpc_url=self._rpc_label())
                    return True, tx_hash_hex, None
                log.error("redemption_tx_failed", tx_hash=tx_hash_hex, rpc_url=self._rpc_label())
                return False, tx_hash_hex, "tx_failed"

            except Exception as e:
                last_err = str(e)
                # If tx hash exists, do not retry broadcasting across RPC endpoints.
                if tx_hash_hex:
                    log.error(
                        "redemption_error_post_send",
                        error=last_err,
                        condition_id=condition_id[:16],
                        tx_hash=tx_hash_hex,
                        rpc_url=self._rpc_label(),
                    )
                    return False, tx_hash_hex, last_err

                retried = False
                if attempt < attempts - 1:
                    retried = self._failover_after_error(stage="redeem", error=e)
                log.warning(
                    "redemption_attempt_failed",
                    attempt=f"{attempt + 1}/{attempts}",
                    will_retry=retried,
                    error=last_err,
                    rpc_url=self._rpc_label(),
                    condition_id=condition_id[:16],
                )
                if retried:
                    continue
                break

        log.error("redemption_error", error=last_err, condition_id=condition_id[:16], rpc_url=self._rpc_label())
        return False, None, last_err

    @staticmethod
    def _normalize_hex(value: str) -> str:
        val = value.strip().lower()
        return val if val.startswith("0x") else f"0x{val}"

    def _encode_redeem_calldata(
        self,
        condition_id: str,
        outcome: str,
        outcome_index: int | None = None,
    ) -> str:
        """Encode CTF redeemPositions(collateral,parent,conditionId,indexSets) calldata."""
        from eth_abi import encode
        from web3 import Web3

        cid = self._normalize_hex(condition_id)
        condition_bytes = bytes.fromhex(cid[2:])
        if len(condition_bytes) != 32:
            raise ValueError("invalid_condition_id_bytes")
        index_sets = self._outcome_to_index_sets(outcome, outcome_index)
        selector = Web3.keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
        encoded_args = encode(
            ["address", "bytes32", "bytes32", "uint256[]"],
            [Web3.to_checksum_address(USDC_ADDRESS), b"\x00" * 32, condition_bytes, index_sets],
        )
        return "0x" + (selector + encoded_args).hex()

    @staticmethod
    def _should_use_neg_risk_adapter(
        *,
        negative_risk: bool,
        outcome: str,
        outcome_index: int | None,
    ) -> bool:
        _ = (outcome, outcome_index)
        return bool(negative_risk)

    @staticmethod
    def _neg_risk_amounts(
        *,
        size: float,
        outcome: str,
        outcome_index: int | None,
    ) -> list[int]:
        shares = Decimal(str(size))
        amount = int((shares * Decimal("1000000")).to_integral_value(rounding=ROUND_DOWN))
        if amount <= 0:
            raise ValueError("invalid_neg_risk_size")

        idx: int | None = None
        if outcome_index is not None and int(outcome_index) in (0, 1):
            idx = int(outcome_index)
        else:
            outcome_lower = (outcome or "").strip().lower()
            if outcome_lower in ("yes", "up", "over", "above"):
                idx = 0
            elif outcome_lower in ("no", "down", "under", "below"):
                idx = 1
        if idx is None:
            raise ValueError("unsupported_neg_risk_outcome")

        amounts = [0, 0]
        amounts[idx] = amount
        return amounts

    def _encode_neg_risk_redeem_calldata(
        self,
        *,
        condition_id: str,
        size: float,
        outcome: str,
        outcome_index: int | None = None,
    ) -> str:
        """Encode neg-risk adapter redeemPositions(conditionId, amounts) calldata."""
        from eth_abi import encode
        from web3 import Web3

        cid = self._normalize_hex(condition_id)
        condition_bytes = bytes.fromhex(cid[2:])
        if len(condition_bytes) != 32:
            raise ValueError("invalid_condition_id_bytes")
        amounts = self._neg_risk_amounts(size=size, outcome=outcome, outcome_index=outcome_index)
        selector = Web3.keccak(text="redeemPositions(bytes32,uint256[])")[:4]
        encoded_args = encode(["bytes32", "uint256[]"], [condition_bytes, amounts])
        return "0x" + (selector + encoded_args).hex()

    def _encode_proxy_batch_data(self, to_address: str, call_data: str) -> str:
        """Encode proxy((uint8,address,uint256,bytes)[]) call for one CALL transaction."""
        from eth_abi import encode
        from eth_utils import to_bytes
        from web3 import Web3

        selector = Web3.keccak(text="proxy((uint8,address,uint256,bytes)[])")[:4]
        call_data_bytes = (
            to_bytes(hexstr=call_data)
            if call_data.startswith("0x")
            else to_bytes(hexstr=f"0x{call_data}")
        )
        tuple_args = [(1, Web3.to_checksum_address(to_address), 0, call_data_bytes)]
        encoded = encode(["(uint8,address,uint256,bytes)[]"], [tuple_args])
        return "0x" + (selector + encoded).hex()

    def _build_proxy_struct_hash(
        self,
        *,
        from_address: str,
        proxy_factory: str,
        encoded_proxy_data: str,
        gas_limit: str,
        nonce: str,
        relay_hub: str,
        relay_address: str,
    ) -> str:
        from eth_utils import keccak, to_bytes
        from hexbytes import HexBytes

        data_bytes = (
            to_bytes(hexstr=encoded_proxy_data)
            if encoded_proxy_data.startswith("0x")
            else to_bytes(hexstr=f"0x{encoded_proxy_data}")
        )
        message = (
            b"rlx:"
            + HexBytes(from_address)
            + HexBytes(proxy_factory)
            + data_bytes
            + int(0).to_bytes(32, "big")  # txFee
            + int(0).to_bytes(32, "big")  # gasPrice
            + int(gas_limit).to_bytes(32, "big")
            + int(nonce).to_bytes(32, "big")
            + HexBytes(relay_hub)
            + HexBytes(relay_address)
        )
        return "0x" + keccak(message).hex()

    def _sign_struct_hash(self, struct_hash: str) -> str:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from hexbytes import HexBytes

        sig = Account.sign_message(
            encode_defunct(HexBytes(struct_hash)),
            self._private_key,
        ).signature.hex()
        return sig if sig.startswith("0x") else f"0x{sig}"

    def _build_relayer_proxy_request(
        self,
        *,
        from_address: str,
        proxy_wallet: str,
        proxy_factory: str,
        relay_hub: str,
        relay_address: str,
        nonce: str,
        encoded_proxy_data: str,
        gas_limit: str,
        metadata: str,
    ) -> dict:
        from web3 import Web3

        from_address = Web3.to_checksum_address(from_address)
        proxy_wallet = Web3.to_checksum_address(proxy_wallet)
        proxy_factory = Web3.to_checksum_address(proxy_factory)
        relay_hub = Web3.to_checksum_address(relay_hub)
        relay_address = Web3.to_checksum_address(relay_address)

        struct_hash = self._build_proxy_struct_hash(
            from_address=from_address,
            proxy_factory=proxy_factory,
            encoded_proxy_data=encoded_proxy_data,
            gas_limit=gas_limit,
            nonce=nonce,
            relay_hub=relay_hub,
            relay_address=relay_address,
        )
        signature = self._sign_struct_hash(struct_hash)
        return {
            "type": "PROXY",
            "from": from_address,
            "to": proxy_factory,
            "proxyWallet": proxy_wallet,
            "data": encoded_proxy_data,
            "signature": signature,
            "signatureParams": {
                "gasPrice": "0",
                "gasLimit": gas_limit,
                "relayerFee": "0",
                "relayHub": relay_hub,
                "relay": relay_address,
            },
            "nonce": nonce,
            "metadata": metadata,
        }

    @staticmethod
    def _serialize_json_body(body: dict | str | None) -> str:
        """Serialize JSON body exactly once so signature and request payload match byte-for-byte."""
        if body is None:
            return ""
        if isinstance(body, str):
            return body
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    def _build_builder_headers(
        self,
        method: str,
        path: str,
        body: dict | str | None = None,
        *,
        timestamp: int | None = None,
    ) -> dict:
        ts = str(int(time.time()))
        if timestamp is not None:
            ts = str(int(timestamp))
        secret = self._config.builder_api_secret
        padding = "=" * ((4 - len(secret) % 4) % 4)
        secret_bytes = base64.urlsafe_b64decode(secret + padding)
        payload = f"{ts}{method}{path}"
        body_json = self._serialize_json_body(body)
        if body_json:
            payload += body_json
        sig = base64.urlsafe_b64encode(
            hmac.new(secret_bytes, payload.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")
        return {
            "POLY_BUILDER_API_KEY": self._config.builder_api_key,
            "POLY_BUILDER_TIMESTAMP": ts,
            "POLY_BUILDER_PASSPHRASE": self._config.builder_api_passphrase,
            "POLY_BUILDER_SIGNATURE": sig,
        }

    def _relayer_url(self, path: str) -> str:
        base = self._config.relayer_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    async def _get_relay_payload(self, signer_address: str) -> dict:
        path = RELAYER_ENDPOINTS["relay_payload"]
        resp = await self._client.get(
            self._relayer_url(path),
            params={"address": signer_address, "type": "PROXY"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_relay_payload_shape")
        return payload

    async def _wait_relayer_transaction(self, tx_id: str) -> dict | None:
        path = RELAYER_ENDPOINTS["transaction"]
        max_polls = max(1, int(self._config.redeem_relayer_max_polls))
        poll_ms = max(500, int(self._config.redeem_relayer_poll_ms))
        for _ in range(max_polls):
            resp = await self._client.get(self._relayer_url(path), params={"id": tx_id})
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list) and payload:
                row = payload[0]
                state = str(row.get("state") or "")
                if state in RELAYER_SUCCESS_STATES or state == RELAYER_FAIL_STATE:
                    return row
            await asyncio.sleep(poll_ms / 1000.0)
        return None

    async def _get_native_balance(self, *, cached: bool) -> float | None:
        """Get signer native MATIC balance with optional short-lived cache."""
        now = time.time()
        if (
            cached
            and self._native_balance_cache is not None
            and now - self._native_balance_cache_at <= self._native_balance_cache_ttl_sec
        ):
            return self._native_balance_cache

        from web3 import Web3

        attempts = self._rpc_attempts()
        last_err = "unknown"
        for attempt in range(attempts):
            if not self._init_web3():
                return None
            try:
                addr = self._account.address if self._account is not None else self._funder
                if not addr:
                    return None
                checksum = Web3.to_checksum_address(addr)
                bal_wei = self._w3.eth.get_balance(checksum)
                bal_matic = float(bal_wei) / 1e18
                self._native_balance_cache = bal_matic
                self._native_balance_cache_at = time.time()
                return bal_matic
            except Exception as e:
                last_err = str(e)
                retried = False
                if attempt < attempts - 1:
                    retried = self._failover_after_error(stage="native_balance", error=e)
                log.warning(
                    "native_balance_attempt_failed",
                    attempt=f"{attempt + 1}/{attempts}",
                    will_retry=retried,
                    error=last_err,
                    rpc_url=self._rpc_label(),
                )
                if retried:
                    continue
                break

        log.error("native_balance_error", error=last_err, rpc_url=self._rpc_label())
        return None

    async def _get_usdc_balance(self) -> float | None:
        """Get USDC balance via Web3."""
        from web3 import Web3

        attempts = self._rpc_attempts()
        last_err = "unknown"
        for attempt in range(attempts):
            if not self._init_web3():
                return None
            try:
                usdc = self._w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_ADDRESS), abi=USDC_ABI
                )
                funder_addr = Web3.to_checksum_address(self._funder)
                balance = usdc.functions.balanceOf(funder_addr).call()
                return balance / 1e6  # USDC has 6 decimals
            except Exception as e:
                last_err = str(e)
                retried = False
                if attempt < attempts - 1:
                    retried = self._failover_after_error(stage="balance", error=e)
                log.warning(
                    "usdc_balance_attempt_failed",
                    attempt=f"{attempt + 1}/{attempts}",
                    will_retry=retried,
                    error=last_err,
                    rpc_url=self._rpc_label(),
                )
                if retried:
                    continue
                break

        log.error("usdc_balance_error", error=last_err, rpc_url=self._rpc_label())
        return None

    async def close(self) -> None:
        await self._client.aclose()
        self._executor.shutdown(wait=False)
