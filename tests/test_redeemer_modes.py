from __future__ import annotations

import json

import pytest

from config import Config
from trading.redeemer import NEG_RISK_ADAPTER_CONTRACT, PositionRedeemer, USDC_ADDRESS


class _Acct:
    def __init__(self, address: str) -> None:
        self.address = address


@pytest.mark.asyncio
async def test_get_redeem_mode_onchain_when_not_proxy():
    cfg = Config(
        poly_private_key="0x" + "11" * 32,
        poly_funder_address="0x1111111111111111111111111111111111111111",
    )
    redeemer = PositionRedeemer(cfg)
    redeemer._account = _Acct("0x1111111111111111111111111111111111111111")  # noqa: SLF001
    async def _not_contract() -> bool:
        return False
    redeemer._is_funder_contract_wallet = _not_contract  # type: ignore[method-assign]

    mode = await redeemer.get_redeem_mode()
    assert mode == "onchain_bot"
    assert await redeemer.needs_native_gas_reserve() is True
    await redeemer.close()


@pytest.mark.asyncio
async def test_get_redeem_mode_manual_proxy_without_builder_creds():
    cfg = Config(
        poly_private_key="0x" + "22" * 32,
        poly_funder_address="0x3333333333333333333333333333333333333333",
        redeem_proxy_via_relayer=True,
        builder_api_key="",
        builder_api_secret="",
        builder_api_passphrase="",
    )
    redeemer = PositionRedeemer(cfg)
    redeemer._account = _Acct("0x4444444444444444444444444444444444444444")  # noqa: SLF001
    async def _is_contract() -> bool:
        return True
    redeemer._is_funder_contract_wallet = _is_contract  # type: ignore[method-assign]

    mode = await redeemer.get_redeem_mode()
    assert mode == "manual_proxy_wallet"
    assert await redeemer.needs_native_gas_reserve() is False
    await redeemer.close()


@pytest.mark.asyncio
async def test_get_redeem_mode_relayer_proxy_when_builder_creds_present():
    cfg = Config(
        poly_private_key="0x" + "33" * 32,
        poly_funder_address="0x5555555555555555555555555555555555555555",
        redeem_proxy_via_relayer=True,
        builder_api_key="key",
        builder_api_secret="c2VjcmV0",  # "secret" base64
        builder_api_passphrase="pass",
        chain_id=137,
    )
    redeemer = PositionRedeemer(cfg)
    redeemer._account = _Acct("0x6666666666666666666666666666666666666666")  # noqa: SLF001
    async def _is_contract() -> bool:
        return True
    redeemer._is_funder_contract_wallet = _is_contract  # type: ignore[method-assign]

    mode = await redeemer.get_redeem_mode()
    assert mode == "relayer_proxy_wallet"
    assert await redeemer.needs_native_gas_reserve() is False
    await redeemer.close()


@pytest.mark.asyncio
async def test_get_redeem_mode_manual_when_builder_key_missing_secret_passphrase():
    cfg = Config(
        poly_private_key="0x" + "44" * 32,
        poly_funder_address="0x7777777777777777777777777777777777777777",
        redeem_proxy_via_relayer=True,
        builder_api_key="builder-key-only",
        builder_api_secret="",
        builder_api_passphrase="",
        chain_id=137,
    )
    redeemer = PositionRedeemer(cfg)
    redeemer._account = _Acct("0x8888888888888888888888888888888888888888")  # noqa: SLF001

    mode = await redeemer.get_redeem_mode()
    assert mode == "manual_proxy_wallet"
    # Explicit key is preserved; no hidden credential replacement should occur.
    assert cfg.builder_api_key == "builder-key-only"
    await redeemer.close()


@pytest.mark.asyncio
async def test_get_redeem_mode_manual_after_relayer_unauthorized():
    cfg = Config(
        poly_private_key="0x" + "55" * 32,
        poly_funder_address="0x9999999999999999999999999999999999999999",
        redeem_proxy_via_relayer=True,
        builder_api_key="key",
        builder_api_secret="c2VjcmV0",
        builder_api_passphrase="pass",
        chain_id=137,
    )
    redeemer = PositionRedeemer(cfg)
    redeemer._account = _Acct("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")  # noqa: SLF001
    redeemer._relayer_auth_unauthorized = True  # noqa: SLF001

    mode = await redeemer.get_redeem_mode()
    assert mode == "manual_proxy_wallet"
    await redeemer.close()


@pytest.mark.asyncio
async def test_redeem_cycle_respects_relayer_submit_cap():
    cfg = Config(
        poly_private_key="0x" + "66" * 32,
        poly_funder_address="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        redeem_relayer_max_submits_per_cycle=2,
        redeem_relayer_submit_spacing_seconds=0.0,
    )
    redeemer = PositionRedeemer(cfg)

    async def _positions():
        return [
            {"conditionId": f"0x{i:064x}", "title": f"M{i}", "outcome": "Yes", "size": 5, "redeemable": True, "curPrice": 1}
            for i in range(4)
        ]

    async def _mode():
        return "relayer_proxy_wallet"

    calls: list[str] = []

    async def _redeem(
        condition_id: str,
        outcome: str,
        outcome_index: int | None = None,
        *,
        size: float = 0.0,
        negative_risk: bool = False,
    ):
        _ = (outcome, outcome_index, size, negative_risk)
        calls.append(condition_id)
        return True, f"0xtx{len(calls)}", None

    async def _balance():
        return 12.34

    redeemer._fetch_positions = _positions  # type: ignore[method-assign]
    redeemer.get_redeem_mode = _mode  # type: ignore[method-assign]
    redeemer._redeem_position_via_relayer = _redeem  # type: ignore[method-assign]
    redeemer._get_usdc_balance = _balance  # type: ignore[method-assign]

    result = await redeemer.check_and_redeem()
    assert result.redeemed == 2
    assert result.failed == 0
    assert len(calls) == 2
    await redeemer.close()


@pytest.mark.asyncio
async def test_relayer_429_backoff_skips_following_cycle():
    cfg = Config(
        poly_private_key="0x" + "77" * 32,
        poly_funder_address="0xcccccccccccccccccccccccccccccccccccccccc",
        redeem_relayer_429_backoff_initial_seconds=120,
        redeem_relayer_429_backoff_max_seconds=600,
    )
    redeemer = PositionRedeemer(cfg)

    async def _positions():
        return [
            {
                "conditionId": "0x" + "11" * 32,
                "title": "Winner",
                "outcome": "Yes",
                "size": 3,
                "redeemable": True,
                "curPrice": 1,
            }
        ]

    async def _mode():
        return "relayer_proxy_wallet"

    attempts = {"count": 0}

    async def _redeem(
        condition_id: str,
        outcome: str,
        outcome_index: int | None = None,
        *,
        size: float = 0.0,
        negative_risk: bool = False,
    ):
        _ = (condition_id, outcome, outcome_index, size, negative_risk)
        attempts["count"] += 1
        redeemer._mark_relayer_rate_limited()  # noqa: SLF001
        return False, None, "relayer_rate_limited_429"

    async def _balance():
        return 9.99

    redeemer._fetch_positions = _positions  # type: ignore[method-assign]
    redeemer.get_redeem_mode = _mode  # type: ignore[method-assign]
    redeemer._redeem_position_via_relayer = _redeem  # type: ignore[method-assign]
    redeemer._get_usdc_balance = _balance  # type: ignore[method-assign]

    first = await redeemer.check_and_redeem()
    assert first.failed == 1
    assert attempts["count"] == 1

    second = await redeemer.check_and_redeem()
    assert second.failed == 0
    assert attempts["count"] == 1
    assert second.usdc_balance == 9.99
    await redeemer.close()


@pytest.mark.asyncio
async def test_redeem_cycle_skips_stale_zero_balance_positions():
    cfg = Config(
        poly_private_key="0x" + "88" * 32,
        poly_funder_address="0xdddddddddddddddddddddddddddddddddddddddd",
    )
    redeemer = PositionRedeemer(cfg)

    async def _positions():
        return [
            {
                "conditionId": "0x" + "22" * 32,
                "title": "Settled Winner",
                "outcome": "Yes",
                "size": 4,
                "redeemable": True,
                "curPrice": 1,
                "asset": "1234",
            }
        ]

    async def _mode():
        return "relayer_proxy_wallet"

    calls = {"redeem": 0}

    async def _redeem(*args, **kwargs):
        _ = (args, kwargs)
        calls["redeem"] += 1
        return True, "0xtx", None

    async def _asset_balance(asset_id):
        assert asset_id == "1234"
        return 0

    async def _balance():
        return 7.77

    redeemer._fetch_positions = _positions  # type: ignore[method-assign]
    redeemer.get_redeem_mode = _mode  # type: ignore[method-assign]
    redeemer._redeem_position_via_relayer = _redeem  # type: ignore[method-assign]
    redeemer._get_position_asset_balance = _asset_balance  # type: ignore[method-assign]
    redeemer._get_usdc_balance = _balance  # type: ignore[method-assign]

    result = await redeemer.check_and_redeem()
    assert result.redeemed == 0
    assert result.failed == 0
    assert calls["redeem"] == 0
    assert result.usdc_balance == 7.77
    await redeemer.close()


@pytest.mark.asyncio
async def test_redeem_cycle_uses_positive_payout_not_curprice():
    cfg = Config(
        poly_private_key="0x" + "89" * 32,
        poly_funder_address="0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    )
    redeemer = PositionRedeemer(cfg)

    async def _positions():
        return [
            {
                "conditionId": "0x" + "33" * 32,
                "title": "Settled Winner",
                "outcome": "Yes",
                "outcomeIndex": 0,
                "size": 5,
                "redeemable": True,
                "curPrice": 0,
                "asset": "4321",
            }
        ]

    async def _mode():
        return "relayer_proxy_wallet"

    calls = {"redeem": 0, "balance": 0}

    async def _redeem(*args, **kwargs):
        _ = (args, kwargs)
        calls["redeem"] += 1
        return True, "0xtx", None

    async def _asset_balance(asset_id):
        assert asset_id == "4321"
        calls["balance"] += 1
        return 1_000_000 if calls["balance"] == 1 else 0

    async def _payout_ratio(condition_id, outcome_index):
        assert condition_id == "0x" + "33" * 32
        assert outcome_index == 0
        return 1, 1

    async def _balance():
        return 8.88

    redeemer._fetch_positions = _positions  # type: ignore[method-assign]
    redeemer.get_redeem_mode = _mode  # type: ignore[method-assign]
    redeemer._redeem_position_via_relayer = _redeem  # type: ignore[method-assign]
    redeemer._get_position_asset_balance = _asset_balance  # type: ignore[method-assign]
    redeemer._get_position_payout_ratio = _payout_ratio  # type: ignore[method-assign]
    redeemer._get_usdc_balance = _balance  # type: ignore[method-assign]

    result = await redeemer.check_and_redeem()
    assert result.redeemed == 1
    assert result.failed == 0
    assert calls["redeem"] == 1
    assert result.usdc_balance == 8.88
    await redeemer.close()


@pytest.mark.asyncio
async def test_fetch_positions_paginates_past_first_page():
    cfg = Config(
        poly_private_key="0x" + "90" * 32,
        poly_funder_address="0xffffffffffffffffffffffffffffffffffffffff",
    )
    redeemer = PositionRedeemer(cfg)

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, params=None):
            _ = url
            self.calls.append(params)
            offset = int(params["offset"])
            limit = int(params["limit"])
            if offset == 0:
                payload = [{"asset": f"a{i}", "conditionId": f"c{i}", "outcomeIndex": 0} for i in range(limit)]
            else:
                payload = [{"asset": f"b{i}", "conditionId": f"d{i}", "outcomeIndex": 1} for i in range(50)]
            return _Resp(payload)

        async def aclose(self):
            return None

    client = _Client()
    redeemer._client = client  # type: ignore[assignment]

    rows = await redeemer._fetch_positions()  # noqa: SLF001
    assert len(rows) == 250
    assert client.calls == [
        {"user": "0xffffffffffffffffffffffffffffffffffffffff", "limit": 200, "offset": 0},
        {"user": "0xffffffffffffffffffffffffffffffffffffffff", "limit": 200, "offset": 200},
    ]
    await redeemer.close()


def test_build_builder_headers_matches_sdk_reference_vector():
    cfg = Config(
        builder_api_key="019894b9-cb40-79c4-b2bd-6aecb6f8c6c5",
        builder_api_secret="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        builder_api_passphrase="1816e5ed89518467ffa78c65a2d6a62d240f6fd6d159cba7b2c4dc510800f75a",
    )
    redeemer = PositionRedeemer(cfg)
    body = (
        '{"deferExec":false,"order":{"salt":718139292476,"maker":"0x6e0c80c90ea6c15917308F820Eac91Ce2724B5b5",'
        '"signer":"0x6e0c80c90ea6c15917308F820Eac91Ce2724B5b5","taker":"0x0000000000000000000000000000000000000000",'
        '"tokenId":"15871154585880608648532107628464183779895785213830018178010423617714102767076","makerAmount":"5000000",'
        '"takerAmount":"10000000","side":"BUY","expiration":"0","nonce":"0","feeRateBps":"1000","signatureType":0,'
        '"signature":"0x64a2b097cf14f9a24403748b4060bedf8f33f3dbe2a38e5f85bc2a5f2b841af633a2afcc9c4d57e60e4ff1d58df2756b2ca469f984ecfd46cb0c8baba8a0d6411b"},'
        '"owner":"5d1c266a-ed39-b9bd-c1f5-f24ae3e14a7b","orderType":"GTC"}'
    )
    headers = redeemer._build_builder_headers("POST", "/order", body, timestamp=1758744060)  # noqa: SLF001
    assert headers["POLY_BUILDER_SIGNATURE"] == "8xh8d0qZHhBcLLYbsKNeiOW3Z0W2N5yNEq1kCVMe5QE="


def test_serialize_json_body_is_minified_and_stable():
    body_dict = {
        "type": "PROXY",
        "nonce": "123",
        "signatureParams": {"gasPrice": "0", "gasLimit": "500000"},
    }
    body_json = PositionRedeemer._serialize_json_body(body_dict)
    assert body_json == json.dumps(body_dict, separators=(",", ":"), ensure_ascii=False)
    assert " " not in body_json


def test_outcome_to_index_sets_prefers_outcome_index():
    assert PositionRedeemer._outcome_to_index_sets("No", 0) == [1, 2]
    assert PositionRedeemer._outcome_to_index_sets("Yes", 1) == [1, 2]


def test_encode_redeem_calldata_matches_ctf_signature():
    pytest.importorskip("eth_abi")
    cfg = Config(
        poly_private_key="0x" + "99" * 32,
        poly_funder_address="0x9999999999999999999999999999999999999999",
    )
    redeemer = PositionRedeemer(cfg)
    condition_id = "0x" + "12" * 32

    calldata = redeemer._encode_redeem_calldata(condition_id, "Over", 1)  # noqa: SLF001
    assert calldata.startswith("0x01b7037c")
    assert "000000000000000000000000" + USDC_ADDRESS[2:].lower() in calldata.lower()
    assert condition_id[2:].lower() in calldata.lower()
    # Binary redemption uses the full 1|2 partition for YES/NO markets.
    assert calldata.lower().endswith("0000000100000000000000000000000000000000000000000000000000000002")


def test_neg_risk_adapter_contract_matches_official_polygon_address():
    assert NEG_RISK_ADAPTER_CONTRACT == "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"


def test_should_use_neg_risk_adapter_for_all_negative_risk_outcomes():
    assert PositionRedeemer._should_use_neg_risk_adapter(  # noqa: SLF001
        negative_risk=True,
        outcome="No",
        outcome_index=1,
    ) is True
    assert PositionRedeemer._should_use_neg_risk_adapter(  # noqa: SLF001
        negative_risk=True,
        outcome="Yes",
        outcome_index=0,
    ) is True
    assert PositionRedeemer._should_use_neg_risk_adapter(  # noqa: SLF001
        negative_risk=False,
        outcome="No",
        outcome_index=1,
    ) is False


def test_neg_risk_amounts_encode_size_in_outcome_slot():
    amounts_no = PositionRedeemer._neg_risk_amounts(  # noqa: SLF001
        size=6.38,
        outcome="No",
        outcome_index=1,
    )
    assert amounts_no == [0, 6_380_000]

    amounts_yes = PositionRedeemer._neg_risk_amounts(  # noqa: SLF001
        size=2.5,
        outcome="Yes",
        outcome_index=0,
    )
    assert amounts_yes == [2_500_000, 0]
