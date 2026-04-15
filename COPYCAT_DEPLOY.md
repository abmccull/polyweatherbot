# Copycat Strategy Deploy Guide

## 1) Bot runtime env

Set these key env vars for copycat mode:

```bash
STRATEGY_MODE=copycat
COPY_ENABLED=true
COPY_POLL_SECONDS=10
COPY_FOLLOWUP_POLL_SECONDS=3
COPY_TRADE_MAX_AGE_SECONDS=300
COPY_SETTLE_MAX_HOURS=72
COPY_BASE_TICKET_USD=5
COPY_PROBATION_MULTIPLIER=0.60
COPY_GMANAS_MULTIPLIER=0.40
COPY_SLIPPAGE_ABS_CAP=0.03
COPY_SLIPPAGE_REL_CAP=0.05
COPY_MAX_MATCH_EXPOSURE_USD=10
COPY_MAX_OPEN_RISK_PCT=0.35
COPY_MIN_CASH_BUFFER_USD=20
COPY_DEDUP_ENABLED=true
COPY_OPPOSITE_POLICY=ignore_until_exit
LEADER_REFRESH_HOURS=24
LEADER_PROBATION_DAYS=14
REDEMPTION_INTERVAL=60

# Proxy-wallet auto redemption through Polymarket relayer
REDEEM_PROXY_VIA_RELAYER=true
RELAYER_URL=https://relayer-v2.polymarket.com
# Optional explicit builder creds (falls back to POLY_API_* if omitted)
# IMPORTANT: relayer auth requires matching key+secret+passphrase.
# A Builder key by itself will return 401 Unauthorized.
BUILDER_API_KEY=
BUILDER_API_SECRET=
BUILDER_API_PASSPHRASE=
REDEEM_RELAYER_GAS_LIMIT=500000
REDEEM_RELAYER_MAX_POLLS=30
REDEEM_RELAYER_POLL_MS=2000
```

Notes:
- In `relayer_proxy_wallet` mode, redemption is gasless and does not require MATIC reserve.
- In `onchain_bot` mode, bot uses direct chain redemption and enforces MATIC reserve.

Run:

```bash
python3 main.py --strategy-mode copycat
```

## 2) Monitoring API runtime

```bash
API_BEARER_TOKEN=replace_me \
DB_PATH=station_sniper.db \
./scripts/run_copycat_api.sh
```

## 3) Vercel dashboard env

In Vercel project settings:

- `API_BASE_URL=https://<your-server-domain-or-ip>:8000`
- `API_BEARER_TOKEN=<same-token-as-api>`

Deploy `dashboard/` as the project root.
