"""Pure utility rules for copycat trade filtering and sizing."""

from __future__ import annotations


def build_match_key(condition_id: str | None, event_slug: str | None, token_id: str) -> str:
    """Build a stable dedup key for one-match exposure lock."""
    if condition_id:
        return f"cond:{condition_id.strip().lower()}"
    if event_slug:
        return f"slug:{event_slug.strip().lower()}"
    return f"token:{token_id.strip().lower()}"


def compute_max_copy_price(
    leader_price: float,
    abs_cap: float,
    rel_cap: float,
) -> float:
    """Compute maximum follower entry price from absolute+relative caps.

    For low-priced contracts, absolute cents caps are more intuitive.
    For higher-priced contracts, relative caps prevent loose chasing.
    """
    if leader_price <= 0:
        return 0.0
    abs_max = leader_price + max(0.0, abs_cap)
    rel_max = leader_price * (1.0 + max(0.0, rel_cap))
    if leader_price <= 0.60:
        return abs_max
    return min(abs_max, rel_max)


def classify_duplicate_reason(existing_side: str, new_side: str) -> str:
    """Return a reason code when a match lock already exists."""
    es = (existing_side or "").upper()
    ns = (new_side or "").upper()
    if es and ns and es != ns:
        return "opposite_side_ignored_locked"
    return "same_match_scalein_ignored"
