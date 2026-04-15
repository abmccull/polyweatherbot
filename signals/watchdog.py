"""Signal funnel watchdog to auto-apply safe expansion when starved."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import Config


@dataclass
class SignalWatchdogResult:
    """Outcome of a watchdog evaluation."""

    triggered: bool = False
    reason: str = ""
    actions: list[str] = field(default_factory=list)
    candidates: int = 0
    emitted: int = 0
    time_gate_ratio: float = 0.0
    exact_ratio: float = 0.0


class SignalWatchdog:
    """Applies staged config fallback actions when opportunities are too scarce."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._applied_actions: set[str] = set()

    def evaluate_and_apply(self, funnel_stats: dict[str, object]) -> SignalWatchdogResult:
        result = SignalWatchdogResult()
        if not self._config.signal_kpi_enabled:
            return result

        candidates = int(funnel_stats.get("candidates", 0) or 0)
        emitted = int(funnel_stats.get("emitted", 0) or 0)
        status_counts = funnel_stats.get("status_counts", {})
        if not isinstance(status_counts, dict):
            status_counts = {}

        result.candidates = candidates
        result.emitted = emitted

        if candidates < self._config.signal_kpi_min_candidates:
            result.reason = "insufficient_candidates"
            return result
        if emitted >= self._config.signal_kpi_min_emitted:
            result.reason = "kpi_healthy"
            return result

        time_blocked = int(status_counts.get("TIME_GATE_BLOCKED", 0) or 0)
        exact_skipped = int(status_counts.get("SKIP_EXACT", 0) or 0)
        non_exact_total = max(1, candidates - exact_skipped)
        time_gate_ratio = time_blocked / non_exact_total
        exact_ratio = exact_skipped / candidates if candidates > 0 else 0.0

        result.triggered = True
        result.reason = "starved_no_emissions"
        result.time_gate_ratio = time_gate_ratio
        result.exact_ratio = exact_ratio

        actions: list[str] = []
        if self._config.signal_kpi_auto_shadow_expand and "shadow_expand" not in self._applied_actions:
            if not self._config.shadow_expansion_enabled:
                self._config.shadow_expansion_enabled = True
                actions.append("enable_shadow_expansion")
            self._applied_actions.add("shadow_expand")

        if (
            self._config.signal_kpi_auto_adaptive_gates
            and time_gate_ratio >= self._config.signal_kpi_max_time_gate_ratio
            and "adaptive_gates" not in self._applied_actions
        ):
            if not self._config.adaptive_time_gates_enabled:
                self._config.adaptive_time_gates_enabled = True
                actions.append("enable_adaptive_time_gates")
            actions.append("accelerate_adaptive_time_gates")
            self._config.adaptive_time_gate_no_emit_cycles = min(
                self._config.adaptive_time_gate_no_emit_cycles,
                20,
            )
            self._applied_actions.add("adaptive_gates")

        if (
            self._config.signal_kpi_auto_exact_pilot
            and exact_ratio >= self._config.signal_kpi_max_exact_ratio
            and "exact_pilot" not in self._applied_actions
        ):
            self._config.shadow_exact_enabled = True
            actions.append("enable_shadow_exact_pilot")
            self._applied_actions.add("exact_pilot")

        result.actions = actions
        return result
