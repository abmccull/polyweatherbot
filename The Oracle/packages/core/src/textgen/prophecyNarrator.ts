import type { ProphecyRecord } from "../state/gameState";
import type { ProphecyContradiction, ProphecyArc, ProphecyArcMilestone } from "../state/prophecy";
import { renderTemplate } from "./templates";

/**
 * Generates atmospheric narration for a prophecy delivery event.
 */
export function narrateProphecyDelivery(
  prophecy: ProphecyRecord,
  factionName: string,
  seed: number
): string {
  const depthAdjective =
    prophecy.depthBand === "oracular" ? "transcendent"
    : prophecy.depthBand === "deep" ? "profound"
    : prophecy.depthBand === "grounded" ? "measured"
    : "murmured";

  const base = renderTemplate("prophecy-delivery", { faction: factionName }, seed);
  const suffix = `The ${depthAdjective} words settle into memory: "${prophecy.text}"`;
  return `${base} ${suffix}`;
}

/**
 * Generates dramatic narration for a detected prophecy contradiction.
 */
export function narrateContradiction(
  contradiction: ProphecyContradiction,
  seed: number
): string {
  const severityLabel =
    contradiction.severity === "catastrophic" ? "catastrophic"
    : contradiction.severity === "major" ? "grave"
    : "troubling";

  const base = renderTemplate("contradiction-detected", {}, seed);
  return `${base} A ${severityLabel} contradiction: ${contradiction.description}.`;
}

/**
 * Generates narration for a prophecy arc milestone completion.
 */
export function narrateArcMilestone(
  arc: ProphecyArc,
  milestone: ProphecyArcMilestone,
  seed: number
): string {
  const outcomeNote = milestone.outcome
    ? ` ${milestone.outcome}`
    : "";

  const base = renderTemplate("arc-milestone", {}, seed);
  return `${base} The "${arc.label}" arc reaches its milestone: ${milestone.label}.${outcomeNote}`;
}
