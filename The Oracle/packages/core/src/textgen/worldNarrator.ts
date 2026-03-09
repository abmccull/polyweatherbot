import type { WorldHistoryEvent } from "../state/worldHistory";
import { renderTemplate } from "./templates";

/**
 * Generates a rich narrative description for a world history event.
 */
export function narrateWorldEvent(
  event: WorldHistoryEvent,
  factions: Record<string, { name: string }>,
  seed: number
): string {
  const factionNames = event.factionIds
    .map((id) => factions[id]?.name ?? id)
    .join(" and ");

  const kindNarrative: Record<string, string> = {
    diplomatic_shift: "The diplomatic landscape shifts",
    alliance_formed: `${factionNames} forge a new alliance`,
    alliance_broken: `The alliance involving ${factionNames} shatters`,
    trade_agreement: `${factionNames} open new trade routes between their lands`,
    embargo: `${factionNames} close their markets in bitter embargo`,
    conflict_started: `War erupts as ${factionNames} take up arms`,
    conflict_resolved: `The conflict involving ${factionNames} comes to its weary end`,
    hegemon_emerged: `${factionNames} rise to hegemonic dominance over the Greek world`,
    hegemon_declined: `The hegemony of ${factionNames} crumbles into memory`,
    revolution: `Revolution engulfs ${factionNames}`,
    regime_change: `A new regime seizes power in ${factionNames}`,
    philosophy_spread: `Philosophical teachings spread through the agora of ${factionNames}`,
    pilgrimage_surge: "A great wave of pilgrims floods the sacred way",
    oracle_triumph: "The oracle's prophecy is vindicated before all of Greece",
    oracle_disgrace: "Whispers of the oracle's failure spread like wildfire",
    crisis_escalation: `Crisis deepens in the lands of ${factionNames}`,
    crisis_resolution: `The crisis involving ${factionNames} finds resolution`
  };

  const eventLine = kindNarrative[event.kind] ?? event.title;
  const base = renderTemplate("event-general", {
    action: eventLine
  }, seed);

  return base || `${eventLine}. ${event.description}`;
}

/**
 * Generates atmospheric narration for an age transition.
 */
export function narrateAgeTransition(
  fromAge: string,
  toAge: string,
  seed: number
): string {
  const context = {
    description: `The ${fromAge} gives way to the ${toAge}, and the world will never be the same.`
  };
  return renderTemplate("age-transition", context, seed);
}

/**
 * Generates narration for a faction revolution event.
 */
export function narrateRevolution(
  factionName: string,
  seed: number
): string {
  return renderTemplate("revolution", { faction: factionName }, seed);
}
