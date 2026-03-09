import {
  advisorDefs,
  buildingDefs,
  consultationQuestions,
  omenDefs,
  philosopherThreatDefById,
  politicalEventDefs,
  rivalOracleOperationDefById,
  wordTiles
} from "@the-oracle/content";

import type {
  AdvisorMessage,
  BuildingDefId,
  ConsultationCurrent,
  ConsequenceCase,
  EventFeedItem,
  FactionAgenda,
  FactionState,
  FactionId,
  GameState,
  OmenReport,
  PhilosopherThreatState,
  PythiaState,
  ResourceId,
  SemanticAction,
  SemanticPolarity,
  SemanticTarget,
  TileSemantics,
  TradeOffer,
  WordTile
} from "../state/gameState";
import type { WorldHistoryState, FactionAlliance, WorldHistoryEvent } from "../state/worldHistory";
import { createWorldHistoryEvent, evaluateHegemon } from "../state/worldHistory";
import { getAbsoluteDay } from "./clock";
import { narrateWorldEvent } from "../textgen/worldNarrator";
import { advancePhilosopherThreats } from "./philosophers";

function hash(seed: number): number {
  let value = seed + 0x6d2b79f5;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function pick<T>(items: T[], seed: number): T {
  return items[Math.floor(hash(seed) * items.length) % items.length]!;
}

type ScoreContext = number | Pick<PythiaState, "attunement" | "mentalClarity" | "tranceDepth" | "needs" | "traits">;
type QuestionDef = typeof consultationQuestions[number];
type OmenDef = typeof omenDefs[number];

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function normalizeScoreContext(context: ScoreContext) {
  if (typeof context === "number") {
    return {
      attunement: context,
      mentalClarity: 60,
      tranceDepth: 55,
      needs: {
        purification: 35,
        rest: 25,
        pilgrimageCooldown: 0
      },
      traits: [] as PythiaState["traits"]
    };
  }

  return context;
}

function hasTrait(pythia: Pick<PythiaState, "traits">, trait: PythiaState["traits"][number]): boolean {
  return pythia.traits.includes(trait);
}

function questionHasTag(question: QuestionDef, tag: string): boolean {
  return question.tags?.includes(tag) ?? false;
}

function roleLabelToId(roleLabel: string): string {
  return roleLabel.toLowerCase().replace(/\s+/g, "_");
}

function chooseConsultationTarget(question: QuestionDef, faction: FactionState, day: number): SemanticTarget {
  if (questionHasTag(question, "fleet") || questionHasTag(question, "harbor")) {
    return "fleet";
  }
  if (questionHasTag(question, "army") || questionHasTag(question, "frontier")) {
    return "army";
  }
  if (questionHasTag(question, "king") || questionHasTag(question, "succession") || questionHasTag(question, "court")) {
    return "king";
  }
  if (questionHasTag(question, "city") || questionHasTag(question, "health")) {
    return "city";
  }
  if (questionHasTag(question, "oracle") || questionHasTag(question, "pilgrims") || questionHasTag(question, "prestige")) {
    return "oracle";
  }
  if (questionHasTag(question, "harvest") || questionHasTag(question, "grain")) {
    return "harvest";
  }
  if (questionHasTag(question, "treasury") || questionHasTag(question, "levy") || questionHasTag(question, "trade")) {
    return "treasury";
  }
  if (questionHasTag(question, "alliance") || questionHasTag(question, "marriage") || questionHasTag(question, "peace") || questionHasTag(question, "diplomacy")) {
    return "alliance";
  }

  if (question.domain === "economic") {
    return day % 2 === 0 ? "treasury" : "harvest";
  }
  if (question.domain === "spiritual") {
    return faction.currentAgenda === "succession" ? "king" : "oracle";
  }
  return faction.currentAgenda === "war" ? "army" : "fleet";
}

function chooseConsultationAction(question: QuestionDef, faction: FactionState, day: number): SemanticAction {
  if (questionHasTag(question, "trade") || questionHasTag(question, "grain") || questionHasTag(question, "harvest")) {
    return faction.debt >= 18 || questionHasTag(question, "levy") ? "withhold" : "prosper";
  }
  if (questionHasTag(question, "marriage") || questionHasTag(question, "legitimacy") || questionHasTag(question, "oath") || questionHasTag(question, "peace")) {
    return faction.currentAgenda === "succession" || day % 3 === 0 ? "fracture" : "endure";
  }
  if (questionHasTag(question, "fleet") || questionHasTag(question, "army") || questionHasTag(question, "campaign")) {
    return faction.credibility >= 52 ? "triumph" : "fall";
  }
  if (questionHasTag(question, "purification") || questionHasTag(question, "rites") || questionHasTag(question, "pilgrims")) {
    return faction.credibility < 46 ? "withhold" : "endure";
  }

  if (question.domain === "economic") {
    return faction.debt >= 20 ? "withhold" : "prosper";
  }
  if (question.domain === "spiritual") {
    return faction.currentAgenda === "succession" ? "fracture" : "endure";
  }
  return day % 2 === 0 ? "triumph" : "fall";
}

function chooseConsultationPolarity(question: QuestionDef, faction: FactionState): SemanticPolarity {
  if (questionHasTag(question, "timing") || questionHasTag(question, "diplomacy") || questionHasTag(question, "treaty")) {
    return "double";
  }
  if (faction.debt >= 22 || faction.credibility < 42) {
    return "warning";
  }
  return question.domain === "spiritual" ? "double" : "favorable";
}

function chooseConsultationAmbiguity(question: QuestionDef): TileSemantics["ambiguity"] {
  if (questionHasTag(question, "succession") || questionHasTag(question, "health") || questionHasTag(question, "levy")) {
    return "specific";
  }
  if (question.domain === "spiritual" || questionHasTag(question, "rites")) {
    return "cryptic";
  }
  return "balanced";
}

function chooseConsultationTimeHorizon(question: QuestionDef): TileSemantics["timeHorizon"] {
  if (questionHasTag(question, "timing") || questionHasTag(question, "river") || questionHasTag(question, "harbor")) {
    return "immediate";
  }
  if (questionHasTag(question, "legitimacy") || questionHasTag(question, "succession") || questionHasTag(question, "prestige")) {
    return "yearly";
  }
  return "seasonal";
}

function buildConsultationSemantics(question: QuestionDef, faction: FactionState, day: number): TileSemantics {
  return {
    target: chooseConsultationTarget(question, faction, day),
    action: chooseConsultationAction(question, faction, day),
    polarity: chooseConsultationPolarity(question, faction),
    ambiguity: chooseConsultationAmbiguity(question),
    timeHorizon: chooseConsultationTimeHorizon(question),
    domain: question.domain
  };
}

function bestPriestSkillForRole(state: GameState, roleLabel: string): number {
  const roleId = roleLabelToId(roleLabel);
  const exactMatch = state.priests.find((priest) => priest.role === roleId);
  if (exactMatch) {
    return exactMatch.skill + 8;
  }

  const matchingDomainPriest = state.priests
    .map((priest) => priest.skill + (priest.role === "attendant" ? 3 : 0))
    .sort((left, right) => right - left)[0];

  return matchingDomainPriest ?? 52;
}

function buildOmenSemantics(base: TileSemantics, omen: OmenDef, day: number, index: number): TileSemantics {
  const shift = (day + index) % 3;

  switch (omen.id) {
    case "ornithomancy":
      return {
        ...base,
        target: base.domain === "military" ? (shift === 0 ? base.target : "fleet") : shift === 1 ? "alliance" : base.target,
        action: base.domain === "military" ? (shift === 2 ? "fall" : "triumph") : shift === 1 ? "endure" : base.action,
        ambiguity: "balanced",
        timeHorizon: shift === 0 ? "immediate" : "seasonal"
      };
    case "pyromancy":
      return {
        ...base,
        target: shift === 0 ? "oracle" : base.target,
        action: base.polarity === "warning" ? "withhold" : shift === 2 ? "endure" : base.action,
        ambiguity: "balanced",
        timeHorizon: base.domain === "spiritual" ? "yearly" : base.timeHorizon
      };
    case "hydromancy":
      return {
        ...base,
        target: base.domain === "economic" ? (shift === 0 ? "harvest" : "treasury") : shift === 1 ? "city" : base.target,
        action: base.domain === "economic" ? (shift === 2 ? "withhold" : "prosper") : shift === 1 ? "endure" : base.action,
        ambiguity: "balanced",
        timeHorizon: "seasonal",
        domain: base.domain === "spiritual" ? "economic" : base.domain
      };
    case "extispicy":
      return {
        ...base,
        action: base.polarity === "warning" ? "fall" : base.action,
        ambiguity: "specific",
        timeHorizon: shift === 0 ? "immediate" : "seasonal"
      };
    case "oneiromancy":
      return {
        ...base,
        target: shift === 0 ? "king" : shift === 1 ? "oracle" : base.target,
        action: shift === 2 ? "fracture" : base.action,
        polarity: "double",
        ambiguity: "cryptic",
        timeHorizon: base.domain === "spiritual" ? "yearly" : base.timeHorizon
      };
    case "astromancy":
      return {
        ...base,
        target: base.domain === "military" ? (shift === 0 ? "fleet" : "army") : shift === 1 ? "king" : base.target,
        action: shift === 2 ? "endure" : base.action,
        ambiguity: "balanced",
        timeHorizon: base.timeHorizon === "immediate" ? "seasonal" : "yearly"
      };
    case "chresmology":
      return {
        ...base,
        target: shift === 0 ? "oracle" : shift === 1 ? "alliance" : base.target,
        action: shift === 2 ? "withhold" : base.action,
        polarity: "double",
        ambiguity: base.domain === "spiritual" ? "cryptic" : "balanced",
        timeHorizon: "yearly"
      };
    default:
      return base;
  }
}

function scoreOmenRelevance(omen: OmenDef, base: TileSemantics, state: GameState, question: QuestionDef): number {
  let score = 0;
  if (omen.family === base.domain) {
    score += 6;
  }
  if (omen.family === "ritual") {
    score += question.domain === "spiritual" ? 5 : 3;
  }
  if (questionHasTag(question, "timing") && (omen.id === "astromancy" || omen.id === "ornithomancy")) {
    score += 3;
  }
  if (questionHasTag(question, "sacrifice") && omen.id === "extispicy") {
    score += 5;
  }
  if (questionHasTag(question, "legitimacy") && (omen.id === "oneiromancy" || omen.id === "chresmology")) {
    score += 4;
  }
  if (questionHasTag(question, "trade") && omen.id === "hydromancy") {
    score += 3;
  }
  score += bestPriestSkillForRole(state, omen.priestRole) * 0.04;
  return score;
}

function buildOmenReports(state: GameState, question: QuestionDef, base: TileSemantics): OmenReport[] {
  const pythia = normalizeScoreContext(state.pythia);
  const strain = pythia.needs.rest * 0.11 + pythia.needs.purification * 0.1;
  const visionaryBonus = hasTrait(pythia, "visionary") ? 4 : 0;
  const calculatingBonus = hasTrait(pythia, "calculating") ? 3 : 0;
  const fragilePenalty = hasTrait(pythia, "fragile") ? Math.max(0, (pythia.needs.rest + pythia.needs.purification - 105) * 0.08) : 0;

  const selectedOmens = [...omenDefs]
    .sort((left, right) => {
      const score = scoreOmenRelevance(right, base, state, question) - scoreOmenRelevance(left, base, state, question);
      if (score !== 0) {
        return score;
      }
      const leftRoll = hash(state.worldSeed + state.clock.day * 31 + left.id.length);
      const rightRoll = hash(state.worldSeed + state.clock.day * 31 + right.id.length);
      if (leftRoll !== rightRoll) {
        return rightRoll - leftRoll;
      }
      return left.id.localeCompare(right.id);
    })
    .slice(0, 4);

  return selectedOmens.map<OmenReport>((omen, index) => {
    const omenSemantics = buildOmenSemantics(base, omen, state.clock.day, index);
    const priestSkill = bestPriestSkillForRole(state, omen.priestRole);
    const tranceBonus = pythia.tranceDepth * (omenSemantics.ambiguity === "cryptic" ? 0.14 : 0.1);
    const clarityBonus = pythia.mentalClarity * (omenSemantics.ambiguity === "specific" ? 0.13 : 0.1);
    const baseReliability =
      34
      + pythia.attunement * 0.18
      + priestSkill * 0.16
      + tranceBonus
      + clarityBonus
      + visionaryBonus
      + calculatingBonus
      - strain
      - fragilePenalty
      + (index === 0 ? 3 : 0);

    return {
      id: `omen-${state.clock.day}-${omen.id}-${index}`,
      sourceRole: omen.priestRole,
      text: omen.templates[(state.clock.day + state.clock.month + index * 2) % omen.templates.length]!,
      semantics: omenSemantics,
      reliability: Math.round(clamp(baseReliability, 28, 92) * 10) / 10
    };
  });
}

function clampStat(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function clampDebt(value: number): number {
  return Math.max(0, Math.min(40, Math.round(value)));
}

function clampRelation(value: number): number {
  return Math.max(-100, Math.min(100, Math.round(value)));
}

function getRelation(factions: Record<FactionId, FactionState>, sourceId: FactionId, targetId: FactionId): number {
  return factions[sourceId]?.relations[targetId] ?? 0;
}

export function applyBilateralRelationDelta(
  factions: Record<FactionId, FactionState>,
  sourceId: FactionId,
  targetId: FactionId,
  delta: number
): Record<FactionId, FactionState> {
  if (sourceId === targetId || delta === 0 || !factions[sourceId] || !factions[targetId]) {
    return factions;
  }

  return {
    ...factions,
    [sourceId]: {
      ...factions[sourceId],
      relations: {
        ...factions[sourceId].relations,
        [targetId]: clampRelation((factions[sourceId].relations[targetId] ?? 0) + delta)
      }
    },
    [targetId]: {
      ...factions[targetId],
      relations: {
        ...factions[targetId].relations,
        [sourceId]: clampRelation((factions[targetId].relations[sourceId] ?? 0) + delta)
      }
    }
  };
}

function appendHistory(history: string[], entry: string): string[] {
  return [entry, ...history].slice(0, 4);
}

function uniqueFactionIds(ids: FactionId[]): FactionId[] {
  return [...new Set(ids)].sort((left, right) => left.localeCompare(right));
}

function sameFactionList(left: FactionId[], right: FactionId[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((entry, index) => entry === right[index]);
}

function describeFactionList(factions: Record<FactionId, FactionState>, ids: FactionId[]): string {
  return ids.map((factionId) => factions[factionId].name).join(", ");
}

function appendOutcomeNote(faction: FactionState, note: string): FactionState {
  return {
    ...faction,
    lastOutcome: `${faction.lastOutcome ?? ""} ${note}`.trim(),
    history: appendHistory(faction.history, note)
  };
}

function factionNarrative(faction: FactionState): string {
  return `${faction.lastOutcome ?? ""} ${faction.history.join(" ")}`.toLowerCase();
}

function averageRelation(faction: FactionState): number {
  const values = Object.values(faction.relations);
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function factionInfluenceScore(faction: FactionState): number {
  return (
    faction.credibility * 0.38
    + faction.favour * 0.28
    + faction.treaties.length * 7
    + (faction.tradeAccess ? 9 : -4)
    - faction.embargoes.length * 6
    - faction.activeConflicts.length * 5
    - faction.debt * 1.1
    - faction.dependence * 0.18
    + averageRelation(faction) * 0.2
  );
}

function consultationStakeScore(faction: FactionState): number {
  const narrative = factionNarrative(faction);
  return (
    faction.debt * 0.8
    + faction.activeConflicts.length * 18
    + Math.max(0, 54 - faction.credibility) * 1.1
    + Math.max(0, 52 - faction.favour) * 0.8
    + Math.max(0, faction.dependence - 30) * 0.4
    + (narrative.includes("dominant bloc") ? 12 : 0)
    + (narrative.includes("government crisis") ? 14 : 0)
    + (narrative.includes("revolutionary agitation") ? 20 : 0)
    + (narrative.includes("creed") || narrative.includes("revival") || narrative.includes("legitimacy") ? 8 : 0)
  );
}

function rotateAgenda(agenda: FactionAgenda, offset: -1 | 1): FactionAgenda {
  const cycle: FactionAgenda[] = ["war", "trade", "faith", "succession"];
  const index = cycle.indexOf(agenda);
  const nextIndex = (index + offset + cycle.length) % cycle.length;
  return cycle[nextIndex] ?? agenda;
}

function agendaDomain(agenda: FactionAgenda): TileSemantics["domain"] {
  if (agenda === "trade") {
    return "economic";
  }
  if (agenda === "faith") {
    return "spiritual";
  }
  return "military";
}

function factionAgendaToSemantics(agenda: FactionAgenda, factionId: FactionId, day: number): TileSemantics {
  const byAgenda: Record<FactionAgenda, TileSemantics> = {
    war: {
      target: day % 2 === 0 ? "army" : "fleet",
      action: day % 3 === 0 ? "fall" : "triumph",
      polarity: day % 3 === 0 ? "warning" : "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    },
    trade: {
      target: day % 2 === 0 ? "treasury" : "alliance",
      action: day % 3 === 0 ? "withhold" : "prosper",
      polarity: day % 3 === 0 ? "warning" : "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    },
    faith: {
      target: factionId === "macedon" ? "king" : "oracle",
      action: day % 2 === 0 ? "endure" : "fracture",
      polarity: day % 2 === 0 ? "favorable" : "warning",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    },
    succession: {
      target: "king",
      action: day % 2 === 0 ? "fracture" : "endure",
      polarity: "double",
      ambiguity: "specific",
      timeHorizon: "yearly",
      domain: "military"
    }
  };

  return byAgenda[agenda];
}

export function generateAdvisorMessages(state: GameState): AdvisorMessage[] {
  const messages: AdvisorMessage[] = [];
  const absoluteDay = getAbsoluteDay(state.clock);
  const bloc = dominantBlocSummary(state.factions);
  const destabilized = Object.values(state.factions)
    .filter((faction) => factionNarrative(faction).includes("government crisis") || factionNarrative(faction).includes("revolutionary agitation"))
    .sort((left, right) => (right.debt + right.activeConflicts.length * 10) - (left.debt + left.activeConflicts.length * 10) || left.id.localeCompare(right.id));

  if (state.resources.incense.amount < 6) {
    messages.push({
      id: `advisor-incense-${state.clock.day}`,
      advisorId: "treasurer",
      severity: "critical",
      text: "The incense coffers are thinning. Another envoy would strip the sanctum bare."
    });
  }

  const brazier = state.buildings.find((building) => building.defId === "eternal_flame_brazier");
  if (brazier && brazier.storedResources.olive_oil !== undefined && brazier.storedResources.olive_oil < 2) {
    messages.push({
      id: `advisor-brazier-${state.clock.day}`,
      advisorId: "hierophant",
      severity: "critical",
      text: "The Eternal Flame gutters. Refill the brazier or suffer a terrible loss of face."
    });
  }

  if (state.pythia.needs.rest > 70) {
    messages.push({
      id: `advisor-rest-${state.clock.day}`,
      advisorId: "hierophant",
      severity: "warn",
      text: "The Pythia grows hollow-eyed. Rest her before the next consultation."
    });
  }

  if (state.pythia.needs.purification > 68) {
    messages.push({
      id: `advisor-purification-${state.clock.day}`,
      advisorId: "hierophant",
      severity: "warn",
      text: "The Pythia is ritually unclean. Purification should come before the god is questioned again."
    });
  }

  if (state.consultation.mode === "pending") {
    messages.push({
      id: `advisor-envoy-${state.clock.day}`,
      advisorId: "diplomat",
      severity: "info",
      text: "A delegation waits beyond the Sacred Way. They will not appreciate delay."
    });
  }

  const sanctum = state.buildings.find((building) => building.defId === "inner_sanctum");
  if (sanctum && (sanctum.storedResources.sacred_water ?? 0) < 1.5) {
    messages.push({
      id: `advisor-water-${state.clock.day}`,
      advisorId: "hierophant",
      severity: "warn",
      text: "The Inner Sanctum is short of sacred water. Purification rites will suffer."
    });
  }

  if (state.tradeOffers.length === 0 && state.clock.month > 1) {
    messages.push({
      id: `advisor-trade-${state.clock.day}`,
      advisorId: "shadow",
      severity: "warn",
      text: "The caravan routes are quiet. We may need to lean on favors to restore trade."
    });
  }

  const closedRoutes = Object.values(state.factions).filter((faction) => !faction.tradeAccess).length;
  if (closedRoutes >= 4) {
    messages.push({
      id: `advisor-routes-${state.clock.day}`,
      advisorId: "shadow",
      severity: "warn",
      text: "Too many caravan lanes are shuttered. A wider political thaw is needed before trade normalizes."
    });
  }

  const embattledDiplomacy = Object.values(state.factions).filter((faction) => faction.embargoes.length >= 2);
  if (embattledDiplomacy.length >= 2) {
    messages.push({
      id: `advisor-embargo-${state.clock.day}`,
      advisorId: "diplomat",
      severity: "warn",
      text: "Embargo blocs are hardening across Hellas. Even friendly cities may tighten terms if this spreads."
    });
  }

  const recentBacklash = [...state.consultation.history]
    .filter((entry) => entry.resolved && (entry.credibilityDelta ?? 0) < 0 && entry.resolvedDay !== undefined)
    .sort((left, right) => (right.resolvedDay ?? 0) - (left.resolvedDay ?? 0))[0];
  if (recentBacklash && absoluteDay - (recentBacklash.resolvedDay ?? absoluteDay) <= 5) {
    messages.push({
      id: `advisor-backlash-${state.clock.day}`,
      advisorId: "diplomat",
      severity: "warn",
      text: `${state.factions[recentBacklash.factionId].name} still remembers a failed reading. Another misstep will cost us dearly.`
    });
  }

  if (bloc.leaderId) {
    messages.push({
      id: `advisor-hegemon-${state.clock.day}`,
      advisorId: "shadow",
      severity: bloc.margin >= 12 ? "critical" : "warn",
      text: `${state.factions[bloc.leaderId].name} is gathering a dominant bloc. Caravan access will increasingly move on that city's pleasure.`
    });
  }

  if (destabilized.length > 0) {
    messages.push({
      id: `advisor-destabilization-${state.clock.day}`,
      advisorId: "diplomat",
      severity: "warn",
      text: `${destabilized[0]!.name} is sliding into open regime instability. Expect envoys to arrive with sharper demands and thinner patience.`
    });
  }

  const philosopherState = state.philosophers;
  const philosopherFlashpoint = philosopherState?.spotlightFactionIds
    .map((factionId) => ({
      faction: state.factions[factionId],
      threat: philosopherState.byFaction[factionId]
    }))
    .filter((entry) =>
      Boolean(entry.faction)
      && Boolean(entry.threat)
      && (entry.threat.stage === "sect" || entry.threat.stage === "crisis" || entry.threat.pressure >= 68)
    )
    .sort((left, right) => right.threat.pressure - left.threat.pressure || left.faction.id.localeCompare(right.faction.id))[0];
  if (philosopherFlashpoint) {
    const definition = philosopherThreatDefById[philosopherFlashpoint.threat.philosopherId];
    messages.push({
      id: `advisor-philosopher-threat-${state.clock.day}`,
      advisorId: "hierophant",
      severity: philosopherFlashpoint.threat.stage === "crisis" ? "critical" : "warn",
      text: `${philosopherFlashpoint.faction.name} is incubating ${definition.name}'s doctrine. Expect consultations to turn on legitimacy, rites, and civic order.`
    });
  } else {
    const ideologicalFlashpoint = Object.values(state.factions)
      .filter((faction) => factionNarrative(faction).includes("revival") || factionNarrative(faction).includes("creed") || factionNarrative(faction).includes("legitimacy"))
      .sort((left, right) => consultationStakeScore(right) - consultationStakeScore(left) || left.id.localeCompare(right.id))[0];
    if (ideologicalFlashpoint) {
      messages.push({
        id: `advisor-philosophy-${state.clock.day}`,
        advisorId: "hierophant",
        severity: "info",
        text: `${ideologicalFlashpoint.name} is turning its politics into doctrine. Expect consultations about legitimacy, rites, and civic order.`
      });
    }
  }

  const rivalFlashpoint = (state.rivalOracles?.spotlightRivalIds.length
    ? state.rivalOracles.spotlightRivalIds
        .map((rivalId) => state.rivalOracles?.roster.find((entry) => entry.id === rivalId))
        .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry))
    : state.rivalOracles?.roster ?? [])
    .filter((rival) => rival.pressure >= 56 || rival.visibility >= 44 || rival.intel >= 44)
    .sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))[0];
  if (rivalFlashpoint) {
    const patronName = rivalFlashpoint.intel >= 64
      ? state.factions[rivalFlashpoint.patronFactionId]?.name ?? "an obscured patron"
      : "an obscured patron";
    const operationLabel = rivalFlashpoint.lastKnownOperationId
      ? rivalOracleOperationDefById[rivalFlashpoint.lastKnownOperationId]?.label.toLowerCase() ?? "pressure work"
      : "shadow work";
    messages.push({
      id: `advisor-rival-oracle-${state.clock.day}`,
      advisorId: "shadow",
      severity: rivalFlashpoint.pressure >= 74 ? "critical" : "warn",
      text: `${rivalFlashpoint.visibility >= 42 ? rivalFlashpoint.name : "A rival oracle"} is building pressure through ${operationLabel} under ${patronName}.`
    });
  }

  // Seasonal warnings
  const season = state.clock.season;
  if (season === "Winter" && state.resources.grain.amount < 15) {
    messages.push({
      id: `advisor-winter-grain-${state.clock.day}`,
      advisorId: "treasurer",
      severity: "critical",
      text: "Winter bites and the granaries thin. Without grain reserves, the precinct faces hunger."
    });
  }
  if (season === "Summer" && state.resources.sacred_water.amount < 10) {
    messages.push({
      id: `advisor-summer-water-${state.clock.day}`,
      advisorId: "hierophant",
      severity: "warn",
      text: "The summer heat dries the spring. Sacred water production will suffer until the rains return."
    });
  }

  // Building condition warnings
  for (const building of state.buildings) {
    const ratio = building.condition / building.maxCondition;
    if (ratio < 0.5) {
      const bDef = buildingDefs[building.defId];
      messages.push({
        id: `advisor-condition-${building.id}-${state.clock.day}`,
        advisorId: "treasurer",
        severity: "warn",
        text: `The ${bDef.name} is deteriorating. Repair it before output drops further.`
      });
      break; // Only warn about the worst building
    }
  }

  return messages.slice(0, advisorDefs.length);
}

export function maybeCreateConsultation(state: GameState): ConsultationCurrent | undefined {
  if (state.consultation.mode !== "idle") {
    return undefined;
  }

  if (state.clock.day < 5 || state.clock.day % 15 !== 0) {
    return undefined;
  }

  const factionList = Object.values(state.factions);
  const factionPool = [...factionList]
    .sort((left, right) => consultationStakeScore(right) - consultationStakeScore(left) || left.id.localeCompare(right.id))
    .slice(0, Math.min(4, factionList.length));
  const faction = pick(factionPool, state.worldSeed + state.clock.day * 3);
  const preferredDomain = preferredAgendaDomain(faction.currentAgenda);
  const questionPool = consultationQuestions.filter((question) => {
    if (question.domain !== preferredDomain) {
      return false;
    }
    if (faction.currentAgenda === "succession") {
      return questionHasTag(question, "legitimacy") || questionHasTag(question, "court") || questionHasTag(question, "alliance");
    }
    if (faction.currentAgenda === "faith") {
      return questionHasTag(question, "rites") || questionHasTag(question, "peace") || questionHasTag(question, "prestige");
    }
    return true;
  });
  const question = pick(questionPool.length > 0 ? questionPool : consultationQuestions, state.worldSeed + state.clock.day * 7);
  const semantics = buildConsultationSemantics(question, faction, state.clock.day);
  const omenReports = buildOmenReports(state, question, semantics);
  const tilePool = buildTilePool(semantics, omenReports, question, state.clock.day, state.pythia);
  const stake = consultationStakeScore(faction);

  return {
    id: `consultation-${state.clock.day}`,
    factionId: faction.id,
    envoyName: `${faction.name} Envoy`,
    mood: stake >= 72 ? "urgent" : faction.favour > 60 ? "hopeful" : faction.credibility < 45 || faction.debt > 20 ? "wary" : "measured",
    paymentOffered: 28 + Math.round(faction.credibility / 2) + (faction.debt > 15 ? 4 : 0) + Math.min(14, Math.round(stake / 10)),
    question: question.text,
    domain: question.domain,
    omenReports,
    tilePool,
    placedTileIds: [],
    scorePreview: {
      clarity: 0,
      value: 0,
      risk: 0
    }
  };
}

function tileRelevance(tile: WordTile, base: TileSemantics): number {
  let score = 0;
  if (tile.semantics.target === base.target) {
    score += 5;
  }
  if (tile.semantics.action === base.action) {
    score += 4;
  }
  if (tile.semantics.domain === base.domain) {
    score += 3;
  }
  if (tile.semantics.timeHorizon === base.timeHorizon) {
    score += 2;
  }
  if (tile.semantics.polarity === base.polarity || tile.semantics.polarity === "double") {
    score += 2;
  }
  if (tile.semantics.ambiguity === base.ambiguity) {
    score += 1;
  }
  return score;
}

function rankedTiles(base: TileSemantics, day: number, source: WordTile[], salt: number): WordTile[] {
  return [...source].sort((left, right) => {
    const relevance = tileRelevance(right, base) - tileRelevance(left, base);
    if (relevance !== 0) {
      return relevance;
    }
    const leftRoll = hash(day * 101 + salt + left.id.length);
    const rightRoll = hash(day * 101 + salt + right.id.length);
    if (leftRoll !== rightRoll) {
      return rightRoll - leftRoll;
    }
    return left.id.localeCompare(right.id);
  });
}

function scoreTileAgainstOmens(tile: WordTile, omenReports: OmenReport[], question: QuestionDef, pythia: ScoreContext): number {
  const profile = normalizeScoreContext(pythia);
  const matchingOmens = omenReports.filter((omen) =>
    tile.semantics.target === omen.semantics.target
    || tile.semantics.action === omen.semantics.action
    || tile.semantics.domain === omen.semantics.domain
  ).length;
  const traitBias =
    (hasTrait(profile, "visionary") && (tile.category === "seal" || tile.semantics.ambiguity === "cryptic") ? 3 : 0)
    + (hasTrait(profile, "calculating") && (tile.category === "subject" || tile.category === "action" || tile.semantics.ambiguity !== "cryptic") ? 2 : 0)
    + (hasTrait(profile, "fragile") && tile.semantics.ambiguity === "specific" ? -2 : 0);
  const questionBias =
    (questionHasTag(question, "timing") && tile.semantics.timeHorizon === "immediate" ? 2 : 0)
    + (questionHasTag(question, "legitimacy") && tile.semantics.target === "king" ? 2 : 0)
    + (questionHasTag(question, "trade") && tile.semantics.domain === "economic" ? 2 : 0)
    + (questionHasTag(question, "peace") && tile.semantics.target === "alliance" ? 2 : 0);

  return matchingOmens * 3 + traitBias + questionBias + profile.tranceDepth * 0.02 + profile.mentalClarity * 0.01;
}

function buildTilePool(
  base: TileSemantics,
  omenReports: OmenReport[],
  question: QuestionDef,
  day: number,
  pythia: ScoreContext
): WordTile[] {
  const categories = {
    subject: wordTiles.filter((tile) => tile.category === "subject"),
    action: wordTiles.filter((tile) => tile.category === "action"),
    condition: wordTiles.filter((tile) => tile.category === "condition"),
    modifier: wordTiles.filter((tile) => tile.category === "modifier"),
    seal: wordTiles.filter((tile) => tile.category === "seal")
  };
  const used = new Set<string>();
  const selected: WordTile[] = [];

  const takeTiles = (source: WordTile[], count: number, salt: number) => {
    const ranked = rankedTiles(base, day, source, salt).sort((left, right) => {
      const omenScore =
        scoreTileAgainstOmens(right, omenReports, question, pythia)
        - scoreTileAgainstOmens(left, omenReports, question, pythia);
      if (omenScore !== 0) {
        return omenScore;
      }
      return 0;
    });

    for (const tile of ranked) {
      if (used.has(tile.id)) {
        continue;
      }
      selected.push(tile);
      used.add(tile.id);
      if (selected.length >= count) {
        break;
      }
    }
  };

  takeTiles(categories.subject, 2, 1);
  takeTiles(categories.action, 4, 3);
  takeTiles(categories.condition, 7, 5);
  takeTiles(categories.modifier, 10, 7);
  takeTiles(categories.seal, 12, 11);

  if (selected.length < 12) {
    takeTiles(wordTiles, 12, 13);
  }

  return selected
    .slice(0, 12)
    .map((tile, index) => ({
      ...tile,
      id: `${tile.id}-${day}-${index}`
    }));
}

function chooseTradeResource(faction: FactionState, seed: number): Exclude<ResourceId, "gold"> {
  const byAgenda: Record<FactionAgenda, Exclude<ResourceId, "gold">[]> = {
    war: ["grain", "olive_oil"],
    trade: ["incense", "grain"],
    faith: ["incense", "sacred_water"],
    succession: ["olive_oil", "grain"]
  };

  const weightedChoices = [
    faction.favoredResource,
    ...byAgenda[faction.currentAgenda],
    faction.favoredResource
  ];

  return pick(weightedChoices, seed);
}

function selectConflictTarget(state: GameState, factionId: FactionId, seed: number): FactionId | undefined {
  const rivals = Object.values(state.factions)
    .map((faction) => faction.id)
    .filter((candidateId) => candidateId !== factionId)
    .sort((left, right) =>
      getRelation(state.factions, factionId, left) - getRelation(state.factions, factionId, right)
      || left.localeCompare(right)
    );
  if (rivals.length === 0) {
    return undefined;
  }
  const worstRelation = getRelation(state.factions, factionId, rivals[0]!);
  const candidates = rivals.filter((candidateId) => getRelation(state.factions, factionId, candidateId) <= worstRelation + 8);
  return pick(candidates, seed);
}

function determineNextAgenda(faction: FactionState, seed: number): FactionAgenda {
  const instability = hash(seed + 17);
  const relationValues = Object.values(faction.relations);
  const worstRelation = relationValues.length > 0 ? Math.min(...relationValues) : 0;
  const bestRelation = relationValues.length > 0 ? Math.max(...relationValues) : 0;

  if (faction.profile === "mercantile" && (faction.debt >= 12 || instability > 0.56)) {
    return "trade";
  }
  if (faction.profile === "devout" && (faction.credibility < 58 || instability > 0.72)) {
    return "faith";
  }
  if (faction.profile === "scheming" && (faction.dependence > 38 || instability > 0.7)) {
    return "succession";
  }
  if (faction.profile === "martial" && worstRelation <= -35 && instability > 0.32) {
    return "war";
  }
  if (faction.profile === "mercantile" && bestRelation >= 24 && instability > 0.34) {
    return "trade";
  }
  if (faction.profile === "martial" && (faction.activeConflicts.length > 0 || instability > 0.76) && faction.debt < 26) {
    return "war";
  }
  if (faction.debt >= 22) {
    return "trade";
  }
  if (faction.activeConflicts.length > 0 && instability > 0.28) {
    return "war";
  }
  if (faction.credibility < 42 && instability > 0.38) {
    return "faith";
  }
  if (faction.id === "macedon" && instability > 0.46) {
    return "succession";
  }
  if (instability > 0.82) {
    return rotateAgenda(faction.currentAgenda, 1);
  }
  if (instability < 0.14) {
    return rotateAgenda(faction.currentAgenda, -1);
  }
  return faction.currentAgenda;
}

type MonthlyFactionUpdate = {
  faction: FactionState;
  salience: number;
  summary: string;
};

type DominantBlocSummary = {
  leaderId?: FactionId;
  leaderScore: number;
  margin: number;
  memberIds: FactionId[];
  rivalIds: FactionId[];
};

type WorldDriftResult = {
  factions: Record<FactionId, FactionState>;
  feedItems: EventFeedItem[];
};

function preferredAgendaDomain(agenda: FactionAgenda): QuestionDef["domain"] {
  if (agenda === "trade") {
    return "economic";
  }
  if (agenda === "war") {
    return "military";
  }
  return "spiritual";
}

function dominantBlocSummary(factions: Record<FactionId, FactionState>): DominantBlocSummary {
  const ranked = Object.values(factions)
    .map((faction) => ({
      faction,
      score: factionInfluenceScore(faction)
    }))
    .sort((left, right) => right.score - left.score || left.faction.id.localeCompare(right.faction.id));
  const leader = ranked[0];
  const runnerUp = ranked[1];

  if (!leader) {
    return {
      leaderId: undefined,
      leaderScore: 0,
      margin: 0,
      memberIds: [],
      rivalIds: []
    };
  }

  const margin = leader.score - (runnerUp?.score ?? 0);
  if (leader.score < 44 || margin < 6) {
    return {
      leaderId: undefined,
      leaderScore: leader.score,
      margin,
      memberIds: [],
      rivalIds: []
    };
  }

  const leaderFaction = leader.faction;
  const memberIds = uniqueFactionIds([
    leaderFaction.id,
    ...Object.values(factions)
      .filter((faction) =>
        faction.id !== leaderFaction.id
        && (
          faction.treaties.includes(leaderFaction.id)
          || leaderFaction.treaties.includes(faction.id)
          || getRelation(factions, faction.id, leaderFaction.id) >= 22
        )
      )
      .map((faction) => faction.id)
  ]);
  const rivalIds = Object.values(factions)
    .filter((faction) =>
      faction.id !== leaderFaction.id
      && !memberIds.includes(faction.id)
      && (
        faction.embargoes.includes(leaderFaction.id)
        || leaderFaction.embargoes.includes(faction.id)
        || faction.activeConflicts.includes(leaderFaction.id)
        || getRelation(factions, faction.id, leaderFaction.id) <= -18
      )
    )
    .map((faction) => faction.id)
    .sort((left, right) => left.localeCompare(right));

  return {
    leaderId: leaderFaction.id,
    leaderScore: leader.score,
    margin,
    memberIds,
    rivalIds
  };
}

function philosophicalDriftTarget(faction: FactionState, seed: number): { agenda: FactionAgenda; label: string } {
  const baseByProfile: Record<FactionState["profile"], { agenda: FactionAgenda; label: string }> = {
    martial: { agenda: "war", label: "a harder martial creed" },
    mercantile: { agenda: "trade", label: "a commercial civic ethos" },
    devout: { agenda: "faith", label: "a pious revival" },
    scheming: { agenda: "succession", label: "a courtly legitimacy cult" }
  };
  const alternateTargets: Array<{ agenda: FactionAgenda; label: string }> = [
    { agenda: "war", label: "a harsh security doctrine" },
    { agenda: "trade", label: "a cosmopolitan market philosophy" },
    { agenda: "faith", label: "an anxious sacred revival" },
    { agenda: "succession", label: "an obsession with legitimacy" }
  ];

  if (hash(seed + faction.name.length) > 0.78) {
    return pick(alternateTargets, seed + 11);
  }

  return baseByProfile[faction.profile];
}

function applyPhilosophicalDrift(
  factions: Record<FactionId, FactionState>,
  state: GameState
): WorldDriftResult {
  const driftStories: Array<{ factionId: FactionId; salience: number; note: string }> = [];
  const nextFactions = Object.fromEntries(
    Object.values(factions).map((faction, index) => {
      const seed = state.worldSeed + state.clock.year * 131 + state.clock.month * 29 + index * 17;
      const drift = philosophicalDriftTarget(faction, seed);
      const instability = hash(seed + 7);
      const stressed = faction.credibility < 50 || faction.debt >= 16 || faction.dependence >= 34;
      const shiftsAgenda = drift.agenda !== faction.currentAgenda && (instability > 0.56 || stressed);
      let nextFaction = { ...faction };
      let salience = 0;

      switch (drift.agenda) {
        case "war":
          nextFaction = {
            ...nextFaction,
            dependence: clampStat(nextFaction.dependence + 1),
            favour: clampStat(nextFaction.favour - (stressed ? 1 : 0))
          };
          salience += 2;
          break;
        case "trade":
          nextFaction = {
            ...nextFaction,
            debt: clampDebt(nextFaction.debt - 1),
            dependence: clampStat(nextFaction.dependence + 1)
          };
          salience += 2;
          break;
        case "faith":
          nextFaction = {
            ...nextFaction,
            credibility: clampStat(nextFaction.credibility + 1),
            favour: clampStat(nextFaction.favour + 1)
          };
          salience += 2;
          break;
        case "succession":
          nextFaction = {
            ...nextFaction,
            dependence: clampStat(nextFaction.dependence + 2),
            credibility: clampStat(nextFaction.credibility - (stressed ? 2 : 1))
          };
          salience += 3;
          break;
      }

      if (shiftsAgenda) {
        nextFaction = {
          ...nextFaction,
          currentAgenda: drift.agenda
        };
        salience += 4;
      }

      if (!shiftsAgenda && !stressed && instability < 0.64) {
        return [faction.id, nextFaction];
      }

      const note = `Month ${state.clock.month}: ${faction.name} drifts toward ${drift.label}, and public talk now leans toward ${drift.agenda}.`;
      driftStories.push({
        factionId: faction.id,
        salience,
        note
      });

      return [faction.id, appendOutcomeNote(nextFaction, note)];
    })
  ) as Record<FactionId, FactionState>;

  return {
    factions: nextFactions,
    feedItems: driftStories
      .sort((left, right) => right.salience - left.salience || left.factionId.localeCompare(right.factionId))
      .slice(0, 2)
      .map((story) => ({
        id: `event-philosophy-${state.clock.year}-${state.clock.month}-${story.factionId}`,
        day: state.clock.day,
        text: story.note
      }))
  };
}

function applyHegemonPressure(
  factions: Record<FactionId, FactionState>,
  state: GameState
): WorldDriftResult & { bloc: DominantBlocSummary } {
  const bloc = dominantBlocSummary(factions);
  if (!bloc.leaderId) {
    return {
      factions,
      feedItems: [],
      bloc
    };
  }

  const leader = factions[bloc.leaderId];
  const nextFactions = { ...factions };
  const leaderNote = `Month ${state.clock.month}: ${leader.name} gathers a dominant bloc and sets the regional terms of trade.`;
  nextFactions[leader.id] = appendOutcomeNote(
    {
      ...leader,
      favour: clampStat(leader.favour + 2),
      credibility: clampStat(leader.credibility + 2),
      tradeAccess: true
    },
    leaderNote
  );

  for (const memberId of bloc.memberIds) {
    if (memberId === leader.id) {
      continue;
    }
    const member = nextFactions[memberId];
    const note = `Month ${state.clock.month}: ${member.name} leans into ${leader.name}'s dominant bloc and finds easier caravan access.`;
    nextFactions[memberId] = appendOutcomeNote(
      {
        ...member,
        favour: clampStat(member.favour + 1),
        credibility: clampStat(member.credibility + 1),
        debt: clampDebt(member.debt - 1),
        tradeAccess: true
      },
      note
    );
  }

  for (const rivalId of bloc.rivalIds) {
    const rival = nextFactions[rivalId];
    const note = `Month ${state.clock.month}: ${leader.name}'s dominant bloc squeezes ${rival.name}, and caravan brokers grow cautious.`;
    nextFactions[rivalId] = appendOutcomeNote(
      {
        ...rival,
        favour: clampStat(rival.favour - 2),
        credibility: clampStat(rival.credibility - 1),
        debt: clampDebt(rival.debt + 2),
        tradeAccess: rival.tradeAccess && rival.debt < 18 && rival.embargoes.length === 0 && rival.activeConflicts.length === 0
      },
      note
    );
  }

  return {
    factions: nextFactions,
    feedItems: [
      {
        id: `event-hegemon-${state.clock.year}-${state.clock.month}-${leader.id}`,
        day: state.clock.day,
        text: leaderNote
      }
    ],
    bloc
  };
}

function applyDestabilizationPressure(
  factions: Record<FactionId, FactionState>,
  state: GameState,
  bloc: DominantBlocSummary
): WorldDriftResult {
  const stories: Array<{ factionId: FactionId; severity: number; note: string }> = [];
  const nextFactions = Object.fromEntries(
    Object.values(factions).map((faction) => {
      const blocRivalPenalty = bloc.rivalIds.includes(faction.id) ? 8 : 0;
      const pressureScore =
        faction.debt * 1.7
        + faction.activeConflicts.length * 18
        + faction.embargoes.length * 8
        + Math.max(0, 52 - faction.credibility) * 1.2
        + Math.max(0, 48 - faction.favour)
        + Math.max(0, faction.dependence - 34) * 0.8
        + blocRivalPenalty;

      if (pressureScore < 72) {
        return [faction.id, faction];
      }

      const revolutionary = pressureScore >= 96 || (faction.credibility <= 32 && faction.favour <= 42);
      const note = revolutionary
        ? `Month ${state.clock.month}: ${faction.name} faces revolutionary agitation as debt, conflict, and broken credibility turn the streets against the regime.`
        : `Month ${state.clock.month}: ${faction.name} slips into government crisis as ministers feud and public confidence sours.`;
      const nextFaction = appendOutcomeNote(
        {
          ...faction,
          currentAgenda: revolutionary ? "succession" : faction.currentAgenda === "trade" ? "faith" : faction.currentAgenda,
          favour: clampStat(faction.favour - (revolutionary ? 6 : 3)),
          credibility: clampStat(faction.credibility - (revolutionary ? 7 : 4)),
          dependence: clampStat(faction.dependence + (revolutionary ? 4 : 2)),
          debt: clampDebt(faction.debt + (revolutionary ? 3 : 1)),
          tradeAccess: false
        },
        note
      );

      stories.push({
        factionId: faction.id,
        severity: pressureScore,
        note
      });

      return [faction.id, nextFaction];
    })
  ) as Record<FactionId, FactionState>;

  return {
    factions: nextFactions,
    feedItems: stories
      .sort((left, right) => right.severity - left.severity || left.factionId.localeCompare(right.factionId))
      .slice(0, 2)
      .map((story) => ({
        id: `event-destabilization-${state.clock.year}-${state.clock.month}-${story.factionId}`,
        day: state.clock.day,
        text: story.note
      }))
  };
}

function resolveMonthlyFaction(state: GameState, faction: FactionState, index: number): MonthlyFactionUpdate {
  const seed = state.worldSeed + state.clock.year * 97 + state.clock.month * 41 + index * 13 + faction.name.length;
  const event = pick(
    politicalEventDefs.filter((entry) => entry.agenda === faction.currentAgenda),
    seed
  );
  const volatility = Math.round((hash(seed + 5) - 0.5) * 10);
  const oracleTailwind = faction.lastOutcome?.includes("hailed") ? 2 : faction.lastOutcome?.includes("embarrassment") ? -2 : 0;

  let credibilityDelta = oracleTailwind;
  let favourDelta = 0;
  let dependenceDelta = 0;
  let debtDelta = 0;

  switch (event.agenda) {
    case "war":
      credibilityDelta += volatility >= 0 ? 1 : -3;
      favourDelta += volatility >= 0 ? 2 : -2;
      dependenceDelta += 1;
      debtDelta += 4 + Math.max(0, volatility);
      break;
    case "trade":
      credibilityDelta += 2 + Math.max(0, volatility);
      favourDelta += 3;
      dependenceDelta += 2;
      debtDelta -= 4;
      break;
    case "faith":
      credibilityDelta += 3;
      favourDelta += 1 + Math.max(0, volatility);
      dependenceDelta -= 1;
      debtDelta -= 1;
      break;
    case "succession":
      credibilityDelta -= 1 + Math.max(0, -volatility);
      favourDelta += volatility;
      dependenceDelta += 3;
      debtDelta += 2;
      break;
  }

  if (state.resources.incense.amount < 8 && event.domain === "spiritual") {
    favourDelta -= 2;
    credibilityDelta -= 1;
  }

  switch (faction.profile) {
    case "martial":
      if (event.agenda === "war") {
        credibilityDelta += 2;
        favourDelta += 1;
      }
      if (event.domain === "economic") {
        debtDelta += 1;
      }
      break;
    case "mercantile":
      debtDelta -= 2;
      if (event.agenda === "trade") {
        favourDelta += 2;
        credibilityDelta += 1;
      }
      break;
    case "devout":
      if (event.domain === "spiritual") {
        credibilityDelta += 2;
        favourDelta += 1;
      }
      if (state.resources.incense.amount < 8) {
        favourDelta -= 1;
      }
      break;
    case "scheming":
      dependenceDelta += 1;
      if (event.agenda === "succession") {
        credibilityDelta += 1;
        favourDelta += 1;
      }
      if (event.agenda === "war") {
        favourDelta -= 1;
      }
      break;
  }

  const provisionalFaction: FactionState = {
    ...faction,
    credibility: clampStat(faction.credibility + credibilityDelta),
    favour: clampStat(faction.favour + favourDelta),
    dependence: clampStat(faction.dependence + dependenceDelta),
    debt: clampDebt(faction.debt + debtDelta)
  };
  const nextAgenda = determineNextAgenda(provisionalFaction, seed);
  const nextConflict = nextAgenda === "war" ? selectConflictTarget(state, faction.id, seed + 11) : undefined;
  const tradeThreshold =
    faction.profile === "mercantile"
      ? 0.08
      : faction.profile === "martial"
        ? 0.24
        : faction.profile === "scheming"
          ? 0.14
          : 0.18;
  const tradeAccess = (
    nextAgenda !== "war"
    && provisionalFaction.debt < (faction.profile === "mercantile" ? 32 : 28)
    && hash(seed + 23) > tradeThreshold
  ) || (faction.profile === "mercantile" && nextAgenda === "trade");
  const outcomeTail = tradeAccess
    ? "Caravan lanes remain negotiable."
    : "Merchants grow wary and routes tighten.";
  const profileTail =
    faction.profile === "martial"
      ? "Its generals keep the city on a hard footing."
      : faction.profile === "mercantile"
        ? "Its merchants press every opening."
        : faction.profile === "devout"
          ? "Its priests shape the public mood."
          : "Its court turns every shift into leverage.";
  const authoredSummary = pick(event.summaries, seed + 29);
  const summary = `Month ${state.clock.month}: ${faction.name} ${authoredSummary}. ${event.label} turns attention toward ${nextAgenda}. ${outcomeTail} ${profileTail}`;

  return {
    faction: {
      ...provisionalFaction,
      currentAgenda: nextAgenda,
      activeConflicts: nextConflict ? [nextConflict] : [],
      tradeAccess,
      lastOutcome: summary,
      history: appendHistory(faction.history, summary)
    },
    salience:
      Math.abs(credibilityDelta)
      + Math.abs(favourDelta)
      + Math.abs(dependenceDelta)
      + Math.abs(debtDelta)
      + (nextAgenda !== faction.currentAgenda ? 3 : 0)
      + (tradeAccess !== faction.tradeAccess ? 3 : 0),
    summary
  };
}

function evolveFactionRelations(factions: Record<FactionId, FactionState>): Record<FactionId, FactionState> {
  let nextFactions = factions;
  const ordered = Object.values(factions).sort((left, right) => left.id.localeCompare(right.id));

  for (let index = 0; index < ordered.length; index += 1) {
    for (let inner = index + 1; inner < ordered.length; inner += 1) {
      const left = nextFactions[ordered[index]!.id];
      const right = nextFactions[ordered[inner]!.id];
      let delta = 0;

      if (left.activeConflicts.includes(right.id) || right.activeConflicts.includes(left.id)) {
        delta -= 6;
      } else if (left.tradeAccess && right.tradeAccess && (
        left.currentAgenda === "trade"
        || right.currentAgenda === "trade"
        || left.profile === "mercantile"
        || right.profile === "mercantile"
      )) {
        delta += 3;
      } else if (left.currentAgenda === "faith" && right.currentAgenda === "faith") {
        delta += 2;
      } else if (left.currentAgenda === "war" && right.currentAgenda === "war") {
        delta -= 1;
      }

      nextFactions = applyBilateralRelationDelta(nextFactions, left.id, right.id, delta);
    }
  }

  return nextFactions;
}

function deriveDiplomaticAccords(factions: Record<FactionId, FactionState>): Record<FactionId, FactionState> {
  const nextFactions = Object.fromEntries(
    Object.values(factions).map((faction) => [
      faction.id,
      {
        ...faction,
        treaties: [],
        embargoes: []
      }
    ])
  ) as unknown as Record<FactionId, FactionState>;
  const ordered = Object.values(nextFactions).sort((left, right) => left.id.localeCompare(right.id));

  for (let index = 0; index < ordered.length; index += 1) {
    for (let inner = index + 1; inner < ordered.length; inner += 1) {
      const left = nextFactions[ordered[index]!.id];
      const right = nextFactions[ordered[inner]!.id];
      const relation = getRelation(nextFactions, left.id, right.id);
      const inConflict = left.activeConflicts.includes(right.id) || right.activeConflicts.includes(left.id);
      const treatyEligible =
        relation >= 45
        && !inConflict
        && (
          left.tradeAccess
          || right.tradeAccess
          || left.currentAgenda === "faith"
          || right.currentAgenda === "faith"
          || left.profile === "mercantile"
          || right.profile === "mercantile"
        );
      const embargoEligible =
        relation <= -40
        && (
          inConflict
          || left.currentAgenda === "trade"
          || right.currentAgenda === "trade"
          || left.profile === "mercantile"
          || right.profile === "mercantile"
        );

      if (treatyEligible) {
        nextFactions[left.id] = {
          ...nextFactions[left.id],
          treaties: uniqueFactionIds([...nextFactions[left.id].treaties, right.id])
        };
        nextFactions[right.id] = {
          ...nextFactions[right.id],
          treaties: uniqueFactionIds([...nextFactions[right.id].treaties, left.id])
        };
      }

      if (embargoEligible) {
        nextFactions[left.id] = {
          ...nextFactions[left.id],
          embargoes: uniqueFactionIds([...nextFactions[left.id].embargoes, right.id])
        };
        nextFactions[right.id] = {
          ...nextFactions[right.id],
          embargoes: uniqueFactionIds([...nextFactions[right.id].embargoes, left.id])
        };
      }
    }
  }

  return nextFactions;
}

function applyDiplomaticTradePressure(factions: Record<FactionId, FactionState>): Record<FactionId, FactionState> {
  return Object.fromEntries(
    Object.values(factions).map((faction) => {
      const treatyCount = faction.treaties.length;
      const embargoCount = faction.embargoes.length;
      const isolated = embargoCount >= 2 && treatyCount === 0;
      const protectedRoutes = treatyCount >= 2 && faction.activeConflicts.length === 0;
      const tradeAccess = protectedRoutes
        ? true
        : isolated
          ? false
          : faction.tradeAccess && embargoCount < 3;

      return [
        faction.id,
        {
          ...faction,
          tradeAccess
        }
      ];
    })
  ) as Record<FactionId, FactionState>;
}

function annotateDiplomaticClimate(
  previousFactions: Record<FactionId, FactionState>,
  factions: Record<FactionId, FactionState>,
  month: number
): Record<FactionId, FactionState> {
  return Object.fromEntries(
    Object.values(factions).map((faction) => {
      const previous = previousFactions[faction.id];
      const treaties = uniqueFactionIds(faction.treaties);
      const embargoes = uniqueFactionIds(faction.embargoes);
      const previousTreaties = uniqueFactionIds(previous.treaties ?? []);
      const previousEmbargoes = uniqueFactionIds(previous.embargoes ?? []);
      const addedTreaties = treaties.filter((entry) => !previousTreaties.includes(entry));
      const liftedTreaties = previousTreaties.filter((entry) => !treaties.includes(entry));
      const addedEmbargoes = embargoes.filter((entry) => !previousEmbargoes.includes(entry));
      const liftedEmbargoes = previousEmbargoes.filter((entry) => !embargoes.includes(entry));
      const notes: string[] = [];

      if (sameFactionList(previousTreaties, treaties) && sameFactionList(previousEmbargoes, embargoes) && treaties.length === 0 && embargoes.length === 0) {
        return [faction.id, faction];
      }

      if (addedTreaties.length > 0) {
        notes.push(`Treaties open with ${describeFactionList(factions, addedTreaties)}.`);
      } else if (treaties.length > 0) {
        notes.push(`Treaties hold with ${describeFactionList(factions, treaties.slice(0, 2))}.`);
      }

      if (addedEmbargoes.length > 0) {
        notes.push(`Embargoes bite against ${describeFactionList(factions, addedEmbargoes)}.`);
      } else if (embargoes.length > 0) {
        notes.push(`Embargoes remain against ${describeFactionList(factions, embargoes.slice(0, 2))}.`);
      }

      if (liftedTreaties.length > 0) {
        notes.push(`A prior treaty with ${describeFactionList(factions, liftedTreaties)} has lapsed.`);
      }

      if (liftedEmbargoes.length > 0) {
        notes.push(`An embargo against ${describeFactionList(factions, liftedEmbargoes)} has eased.`);
      }

      if (notes.length === 0) {
        return [faction.id, faction];
      }

      const diplomacyNote = `Month ${month}: ${notes.join(" ")}`;
      const history = appendHistory(faction.history, diplomacyNote);
      const lastOutcome = faction.lastOutcome?.includes("Treaties") || faction.lastOutcome?.includes("Embargoes")
        ? diplomacyNote
        : `${faction.lastOutcome ?? ""} ${diplomacyNote}`.trim();

      return [
        faction.id,
        {
          ...faction,
          history,
          lastOutcome
        }
      ];
    })
  ) as Record<FactionId, FactionState>;
}

function normalizeFactionConflicts(factions: Record<FactionId, FactionState>): Record<FactionId, FactionState> {
  const normalized = { ...factions };

  for (const faction of Object.values(factions)) {
    for (const rivalId of faction.activeConflicts) {
      const rival = normalized[rivalId];
      if (!rival) {
        continue;
      }
      if (!rival.activeConflicts.includes(faction.id)) {
        normalized[rivalId] = {
          ...rival,
          activeConflicts: [...rival.activeConflicts, faction.id].sort((left, right) => left.localeCompare(right))
        };
      }
    }
  }

  return normalized;
}

export function advancePoliticalClimate(state: GameState): {
  factions: Record<FactionId, FactionState>;
  philosophers: GameState["philosophers"];
  tradeOffers: TradeOffer[];
  feedItems: EventFeedItem[];
} {
  const orderedFactions = Object.values(state.factions).sort((left, right) => left.id.localeCompare(right.id));
  const updates = orderedFactions.map((faction, index) => resolveMonthlyFaction(state, faction, index));
  let factions = Object.fromEntries(updates.map((update) => [update.faction.id, update.faction])) as Record<FactionId, FactionState>;
  const philosopherUpdate = advancePhilosopherThreats({
    ...state,
    factions
  });
  factions = philosopherUpdate.factions;
  const philosophyUpdate = applyPhilosophicalDrift(factions, state);
  factions = normalizeFactionConflicts(philosophyUpdate.factions);
  factions = evolveFactionRelations(factions);
  factions = deriveDiplomaticAccords(factions);
  const hegemonUpdate = applyHegemonPressure(factions, state);
  factions = hegemonUpdate.factions;
  const destabilizationUpdate = applyDestabilizationPressure(factions, state, hegemonUpdate.bloc);
  factions = destabilizationUpdate.factions;
  factions = applyDiplomaticTradePressure(factions);
  factions = annotateDiplomaticClimate(state.factions, factions, state.clock.month);
  const openRoutes = Object.values(factions).filter((faction) => faction.tradeAccess);
  if (openRoutes.length < 2) {
    const fallbackPartners = Object.values(factions)
      .sort((left, right) => (right.favour + right.credibility - right.debt) - (left.favour + left.credibility - left.debt))
      .slice(0, 2);

    factions = {
      ...factions,
      ...Object.fromEntries(fallbackPartners.map((faction) => [
        faction.id,
        {
          ...faction,
          tradeAccess: true,
          lastOutcome: `${faction.lastOutcome} Delphi keeps a narrow caravan accord alive.`,
          history: appendHistory(faction.history, `Month ${state.clock.month}: Delphi keeps a narrow caravan accord alive for ${faction.name}.`)
        }
      ]))
    };
  }
  const tradeOffers = buildMonthlyTradeOffers({
    ...state,
    factions
  });
  const keyStories = [...updates]
    .sort((left, right) => right.salience - left.salience)
    .slice(0, 2);
  const feedItems: EventFeedItem[] = [
    ...hegemonUpdate.feedItems,
    ...destabilizationUpdate.feedItems,
    ...philosopherUpdate.feedItems,
    ...philosophyUpdate.feedItems,
    ...keyStories.map((story) => ({
      id: `event-month-${state.clock.year}-${state.clock.month}-${story.faction.id}`,
      day: state.clock.day,
      text: story.summary
    })),
    {
      id: `event-trade-${state.clock.year}-${state.clock.month}`,
      day: state.clock.day,
      text: `Month ${state.clock.month}: ${tradeOffers.length} caravans offer terms to Delphi.`
    }
  ];

  return {
    factions,
    philosophers: philosopherUpdate.philosophers,
    tradeOffers,
    feedItems
  };
}

export function buildMonthlyTradeOffers(state: GameState): TradeOffer[] {
  const bloc = dominantBlocSummary(state.factions);
  return Object.values(state.factions)
    .filter((faction) => faction.tradeAccess)
    .sort((left, right) =>
      (right.favour + right.credibility - right.debt + (right.profile === "mercantile" ? 6 : 0))
      - (left.favour + left.credibility - left.debt + (left.profile === "mercantile" ? 6 : 0))
    )
    .slice(0, 4)
    .map((faction, index) => {
      const resourceId = chooseTradeResource(faction, state.worldSeed + state.clock.month * 19 + index);
      const amountBonus =
        (faction.profile === "mercantile" ? 3 : 0)
        + (resourceId === faction.favoredResource ? 2 : 0)
        + (faction.profile === "devout" && resourceId === "incense" ? 1 : 0)
        + faction.treaties.length
        - faction.embargoes.length
        + (bloc.leaderId === faction.id ? 3 : bloc.memberIds.includes(faction.id) ? 2 : 0)
        - (bloc.rivalIds.includes(faction.id) ? 1 : 0);
      const priceBias =
        (faction.profile === "mercantile" ? -0.7 : 0)
        + (faction.profile === "scheming" ? 0.6 : 0)
        + (resourceId === faction.favoredResource ? -0.5 : 0)
        - faction.treaties.length * 0.35
        + faction.embargoes.length * 0.75
        - (bloc.leaderId === faction.id ? 0.8 : bloc.memberIds.includes(faction.id) ? 0.45 : 0)
        + (bloc.rivalIds.includes(faction.id) ? 1 : 0);

      return {
        id: `trade-${state.clock.month}-${index}`,
        factionId: faction.id,
        resourceId,
        amount: Math.max(4, 7 + index * 2 + amountBonus + Math.round((faction.favour + faction.credibility) / 25)),
        price: Math.max(5, 8 + Math.max(0, 58 - faction.favour) / 6 + faction.debt / 8 + priceBias)
      };
    });
}

export function createConsequence(state: GameState, prophecyId: string, factionId: FactionId, semantics: TileSemantics[]): ConsequenceCase {
  const base = semantics[0] ?? factionAgendaToSemantics(state.factions[factionId].currentAgenda, factionId, state.clock.day);
  const absoluteDay = getAbsoluteDay(state.clock);
  const outcome: TileSemantics = {
    ...base,
    action: pick<SemanticAction>(["triumph", "fall", "fracture", "endure", "prosper", "withhold"], state.worldSeed + state.clock.day * 13),
    polarity: pick<SemanticPolarity>(["favorable", "warning", "double"], state.worldSeed + state.clock.day * 17),
    target: pick<SemanticTarget>(["army", "fleet", "king", "city", "oracle", "harvest", "treasury", "alliance"], state.worldSeed + state.clock.day * 19)
  };

  return {
    id: `consequence-${prophecyId}`,
    prophecyId,
    factionId,
    dueDay: absoluteDay + 20,
    outcome,
    resolved: false
  };
}

export type ObserverReaction = {
  factionId: FactionId;
  counterpartyFactionId: FactionId;
  favourDelta: number;
  credibilityDelta: number;
  dependenceDelta: number;
  relationDelta: number;
  salience: number;
  historyEntry: string;
};

function dominantPolarityForSemantics(prophecySemantics: TileSemantics[]): SemanticPolarity {
  const polarityCounts = prophecySemantics.reduce(
    (counts, semantics) => ({
      favorable: counts.favorable + (semantics.polarity === "favorable" ? 1 : 0),
      warning: counts.warning + (semantics.polarity === "warning" ? 1 : 0),
      double: counts.double + (semantics.polarity === "double" ? 1 : 0)
    }),
    { favorable: 0, warning: 0, double: 0 }
  );

  return polarityCounts.warning > polarityCounts.favorable && polarityCounts.warning >= polarityCounts.double
    ? "warning"
    : polarityCounts.double > polarityCounts.favorable
      ? "double"
      : "favorable";
}

export function evaluateDeliveryObservers(
  state: GameState,
  consultedFactionId: FactionId,
  prophecySemantics: TileSemantics[],
  score: { value: number; risk: number }
): ObserverReaction[] {
  const consultedFaction = state.factions[consultedFactionId];
  const dominantDomain = prophecySemantics[0]?.domain ?? agendaDomain(consultedFaction.currentAgenda);
  const dominantPolarity = dominantPolarityForSemantics(prophecySemantics);

  return Object.values(state.factions)
    .filter((faction) => faction.id !== consultedFactionId)
    .map<ObserverReaction | undefined>((faction) => {
      const rival = faction.activeConflicts.includes(consultedFactionId);
      const aligned = agendaDomain(faction.currentAgenda) === dominantDomain;
      let favourDelta = 0;
      let credibilityDelta = 0;
      let dependenceDelta = 0;
      let relationDelta = 0;
      let historyEntry: string | undefined;

      if (rival) {
        if (dominantPolarity === "favorable") {
          favourDelta -= score.value >= 75 ? 3 : 2;
          credibilityDelta -= score.risk >= 80 ? 1 : 0;
          relationDelta -= 4;
          historyEntry = `Day ${state.clock.day}: ${faction.name} bristles at Delphi's favorable answer to ${consultedFaction.name}.`;
        } else if (dominantPolarity === "warning") {
          favourDelta += 2;
          relationDelta -= 2;
          historyEntry = `Day ${state.clock.day}: ${faction.name} quietly approves Delphi's warning to ${consultedFaction.name}.`;
        } else {
          favourDelta -= 1;
          relationDelta -= 1;
          historyEntry = `Day ${state.clock.day}: ${faction.name} studies Delphi's answer to ${consultedFaction.name} for hidden intent.`;
        }
      } else if (aligned && score.value >= 72) {
        favourDelta += dominantPolarity === "warning" ? 0 : 1;
        dependenceDelta += faction.tradeAccess ? 1 : 0;
        relationDelta += 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} reads Delphi's answer to ${consultedFaction.name} as a sign for its own ${dominantDomain} plans.`;
      } else if (faction.tradeAccess && score.value >= 82 && score.risk < 78) {
        favourDelta += 1;
        relationDelta += 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} sees steadier advantage in Delphi after the answer given to ${consultedFaction.name}.`;
      } else if (score.risk >= 85 && dominantPolarity === "favorable") {
        credibilityDelta -= 1;
        relationDelta -= 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} suspects Delphi is taking too bold a side in ${consultedFaction.name}'s affairs.`;
      }

      if (!historyEntry || (favourDelta === 0 && credibilityDelta === 0 && dependenceDelta === 0 && relationDelta === 0)) {
        return undefined;
      }

      return {
        factionId: faction.id,
        counterpartyFactionId: consultedFactionId,
        favourDelta,
        credibilityDelta,
        dependenceDelta,
        relationDelta,
        salience: Math.abs(favourDelta) + Math.abs(credibilityDelta) + Math.abs(dependenceDelta) + Math.abs(relationDelta) + (rival ? 2 : 0),
        historyEntry
      };
    })
    .filter((reaction): reaction is ObserverReaction => Boolean(reaction))
    .sort((left, right) => right.salience - left.salience || left.factionId.localeCompare(right.factionId))
    .slice(0, 3);
}

export function evaluateResolutionObservers(
  state: GameState,
  consultedFactionId: FactionId,
  outcome: TileSemantics,
  delta: number
): ObserverReaction[] {
  const consultedFaction = state.factions[consultedFactionId];
  const vindicated = delta >= 8;
  const backlash = delta <= -6;
  const dominantDomain = outcome.domain ?? agendaDomain(consultedFaction.currentAgenda);
  const dominantPolarity = outcome.polarity;

  return Object.values(state.factions)
    .filter((faction) => faction.id !== consultedFactionId)
    .map<ObserverReaction | undefined>((faction) => {
      const rival = faction.activeConflicts.includes(consultedFactionId);
      const aligned = agendaDomain(faction.currentAgenda) === dominantDomain;
      let favourDelta = 0;
      let credibilityDelta = 0;
      let dependenceDelta = 0;
      let relationDelta = 0;
      let historyEntry: string | undefined;

      if (vindicated && rival) {
        favourDelta -= dominantPolarity === "favorable" ? 3 : 2;
        credibilityDelta += 1;
        relationDelta -= 3;
        historyEntry = `Day ${state.clock.day}: ${faction.name} resents Delphi's vindication of ${consultedFaction.name}.`;
      } else if (backlash && rival) {
        favourDelta += 2;
        credibilityDelta -= 1;
        relationDelta -= 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} savors ${consultedFaction.name}'s embarrassment and doubts Delphi's judgment.`;
      } else if (vindicated && aligned) {
        favourDelta += dominantPolarity === "warning" ? 0 : 1;
        dependenceDelta += faction.tradeAccess ? 1 : 0;
        credibilityDelta += delta >= 12 ? 1 : 0;
        relationDelta += 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} treats Delphi's vindicated answer to ${consultedFaction.name} as guidance for its own ${dominantDomain} ambitions.`;
      } else if (backlash && aligned) {
        credibilityDelta -= 1;
        relationDelta -= 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} worries Delphi has misread the omens touching ${dominantDomain} affairs.`;
      } else if (vindicated && faction.tradeAccess && dominantDomain === "economic") {
        favourDelta += 1;
        dependenceDelta += 1;
        relationDelta += 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} sees safer profit in Delphi after ${consultedFaction.name}'s prophecy proves sound.`;
      } else if (backlash && faction.tradeAccess) {
        credibilityDelta -= delta <= -10 ? 2 : 1;
        relationDelta -= 1;
        historyEntry = `Day ${state.clock.day}: ${faction.name} trims its trust in Delphi after ${consultedFaction.name}'s failed prophecy ripens.`;
      }

      if (!historyEntry || (favourDelta === 0 && credibilityDelta === 0 && dependenceDelta === 0 && relationDelta === 0)) {
        return undefined;
      }

      return {
        factionId: faction.id,
        counterpartyFactionId: consultedFactionId,
        favourDelta,
        credibilityDelta,
        dependenceDelta,
        relationDelta,
        salience:
          Math.abs(favourDelta)
          + Math.abs(credibilityDelta)
          + Math.abs(dependenceDelta)
          + Math.abs(relationDelta)
          + (rival ? 2 : 0)
          + (vindicated ? 1 : 0),
        historyEntry
      };
    })
    .filter((reaction): reaction is ObserverReaction => Boolean(reaction))
    .sort((left, right) => right.salience - left.salience || left.factionId.localeCompare(right.factionId))
    .slice(0, 3);
}

export function scorePlacedTiles(tiles: WordTile[], context: ScoreContext): { clarity: number; value: number; risk: number } {
  if (tiles.length === 0) {
    return { clarity: 0, value: 0, risk: 0 };
  }

  const pythia = normalizeScoreContext(context);
  const categories = new Set(tiles.map((tile) => tile.category));
  const domainSpread = new Set(tiles.map((tile) => tile.semantics.domain)).size;
  const targetSpread = new Set(tiles.map((tile) => tile.semantics.target)).size;
  const specificCount = tiles.filter((tile) => tile.semantics.ambiguity === "specific").length;
  const balancedCount = tiles.filter((tile) => tile.semantics.ambiguity === "balanced").length;
  const crypticCount = tiles.filter((tile) => tile.semantics.ambiguity === "cryptic").length;
  const timeSpread = new Set(tiles.map((tile) => tile.semantics.timeHorizon)).size;
  const visionary = hasTrait(pythia, "visionary");
  const calculating = hasTrait(pythia, "calculating");
  const diplomatic = hasTrait(pythia, "diplomatic");
  const fragile = hasTrait(pythia, "fragile");
  const structureBonus =
    (categories.has("subject") ? 16 : 0)
    + (categories.has("action") ? 16 : 0)
    + (categories.has("condition") ? 8 : 0)
    + (categories.has("modifier") ? 6 : 0)
    + (categories.has("seal") ? 8 : 0);
  const coherenceBonus = domainSpread === 1 ? 12 : 6 - (domainSpread - 1) * 4;
  const pythiaComposure = pythia.mentalClarity * 0.18 + pythia.attunement * 0.14 + pythia.tranceDepth * 0.12;
  const strainPenalty = pythia.needs.rest * 0.16 + pythia.needs.purification * 0.14;
  const clarity = Math.max(
    20,
    Math.min(
      100,
      18
        + structureBonus
        + balancedCount * 7
        + specificCount * 4
        + crypticCount * 2
        + coherenceBonus
        + (calculating ? 8 : 0)
        + (visionary ? crypticCount * 2 : 0)
        + pythiaComposure * 0.12
        - Math.max(0, timeSpread - 1) * 4
        - Math.max(0, targetSpread - 1) * 10
        - strainPenalty * 0.08
    )
  );
  const value = Math.max(
    20,
    Math.min(
      100,
      clarity
        + (categories.has("seal") ? 10 : 0)
        + (domainSpread === 1 ? 12 : 4)
        + pythia.attunement * 0.08
        + pythia.tranceDepth * 0.08
        + (visionary ? 6 : 0)
        + (diplomatic ? 4 : 0)
        - (fragile ? Math.max(0, pythia.needs.rest - 55) * 0.08 : 0)
    )
  );
  const risk = Math.max(
    5,
    Math.min(
      100,
      16
        + specificCount * 18
        + Math.max(0, targetSpread - 1) * 12
        + Math.max(0, domainSpread - 1) * 8
        + Math.max(0, timeSpread - 1) * 6
        + (categories.has("subject") && categories.has("action") ? 0 : 12)
        - crypticCount * 5
        - pythia.attunement * 0.05
        - pythia.mentalClarity * 0.06
        - pythia.tranceDepth * 0.04
        - (calculating ? 6 : 0)
        - (diplomatic ? 3 : 0)
        + (fragile ? 7 : 0)
        + Math.max(0, pythia.needs.rest - 50) * 0.12
        + Math.max(0, pythia.needs.purification - 50) * 0.1
    )
  );

  return {
    clarity: Math.round(clarity),
    value: Math.round(value),
    risk: Math.round(risk)
  };
}

export function resolveConsequence(consequence: ConsequenceCase, prophecyTiles: TileSemantics[]): { report: string; delta: number } {
  const specificCount = prophecyTiles.filter((tile) => tile.ambiguity === "specific").length;
  const crypticCount = prophecyTiles.filter((tile) => tile.ambiguity === "cryptic").length;
  const factionBias: Partial<Record<FactionId, number>> = {
    athens: -1,
    sparta: -2,
    corinth: 1,
    thebes: 1,
    miletus: 1,
    macedon: -1
  };
  const matchScores = prophecyTiles.map((tile) => {
    let score = 0;
    if (tile.target === consequence.outcome.target) {
      score += 5;
    }
    if (tile.action === consequence.outcome.action) {
      score += 4;
    }
    if (tile.domain === consequence.outcome.domain) {
      score += 3;
    }
    if (tile.polarity === consequence.outcome.polarity || tile.polarity === "double") {
      score += 2;
    }
    if (tile.timeHorizon === consequence.outcome.timeHorizon) {
      score += 1;
    }
    if (tile.ambiguity === "specific" && tile.target === consequence.outcome.target && tile.action === consequence.outcome.action) {
      score += 2;
    }
    if (tile.ambiguity === "cryptic" && score > 0) {
      score += 1;
    }
    return score;
  });
  const bestMatch = matchScores.length > 0 ? Math.max(...matchScores) : 0;
  const supportingMatches = matchScores.filter((score) => score >= 6).length;
  let delta =
    bestMatch >= 12
      ? 14 + supportingMatches * 2
      : bestMatch >= 8
        ? 7 + supportingMatches + crypticCount
        : bestMatch >= 5
          ? 2 + crypticCount - specificCount
          : -8 - specificCount * 4 + crypticCount * 2;
  delta += factionBias[consequence.factionId] ?? 0;
  delta = Math.max(-18, Math.min(18, delta));
  const tone =
    delta >= 12
      ? "The city proclaims the oracle vindicated."
      : delta >= 4
        ? "The prophecy is remembered as cleverly judged."
        : delta > -6
          ? "The words were considered murky, but not ruinous."
          : "The city remembers the prophecy as a public humiliation.";
  const confidence =
    specificCount > crypticCount ? "precise" : crypticCount > specificCount ? "veiled" : "balanced";
  const report = `Day ${consequence.dueDay}: ${tone} ${consequence.factionId} faced ${consequence.outcome.target} and ${consequence.outcome.action}; the ${confidence} prophecy scored ${bestMatch}.`;

  return {
    report,
    delta
  };
}

// ---------------------------------------------------------------------------
// World History Integration
// ---------------------------------------------------------------------------

function computeFactionPressureScore(faction: FactionState): number {
  return (
    faction.debt * 1.7
    + faction.activeConflicts.length * 18
    + faction.embargoes.length * 8
    + Math.max(0, 52 - faction.credibility) * 1.2
    + Math.max(0, 48 - faction.favour)
    + Math.max(0, faction.dependence - 34) * 0.8
  );
}

function shouldFormAlliance(
  factionA: FactionState,
  factionB: FactionState,
  existingAlliances: FactionAlliance[]
): { form: boolean; kind: FactionAlliance["kind"] } {
  const relation = factionA.relations[factionB.id] ?? 0;
  const alreadyAllied = existingAlliances.some(
    (a) =>
      !a.brokenDay
      && ((a.factionA === factionA.id && a.factionB === factionB.id)
        || (a.factionA === factionB.id && a.factionB === factionA.id))
  );

  if (alreadyAllied || relation < 40) {
    return { form: false, kind: "trade" };
  }

  const bothTrade =
    factionA.tradeAccess && factionB.tradeAccess
    && (factionA.currentAgenda === "trade" || factionB.currentAgenda === "trade"
      || factionA.profile === "mercantile" || factionB.profile === "mercantile");
  const bothMilitary =
    factionA.currentAgenda === "war" && factionB.currentAgenda === "war"
    && !factionA.activeConflicts.includes(factionB.id);
  const bothFaith =
    factionA.currentAgenda === "faith" && factionB.currentAgenda === "faith";

  if (relation >= 60 && bothMilitary) {
    return { form: true, kind: "military" };
  }
  if (relation >= 50 && bothTrade) {
    return { form: true, kind: "trade" };
  }
  if (relation >= 45 && bothFaith) {
    return { form: true, kind: "cultural" };
  }

  return { form: false, kind: "trade" };
}

function shouldBreakAlliance(
  alliance: FactionAlliance,
  factions: Record<FactionId, FactionState>
): { fracture: boolean; reason: string } {
  const factionA = factions[alliance.factionA];
  const factionB = factions[alliance.factionB];
  if (!factionA || !factionB) {
    return { fracture: true, reason: "faction no longer exists" };
  }

  const relation = factionA.relations[factionB.id] ?? 0;
  if (relation <= -10) {
    return { fracture: true, reason: "relations have soured beyond repair" };
  }
  if (factionA.activeConflicts.includes(factionB.id) || factionB.activeConflicts.includes(factionA.id)) {
    return { fracture: true, reason: "open conflict has erupted between allies" };
  }
  if (alliance.kind === "trade" && (!factionA.tradeAccess || !factionB.tradeAccess)) {
    return { fracture: true, reason: "trade routes have been severed" };
  }

  return { fracture: false, reason: "" };
}

function advanceAlliances(
  state: GameState,
  wh: WorldHistoryState,
  day: number
): { alliances: FactionAlliance[]; events: WorldHistoryEvent[] } {
  const seed = state.worldSeed;
  const factions = state.factions;
  const orderedFactions = Object.values(factions).sort((a, b) => a.id.localeCompare(b.id));
  let alliances = [...wh.alliances];
  const newEvents: WorldHistoryEvent[] = [];
  let eventCounter = 0;

  // Check for alliance fractures
  for (const alliance of alliances) {
    if (alliance.brokenDay) {
      continue;
    }
    const { fracture, reason } = shouldBreakAlliance(alliance, factions);
    if (fracture) {
      alliances = alliances.map((a) =>
        a.id === alliance.id ? { ...a, brokenDay: day, breakReason: reason } : a
      );
      const fA = factions[alliance.factionA];
      const fB = factions[alliance.factionB];
      newEvents.push(
        createWorldHistoryEvent(
          `wh-alliance-broken-${day}-${eventCounter++}`,
          day,
          state.clock.year,
          state.clock.month,
          "alliance_broken",
          [alliance.factionA, alliance.factionB],
          `${fA?.name ?? alliance.factionA}–${fB?.name ?? alliance.factionB} ${alliance.kind} pact dissolves`,
          `The ${alliance.kind} alliance between ${fA?.name ?? alliance.factionA} and ${fB?.name ?? alliance.factionB} has broken: ${reason}.`,
          {
            factionRelationDeltas: [{ factionA: alliance.factionA, factionB: alliance.factionB, delta: -8 }]
          }
        )
      );
    }
  }

  // Check for new alliances
  for (let i = 0; i < orderedFactions.length; i += 1) {
    for (let j = i + 1; j < orderedFactions.length; j += 1) {
      const fA = orderedFactions[i]!;
      const fB = orderedFactions[j]!;
      const { form, kind } = shouldFormAlliance(fA, fB, alliances);
      if (!form) {
        continue;
      }

      // Use seeded RNG to add some stochasticity — only 40% chance per eligible pair per month
      const roll = hash(seed + day * 37 + fA.id.length * 13 + fB.id.length * 7);
      if (roll > 0.4) {
        continue;
      }

      const newAlliance: FactionAlliance = {
        id: `alliance-${fA.id}-${fB.id}-${day}`,
        factionA: fA.id,
        factionB: fB.id,
        formedDay: day,
        kind,
        strength: Math.round(((fA.relations[fB.id] ?? 0) + 50) / 2),
        oracleMediated: false
      };
      alliances = [...alliances, newAlliance];
      newEvents.push(
        createWorldHistoryEvent(
          `wh-alliance-formed-${day}-${eventCounter++}`,
          day,
          state.clock.year,
          state.clock.month,
          "alliance_formed",
          [fA.id, fB.id],
          `${fA.name}–${fB.name} ${kind} pact formed`,
          `${fA.name} and ${fB.name} have entered a ${kind} alliance, strengthening their mutual position.`,
          {
            factionRelationDeltas: [{ factionA: fA.id, factionB: fB.id, delta: 5 }]
          }
        )
      );
    }
  }

  return { alliances, events: newEvents };
}

function advancePhilosophySpreads(
  state: GameState,
  day: number
): WorldHistoryEvent[] {
  const philosophers = state.philosophers;
  if (!philosophers) {
    return [];
  }

  const events: WorldHistoryEvent[] = [];
  let eventCounter = 0;
  const orderedEntries = (Object.entries(philosophers.byFaction) as [FactionId, PhilosopherThreatState][])
    .sort(([a], [b]) => a.localeCompare(b));

  for (const [factionId, threat] of orderedEntries) {
    if (!threat.active) {
      continue;
    }
    // "circle" or higher = circle, sect, crisis
    const isSpreadStage = threat.stage === "circle" || threat.stage === "sect" || threat.stage === "crisis";
    if (!isSpreadStage) {
      continue;
    }

    // Only generate once per philosopher reaching a new stage — use lastShiftMonth to avoid duplicates
    if (threat.lastShiftMonth === state.clock.month) {
      continue;
    }

    // Use seeded RNG — only 50% chance per qualifying philosopher per month
    const roll = hash(state.worldSeed + day * 43 + factionId.length * 11);
    if (roll > 0.5) {
      continue;
    }

    const faction = state.factions[factionId];
    const regionNode = state.campaign.worldMap.nodes.find(
      (n) => n.controllingFactionId === factionId
    );

    events.push(
      createWorldHistoryEvent(
        `wh-philosophy-spread-${day}-${eventCounter++}`,
        day,
        state.clock.year,
        state.clock.month,
        "philosophy_spread",
        [factionId],
        `${threat.worldview} philosophy spreads in ${faction?.name ?? factionId}`,
        `The ${threat.worldview} movement has reached the "${threat.stage}" stage in ${faction?.name ?? factionId}${regionNode ? ` (${regionNode.label})` : ""}, drawing new adherents and challenging oracle authority.`,
        {
          credibilityDelta: threat.stage === "crisis" ? -5 : threat.stage === "sect" ? -3 : -1,
          pressureDelta: threat.pressure
        },
        false,
        undefined,
        regionNode?.id
      )
    );
  }

  return events;
}

function advancePilgrimSurges(
  state: GameState,
  day: number
): WorldHistoryEvent[] {
  const events: WorldHistoryEvent[] = [];
  const avgCredibility =
    Object.values(state.factions).reduce((sum, f) => sum + f.credibility, 0)
    / Math.max(1, Object.values(state.factions).length);

  if (avgCredibility < 55) {
    return events;
  }

  // Check for a major prophecy delivered recently (within last 30 days)
  const recentProphecy = state.consultation.history.find(
    (p) => p.resolved && p.dayIssued >= day - 30 && (p.credibilityDelta ?? 0) >= 3
  );
  if (!recentProphecy) {
    return events;
  }

  const faction = state.factions[recentProphecy.factionId];
  events.push(
    createWorldHistoryEvent(
      `wh-pilgrimage-surge-${day}`,
      day,
      state.clock.year,
      state.clock.month,
      "pilgrimage_surge",
      [recentProphecy.factionId],
      `Pilgrim surge following prophecy for ${faction?.name ?? recentProphecy.factionId}`,
      `Word of the oracle's accurate prophecy for ${faction?.name ?? recentProphecy.factionId} has spread. Pilgrims flock to Delphi, increasing consultation frequency and offerings.`,
      {
        pilgrimDelta: Math.round(avgCredibility / 10),
        credibilityDelta: 2
      },
      true,
      recentProphecy.id
    )
  );

  return events;
}

function advanceRevolutions(
  state: GameState,
  day: number
): { events: WorldHistoryEvent[]; factions: Record<FactionId, FactionState> } {
  const seed = state.worldSeed;
  const events: WorldHistoryEvent[] = [];
  let factions = state.factions;
  let eventCounter = 0;
  const orderedFactions = Object.values(factions).sort((a, b) => a.id.localeCompare(b.id));

  for (const faction of orderedFactions) {
    const pressureScore = computeFactionPressureScore(faction);

    if (pressureScore < 80) {
      continue;
    }

    // Seeded probability — higher pressure = higher chance, but max 30%
    const revolutionChance = Math.min(0.3, (pressureScore - 80) / 100);
    const roll = hash(seed + day * 53 + faction.id.length * 17);
    if (roll > revolutionChance) {
      continue;
    }

    const agendaRotation: Record<FactionAgenda, FactionAgenda> = {
      war: "trade",
      trade: "faith",
      faith: "succession",
      succession: "war"
    };
    const newAgenda = agendaRotation[faction.currentAgenda];
    const regionNode = state.campaign.worldMap.nodes.find(
      (n) => n.controllingFactionId === faction.id
    );

    events.push(
      createWorldHistoryEvent(
        `wh-revolution-${day}-${eventCounter++}`,
        day,
        state.clock.year,
        state.clock.month,
        "revolution",
        [faction.id],
        `Revolution erupts in ${faction.name}`,
        `Internal pressure has boiled over in ${faction.name}. The old regime falls as new leaders seize power, shifting the faction's agenda from ${faction.currentAgenda} to ${newAgenda}.`,
        {
          pressureDelta: -pressureScore,
          credibilityDelta: -8
        },
        false,
        undefined,
        regionNode?.id
      )
    );

    factions = {
      ...factions,
      [faction.id]: {
        ...faction,
        currentAgenda: newAgenda,
        credibility: Math.max(0, Math.round(faction.credibility * 0.6)),
        favour: Math.max(0, Math.round(faction.favour * 0.7)),
        debt: Math.max(0, Math.round(faction.debt * 0.5)),
        dependence: Math.min(100, Math.round(faction.dependence + 10)),
        activeConflicts: [],
        history: [
          `Month ${state.clock.month}: Revolution! ${faction.name} sees regime change. New agenda: ${newAgenda}.`,
          ...faction.history
        ].slice(0, 4)
      }
    };
  }

  return { events, factions };
}

export function advanceWorldHistory(state: GameState): GameState {
  const wh = state.worldHistory ?? {
    events: [],
    alliances: [],
    hegemon: { hegemonScore: {}, hegemonHistory: [] },
    revolutionCount: 0,
    lastEventDay: 0
  };

  const day = getAbsoluteDay(state.clock);
  const newEvents: WorldHistoryEvent[] = [];

  // 1. Alliance formation/fracture
  const allianceResult = advanceAlliances(state, wh, day);
  newEvents.push(...allianceResult.events);

  // 2. Hegemon evaluation
  const factionScores = Object.fromEntries(
    Object.values(state.factions).map((f) => [f.id, factionInfluenceScore(f)])
  );
  const prevHegemonId = wh.hegemon.currentHegemonId;
  const newHegemon = evaluateHegemon(wh.hegemon, factionScores, day);

  if (newHegemon.currentHegemonId !== prevHegemonId) {
    if (newHegemon.currentHegemonId) {
      const hegemonFaction = state.factions[newHegemon.currentHegemonId];
      newEvents.push(
        createWorldHistoryEvent(
          `wh-hegemon-emerged-${day}`,
          day,
          state.clock.year,
          state.clock.month,
          "hegemon_emerged",
          [newHegemon.currentHegemonId],
          `${hegemonFaction?.name ?? newHegemon.currentHegemonId} emerges as hegemon`,
          `${hegemonFaction?.name ?? newHegemon.currentHegemonId} has accumulated enough influence to dominate the Greek world as hegemon.`,
          { credibilityDelta: 3 }
        )
      );
    } else if (prevHegemonId) {
      const prevFaction = state.factions[prevHegemonId];
      newEvents.push(
        createWorldHistoryEvent(
          `wh-hegemon-declined-${day}`,
          day,
          state.clock.year,
          state.clock.month,
          "hegemon_declined",
          [prevHegemonId],
          `${prevFaction?.name ?? prevHegemonId} loses hegemonic status`,
          `${prevFaction?.name ?? prevHegemonId} has declined and can no longer maintain its hegemonic grip on the Greek world.`,
          { credibilityDelta: -3 }
        )
      );
    }
  }

  // 3. Philosophy spread
  const philosophyEvents = advancePhilosophySpreads(state, day);
  newEvents.push(...philosophyEvents);

  // 4. Pilgrim surges
  const pilgrimEvents = advancePilgrimSurges(state, day);
  newEvents.push(...pilgrimEvents);

  // 5. Revolutions
  const revolutionResult = advanceRevolutions(state, day);
  newEvents.push(...revolutionResult.events);

  const revolutionCount = wh.revolutionCount + revolutionResult.events.length;

  // Enrich new events with narrated descriptions
  const factionNameMap: Record<string, { name: string }> = {};
  for (const f of Object.values(state.factions)) {
    factionNameMap[f.id] = { name: f.name };
  }
  const enrichedEvents = newEvents.map((event, index) => ({
    ...event,
    description: narrateWorldEvent(event, factionNameMap, state.worldSeed + day * 17 + index * 53)
  }));

  const updatedWorldHistory: WorldHistoryState = {
    events: [...wh.events, ...enrichedEvents],
    alliances: allianceResult.alliances,
    hegemon: newHegemon,
    revolutionCount,
    lastEventDay: enrichedEvents.length > 0 ? day : wh.lastEventDay
  };

  return {
    ...state,
    factions: revolutionResult.factions,
    worldHistory: updatedWorldHistory
  };
}

export function buildingDisplayColor(defId: BuildingDefId): number {
  return buildingDefs[defId]?.color ?? 0xb98b4d;
}
