import { buildingDefs, scenarioDefs } from "@the-oracle/content";

import type {
  BuildingDefId,
  CampaignPressureIndicator,
  CampaignState,
  CrisisChainState,
  EventFeedItem,
  FactionId,
  FactionState,
  GameState,
  ReputationState,
  ReputationTierId,
  ScenarioId,
  WorldMapLink,
  WorldMapNode
} from "./gameState";

const TIER_ORDER: ReputationTierId[] = ["obscure", "recognized", "revered", "panhellenic"];
const NODE_FALLBACKS: Record<FactionId, string> = {
  athens: "athens",
  sparta: "sparta",
  corinth: "corinth",
  thebes: "thebes",
  argos: "sparta",
  miletus: "athens",
  syracuse: "corinth",
  macedon: "thebes"
};

export const DEFAULT_REPUTATION_THRESHOLDS: Record<ReputationTierId, number> = {
  obscure: 0,
  recognized: 20,
  revered: 50,
  panhellenic: 85
};

function tierRank(tier: ReputationTierId): number {
  return TIER_ORDER.indexOf(tier);
}

export function resolveReputationTier(
  score: number,
  thresholds: Record<ReputationTierId, number> = DEFAULT_REPUTATION_THRESHOLDS
): ReputationTierId {
  let currentTier: ReputationTierId = "obscure";

  for (const tier of TIER_ORDER) {
    if (score >= thresholds[tier]) {
      currentTier = tier;
    }
  }

  return currentTier;
}

function nextTier(currentTier: ReputationTierId): ReputationTierId | undefined {
  const index = TIER_ORDER.indexOf(currentTier);
  return index >= 0 ? TIER_ORDER[index + 1] : undefined;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function scenarioFor(scenarioId: ScenarioId) {
  return scenarioDefs.find((scenario) => scenario.id === scenarioId) ?? scenarioDefs[0]!;
}

function createWinConditionForScenario(scenarioId: ScenarioId): CampaignState["winCondition"] {
  if (scenarioId === "sandbox") {
    return {
      id: "sandbox-endless",
      label: "Sandbox has no fixed ending.",
      completed: false,
      summary: "Open-ended mode. Build freely without a campaign victory target."
    };
  }

  return {
    id: "rise-to-revered",
    label: "Reach Revered reputation and survive the first regional crisis chain.",
    completed: false,
    summary: "Tier Obscure / Revered, dedications 0, crises resolved 0 / 1."
  };
}

function defaultWorldNodes(): WorldMapNode[] {
  return [
    {
      id: "delphi",
      label: "Delphi",
      position: { x: 42, y: 28 },
      tags: ["sanctuary", "city"],
      pressure: 18,
      unrest: 8,
      connectedNodeIds: ["corinth", "thebes"]
    },
    {
      id: "athens",
      label: "Athens",
      position: { x: 63, y: 45 },
      controllingFactionId: "athens",
      tags: ["city", "port"],
      pressure: 24,
      unrest: 21,
      connectedNodeIds: ["corinth", "miletus"]
    },
    {
      id: "sparta",
      label: "Sparta",
      position: { x: 28, y: 58 },
      controllingFactionId: "sparta",
      tags: ["city", "pass"],
      pressure: 22,
      unrest: 17,
      connectedNodeIds: ["argos", "corinth"]
    },
    {
      id: "corinth",
      label: "Corinth",
      position: { x: 51, y: 48 },
      controllingFactionId: "corinth",
      tags: ["city", "port"],
      pressure: 26,
      unrest: 12,
      connectedNodeIds: ["delphi", "athens", "sparta", "argos"]
    },
    {
      id: "thebes",
      label: "Thebes",
      position: { x: 53, y: 34 },
      controllingFactionId: "thebes",
      tags: ["city"],
      pressure: 20,
      unrest: 10,
      connectedNodeIds: ["delphi", "athens"]
    }
  ];
}

function defaultWorldLinks(): WorldMapLink[] {
  return [
    { id: "delphi-corinth", fromNodeId: "delphi", toNodeId: "corinth", kind: "road", tradeFlow: 52, risk: 12 },
    { id: "delphi-thebes", fromNodeId: "delphi", toNodeId: "thebes", kind: "road", tradeFlow: 38, risk: 15 },
    { id: "corinth-athens", fromNodeId: "corinth", toNodeId: "athens", kind: "sea", tradeFlow: 64, risk: 18 },
    { id: "corinth-sparta", fromNodeId: "corinth", toNodeId: "sparta", kind: "road", tradeFlow: 44, risk: 23 },
    { id: "athens-thebes", fromNodeId: "athens", toNodeId: "thebes", kind: "road", tradeFlow: 41, risk: 17 }
  ];
}

function unlockedBuildingIdsForTier(currentTier: ReputationTierId): BuildingDefId[] {
  return Object.values(buildingDefs)
    .filter((building) => !building.unlockTier || tierRank(building.unlockTier) <= tierRank(currentTier))
    .map((building) => building.id);
}

function unlockedScenarioIdsForTier(currentTier: ReputationTierId): ScenarioId[] {
  return scenarioDefs
    .filter((scenario) => !scenario.recommendedStartingTier || tierRank(scenario.recommendedStartingTier) <= tierRank(currentTier))
    .map((scenario) => scenario.id);
}

function normalizeReputationState(reputation?: Partial<ReputationState>): ReputationState {
  const score = Math.max(0, Math.round(reputation?.score ?? 0));
  const thresholds = {
    ...DEFAULT_REPUTATION_THRESHOLDS,
    ...(reputation?.thresholds ?? {})
  };
  const currentTier = resolveReputationTier(score, thresholds);

  return {
    score,
    currentTier,
    nextTier: nextTier(currentTier),
    thresholds,
    unlockedBuildingIds: unlockedBuildingIdsForTier(currentTier),
    unlockedScenarioIds: unlockedScenarioIdsForTier(currentTier),
    lastTierChangeDay: reputation?.lastTierChangeDay
  };
}

export function syncCampaignState(campaign: CampaignState, absoluteDay?: number): CampaignState {
  const reputation = normalizeReputationState(campaign.reputation);
  const tierChanged = campaign.reputation.currentTier !== reputation.currentTier;

  return {
    ...campaign,
    reputation: {
      ...reputation,
      lastTierChangeDay: tierChanged ? absoluteDay ?? campaign.reputation.lastTierChangeDay : campaign.reputation.lastTierChangeDay
    }
  };
}

export function createInitialCampaignState(scenarioId: ScenarioId = "rising-oracle"): CampaignState {
  const base: CampaignState = {
    scenarioId,
    reputation: normalizeReputationState({
      score: 0
    }),
    treasury: {
      completed: 0,
      totalGoldInvested: 0,
      nextMilestoneGold: 150
    },
    patronMilestones: [],
    worldMap: {
      selectedNodeId: "delphi",
      nodes: defaultWorldNodes(),
      links: defaultWorldLinks(),
      activePressures: [],
      crisisChains: []
    },
    winCondition: createWinConditionForScenario(scenarioId)
  };

  return syncCampaignState(base);
}

export function normalizeCampaignState(campaign?: Partial<CampaignState>): CampaignState {
  const base = createInitialCampaignState(campaign?.scenarioId ?? "rising-oracle");

  return syncCampaignState({
    scenarioId: campaign?.scenarioId ?? base.scenarioId,
    reputation: normalizeReputationState(campaign?.reputation),
    treasury: {
      ...base.treasury,
      ...(campaign?.treasury ?? {})
    },
    patronMilestones: [...(campaign?.patronMilestones ?? base.patronMilestones)],
    worldMap: {
      selectedNodeId: campaign?.worldMap?.selectedNodeId ?? base.worldMap.selectedNodeId,
      nodes: [...(campaign?.worldMap?.nodes ?? base.worldMap.nodes)],
      links: [...(campaign?.worldMap?.links ?? base.worldMap.links)],
      activePressures: [...(campaign?.worldMap?.activePressures ?? base.worldMap.activePressures)],
      crisisChains: [...(campaign?.worldMap?.crisisChains ?? base.worldMap.crisisChains)]
    },
    winCondition: {
      ...base.winCondition,
      ...(campaign?.winCondition ?? {})
    }
  });
}

function pushMilestone(milestones: string[], entry: string): string[] {
  const next = [entry, ...milestones.filter((value) => value !== entry)];
  return next.slice(0, 8);
}

function prestigeEffectWeight(kind: NonNullable<(typeof buildingDefs)[BuildingDefId]["passiveEffects"]>[number]["kind"]): number {
  switch (kind) {
    case "prestige":
      return 5;
    case "omen_quality":
      return 4;
    case "trade_income":
      return 3;
    case "pilgrim_capacity":
      return 2;
    case "donation":
      return 3;
    case "storage_buffer":
    default:
      return 1.5;
  }
}

function precinctPrestige(state: GameState): number {
  return state.buildings.reduce((total, building) => {
    const def = buildingDefs[building.defId];
    const conditionRatio = building.maxCondition > 0 ? building.condition / building.maxCondition : 1;
    const categoryWeight =
      def.category === "ritual"
        ? 3.5
        : def.category === "trade"
          ? 2.8
          : def.category === "hospitality"
            ? 2.4
            : def.category === "production"
              ? 1.8
              : def.category === "storage"
                ? 1.4
                : def.category === "housing"
                  ? 1.2
                  : 0.8;
    const staffingWeight = building.assignedPriestIds.length > 0 ? 2.5 : 0;
    const passiveWeight = (def.passiveEffects ?? []).reduce((sum, effect) => sum + effect.value * prestigeEffectWeight(effect.kind), 0);

    return total + (categoryWeight + staffingWeight + passiveWeight) * conditionRatio;
  }, 0);
}

function recentConsultationMomentum(state: GameState, absoluteDay: number): number {
  return state.consultation.history.reduce((score, entry) => {
    const day = entry.resolvedDay ?? entry.dayIssued;
    if (absoluteDay - day > 45) {
      return score;
    }
    const delta = entry.credibilityDelta ?? 0;
    const deliveryWeight = 2;
    const outcomeWeight = delta > 0 ? delta * 0.6 : delta * 0.8;
    return score + deliveryWeight + outcomeWeight;
  }, 0);
}

function diplomaticReach(factions: Record<FactionId, FactionState>): number {
  return Object.values(factions).reduce((score, faction) => (
    score
      + (faction.tradeAccess ? 2 : 0)
      + faction.treaties.length
      - faction.embargoes.length
      + faction.activeConflicts.length
  ), 0);
}

function dedicationText(count: number): string {
  if (count === 1) {
    return "Dedicated the first treasury offering and drew the notice of major patrons.";
  }
  if (count === 2) {
    return "A second dedication secured regional patronage and steadier offerings.";
  }
  if (count === 3) {
    return "A third dedication made Delphi a fixture in Greek political theater.";
  }
  return `Treasury dedication ${count} deepened Delphi's political patronage network.`;
}

function calculateTreasuryContribution(state: GameState, factions: Record<FactionId, FactionState>): number {
  const prestige = precinctPrestige(state);
  const consultation = recentConsultationMomentum(
    state,
    (state.clock.year - 1) * 360 + (state.clock.month - 1) * 30 + state.clock.day
  );
  const surplusGold = Math.max(0, state.resources.gold.amount - 90);
  const openRoutes = Object.values(factions).filter((faction) => faction.tradeAccess).length;
  const treatyCount = Object.values(factions).reduce((total, faction) => total + faction.treaties.length, 0);
  const penalty = Object.values(factions).reduce((total, faction) => total + faction.embargoes.length + faction.activeConflicts.length, 0);

  return clamp(
    Math.round(surplusGold / 5 + prestige * 0.55 + consultation * 0.45 + openRoutes * 3 + treatyCount * 0.4 - penalty * 1.2),
    0,
    90
  );
}

function advanceTreasury(
  campaign: CampaignState,
  state: GameState,
  factions: Record<FactionId, FactionState>,
  absoluteDay: number
): { treasury: CampaignState["treasury"]; milestones: string[]; feedItems: EventFeedItem[] } {
  const scenario = scenarioFor(campaign.scenarioId);
  if (campaign.scenarioId === "sandbox") {
    return {
      treasury: campaign.treasury,
      milestones: [],
      feedItems: []
    };
  }

  const contribution = calculateTreasuryContribution(state, factions);
  const thresholds = scenario.dedicationMilestones ?? [campaign.treasury.nextMilestoneGold];
  let completed = campaign.treasury.completed;
  let nextMilestoneGold = campaign.treasury.nextMilestoneGold;
  let lastDedicationDay = campaign.treasury.lastDedicationDay;
  const totalGoldInvested = campaign.treasury.totalGoldInvested + contribution;
  const milestones: string[] = [];
  const feedItems: EventFeedItem[] = [];

  while (totalGoldInvested >= nextMilestoneGold) {
    completed += 1;
    lastDedicationDay = absoluteDay;
    const milestone = dedicationText(completed);
    milestones.push(milestone);
    feedItems.push({
      id: `event-dedication-${absoluteDay}-${completed}`,
      day: state.clock.day,
      text: milestone
    });
    nextMilestoneGold = thresholds[completed] ?? nextMilestoneGold + 150 + completed * 30;
  }

  return {
    treasury: {
      completed,
      totalGoldInvested,
      nextMilestoneGold,
      lastDedicationDay
    },
    milestones,
    feedItems
  };
}

function calculateReputationDelta(
  state: GameState,
  campaign: CampaignState,
  factions: Record<FactionId, FactionState>,
  absoluteDay: number
): number {
  const prestige = precinctPrestige(state);
  const consultation = recentConsultationMomentum(state, absoluteDay);
  const diplomatic = diplomaticReach(factions);
  const openRoutes = Object.values(factions).filter((faction) => faction.tradeAccess).length;
  const unresolvedConsequences = state.consequences.filter((entry) => !entry.resolved).length;
  const criticalEmbargoBlocs = Object.values(factions).filter((faction) => faction.embargoes.length >= 2).length;
  const ritualPenalty = state.resources.incense.amount < 6 ? 4 : 0;
  const emptySanctumPenalty = state.buildings.some((building) => building.defId === "inner_sanctum") ? 0 : 5;
  const baseDelta =
    prestige * 0.16
    + consultation * 0.32
    + diplomatic * 0.22
    + campaign.treasury.completed * 3.5
    + openRoutes * 0.8
    - unresolvedConsequences * 2.4
    - criticalEmbargoBlocs * 1.5
    - ritualPenalty
    - emptySanctumPenalty;

  return clamp(Math.round(baseDelta), -6, 16);
}

function factionNarrative(faction: FactionState): string {
  return `${faction.lastOutcome ?? ""} ${faction.history.join(" ")}`.toLowerCase();
}

function factionInfluenceScore(faction: FactionState): number {
  const relationValues = Object.values(faction.relations);
  const averageRelation = relationValues.length > 0
    ? relationValues.reduce((sum, value) => sum + value, 0) / relationValues.length
    : 0;

  return (
    faction.credibility * 0.38
    + faction.favour * 0.28
    + faction.treaties.length * 7
    + (faction.tradeAccess ? 9 : -4)
    - faction.embargoes.length * 6
    - faction.activeConflicts.length * 5
    - faction.debt * 1.1
    - faction.dependence * 0.18
    + averageRelation * 0.2
  );
}

function dominantBlocLeaderId(factions: Record<FactionId, FactionState>): FactionId | undefined {
  const ranked = Object.values(factions)
    .map((faction) => ({
      id: faction.id,
      score: factionInfluenceScore(faction)
    }))
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id));
  const leader = ranked[0];
  const runnerUp = ranked[1];

  if (!leader) {
    return undefined;
  }

  return leader.score >= 44 && leader.score - (runnerUp?.score ?? 0) >= 6 ? leader.id : undefined;
}

function factionUnderHegemonPressure(faction: FactionState, factions: Record<FactionId, FactionState>): boolean {
  const leaderId = dominantBlocLeaderId(factions);
  if (!leaderId || leaderId === faction.id) {
    return false;
  }
  const leader = factions[leaderId];
  return (
    factionNarrative(faction).includes("dominant bloc squeezes")
    || faction.embargoes.includes(leaderId)
    || leader.embargoes.includes(faction.id)
    || faction.activeConflicts.includes(leaderId)
  );
}

function factionDestabilizationScore(faction: FactionState): number {
  const narrative = factionNarrative(faction);
  return (
    faction.debt * 1.5
    + faction.activeConflicts.length * 18
    + faction.embargoes.length * 6
    + Math.max(0, 52 - faction.credibility) * 1.2
    + Math.max(0, 48 - faction.favour)
    + (narrative.includes("government crisis") ? 18 : 0)
    + (narrative.includes("revolutionary agitation") ? 28 : 0)
  );
}

function factionIdeologyPressure(faction: FactionState): number {
  const narrative = factionNarrative(faction);
  return (
    (narrative.includes("creed") ? 10 : 0)
    + (narrative.includes("revival") ? 10 : 0)
    + (narrative.includes("legitimacy") ? 12 : 0)
    + (faction.currentAgenda === "faith" ? 6 : 0)
    + (faction.currentAgenda === "succession" ? 8 : 0)
  );
}

function pressureKindForFaction(
  faction: FactionState,
  factions: Record<FactionId, FactionState>
): CampaignPressureIndicator["kind"] {
  if (faction.activeConflicts.length > 0 || faction.currentAgenda === "war" || factionDestabilizationScore(faction) >= 78) {
    return "conflict";
  }
  if (!faction.tradeAccess || faction.debt >= 14 || faction.embargoes.length > 0 || factionUnderHegemonPressure(faction, factions)) {
    return "trade";
  }
  if (factionIdeologyPressure(faction) >= 14) {
    return "consultation";
  }
  return "consultation";
}

function pressureNodeId(nodes: WorldMapNode[], factionId: FactionId, kind: CampaignPressureIndicator["kind"]): string {
  if (kind === "consultation") {
    return "delphi";
  }

  const direct = nodes.find((node) => node.controllingFactionId === factionId);
  if (direct) {
    return direct.id;
  }

  return NODE_FALLBACKS[factionId];
}

function pressureValueForFaction(
  state: GameState,
  campaign: CampaignState,
  factions: Record<FactionId, FactionState>,
  faction: FactionState,
  absoluteDay: number
): number {
  const unresolvedForFaction = state.consequences.filter((entry) => !entry.resolved && entry.factionId === faction.id).length;
  const recentProphecies = state.consultation.history.filter((entry) => entry.factionId === faction.id && absoluteDay - entry.dayIssued <= 60).length;
  const relief = campaign.reputation.score * 0.2 + campaign.treasury.completed * 8;
  const kind = pressureKindForFaction(faction, factions);
  const destabilization = factionDestabilizationScore(faction);
  const ideologyPressure = factionIdeologyPressure(faction);
  const hegemonPressure = factionUnderHegemonPressure(faction, factions) ? 14 : 0;
  let base =
    faction.dependence * 0.45
    + Math.max(0, 52 - faction.favour) * 0.35
    + Math.max(0, 50 - faction.credibility) * 0.28
    + destabilization * 0.15
    + ideologyPressure * 0.25;

  if (kind === "conflict") {
    base += 24 + faction.activeConflicts.length * 14 + faction.embargoes.length * 7 + destabilization * 0.22;
  } else if (kind === "trade") {
    base += 18 + faction.debt * 1.35 + (faction.tradeAccess ? 0 : 12) + faction.embargoes.length * 8 + hegemonPressure;
  } else {
    base += 16 + unresolvedForFaction * 14 + recentProphecies * 6 + ideologyPressure + (faction.currentAgenda === "faith" ? 8 : 0);
  }

  return clamp(Math.round(base - relief), 0, 100);
}

function buildPressureIndicators(
  state: GameState,
  campaign: CampaignState,
  factions: Record<FactionId, FactionState>,
  absoluteDay: number
): CampaignPressureIndicator[] {
  return Object.values(factions)
    .map((faction) => {
      const kind = pressureKindForFaction(faction, factions);
      const value = pressureValueForFaction(state, campaign, factions, faction, absoluteDay);
      const severity =
        value >= 75
          ? "critical"
          : value >= 55
            ? "rising"
            : "low";

      return {
        id: `pressure-${faction.id}-${kind}`,
        factionId: faction.id,
        nodeId: pressureNodeId(campaign.worldMap.nodes, faction.id, kind),
        kind,
        severity,
        value
      } satisfies CampaignPressureIndicator;
    })
    .filter((pressure) => pressure.value >= 35)
    .sort((left, right) => right.value - left.value || left.id.localeCompare(right.id))
    .slice(0, 4);
}

function updateWorldNodes(
  campaign: CampaignState,
  factions: Record<FactionId, FactionState>,
  activePressures: CampaignPressureIndicator[]
): WorldMapNode[] {
  return campaign.worldMap.nodes.map((node) => {
    const controllingFaction = node.controllingFactionId ? factions[node.controllingFactionId] : undefined;
    const nodePressures = activePressures.filter((pressure) => pressure.nodeId === node.id);
    const externalPressure = nodePressures.reduce((sum, pressure) => sum + pressure.value * (pressure.kind === "conflict" ? 0.42 : pressure.kind === "trade" ? 0.35 : 0.28), 0);
    const factionPressure = controllingFaction
      ? controllingFaction.debt * 0.4
        + controllingFaction.activeConflicts.length * 12
        + controllingFaction.embargoes.length * 5
        + factionDestabilizationScore(controllingFaction) * 0.08
        - controllingFaction.treaties.length * 3
      : 0;
    const unrest = controllingFaction
      ? clamp(
          Math.round(
            10
            + controllingFaction.debt * 0.55
            + controllingFaction.activeConflicts.length * 10
            + controllingFaction.embargoes.length * 4
            + factionDestabilizationScore(controllingFaction) * 0.14
            + factionIdeologyPressure(controllingFaction) * 0.25
            - controllingFaction.favour * 0.12
          ),
          0,
          100
        )
      : clamp(Math.round(8 + externalPressure * 0.2 - campaign.reputation.score * 0.08), 0, 100);

    return {
      ...node,
      pressure: clamp(Math.round(12 + factionPressure + externalPressure - campaign.reputation.score * 0.1 - campaign.treasury.completed * 4), 0, 100),
      unrest
    };
  });
}

function crisisLabel(pressure: CampaignPressureIndicator, factions: Record<FactionId, FactionState>): string {
  const factionName = factions[pressure.factionId]?.name ?? "Regional";
  const faction = factions[pressure.factionId];
  const narrative = faction ? factionNarrative(faction) : "";
  if (pressure.kind === "conflict") {
    if (narrative.includes("revolutionary agitation") || narrative.includes("government crisis")) {
      return `${factionName} Regime Crisis`;
    }
    return `${factionName} League Fracture`;
  }
  if (pressure.kind === "trade") {
    if (narrative.includes("dominant bloc") || factionUnderHegemonPressure(faction, factions)) {
      return `${factionName} Hegemon Squeeze`;
    }
    return `${factionName} Caravan Squeeze`;
  }
  if (narrative.includes("creed") || narrative.includes("revival") || narrative.includes("legitimacy")) {
    return `${factionName} Creed Schism`;
  }
  return `${factionName} Consultation Surge`;
}

function advanceCrisisChains(
  campaign: CampaignState,
  factions: Record<FactionId, FactionState>,
  activePressures: CampaignPressureIndicator[],
  absoluteDay: number,
  currentDay: number
): { crisisChains: CrisisChainState[]; feedItems: EventFeedItem[] } {
  const previousChains = new Map(campaign.worldMap.crisisChains.map((chain) => [chain.id, chain]));
  const feedItems: EventFeedItem[] = [];
  const nextChains: CrisisChainState[] = [];
  const relief = campaign.reputation.score + campaign.treasury.completed * 18;
  const seen = new Set<string>();

  for (const pressure of activePressures.filter((entry) => entry.severity !== "low")) {
    const id = `crisis-${pressure.nodeId}-${pressure.kind}`;
    const previous = previousChains.get(id);
    const netPressure = pressure.value - relief * 0.28;
    let stage: CrisisChainState["stage"];
    let stepsCompleted: number;
    let resolvedDay = previous?.resolvedDay;

    if (!previous) {
      stage = pressure.severity === "critical" || netPressure >= 48 ? "active" : "rumor";
      stepsCompleted = stage === "active" ? 2 : 1;
      feedItems.push({
        id: `event-crisis-${id}-${absoluteDay}`,
        day: currentDay,
        text: stage === "active"
          ? `${crisisLabel(pressure, factions)} breaks into an active regional crisis.`
          : `${crisisLabel(pressure, factions)} begins to gather as rumor and pressure.`
      });
    } else if (previous.stage !== "resolution" && (pressure.severity === "critical" || netPressure >= 50)) {
      stage = "active";
      stepsCompleted = Math.min(4, Math.max(previous.stepsCompleted + 1, 2));
      if (previous.stage !== "active") {
        feedItems.push({
          id: `event-crisis-escalation-${id}-${absoluteDay}`,
          day: currentDay,
          text: `${crisisLabel(pressure, factions)} escalates beyond rumor and now strains the wider region.`
        });
      }
    } else if (previous.stage === "active" && relief >= pressure.value - 6) {
      stage = "resolution";
      stepsCompleted = Math.max(previous.stepsCompleted + 1, 3);
      resolvedDay = absoluteDay;
      feedItems.push({
        id: `event-crisis-resolution-${id}-${absoluteDay}`,
        day: currentDay,
        text: `${crisisLabel(pressure, factions)} begins to ease as Delphi's standing steadies the region.`
      });
    } else {
      stage = previous.stage;
      stepsCompleted = previous.stepsCompleted;
    }

    nextChains.push({
      id,
      label: crisisLabel(pressure, factions),
      nodeId: pressure.nodeId,
      factionId: pressure.factionId,
      stage,
      pressure: pressure.value,
      stepsCompleted,
      resolvedDay
    });
    seen.add(id);
  }

  for (const chain of campaign.worldMap.crisisChains) {
    if (seen.has(chain.id)) {
      continue;
    }
    if (chain.stage === "resolution") {
      nextChains.push(chain);
      continue;
    }
    if (relief >= chain.pressure + 4) {
      nextChains.push({
        ...chain,
        stage: "resolution",
        stepsCompleted: Math.max(chain.stepsCompleted + 1, 3),
        resolvedDay: chain.resolvedDay ?? absoluteDay,
        pressure: Math.max(12, chain.pressure - 18)
      });
      feedItems.push({
        id: `event-crisis-fade-${chain.id}-${absoluteDay}`,
        day: currentDay,
        text: `${chain.label} recedes before it can fully ignite.`
      });
    } else {
      nextChains.push({
        ...chain,
        pressure: Math.max(18, chain.pressure - 8)
      });
    }
  }

  return {
    crisisChains: nextChains
      .sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))
      .slice(0, 4),
    feedItems
  };
}

function updateWinCondition(campaign: CampaignState, absoluteDay: number): CampaignState["winCondition"] {
  const scenario = scenarioFor(campaign.scenarioId);
  if (campaign.scenarioId === "sandbox") {
    return {
      ...campaign.winCondition,
      label: "Sandbox has no fixed ending.",
      summary: "Open-ended mode. Build freely without a campaign victory target.",
      completed: false,
      completedDay: undefined
    };
  }

  const winningTier = scenario.winningTier ?? "revered";
  const requiredResolvedCrises = scenario.requiredResolvedCrises ?? 1;
  const resolvedCrises = campaign.worldMap.crisisChains.filter((chain) => chain.stage === "resolution").length;
  const completed =
    tierRank(campaign.reputation.currentTier) >= tierRank(winningTier)
    && campaign.treasury.completed >= 1
    && resolvedCrises >= requiredResolvedCrises;

  return {
    ...campaign.winCondition,
    label: "Reach Revered reputation and survive the first regional crisis chain.",
    summary: `Tier ${campaign.reputation.currentTier} / ${winningTier}, dedications ${campaign.treasury.completed}, crises resolved ${resolvedCrises} / ${requiredResolvedCrises}.`,
    completed,
    completedDay: completed ? campaign.winCondition.completedDay ?? absoluteDay : campaign.winCondition.completedDay
  };
}

export function advanceCampaignState(
  state: GameState,
  factions: Record<FactionId, FactionState>
): { campaign: CampaignState; feedItems: EventFeedItem[] } {
  const absoluteDay = (state.clock.year - 1) * 360 + (state.clock.month - 1) * 30 + state.clock.day;
  const feedItems: EventFeedItem[] = [];
  const tierBefore = state.campaign.reputation.currentTier;
  const treasuryUpdate = advanceTreasury(state.campaign, state, factions, absoluteDay);
  let patronMilestones = [...state.campaign.patronMilestones];

  for (const milestone of treasuryUpdate.milestones) {
    patronMilestones = pushMilestone(patronMilestones, milestone);
  }

  let nextCampaign = syncCampaignState(
    {
      ...state.campaign,
      treasury: treasuryUpdate.treasury,
      patronMilestones
    },
    absoluteDay
  );

  const reputationDelta = calculateReputationDelta(state, nextCampaign, factions, absoluteDay);
  nextCampaign = syncCampaignState(
    {
      ...nextCampaign,
      reputation: {
        ...nextCampaign.reputation,
        score: nextCampaign.reputation.score + reputationDelta
      }
    },
    absoluteDay
  );

  if (nextCampaign.reputation.currentTier !== tierBefore) {
    const tierMilestone = `Delphi has risen to ${nextCampaign.reputation.currentTier} reputation.`;
    nextCampaign = {
      ...nextCampaign,
      patronMilestones: pushMilestone(nextCampaign.patronMilestones, tierMilestone)
    };
    feedItems.push({
      id: `event-tier-${absoluteDay}-${nextCampaign.reputation.currentTier}`,
      day: state.clock.day,
      text: `${tierMilestone} New precinct works are now within reach.`
    });
  }

  const activePressures = buildPressureIndicators(state, nextCampaign, factions, absoluteDay);
  const nodes = updateWorldNodes(nextCampaign, factions, activePressures);
  const crisisUpdate = advanceCrisisChains(nextCampaign, factions, activePressures, absoluteDay, state.clock.day);
  const selectedNodeId =
    activePressures[0]?.nodeId
    ?? nextCampaign.worldMap.selectedNodeId
    ?? nodes[0]?.id;

  nextCampaign = {
    ...nextCampaign,
    worldMap: {
      ...nextCampaign.worldMap,
      selectedNodeId,
      nodes,
      activePressures,
      crisisChains: crisisUpdate.crisisChains
    }
  };
  nextCampaign = {
    ...nextCampaign,
    winCondition: updateWinCondition(nextCampaign, absoluteDay)
  };

  if (nextCampaign.winCondition.completed && !state.campaign.winCondition.completed) {
    feedItems.push({
      id: `event-win-${absoluteDay}`,
      day: state.clock.day,
      text: "The Rising Oracle is fulfilled: Delphi is revered and has weathered its first regional crisis."
    });
  }

  return {
    campaign: nextCampaign,
    feedItems: [...treasuryUpdate.feedItems, ...crisisUpdate.feedItems, ...feedItems]
  };
}
