import {
  ageDefs,
  ageDefById,
  buildingDefs,
  legendaryConsultationDefs,
  rivalOracleOperationDefById,
  type AgeDef,
  type LegendaryConsultationDef
} from "@the-oracle/content";

import type {
  BuildingInstance,
  GameState,
  LegendaryConsultationProgress,
  NamedCharacterState,
  OmenReport,
  PhilosopherThreatState,
  ProphecyDepthBand,
  ProphecyInterpretation,
  ProphecyRecord,
  ProphecyScaffoldPart,
  PythiaState,
  WalkerInstance,
  WordTile
} from "../state/gameState";
import type { LegacyArtifact, LegacyPhase } from "../state/legacy";
import { computeLegacyScore } from "../state/legacy";
import type { RunRecord, CarryoverBonus, BurdenId } from "../state/lineage";
import { createInitialLineageState } from "../state/lineage";
import { challengeSeeds } from "../state/challengeSeeds";
import type { ChallengeSeed } from "../state/challengeSeeds";
import type {
  ExcavationSite,
  Relic,
  RelicEffect,
  SacredSite
} from "../state/excavation";
import { getAbsoluteDay } from "../simulation/clock";
import { buildRunSetupOriginOptions, buildRunSetupPreview } from "../state/worldGen";

export function selectSelectedBuilding(state: GameState): BuildingInstance | undefined {
  if (state.ui.selectedEntityKind !== "building") {
    return undefined;
  }
  return state.buildings.find((building) => building.id === state.ui.selectedEntityId);
}

export function selectSelectedWalker(state: GameState): WalkerInstance | undefined {
  if (state.ui.selectedEntityKind !== "walker") {
    return undefined;
  }
  return state.walkers.find((walker) => walker.id === state.ui.selectedEntityId);
}

export function selectTopAdvisorMessage(state: GameState) {
  return state.advisorMessages[0];
}

export function selectBuildingSummary(state: GameState) {
  return state.buildings.map((building) => ({
    id: building.id,
    name: buildingDefs[building.defId].name,
    tile: building.position,
    condition: Math.round(building.condition),
    assignedPriests: building.assignedPriestIds.length
  }));
}

export function selectPrecinctHealth(state: GameState): number {
  if (state.buildings.length === 0) {
    return 100;
  }
  const total = state.buildings.reduce((sum, building) => sum + building.condition / building.maxCondition, 0);
  return Math.round((total / state.buildings.length) * 100);
}

export function selectCarrierSummary(state: GameState) {
  const carriers = state.walkers.filter((walker) => walker.role === "carrier");
  if (carriers.length === 0) {
    return {
      count: 0,
      averageFatigue: 0,
      highestFatigue: 0,
      activeJobs: 0,
      strainedCount: 0,
      distinctRadii: [] as number[]
    };
  }

  const averageFatigue = carriers.reduce((sum, walker) => sum + (walker.fatigue ?? 0), 0) / carriers.length;
  const highestFatigue = Math.max(...carriers.map((walker) => walker.fatigue ?? 0));
  const distinctRadii = [...new Set(carriers.map((walker) => walker.supplyRadius ?? 0))].sort((left, right) => left - right);

  return {
    count: carriers.length,
    averageFatigue: Math.round(averageFatigue * 10) / 10,
    highestFatigue: Math.round(highestFatigue * 10) / 10,
    activeJobs: carriers.filter((walker) => Boolean(walker.assignedJobId)).length,
    strainedCount: carriers.filter((walker) => (walker.fatigue ?? 0) >= 45).length,
    distinctRadii
  };
}

function stableCharacterRoster(state: GameState): NamedCharacterState[] {
  return [...(state.characters?.roster ?? [])]
    .sort((left, right) => right.prominence - left.prominence || right.influence - left.influence || left.id.localeCompare(right.id));
}

function stablePriestCharacterRoster(state: GameState): NamedCharacterState[] {
  return stableCharacterRoster(state).filter((character) => character.role === "priest");
}

function describeCharacterTone(character: NamedCharacterState): "trusted" | "watchful" | "volatile" | "cold" {
  if (character.relationship.trust >= 55 && character.relationship.hostility <= 18) {
    return "trusted";
  }
  if (character.relationship.hostility >= 40 || character.relationship.fear >= 48) {
    return "volatile";
  }
  if (character.relationship.trust <= 20 && character.relationship.familiarity <= 20) {
    return "cold";
  }
  return "watchful";
}

export function selectCharacterRoster(state: GameState) {
  return stableCharacterRoster(state).map((character) => ({
    id: character.id,
    displayName: character.displayName,
    role: character.role,
    cadence: character.cadence,
    factionId: character.homeFactionId ?? null,
    factionName: character.homeFactionId ? state.factions[character.homeFactionId]?.name ?? null : null,
    regionId: character.anchorRegionId ?? null,
    regionLabel: character.anchorRegionId
      ? state.worldGeneration.regions.find((region) => region.id === character.anchorRegionId)?.label ?? null
      : null,
    influence: character.influence,
    prominence: character.prominence,
    trust: character.relationship.trust,
    fear: character.relationship.fear,
    hostility: character.relationship.hostility,
    familiarity: character.relationship.familiarity,
    visitCount: character.memory.visitCount,
    lastSeenDay: character.memory.lastSeenDay ?? null,
    lastImpression: character.memory.lastImpression ?? null,
    tone: describeCharacterTone(character),
    tags: character.tags
  }));
}

export function selectCharacterSpotlight(state: GameState) {
  const roster = stableCharacterRoster(state);
  if (roster.length === 0) {
    return [];
  }

  const savedSpotlightIds = state.characters?.spotlightCharacterIds ?? [];
  const spotlightIds = savedSpotlightIds.length
    ? savedSpotlightIds
    : roster.slice(0, 3).map((character) => character.id);

  return spotlightIds
    .map((characterId) => roster.find((entry) => entry.id === characterId))
    .filter((character): character is NamedCharacterState => Boolean(character))
    .map((character) => ({
      id: character.id,
      displayName: character.displayName,
      role: character.role,
      factionName: character.homeFactionId ? state.factions[character.homeFactionId]?.name ?? null : null,
      regionLabel: character.anchorRegionId
        ? state.worldGeneration.regions.find((region) => region.id === character.anchorRegionId)?.label ?? null
        : null,
      prominence: character.prominence,
      influence: character.influence,
      tone: describeCharacterTone(character),
      memoryHook: character.memory.notableMoments[0] ?? character.memory.lastImpression ?? null
    }));
}

export function selectPriestPoliticsOverview(state: GameState) {
  const featuredIds = state.priestPolitics?.featuredCharacterIds?.length
    ? state.priestPolitics.featuredCharacterIds
    : stablePriestCharacterRoster(state).slice(0, 2).map((character) => character.id);
  const politics = state.priestPolitics;
  const featuredCharacters = stablePriestCharacterRoster(state)
    .filter((character) => featuredIds.includes(character.id))
    .map((character) => ({
      id: character.id,
      displayName: character.displayName,
      influence: character.influence,
      prominence: character.prominence,
      tone: describeCharacterTone(character),
      factionName: character.homeFactionId ? state.factions[character.homeFactionId]?.name ?? null : null
    }));

  if (!politics) {
    return {
      overallPressure: 0,
      unity: 0,
      status: "calm" as const,
      dominantBlocId: "pythia" as const,
      dominantBlocLabel: "Pythia Loyalists",
      currentIssue: "The college has not formed a readable political picture yet.",
      rumor: null,
      featuredCharacters,
      blocs: []
    };
  }

  const dominantBloc = politics.blocs.find((bloc) => bloc.id === politics.dominantBlocId);

  return {
    overallPressure: politics.overallPressure,
    unity: politics.unity,
    status: politics.status,
    dominantBlocId: politics.dominantBlocId,
    dominantBlocLabel: dominantBloc?.label ?? "Pythia Loyalists",
    currentIssue: politics.currentIssue,
    rumor: politics.rumor,
    featuredCharacters,
    blocs: politics.blocs
  };
}

export function selectPriestRosterInsights(state: GameState) {
  return state.priests.map((priest) => {
    const walker = state.walkers.find((candidate) => candidate.id === priest.walkerId);
    const profile = state.priestPolitics?.priests?.[priest.id];
    const anchor = profile?.anchorCharacterId
      ? state.characters?.roster.find((character) => character.id === profile.anchorCharacterId)
      : undefined;
    const favoredFaction = profile?.favoredFactionId ? state.factions[profile.favoredFactionId] : undefined;
    const assignment = priest.currentAssignmentBuildingId
      ? state.buildings.find((building) => building.id === priest.currentAssignmentBuildingId)
      : undefined;

    return {
      id: priest.id,
      name: walker?.name ?? priest.id,
      role: priest.role,
      skill: priest.skill,
      morale: priest.morale,
      assignmentId: priest.currentAssignmentBuildingId ?? null,
      assignmentLabel: assignment ? buildingDefs[assignment.defId].name : null,
      temperament: profile?.temperament ?? "steady",
      ambition: profile?.ambition ?? "guardian",
      stance: profile?.stance ?? "loyalist",
      influence: profile?.influence ?? priest.skill,
      loyalty: profile?.loyalty ?? priest.morale,
      dissent: profile?.dissent ?? Math.max(0, 100 - priest.morale),
      favoredFactionName: favoredFaction?.name ?? null,
      anchorName: anchor?.displayName ?? null,
      note: profile?.note ?? "Temple politics around this priest have not crystallized yet."
    };
  });
}

export function selectEspionageOverview(state: GameState) {
  const espionage = state.espionage;
  if (!espionage) {
    return {
      agents: [],
      activeOperations: [],
      recentCounterIntel: [],
      networkStrength: 0
    };
  }

  return {
    agents: espionage.agents.map((agent) => ({
      id: agent.id,
      name: agent.name,
      cover: agent.cover,
      targetFactionId: agent.targetFactionId,
      targetFactionName: state.factions[agent.targetFactionId]?.name ?? agent.targetFactionId,
      skill: agent.skill,
      loyalty: agent.loyalty,
      compromised: agent.compromised,
      recruitedDay: agent.recruitedDay
    })),
    activeOperations: espionage.operations
      .filter((op) => op.status === "active")
      .map((op) => {
        const agent = espionage.agents.find((a) => a.id === op.agentId);
        const elapsed = state.clock.day - op.startDay;
        const progress = Math.min(100, Math.round((elapsed / op.duration) * 100));
        return {
          id: op.id,
          kind: op.kind,
          agentName: agent?.name ?? "Unknown",
          targetId: op.targetId,
          progress,
          daysRemaining: Math.max(0, op.duration - elapsed),
          status: op.status
        };
      }),
    recentCounterIntel: espionage.counterIntelEvents.slice(0, 5),
    networkStrength: espionage.networkStrength
  };
}

export function selectPriestSecrets(state: GameState) {
  return state.priests.map((priest) => {
    const walker = state.walkers.find((w) => w.id === priest.walkerId);
    const discoveredSecrets = (priest.secrets ?? []).filter((s) => s.discoveredDay != null);
    const hiddenCount = (priest.secrets ?? []).filter((s) => s.discoveredDay == null).length;

    return {
      priestId: priest.id,
      priestName: walker?.name ?? priest.id,
      discoveredSecrets: discoveredSecrets.map((s) => ({
        id: s.id,
        kind: s.kind,
        severity: s.severity,
        discoveredDay: s.discoveredDay!,
        exposed: s.exposedDay != null
      })),
      hiddenCount,
      successionRank: priest.successionRank
    };
  });
}

export function selectSuccessionContest(state: GameState) {
  const contest = state.priestPolitics?.successionContest;
  if (!contest) {
    return null;
  }

  const candidates = contest.candidates.map((id) => {
    const priest = state.priests.find((p) => p.id === id);
    const walker = priest ? state.walkers.find((w) => w.id === priest.walkerId) : undefined;
    const profile = state.priestPolitics?.priests?.[id];
    return {
      priestId: id,
      name: walker?.name ?? id,
      influence: profile?.influence ?? 0,
      loyalty: profile?.loyalty ?? 0,
      isFrontRunner: id === contest.frontRunnerId
    };
  });

  return {
    active: contest.active,
    candidates,
    frontRunnerName: candidates.find((c) => c.isFrontRunner)?.name ?? null,
    startDay: contest.startDay,
    daysSinceStart: state.clock.day - contest.startDay
  };
}

function summarizeOmenConsensus(omenReports: OmenReport[]) {
  const averageReliability = omenReports.length > 0
    ? omenReports.reduce((sum, omen) => sum + omen.reliability, 0) / omenReports.length
    : 0;
  const strongestOmen = [...omenReports].sort((left, right) => right.reliability - left.reliability || left.id.localeCompare(right.id))[0];
  const weakestOmen = [...omenReports].sort((left, right) => left.reliability - right.reliability || left.id.localeCompare(right.id))[0];
  const domains = new Set(omenReports.map((omen) => omen.semantics.domain)).size;
  const targets = new Set(omenReports.map((omen) => omen.semantics.target)).size;
  const polarities = new Set(
    omenReports
      .map((omen) => omen.semantics.polarity)
      .filter((polarity) => polarity !== "double")
  ).size;

  let consensus: "aligned" | "mixed" | "contradictory" = "mixed";
  if (domains === 1 && targets <= 2 && polarities <= 1) {
    consensus = "aligned";
  } else if ((domains >= 2 && polarities >= 2) || targets >= 3) {
    consensus = "contradictory";
  }

  const reliabilityBand = averageReliability >= 70 ? "clear" : averageReliability >= 58 ? "steady" : "fragile";

  return {
    averageReliability,
    strongestOmen,
    weakestOmen,
    consensus,
    reliabilityBand
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function orderedPlacedTiles(tilePool: WordTile[], placedTileIds: string[]) {
  const tileById = new Map(tilePool.map((tile) => [tile.id, tile]));
  return placedTileIds
    .map((tileId) => tileById.get(tileId))
    .filter((tile): tile is WordTile => Boolean(tile));
}

function summarizePlacedTiles(tilePool: WordTile[], placedTileIds: string[]) {
  const placed = orderedPlacedTiles(tilePool, placedTileIds);
  const categories = new Set(placed.map((tile) => tile.category));

  return {
    placed,
    categories,
    hasSubject: categories.has("subject"),
    hasAction: categories.has("action")
  };
}

function depthBandFor(score: number): ProphecyDepthBand {
  if (score >= 82) {
    return "oracular";
  }
  if (score >= 66) {
    return "deep";
  }
  if (score >= 48) {
    return "grounded";
  }
  return "shallow";
}

function depthTextFor(band: ProphecyDepthBand): string {
  switch (band) {
    case "oracular":
      return "The reading has layered authority. Later interpreters will have room to defend it without sounding evasive.";
    case "deep":
      return "The reading carries real secondary meaning. It can survive drift if the outcome lands near the core image.";
    case "grounded":
      return "The structure is serviceable, but the prophecy still leans on its literal surface.";
    default:
      return "The reading is still shallow. It needs a firmer hinge or seal before it can bear interpretation.";
  }
}

export function buildProphecyScaffold(tiles: WordTile[]): ProphecyScaffoldPart[] {
  const spineTiles = tiles.filter((tile) => tile.category === "subject" || tile.category === "action");
  const hingeTiles = tiles.filter((tile) => tile.category === "condition" || tile.category === "modifier");
  const sealTiles = tiles.filter((tile) => tile.category === "seal");

  return [
    {
      kind: "spine",
      label: "Spine",
      text: spineTiles.length > 0 ? spineTiles.map((tile) => tile.text).join(" ") : "Missing subject or action",
      tileIds: spineTiles.map((tile) => tile.id),
      state: spineTiles.length >= 2 ? "stable" : spineTiles.length === 1 ? "forming" : "missing"
    },
    {
      kind: "hinge",
      label: "Hinge",
      text: hingeTiles.length > 0 ? hingeTiles.map((tile) => tile.text).join(" ") : "No condition or modifier yet",
      tileIds: hingeTiles.map((tile) => tile.id),
      state: hingeTiles.length >= 2 ? "stable" : hingeTiles.length === 1 ? "forming" : "missing"
    },
    {
      kind: "seal",
      label: "Seal",
      text: sealTiles.length > 0 ? sealTiles.map((tile) => tile.text).join(" ") : "Unsealed",
      tileIds: sealTiles.map((tile) => tile.id),
      state: sealTiles.length > 0 ? "charged" : "missing"
    }
  ];
}

export function buildProphecyDepthSummary({
  tiles,
  omenReports,
  score,
  pythia
}: {
  tiles: WordTile[];
  omenReports: OmenReport[];
  score: { clarity: number; value: number; risk: number };
  pythia: Pick<PythiaState, "attunement" | "mentalClarity" | "tranceDepth" | "needs" | "traits">;
}) {
  const scaffold = buildProphecyScaffold(tiles);
  const omenSummary = summarizeOmenConsensus(omenReports);
  const categories = new Set(tiles.map((tile) => tile.category));
  const structure =
    (categories.has("subject") && categories.has("action") ? 18 : 0)
    + (categories.has("condition") ? 6 : 0)
    + (categories.has("modifier") ? 4 : 0)
    + (categories.has("seal") ? 6 : 0);
  const pythiaCharge =
    pythia.tranceDepth * 0.1
    + pythia.mentalClarity * 0.05
    + Math.max(0, pythia.attunement - 50) * 0.04
    - pythia.needs.rest * 0.05
    - pythia.needs.purification * 0.05
    + (pythia.traits.includes("visionary") ? 4 : 0)
    + (pythia.traits.includes("calculating") ? 2 : 0)
    - (pythia.traits.includes("fragile") ? 3 : 0);
  const omenCharge =
    omenSummary.averageReliability * 0.09
    + (omenSummary.consensus === "aligned" ? 6 : omenSummary.consensus === "mixed" ? 1 : -5);
  const scoreWeight = score.clarity * 0.05 + score.value * 0.05 - score.risk * 0.09;
  const symbolicCharge = Math.min(8, tiles.length * 1.5) + scaffold.filter((part) => part.state === "charged" || part.state === "stable").length * 2;
  const depth = clamp(Math.round(structure + pythiaCharge + omenCharge + scoreWeight + symbolicCharge - 8), 0, 100);
  const depthBand = depthBandFor(depth);

  return {
    depth,
    depthBand,
    depthText: depthTextFor(depthBand),
    omenReliability: Math.round(omenSummary.averageReliability * 10) / 10,
    omenConsensus: omenSummary.consensus,
    scaffold
  };
}

function actionVerb(action: ProphecyRecord["semantics"][number]["action"] | undefined): string {
  switch (action) {
    case "triumph":
      return "gain advantage";
    case "fall":
      return "suffer collapse";
    case "fracture":
      return "split under strain";
    case "endure":
      return "hold through pressure";
    case "prosper":
      return "grow in fortune";
    case "withhold":
      return "be denied";
    default:
      return "shift";
  }
}

function fulfillmentWindowText(record: Pick<ProphecyRecord, "semantics" | "dayIssued" | "dueDay">): string {
  const horizons = new Set(record.semantics.map((semantic) => semantic.timeHorizon));
  if (horizons.has("immediate")) {
    return `The record expects a near-term sign, likely before day ${Math.min(record.dueDay, record.dayIssued + 7)}.`;
  }
  if (horizons.has("yearly")) {
    return `The record points to a long arc and remains open until at least day ${record.dueDay}.`;
  }
  return `The record is tuned for a seasonal turn and should be judged against developments before day ${record.dueDay}.`;
}

function rivalContextText(state: GameState, factionId: ProphecyRecord["factionId"]): string | null {
  const relations = Object.entries(state.factions[factionId].relations ?? {}) as [keyof GameState["factions"], number][];
  const rival = [...relations].sort((left, right) => left[1] - right[1] || String(left[0]).localeCompare(String(right[0])))[0];
  if (!rival) {
    return null;
  }

  const rivalFaction = state.factions[rival[0]];
  return `${rivalFaction.name} is the likeliest hostile witness to this reading. If events blur, their priests will press the harshest interpretation first.`;
}

export function buildProphecyInterpretation(
  state: GameState,
  record: Pick<ProphecyRecord, "factionId" | "semantics" | "depth" | "depthBand" | "risk" | "value" | "dayIssued" | "dueDay">
): ProphecyInterpretation {
  const faction = state.factions[record.factionId];
  const primary = record.semantics[0];
  const band = record.depthBand ?? depthBandFor(record.depth ?? 0);
  const summary = primary
    ? `${faction.name} received a ${band} reading around the ${primary.target}; it implies the matter may ${actionVerb(primary.action)} if the visible signs hold.`
    : `${faction.name} received a ${band} reading that still depends on later interpretation.`;
  const politicalReading = primary
    ? primary.polarity === "warning"
      ? `Politically, this gives Delphi room to caution ${faction.name} without openly humiliating them.`
      : `Politically, this lets Delphi affirm ${faction.name} while keeping the claim tied to ${primary.domain} signs rather than pure flattery.`
    : `Politically, the wording remains broad enough to defend from several directions.`;
  const caution = record.risk >= 82
    ? "The wording is sharp enough that a clean miss will be remembered as failure, not mystery."
    : (record.depth ?? 0) < 48
      ? "The wording is still literal-heavy. Without a stronger hinge, later interpreters will have little room to rescue it."
      : "The wording has enough depth to absorb some drift, but only if the central image survives.";

  return {
    summary,
    politicalReading,
    caution,
    fulfillmentWindow: fulfillmentWindowText(record),
    rivalContext: rivalContextText(state, record.factionId)
  };
}

export function selectConsultationInsights(state: GameState) {
  const current = state.consultation.current;
  if (!current) {
    return undefined;
  }

  const omenSummary = summarizeOmenConsensus(current.omenReports);
  const placedSummary = summarizePlacedTiles(current.tilePool, current.placedTileIds);
  const depthSummary = buildProphecyDepthSummary({
    tiles: placedSummary.placed,
    omenReports: current.omenReports,
    score: current.scorePreview,
    pythia: state.pythia
  });
  const pythia = state.pythia;
  const pythiaStrained = pythia.needs.rest >= 70 || pythia.needs.purification >= 68;
  const shallowTrance = pythia.tranceDepth < 42;
  const clearHead = pythia.mentalClarity >= 72;
  const calculating = pythia.traits.includes("calculating");
  const visionary = pythia.traits.includes("visionary");
  const diplomatic = pythia.traits.includes("diplomatic");
  const fragile = pythia.traits.includes("fragile");
  let riskWarning: string | null = null;

  if (current.scorePreview.risk >= 80) {
    riskWarning = omenSummary.averageReliability < 58 || pythiaStrained
      ? "The signs are weak and the wording is dangerously exact. A miss here will damage Delphi twice over."
      : "This reading is dangerously exact. If events drift even slightly, the envoy will remember the miss.";
  } else if (current.scorePreview.clarity < 45 && current.placedTileIds.length >= 2) {
    riskWarning = !placedSummary.hasSubject || !placedSummary.hasAction
      ? "The answer still lacks a clear subject and action. Give the prophecy a firmer spine before you deliver it."
      : "The wording is still muddy. The omens are crossing each other instead of converging.";
  } else if (depthSummary.depth < 46 && current.placedTileIds.length >= 3) {
    riskWarning = "The reading can be spoken, but it is still too shallow to carry much reinterpretation later.";
  } else if (current.scorePreview.risk >= 68 && current.placedTileIds.length >= 4) {
    riskWarning = "This reading is becoming fragile at the edges. Another sharp tile may make the promise too brittle.";
  } else if (shallowTrance && current.placedTileIds.length >= 2) {
    riskWarning = "The trance is still shallow. Lean on a condition or seal before you force a verdict.";
  } else if (pythia.needs.purification >= 72) {
    riskWarning = "Purification strain is bleeding into the reading. A cleaner rite would make this safer.";
  } else if (current.placedTileIds.length >= 3 && omenSummary.averageReliability < 58) {
    riskWarning = "The omens are fragile. A bold promise now would outrun what the signs can honestly bear.";
  } else if (fragile && current.scorePreview.risk >= 68) {
    riskWarning = "The Pythia is running hot and brittle. Another exact tile could push this reading past safe limits.";
  } else if (current.placedTileIds.length >= 4) {
    riskWarning = "The reading is growing fragile through accumulation. More clauses make the promise harder to survive.";
  }

  let guidance = "The god is whispering through mixed signs. Balance conviction with room to maneuver.";
  if (omenSummary.consensus === "aligned" && omenSummary.reliabilityBand === "clear") {
    guidance = calculating
      ? "The omens align cleanly and the Pythia is holding the line. Build a sharp subject-action spine."
      : visionary
        ? "The omens align cleanly. Follow the strongest image and let the seal carry the rest."
        : "The omens align cleanly. You can afford a sharper answer if the score stays under control.";
  } else if (omenSummary.consensus === "contradictory") {
    guidance = "The omens clash. Lean on modifiers or seals so the prophecy can survive the contradictions.";
  } else if (shallowTrance) {
    guidance = "The trance has not deepened yet. Use a seal or softer modifier rather than a hard claim.";
  } else if (depthSummary.depth < 48 && current.placedTileIds.length >= 2) {
    guidance = "The scaffold is still shallow. Give the spine a hinge or a seal before you trust it to travel.";
  } else if (pythiaStrained) {
    guidance = "The Pythia is strained. Favor breadth and survivable wording over a bold promise.";
  } else if (omenSummary.reliabilityBand === "fragile") {
    guidance = "The signs are thin. Favor breadth and ambiguity over a narrow promise.";
  } else if (diplomatic && current.scorePreview.value >= 68) {
    guidance = "The envoy will hear nuance well today. Keep the structure clear and let the softer edges do the work.";
  } else if (current.scorePreview.value >= 75) {
    guidance = "This answer already carries weight with the envoy. One more careful tile may be enough.";
  } else if (clearHead && placedSummary.categories.has("subject") && placedSummary.categories.has("action")) {
    guidance = "The spine is in place. Now add one condition or modifier that keeps the answer honest.";
  }

  return {
    averageReliability: Math.round(omenSummary.averageReliability * 10) / 10,
    strongestReliability: omenSummary.strongestOmen ? Math.round(omenSummary.strongestOmen.reliability * 10) / 10 : 0,
    weakestReliability: omenSummary.weakestOmen ? Math.round(omenSummary.weakestOmen.reliability * 10) / 10 : 0,
    consensus: omenSummary.consensus,
    reliabilityBand: omenSummary.reliabilityBand,
    depth: depthSummary.depth,
    depthBand: depthSummary.depthBand,
    depthText: depthSummary.depthText,
    scaffold: depthSummary.scaffold,
    omenSummaryText: `Consensus ${omenSummary.consensus} · Average reliability ${Math.round(omenSummary.averageReliability * 10) / 10} · Depth ${depthSummary.depth} (${depthSummary.depthBand})`,
    guidance,
    riskWarning
  };
}

export function selectSacredRecordEntries(state: GameState) {
  return [...state.consultation.history]
    .sort((left, right) => right.dayIssued - left.dayIssued || left.id.localeCompare(right.id))
    .map((entry) => {
      const depthSummary = entry.depth !== undefined && entry.depthBand
        ? {
            depth: entry.depth,
            depthBand: entry.depthBand,
            depthText: depthTextFor(entry.depthBand),
            omenReliability: entry.omenReliability ?? 0,
            omenConsensus: entry.omenConsensus ?? "mixed",
            scaffold: entry.scaffold ?? buildProphecyScaffold([])
          }
        : buildProphecyDepthSummary({
            tiles: [],
            omenReports: [],
            score: entry,
            pythia: state.pythia
          });
      const interpretation = entry.interpretation ?? buildProphecyInterpretation(state, {
        factionId: entry.factionId,
        semantics: entry.semantics,
        depth: depthSummary.depth,
        depthBand: depthSummary.depthBand,
        risk: entry.risk,
        value: entry.value,
        dayIssued: entry.dayIssued,
        dueDay: entry.dueDay
      });
      const factionName = state.factions[entry.factionId].name;
      const status = !entry.resolved
        ? "awaiting"
        : (entry.credibilityDelta ?? 0) > 0
          ? "vindicated"
          : (entry.credibilityDelta ?? 0) < 0
            ? "broken"
            : "contested";

      return {
        id: entry.id,
        title: `${factionName} record`,
        factionId: entry.factionId,
        factionName,
        dayIssued: entry.dayIssued,
        dueDay: entry.dueDay,
        resolvedDay: entry.resolvedDay ?? null,
        text: entry.text,
        clarity: entry.clarity,
        value: entry.value,
        risk: entry.risk,
        depth: depthSummary.depth,
        depthBand: depthSummary.depthBand,
        depthText: depthSummary.depthText,
        omenReliability: entry.omenReliability ?? depthSummary.omenReliability,
        omenConsensus: entry.omenConsensus ?? depthSummary.omenConsensus,
        scaffold: entry.scaffold ?? depthSummary.scaffold,
        interpretation,
        resolutionReport: entry.resolutionReport ?? null,
        credibilityDelta: entry.credibilityDelta ?? null,
        status
      };
    });
}

export function selectChronicleEntries(state: GameState) {
  const records = selectSacredRecordEntries(state);
  const prophecies = records.map((entry) => ({
    id: `prophecy-${entry.id}`,
    kind: "prophecy" as const,
    day: entry.dayIssued,
    factionId: entry.factionId,
    title: `${entry.factionName} prophecy`,
    text: `${entry.text} · ${entry.depthBand} depth`,
    delta: entry.credibilityDelta
  }));
  const consequences = records
    .filter((entry) => entry.resolutionReport && entry.resolvedDay !== null)
    .map((entry) => ({
      id: `consequence-${entry.id}`,
      kind: "consequence" as const,
      day: entry.resolvedDay ?? entry.dueDay,
      factionId: entry.factionId,
      title: `${entry.factionName} consequence`,
      text: entry.resolutionReport ?? "",
      delta: entry.credibilityDelta
    }));

  return [...prophecies, ...consequences]
    .sort((left, right) => right.day - left.day)
    .slice(0, 4);
}

function summarizeFactionRelations(state: GameState, factionId: keyof GameState["factions"]) {
  const faction = state.factions[factionId];
  const entries = Object.entries(faction.relations) as [keyof GameState["factions"], number][];
  if (entries.length === 0) {
    return {
      ally: null,
      rival: null
    };
  }

  const strongest = [...entries].sort((left, right) => right[1] - left[1] || String(left[0]).localeCompare(String(right[0])))[0];
  const weakest = [...entries].sort((left, right) => left[1] - right[1] || String(left[0]).localeCompare(String(right[0])))[0];

  return {
    ally: strongest ? `${state.factions[strongest[0]].name} (${strongest[1]})` : null,
    rival: weakest ? `${state.factions[weakest[0]].name} (${weakest[1]})` : null
  };
}

function summarizeFactionDiplomacy(state: GameState, factionId: keyof GameState["factions"]) {
  const faction = state.factions[factionId];
  const treaties = (faction.treaties ?? []).map((entry) => state.factions[entry].name);
  const embargoes = (faction.embargoes ?? []).map((entry) => state.factions[entry].name);

  return {
    treaties,
    embargoes
  };
}

function factionNarrative(faction: GameState["factions"][keyof GameState["factions"]]) {
  return `${faction.lastOutcome ?? ""} ${faction.history.join(" ")}`.toLowerCase();
}

function factionInfluenceScore(faction: GameState["factions"][keyof GameState["factions"]]) {
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

function rivalDiscoveryBand(visibility: number, intel: number): "shadow" | "suspected" | "confirmed" {
  const combined = visibility * 0.55 + intel * 0.45;
  if (combined >= 64 || (visibility >= 60 && intel >= 56)) {
    return "confirmed";
  }
  if (combined >= 34) {
    return "suspected";
  }
  return "shadow";
}

export function selectRivalOracleSummary(state: GameState) {
  const regionMap = new Map(state.worldGeneration.regions.map((region) => [region.id, region]));
  const rivals = [...(state.rivalOracles?.roster ?? [])]
    .sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id));

  return rivals.map((rival) => {
    const discovery = rivalDiscoveryBand(rival.visibility, rival.intel);
    const patron = state.factions[rival.patronFactionId];
    const operation = rival.lastKnownOperationId ? rivalOracleOperationDefById[rival.lastKnownOperationId] : undefined;
    const lastIncident = state.rivalOracles?.incidents.find((incident) => incident.rivalId === rival.id);
    const targetRegion = regionMap.get(rival.lastTargetRegionId ?? rival.homeRegionId);

    return {
      id: rival.id,
      name: discovery === "shadow" ? "Unseen rival oracle" : rival.name,
      title: rival.title,
      pressure: rival.pressure,
      visibility: rival.visibility,
      intel: rival.intel,
      patronLabel: discovery === "confirmed"
        ? patron?.name ?? "Unknown patron"
        : discovery === "suspected"
          ? `${patron?.name ?? "Unknown"} suspected`
          : "Patron obscured",
      homeRegionId: rival.homeRegionId,
      homeRegionLabel: regionMap.get(rival.homeRegionId)?.label ?? rival.homeRegionId,
      targetRegionId: rival.lastTargetRegionId ?? rival.homeRegionId,
      targetRegionLabel: targetRegion?.label ?? rival.homeRegionId,
      discovery,
      operationLabel: operation?.label ?? "Obscured movement",
      lastSummary: lastIncident?.summary ?? null,
      patronage: rival.patronage,
      intrigue: rival.intrigue
    };
  });
}

export function selectWorldSummary(state: GameState) {
  const factions = Object.values(state.factions).sort((left, right) => left.id.localeCompare(right.id));
  const rivalSummary = selectRivalOracleSummary(state);
  const ranked = [...factions]
    .map((faction) => ({
      faction,
      score: factionInfluenceScore(faction)
    }))
    .sort((left, right) => right.score - left.score || left.faction.id.localeCompare(right.faction.id));
  const leader = ranked[0];
  const runnerUp = ranked[1];
  const hegemon = leader && leader.score >= 44 && leader.score - (runnerUp?.score ?? 0) >= 6
    ? `${leader.faction.name} leads the dominant bloc`
    : "No single hegemon holds the region together yet";
  const destabilized = factions
    .filter((faction) => factionNarrative(faction).includes("government crisis") || factionNarrative(faction).includes("revolutionary agitation"))
    .map((faction) => faction.name)
    .slice(0, 3);
  const ideologicalDrift = factions
    .filter((faction) => factionNarrative(faction).includes("creed") || factionNarrative(faction).includes("revival") || factionNarrative(faction).includes("legitimacy"))
    .map((faction) => `${faction.name}: ${faction.currentAgenda}`)
    .slice(0, 3);
  const leadingRivals = rivalSummary
    .slice(0, 3)
    .map((rival) => `${rival.name} at ${rival.targetRegionLabel} (${rival.pressure})`);

  return {
    origin: state.worldGeneration.originTitle,
    seed: state.worldSeed,
    seedText: state.worldSeedText,
    climate: state.worldGeneration.climate.value,
    economicClimate: state.worldGeneration.economicClimate,
    divineMood: state.worldGeneration.divineMood.value,
    oracleDensity: state.worldGeneration.oracleDensity.value,
    politicalClimate: state.worldGeneration.politicalClimate,
    challengeTags: state.worldGeneration.challengeTags,
    hegemon,
    destabilized,
    ideologicalDrift,
    rivalOraclePressure: state.rivalOracles?.totalPressure ?? 0,
    espionageRisk: rivalSummary.some((rival) => rival.discovery === "shadow" && rival.pressure >= 55)
      ? "obscured"
      : rivalSummary.some((rival) => rival.discovery !== "confirmed" && rival.pressure >= 45)
        ? "uncertain"
        : "mapped",
    leadingRivals,
    openTradeLanes: factions.filter((faction) => faction.tradeAccess).length,
    activeCrises: state.campaign.worldMap.crisisChains.map((chain) => chain.label).slice(0, 3),
    regionalOpenings: state.worldGeneration.regions.slice(0, 5).map((region) => ({
      id: region.id,
      label: region.label,
      pressure: region.pressure,
      unrest: region.unrest,
      hegemon: region.hegemon,
      philosophy: region.philosophy
    }))
  };
}

export function selectRunSetupOriginOptions(seed?: number | string) {
  return buildRunSetupOriginOptions(seed);
}

export function selectRunSetupPreview(seed?: number | string, originId?: GameState["originId"]) {
  return buildRunSetupPreview({
    seed,
    originId
  });
}

export function selectCurrentRunSetupPreview(state: GameState) {
  return buildRunSetupPreview({
    seed: state.worldSeedText,
    originId: state.originId
  });
}

export function renderGameToText(state: GameState): string {
  const carrierSummary = selectCarrierSummary(state);
  const characterSpotlight = selectCharacterSpotlight(state);
  const consultationInsights = selectConsultationInsights(state);
  const priestPolitics = selectPriestPoliticsOverview(state);
  const sacredRecord = selectSacredRecordEntries(state);
  const payload = {
    coordinate_system: "origin top-left, +x east, +y south, isometric display projected from tile coordinates",
    clock: {
      day: state.clock.day,
      month: state.clock.month,
      year: state.clock.year,
      absoluteDay: getAbsoluteDay(state.clock),
      season: state.clock.season,
      speed: state.clock.speed,
      paused: state.clock.paused
    },
    run: {
      seed: state.worldSeed,
      seedText: state.worldSeedText,
      originId: state.originId,
      originTitle: state.worldGeneration.originTitle,
      challengeTags: state.worldGeneration.challengeTags,
      scoreModifier: state.worldGeneration.scoreModifier,
      disabledSystems: state.worldGeneration.disabledSystems
    },
    tool: state.ui.activeTool,
    hovered_tile: state.ui.hoveredTile,
    selected: state.ui.selectedEntityId,
    resources: Object.fromEntries(
      Object.entries(state.resources).map(([key, value]) => [key, Math.round(value.amount * 10) / 10])
    ),
    pythia: {
      attunement: state.pythia.attunement,
      physicalHealth: state.pythia.physicalHealth,
      mentalClarity: state.pythia.mentalClarity,
      tranceDepth: state.pythia.tranceDepth,
      needs: {
        rest: Math.round(state.pythia.needs.rest),
        purification: Math.round(state.pythia.needs.purification),
        pilgrimageCooldown: Math.round(state.pythia.needs.pilgrimageCooldown)
      }
    },
    buildings: state.buildings.map((building) => ({
      id: building.id,
      defId: building.defId,
      name: buildingDefs[building.defId].name,
      x: building.position.x,
      y: building.position.y,
      condition: Math.round(building.condition),
      stored: building.storedResources
    })),
    walkers: state.walkers.map((walker) => ({
      id: walker.id,
      role: walker.role,
      x: walker.tile.x,
      y: walker.tile.y,
      state: walker.state,
      path_length: walker.path.length,
      carrying: walker.carrying ?? null,
      carrying_amount: walker.carryingAmount ?? 0,
      assigned_job: walker.assignedJobId ?? null,
      fatigue: walker.role === "carrier" ? Math.round((walker.fatigue ?? 0) * 10) / 10 : null,
      hauling_skill: walker.role === "carrier" ? walker.haulingSkill ?? null : null,
      supply_radius: walker.role === "carrier" ? walker.supplyRadius ?? null : null
    })),
    resource_jobs: state.resourceJobs.map((job) => ({
      id: job.id,
      resource: job.resourceId,
      amount: job.amount,
      source: job.sourceBuildingId,
      target: job.targetBuildingId,
      priority: job.priority,
      phase: job.phase,
      assigned: job.assignedWalkerId ?? null
    })).sort((left, right) =>
      (right.priority === "critical" ? 3 : right.priority === "high" ? 2 : 1)
      - (left.priority === "critical" ? 3 : left.priority === "high" ? 2 : 1)
      || left.id.localeCompare(right.id)
    ),
    trade_offers: state.tradeOffers.map((offer) => ({
      id: offer.id,
      factionId: offer.factionId,
      resource: offer.resourceId,
      amount: offer.amount,
      price: Math.round(offer.price * 10) / 10
    })),
    factions: Object.values(state.factions)
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((faction) => ({
        id: faction.id,
        profile: faction.profile,
        favoredResource: faction.favoredResource,
        agenda: faction.currentAgenda,
        credibility: faction.credibility,
        favour: faction.favour,
        debt: faction.debt,
        dependence: faction.dependence,
        tradeAccess: faction.tradeAccess,
        conflicts: faction.activeConflicts,
        treaties: summarizeFactionDiplomacy(state, faction.id).treaties,
        embargoes: summarizeFactionDiplomacy(state, faction.id).embargoes,
        lastOutcome: faction.lastOutcome ?? null,
        history: faction.history.slice(0, 2),
        ...summarizeFactionRelations(state, faction.id)
    })),
    rival_oracles: selectRivalOracleSummary(state),
    characters: selectCharacterRoster(state).map((character) => ({
      id: character.id,
      name: character.displayName,
      role: character.role,
      cadence: character.cadence,
      faction: character.factionName,
      region: character.regionLabel,
      influence: character.influence,
      prominence: character.prominence,
      trust: character.trust,
      fear: character.fear,
      hostility: character.hostility,
      familiarity: character.familiarity,
      visits: character.visitCount,
      tone: character.tone
    })),
    character_spotlight: characterSpotlight,
    priest_politics: {
      overallPressure: priestPolitics.overallPressure,
      unity: priestPolitics.unity,
      status: priestPolitics.status,
      dominantBloc: priestPolitics.dominantBlocLabel,
      currentIssue: priestPolitics.currentIssue,
      rumor: priestPolitics.rumor,
      blocs: priestPolitics.blocs.map((bloc) => ({
        id: bloc.id,
        label: bloc.label,
        support: bloc.support,
        tension: bloc.tension
      }))
    },
    consultation: state.consultation.current
      ? {
          mode: state.consultation.mode,
          factionId: state.consultation.current.factionId,
          question: state.consultation.current.question,
          placedTiles: state.consultation.current.placedTileIds.length,
          scorePreview: {
            ...state.consultation.current.scorePreview,
            depth: consultationInsights?.depth ?? state.consultation.current.scorePreview.depth ?? 0,
            depthBand: consultationInsights?.depthBand ?? state.consultation.current.scorePreview.depthBand ?? null
          },
          omenSummary: consultationInsights?.omenSummaryText ?? null,
          depthText: consultationInsights?.depthText ?? null,
          scaffold: consultationInsights?.scaffold ?? [],
          guidance: consultationInsights?.guidance ?? null,
          riskWarning: consultationInsights?.riskWarning ?? null
        }
      : { mode: state.consultation.mode },
    campaign: {
      scenarioId: state.campaign.scenarioId,
      reputation: {
        score: state.campaign.reputation.score,
        currentTier: state.campaign.reputation.currentTier,
        nextTier: state.campaign.reputation.nextTier ?? null,
        unlockedBuildings: state.campaign.reputation.unlockedBuildingIds,
        unlockedScenarios: state.campaign.reputation.unlockedScenarioIds
      },
      treasury: state.campaign.treasury,
      patronMilestones: state.campaign.patronMilestones,
      selectedWorldNode: state.campaign.worldMap.selectedNodeId ?? null,
      worldNodes: state.campaign.worldMap.nodes.map((node) => ({
        id: node.id,
        label: node.label,
        x: node.position.x,
        y: node.position.y,
        pressure: node.pressure,
        unrest: node.unrest,
        tags: node.tags,
        factionId: node.controllingFactionId ?? null
      })),
      activePressures: state.campaign.worldMap.activePressures,
      crisisChains: state.campaign.worldMap.crisisChains,
      winCondition: state.campaign.winCondition
    },
    consequences_due: state.consequences.filter((entry) => !entry.resolved).map((entry) => ({
      factionId: entry.factionId,
      dueDay: entry.dueDay
    })),
    chronicle: selectChronicleEntries(state),
    sacred_record: sacredRecord.slice(0, 5).map((entry) => ({
      id: entry.id,
      factionId: entry.factionId,
      status: entry.status,
      dayIssued: entry.dayIssued,
      dueDay: entry.dueDay,
      resolvedDay: entry.resolvedDay,
      depth: entry.depth,
      depthBand: entry.depthBand,
      omenConsensus: entry.omenConsensus,
      omenReliability: entry.omenReliability,
      text: entry.text,
      resolutionReport: entry.resolutionReport
    })),
    world_summary: selectWorldSummary(state),
    world_generation: {
      summary: state.worldGeneration.summary,
      note: state.worldGeneration.note,
      climate: state.worldGeneration.climate,
      economic: state.worldGeneration.economic,
      divineMood: state.worldGeneration.divineMood,
      oracleDensity: state.worldGeneration.oracleDensity,
      factionMix: state.worldGeneration.factionMix,
      pressures: state.worldGeneration.pressures,
      regions: state.worldGeneration.regions.map((region) => ({
        id: region.id,
        label: region.label,
        pressure: region.pressure,
        unrest: region.unrest,
        hegemon: region.hegemon,
        philosophy: region.philosophy,
        divineMood: region.divineMood,
        oracleDensity: region.oracleDensity
      }))
    },
    precinct_health: selectPrecinctHealth(state),
    carriers: carrierSummary.count,
    carrier_summary: carrierSummary,
    top_advisor_message: selectTopAdvisorMessage(state)?.text ?? null,
    age: state.age
      ? {
          currentAgeId: state.age.currentAgeId,
          currentAgeIndex: state.age.currentAgeIndex,
          ageHistory: state.age.ageHistory,
          lastAgeCheckYear: state.age.lastAgeCheckYear
        }
      : null,
    legendary_consultations: (state.legendaryProgress ?? []).map((lp) => ({
      consultationId: lp.consultationId,
      currentStage: lp.currentStage,
      completed: lp.completed
    })),
    legacy: state.legacy
      ? {
          phase: state.legacy.phase,
          declineSeverity: state.legacy.declineSeverity,
          comebackAttempts: state.legacy.comebackAttempts,
          legacyScore: state.legacy.legacyScore
        }
      : null,
    espionage_overview: state.espionage
      ? {
          agents: state.espionage.agents.length,
          activeOperations: state.espionage.operations.filter((op) => op.status === "active").length,
          completedOperations: state.espionage.operations.filter((op) => op.status === "success").length,
          detectedOperations: state.espionage.operations.filter((op) => op.status === "detected").length
        }
      : null,
    excavation_sites: state.excavation
      ? {
          sites: state.excavation.sites.map((site) => ({
            id: site.id,
            status: site.status,
            depth: site.depth,
            maxDepth: site.maxDepth,
            tile: site.tile
          })),
          relicsFound: state.excavation.relics.length,
          sacredSites: state.excavation.sacredSites.map((ss) => ({
            id: ss.id,
            kind: ss.kind,
            discovered: ss.discovered,
            active: ss.active
          }))
        }
      : null,
    prophecy_arcs: state.prophecyArcs
      ? {
          activeArcs: state.prophecyArcs.arcs.filter((arc) => arc.status === "active").length,
          totalArcs: state.prophecyArcs.arcs.length,
          arcs: state.prophecyArcs.arcs.slice(0, 5).map((arc) => ({
            id: arc.id,
            label: arc.label,
            domain: arc.domain,
            factionId: arc.factionId,
            status: arc.status,
            milestones: arc.milestones.length
          }))
        }
      : null,
    world_history_summary: state.worldHistory
      ? {
          totalEvents: state.worldHistory.events.length,
          recentEvents: state.worldHistory.events.slice(-5).map((e) => ({
            kind: e.kind,
            title: e.title,
            day: e.day,
            year: e.year,
            oracleInvolved: e.oracleInvolved
          })),
          activeAlliances: state.worldHistory.alliances.filter((a) => !a.brokenDay).length,
          hegemon: state.worldHistory.hegemon?.currentHegemonId ?? null
        }
      : null
  };

  return JSON.stringify(payload, null, 2);
}

export function selectPhilosopherThreats(state: GameState) {
  if (!state.philosophers) return [];
  const factionEntries = Object.entries(state.philosophers.byFaction) as Array<[string, PhilosopherThreatState]>;
  return factionEntries
    .filter(([, threat]) => threat.active)
    .map(([factionId, threat]) => ({
      factionId,
      factionName: state.factions[factionId as keyof typeof state.factions]?.name ?? factionId,
      philosopherId: threat.philosopherId,
      worldview: threat.worldview,
      influence: threat.influence,
      suspicion: threat.suspicion,
      pressure: threat.pressure,
      stage: threat.stage as string
    }))
    .sort((a, b) => b.pressure - a.pressure);
}

export function selectConsequenceTracker(state: GameState) {
  const all = state.consequences ?? [];
  const pending = all.filter((c) => !c.resolved);
  const resolved = all.filter((c) => c.resolved);
  const currentDay = getAbsoluteDay(state.clock);
  const urgentCount = pending.filter((c) => c.dueDay - currentDay <= 30).length;
  return { pending, resolved, urgentCount };
}

export function selectCrisisChains(state: GameState) {
  const chains = state.campaign?.worldMap?.crisisChains ?? [];
  return chains
    .filter((c) => !c.resolvedDay)
    .map((c) => ({
      id: c.id,
      label: c.label,
      nodeId: c.nodeId,
      factionId: c.factionId,
      factionName: c.factionId ? (state.factions[c.factionId as keyof typeof state.factions]?.name ?? c.factionId) : undefined,
      stage: c.stage,
      pressure: c.pressure,
      stepsCompleted: c.stepsCompleted
    }))
    .sort((a, b) => b.pressure - a.pressure);
}

export function selectActiveProphecyArcs(state: GameState) {
  const arcState = state.prophecyArcs;
  if (!arcState) return [];

  const currentDay = getAbsoluteDay(state.clock);

  return arcState.arcs
    .filter((arc) => arc.status === "active")
    .map((arc) => {
      const faction = state.factions[arc.factionId as keyof typeof state.factions];
      const completedMilestones = arc.milestones.filter((m) => m.completed).length;
      const totalMilestones = arc.milestones.length;
      const daysRemaining = Math.max(0, arc.expectedEndDay - currentDay);

      return {
        id: arc.id,
        label: arc.label,
        rootProphecyId: arc.rootProphecyId,
        domain: arc.domain,
        factionId: arc.factionId,
        factionName: faction?.name ?? arc.factionId,
        startDay: arc.startDay,
        expectedEndDay: arc.expectedEndDay,
        daysRemaining,
        status: arc.status,
        milestoneProgress: `${completedMilestones}/${totalMilestones}`,
        completedMilestones,
        totalMilestones,
        milestones: arc.milestones,
        interpretationBranches: arc.interpretationBranches,
        followUpObligations: arc.followUpObligations
      };
    })
    .sort((a, b) => a.daysRemaining - b.daysRemaining);
}

export function selectProphecyContradictions(state: GameState) {
  const arcState = state.prophecyArcs;
  if (!arcState) return [];

  return arcState.contradictions
    .filter((c) => !c.resolved)
    .sort((a, b) => {
      const severityOrder = { catastrophic: 0, major: 1, minor: 2 };
      return (severityOrder[a.severity] ?? 2) - (severityOrder[b.severity] ?? 2);
    })
    .map((c) => ({
      id: c.id,
      prophecyIdA: c.prophecyIdA,
      prophecyIdB: c.prophecyIdB,
      domain: c.domain,
      description: c.description,
      detectedDay: c.detectedDay,
      severity: c.severity,
      credibilityImpact: c.credibilityImpact
    }));
}

// --- Excavation selectors ---

export type ExcavationSiteOverview = {
  id: string;
  tile: { x: number; y: number };
  status: ExcavationSite["status"];
  depth: number;
  maxDepth: number;
  progress: number;
  assignedPriestId?: string;
  unclaimedRelicCount: number;
  revealedLayerCount: number;
};

export function selectExcavationOverview(state: GameState): ExcavationSiteOverview[] {
  const excavation = state.excavation;
  if (!excavation) {
    return [];
  }
  return excavation.sites.map((site) => ({
    id: site.id,
    tile: site.tile,
    status: site.status,
    depth: site.depth,
    maxDepth: site.maxDepth,
    progress: site.maxDepth > 0 ? Math.min(100, (site.depth / site.maxDepth) * 100) : 0,
    assignedPriestId: site.assignedPriestId,
    unclaimedRelicCount: site.layers.filter((l) => l.revealed && l.relicId).length,
    revealedLayerCount: site.layers.filter((l) => l.revealed).length
  }));
}

export type RelicCollectionEntry = {
  id: string;
  name: string;
  kind: Relic["kind"];
  domain: Relic["domain"];
  effect: RelicEffect;
  discoveredDay: number;
};

export function selectRelicCollection(state: GameState): RelicCollectionEntry[] {
  const excavation = state.excavation;
  if (!excavation) {
    return [];
  }
  return excavation.relics
    .filter((r) => r.discoveredDay > 0)
    .map((r) => ({
      id: r.id,
      name: r.name,
      kind: r.kind,
      domain: r.domain,
      effect: r.effect,
      discoveredDay: r.discoveredDay
    }));
}

export type SacredSiteEntry = {
  id: string;
  tile: { x: number; y: number };
  kind: SacredSite["kind"];
  discovered: boolean;
  active: boolean;
  bonuses: RelicEffect[];
};

export function selectSacredSites(state: GameState): SacredSiteEntry[] {
  const excavation = state.excavation;
  if (!excavation) {
    return [];
  }
  return excavation.sacredSites
    .filter((s) => s.discovered)
    .map((s) => ({
      id: s.id,
      tile: s.tile,
      kind: s.kind,
      discovered: s.discovered,
      active: s.active,
      bonuses: s.bonuses
    }));
}

export function selectCurrentAge(state: GameState): {
  id: string;
  name: string;
  description: string;
  modifiers: Pick<AgeDef, "consultationWeightMod" | "philosopherPrevalence" | "tradeActivityMod" | "rivalAggressionMod" | "factionModifiers">;
  yearsInAge: number;
} {
  const ageState = state.age;
  const currentDef = ageState ? ageDefById[ageState.currentAgeId] : ageDefs[0]!;
  const ageEntry = ageState?.ageHistory.find((h) => h.ageId === currentDef.id);
  const startYear = ageEntry?.startYear ?? 1;
  return {
    id: currentDef.id,
    name: currentDef.name,
    description: currentDef.description,
    modifiers: {
      consultationWeightMod: currentDef.consultationWeightMod,
      philosopherPrevalence: currentDef.philosopherPrevalence,
      tradeActivityMod: currentDef.tradeActivityMod,
      rivalAggressionMod: currentDef.rivalAggressionMod,
      factionModifiers: currentDef.factionModifiers
    },
    yearsInAge: Math.max(0, state.clock.year - startYear)
  };
}

// --- Legacy selectors ---

export type LegacyStatusView = {
  phase: LegacyPhase;
  declineSeverity: number;
  comebackAttempts: number;
  scorePreview: number;
  declineStartDay?: number;
};

export function selectLegacyStatus(state: GameState): LegacyStatusView {
  const legacy = state.legacy;
  return {
    phase: legacy?.phase ?? "thriving",
    declineSeverity: legacy?.declineSeverity ?? 0,
    comebackAttempts: legacy?.comebackAttempts ?? 0,
    scorePreview: legacy?.legacyScore ?? computeLegacyScore(state),
    declineStartDay: legacy?.declineStartDay
  };
}

export function selectLegacyArtifact(state: GameState): LegacyArtifact | undefined {
  return state.legacy?.legacyArtifact;
}

// --- Legendary Consultations ---

export type AvailableLegendaryView = {
  def: LegendaryConsultationDef;
  progress?: LegendaryConsultationProgress;
};

export function selectAvailableLegendaryConsultations(state: GameState): AvailableLegendaryView[] {
  const available = state.availableLegendary ?? [];
  const progressList = state.legendaryProgress ?? [];
  const completedIds = new Set(
    progressList.filter((p) => p.completed).map((p) => p.consultationId)
  );

  return available
    .filter((id) => !completedIds.has(id))
    .map((id) => {
      const def = legendaryConsultationDefs[id];
      const progress = progressList.find((p) => p.consultationId === id);
      return { def, progress };
    });
}

export type LegendaryProgressView = {
  def: LegendaryConsultationDef;
  progress: LegendaryConsultationProgress;
};

export function selectLegendaryProgress(state: GameState): {
  inProgress: LegendaryProgressView[];
  completed: LegendaryProgressView[];
} {
  const progressList = state.legendaryProgress ?? [];
  const inProgress: LegendaryProgressView[] = [];
  const completed: LegendaryProgressView[] = [];

  for (const progress of progressList) {
    const def = legendaryConsultationDefs[progress.consultationId];
    if (!def) continue;
    if (progress.completed) {
      completed.push({ def, progress });
    } else {
      inProgress.push({ def, progress });
    }
  }

  return { inProgress, completed };
}

// --- Lineage selectors ---

export type LineageOverview = {
  totalRuns: number;
  unlockedOrigins: string[];
  unlockedBurdens: BurdenId[];
  lineageScore: number;
  carryoverBonuses: CarryoverBonus[];
  runHistory: RunRecord[];
};

export function selectLineageOverview(state: GameState): LineageOverview {
  const lineage = state.lineage ?? createInitialLineageState();
  return {
    totalRuns: lineage.totalRuns,
    unlockedOrigins: [...lineage.unlockedOrigins],
    unlockedBurdens: [...lineage.unlockedBurdens],
    lineageScore: lineage.lineageScore,
    carryoverBonuses: [...lineage.carryoverBonuses],
    runHistory: [...lineage.runHistory]
  };
}

export type ChallengeSeedView = {
  challenge: ChallengeSeed;
  completed: boolean;
  bestScore?: number;
};

export function selectChallengeSeeds(state: GameState): ChallengeSeedView[] {
  const lineage = state.lineage ?? createInitialLineageState();
  return challengeSeeds.map((challenge) => {
    const matchingRuns = lineage.runHistory.filter(
      (r) => r.originId === challenge.originId && r.seedText === challenge.seedText
    );
    const bestScore = matchingRuns.length > 0
      ? Math.max(...matchingRuns.map((r) => r.finalScore))
      : undefined;
    const completed = bestScore !== undefined && bestScore >= challenge.targetScore;
    return { challenge, completed, bestScore };
  });
}

// --- Enriched Event Feed ---

export type EnrichedEventFeedItem = {
  id: string;
  day: number;
  text: string;
  narratedDescription?: string;
};

/**
 * Returns the event feed with narrated descriptions where available.
 * World history events that occurred on the same day are matched to
 * feed items and their enriched descriptions are attached.
 */
export function selectEnrichedEventFeed(state: GameState): EnrichedEventFeedItem[] {
  const worldEvents = state.worldHistory?.events ?? [];

  // Build a map of day -> world event descriptions for quick lookup
  const descriptionsByDay = new Map<number, string[]>();
  for (const event of worldEvents) {
    const existing = descriptionsByDay.get(event.day);
    if (existing) {
      existing.push(event.description);
    } else {
      descriptionsByDay.set(event.day, [event.description]);
    }
  }

  return state.eventFeed.map((item) => {
    const dayDescriptions = descriptionsByDay.get(item.day);
    const narratedDescription = dayDescriptions?.shift();
    return {
      ...item,
      narratedDescription
    };
  });
}
