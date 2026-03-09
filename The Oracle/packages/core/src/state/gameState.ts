import type {
  BuildingDefId,
  DomainTag,
  FactionAgenda,
  FactionId,
  FactionProfile,
  LegendaryConsultationId,
  NamedCharacterArchetypeId,
  NamedCharacterCadence as ContentNamedCharacterCadence,
  NamedCharacterRole as ContentNamedCharacterRole,
  OriginId,
  PhilosopherId,
  PhilosopherThreatStage as ContentPhilosopherThreatStage,
  PriestRole as ContentPriestRole,
  PythiaTraitId,
  ReputationTierId,
  RivalOracleId,
  RivalOracleOperationId,
  ResourceId,
  ScenarioId,
  TileSemantics
} from "@the-oracle/content";

export type {
  BuildingDefId,
  DomainTag,
  FactionAgenda,
  FactionId,
  FactionProfile,
  LegendaryConsultationId,
  NamedCharacterArchetypeId,
  OriginId,
  PhilosopherId,
  PythiaTraitId,
  ReputationTierId,
  RivalOracleId,
  RivalOracleOperationId,
  ResourceId,
  ScenarioId,
  TileSemantics
};
export type { BurdenId } from "./lineage";
import type { AgeState } from "./ages";
import type { EspionageState } from "./espionage";
import type { ExcavationState } from "./excavation";
import type { LegacyState } from "./legacy";
import type { LineageState, BurdenId } from "./lineage";
import type { ProphecyArcState } from "./prophecy";
import type { WorldGenerationState } from "./worldGen";
import type { WorldHistoryState } from "./worldHistory";

export type Season = "Spring" | "Summer" | "Autumn" | "Winter";

export type PriestRole = ContentPriestRole;
export type PhilosopherThreatStage = ContentPhilosopherThreatStage;
export type NamedCharacterRole = ContentNamedCharacterRole;
export type NamedCharacterCadence = ContentNamedCharacterCadence;

export type WalkerRole = "priest" | "pilgrim" | "custodian" | "carrier";

export type SemanticTarget =
  | "army"
  | "fleet"
  | "king"
  | "city"
  | "oracle"
  | "harvest"
  | "treasury"
  | "alliance";

export type SemanticAction =
  | "triumph"
  | "fall"
  | "fracture"
  | "endure"
  | "prosper"
  | "withhold";

export type SemanticPolarity = "favorable" | "warning" | "double";

export type AmbiguityBand = "cryptic" | "balanced" | "specific";

export type TimeHorizon = "immediate" | "seasonal" | "yearly";

export type Coord = {
  x: number;
  y: number;
};

export type PlacementTool = "select" | BuildingDefId;

export type WorldClock = {
  tick: number;
  day: number;
  month: number;
  year: number;
  season: Season;
  tickOfDay: number;
  ticksPerDay: number;
  speed: 0 | 1 | 2 | 3;
  paused: boolean;
};

export type ResourceState = {
  amount: number;
  capacity: number;
  trend: number;
};

export type BuildingInstance = {
  id: string;
  defId: BuildingDefId;
  position: Coord;
  condition: number;
  maxCondition: number;
  requiresPriest: boolean;
  assignedPriestIds: string[];
  storedResources: Partial<Record<ResourceId, number>>;
  connectedToRoad: boolean;
};

export type WalkerState = "idle" | "moving" | "working" | "repairing" | "visiting" | "hauling" | "delivering";

export type WalkerInstance = {
  id: string;
  role: WalkerRole;
  name: string;
  tile: Coord;
  state: WalkerState;
  homeBuildingId?: string;
  assignmentBuildingId?: string;
  path: Coord[];
  moveCooldown: number;
  destination?: Coord;
  carrying?: ResourceId;
  carryingAmount?: number;
  assignedJobId?: string;
  fatigue?: number;
  haulingSkill?: number;
  supplyRadius?: number;
};

export type PriestSecretKind = "corruption" | "heresy" | "rivalry" | "forbidden_knowledge";

export type PriestSecret = {
  id: string;
  kind: PriestSecretKind;
  severity: number;
  discoveredDay?: number;
  exposedDay?: number;
};

export type PriestState = {
  id: string;
  walkerId: string;
  role: PriestRole;
  skill: number;
  morale: number;
  range: number;
  currentAssignmentBuildingId?: string;
  homeBuildingId?: string;
  secrets?: PriestSecret[];
  successionRank?: number;
};

export type PriestTemperament = "steady" | "zealous" | "cunning" | "scholarly";

export type PriestAmbition = "guardian" | "broker" | "reformer" | "mystic";

export type PriestPoliticalStance = "loyalist" | "traditionalist" | "broker" | "reformer";

export type PriestCouncilBlocId = "pythia" | "rites" | "patrons" | "reformers";

export type PriestPoliticalProfile = {
  priestId: string;
  temperament: PriestTemperament;
  ambition: PriestAmbition;
  stance: PriestPoliticalStance;
  influence: number;
  loyalty: number;
  dissent: number;
  favoredFactionId?: FactionId;
  anchorCharacterId?: string;
  note: string;
};

export type PriestCouncilBlocState = {
  id: PriestCouncilBlocId;
  label: string;
  support: number;
  tension: number;
  note: string;
};

export type PriestPoliticsStatus = "calm" | "restless" | "fractured" | "crisis";

export type SuccessionContest = {
  active: boolean;
  candidates: string[];
  frontRunnerId?: string;
  startDay: number;
};

export type PriestPoliticsState = {
  overallPressure: number;
  unity: number;
  dominantBlocId: PriestCouncilBlocId;
  status: PriestPoliticsStatus;
  currentIssue: string;
  rumor: string;
  featuredCharacterIds: string[];
  lastUpdatedDay: number;
  blocs: PriestCouncilBlocState[];
  priests: Record<string, PriestPoliticalProfile>;
  successionContest?: SuccessionContest;
};

export type PythiaState = {
  name: string;
  attunement: number;
  physicalHealth: number;
  mentalClarity: number;
  tranceDepth: number;
  prestige: number;
  needs: {
    purification: number;
    rest: number;
    pilgrimageCooldown: number;
  };
  traits: PythiaTraitId[];
};

export type FactionState = {
  id: FactionId;
  name: string;
  profile: FactionProfile;
  favoredResource: Exclude<ResourceId, "gold">;
  relations: Partial<Record<FactionId, number>>;
  treaties: FactionId[];
  embargoes: FactionId[];
  credibility: number;
  favour: number;
  dependence: number;
  debt: number;
  currentAgenda: FactionAgenda;
  activeConflicts: FactionId[];
  tradeAccess: boolean;
  lastOutcome?: string;
  history: string[];
};

export type PhilosopherThreatLevel = "dormant" | PhilosopherThreatStage;

export type PhilosopherThreatState = {
  philosopherId: PhilosopherId;
  worldview: string;
  influence: number;
  suspicion: number;
  pressure: number;
  stage: PhilosopherThreatLevel;
  active: boolean;
  lastShiftMonth?: number;
  lastEventDay?: number;
};

export type PhilosophersState = {
  byFaction: Record<FactionId, PhilosopherThreatState>;
  spotlightFactionIds: FactionId[];
};

export type RivalOracleDiscovery = "shadow" | "suspected" | "confirmed";

export type RivalOracleOperationState = {
  id: RivalOracleOperationId;
  successCount: number;
  lastExecutedDay?: number;
};

export type RivalOracleIncident = {
  id: string;
  day: number;
  rivalId: RivalOracleId;
  operationId: RivalOracleOperationId;
  targetRegionId: string;
  targetFactionId?: FactionId;
  discovery: RivalOracleDiscovery;
  pressureDelta: number;
  visibility: number;
  intel: number;
  summary: string;
};

export type RivalOracleState = {
  id: RivalOracleId;
  name: string;
  title: string;
  homeRegionId: string;
  favoredDomain: DomainTag;
  active: boolean;
  patronFactionId: FactionId;
  patronage: number;
  pressure: number;
  visibility: number;
  intel: number;
  intrigue: number;
  pressureCap: number;
  lastOperationDay?: number;
  lastKnownOperationId?: RivalOracleOperationId;
  lastTargetRegionId?: string;
  operations: RivalOracleOperationState[];
};

export type RivalOraclesState = {
  roster: RivalOracleState[];
  spotlightRivalIds: RivalOracleId[];
  incidents: RivalOracleIncident[];
  totalPressure: number;
  lastPressureDay?: number;
};

export type CharacterRelationshipState = {
  trust: number;
  fear: number;
  hostility: number;
  familiarity: number;
};

export type CharacterMemoryState = {
  knownSinceDay: number;
  visitCount: number;
  successfulVisits: number;
  rebuffedVisits: number;
  lastSeenDay?: number;
  lastInteractionDay?: number;
  lastLocationId?: string;
  lastImpression?: string;
  notableMoments: string[];
};

export type NamedCharacterStatus = "active" | "dormant" | "legendary";

export type NamedCharacterState = {
  id: string;
  defId: NamedCharacterArchetypeId;
  role: NamedCharacterRole;
  cadence: NamedCharacterCadence;
  name: string;
  title: string;
  epithet: string;
  displayName: string;
  homeFactionId?: FactionId;
  anchorRegionId?: string;
  influence: number;
  prominence: number;
  status: NamedCharacterStatus;
  tags: string[];
  relationship: CharacterRelationshipState;
  memory: CharacterMemoryState;
};

export type CharactersState = {
  roster: NamedCharacterState[];
  spotlightCharacterIds: string[];
};

export type OmenReport = {
  id: string;
  sourceRole: string;
  text: string;
  semantics: TileSemantics;
  reliability: number;
};

export type WordTile = {
  id: string;
  text: string;
  category: "subject" | "action" | "condition" | "modifier" | "seal";
  semantics: TileSemantics;
};

export type ProphecyDepthBand = "shallow" | "grounded" | "deep" | "oracular";

export type ProphecyScaffoldPart = {
  kind: "spine" | "hinge" | "seal";
  label: string;
  text: string;
  tileIds: string[];
  state: "missing" | "forming" | "stable" | "charged";
};

export type ProphecyInterpretation = {
  summary: string;
  politicalReading: string;
  caution: string;
  fulfillmentWindow: string;
  rivalContext?: string | null;
};

export type ProphecyRecord = {
  id: string;
  factionId: FactionId;
  dayIssued: number;
  text: string;
  tileIds: string[];
  semantics: TileSemantics[];
  clarity: number;
  value: number;
  risk: number;
  depth?: number;
  depthBand?: ProphecyDepthBand;
  omenReliability?: number;
  omenConsensus?: "aligned" | "mixed" | "contradictory";
  scaffold?: ProphecyScaffoldPart[];
  interpretation?: ProphecyInterpretation;
  dueDay: number;
  resolved: boolean;
  resolvedDay?: number;
  resolutionReport?: string;
  credibilityDelta?: number;
};

export type ConsequenceCase = {
  id: string;
  prophecyId: string;
  factionId: FactionId;
  dueDay: number;
  outcome: TileSemantics;
  resolved: boolean;
  report?: string;
  credibilityDelta?: number;
};

export type ConsultationCurrent = {
  id: string;
  factionId: FactionId;
  envoyName: string;
  mood: string;
  paymentOffered: number;
  question: string;
  domain: DomainTag;
  omenReports: OmenReport[];
  tilePool: WordTile[];
  placedTileIds: string[];
  scorePreview: {
    clarity: number;
    value: number;
    risk: number;
    depth?: number;
    depthBand?: ProphecyDepthBand;
  };
};

export type ConsultationState = {
  mode: "idle" | "pending" | "open";
  current?: ConsultationCurrent;
  history: ProphecyRecord[];
};

export type AdvisorMessage = {
  id: string;
  advisorId: string;
  text: string;
  severity: "info" | "warn" | "critical";
};

export type EventFeedItem = {
  id: string;
  day: number;
  text: string;
};

export type TradeOffer = {
  id: string;
  factionId: FactionId;
  resourceId: ResourceId;
  amount: number;
  price: number;
};

export type ResourceTransferJob = {
  id: string;
  resourceId: Exclude<ResourceId, "gold">;
  amount: number;
  sourceBuildingId: string;
  targetBuildingId: string;
  priority: "critical" | "high" | "routine";
  assignedWalkerId?: string;
  phase: "to_source" | "to_target";
};

export type ReputationState = {
  score: number;
  currentTier: ReputationTierId;
  nextTier?: ReputationTierId;
  thresholds: Record<ReputationTierId, number>;
  unlockedBuildingIds: BuildingDefId[];
  unlockedScenarioIds: ScenarioId[];
  lastTierChangeDay?: number;
};

export type TreasuryDedicationState = {
  completed: number;
  totalGoldInvested: number;
  nextMilestoneGold: number;
  lastDedicationDay?: number;
};

export type WorldNodeTag = "sanctuary" | "city" | "port" | "pass";

export type WorldMapNode = {
  id: string;
  label: string;
  position: Coord;
  controllingFactionId?: FactionId;
  tags: WorldNodeTag[];
  pressure: number;
  unrest: number;
  connectedNodeIds: string[];
};

export type WorldMapLink = {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  kind: "road" | "sea" | "mountain";
  tradeFlow: number;
  risk: number;
};

export type CampaignPressureIndicator = {
  id: string;
  factionId: FactionId;
  nodeId: string;
  kind: "consultation" | "trade" | "conflict";
  severity: "low" | "rising" | "critical";
  value: number;
};

export type CrisisChainState = {
  id: string;
  label: string;
  nodeId: string;
  factionId?: FactionId;
  stage: "rumor" | "active" | "resolution";
  pressure: number;
  stepsCompleted: number;
  resolvedDay?: number;
};

export type CampaignState = {
  scenarioId: ScenarioId;
  reputation: ReputationState;
  treasury: TreasuryDedicationState;
  patronMilestones: string[];
  worldMap: {
    selectedNodeId?: string;
    nodes: WorldMapNode[];
    links: WorldMapLink[];
    activePressures: CampaignPressureIndicator[];
    crisisChains: CrisisChainState[];
  };
  winCondition: {
    id: string;
    label: string;
    completed: boolean;
    summary?: string;
    completedDay?: number;
  };
};

export type LegendaryConsultationProgress = {
  consultationId: LegendaryConsultationId;
  currentStage: number;
  startDay: number;
  completed: boolean;
  completedDay?: number;
};

export type GameEvent =
  | { type: "DayAdvanced"; day: number }
  | { type: "BuildingPlaced"; buildingId: string; defId: BuildingDefId }
  | { type: "WalkerAssigned"; walkerId: string; buildingId: string }
  | { type: "ResourceConsumed"; resourceId: ResourceId; amount: number }
  | { type: "BuildingDegraded"; buildingId: string; condition: number }
  | { type: "ConsultationStarted"; consultationId: string; factionId: FactionId }
  | { type: "ProphecyDelivered"; prophecyId: string; factionId: FactionId }
  | { type: "CredibilityChanged"; factionId: FactionId; delta: number }
  | { type: "ConsequenceResolved"; consequenceId: string; factionId: FactionId; delta: number }
  | { type: "RivalOracleOperation"; rivalId: RivalOracleId; operationId: RivalOracleOperationId; day: number }
  | { type: "TradePurchased"; offerId: string; resourceId: ResourceId; amount: number }
  | { type: "AutosaveTriggered"; day: number };

export type GameSnapshot = {
  version: 1;
  state: GameState;
  recentEvents: GameEvent[];
};

export type GameState = {
  worldSeed: number;
  worldSeedText: string;
  originId: OriginId;
  worldGeneration: WorldGenerationState;
  clock: WorldClock;
  grid: {
    width: number;
    height: number;
    roads: Coord[];
  };
  resources: Record<ResourceId, ResourceState>;
  buildings: BuildingInstance[];
  walkers: WalkerInstance[];
  priests: PriestState[];
  priestPolitics?: PriestPoliticsState;
  pythia: PythiaState;
  factions: Record<FactionId, FactionState>;
  characters?: CharactersState;
  philosophers?: PhilosophersState;
  rivalOracles?: RivalOraclesState;
  age?: AgeState;
  espionage?: EspionageState;
  excavation?: ExcavationState;
  legacy?: LegacyState;
  prophecyArcs?: ProphecyArcState;
  worldHistory?: WorldHistoryState;
  lineage?: LineageState;
  endlessMode?: boolean;
  activeBurdens?: BurdenId[];
  legendaryProgress?: LegendaryConsultationProgress[];
  availableLegendary?: LegendaryConsultationId[];
  consultation: ConsultationState;
  consequences: ConsequenceCase[];
  advisorMessages: AdvisorMessage[];
  eventFeed: EventFeedItem[];
  tradeOffers: TradeOffer[];
  resourceJobs: ResourceTransferJob[];
  campaign: CampaignState;
  ui: {
    selectedEntityId?: string;
    selectedEntityKind?: "building" | "walker";
    activeTool: PlacementTool;
    hoveredTile?: Coord;
  };
  lastAutosaveDay: number;
  nextId: number;
};
