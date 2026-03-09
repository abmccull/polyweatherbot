export type ResourceId =
  | "gold"
  | "sacred_water"
  | "olive_oil"
  | "incense"
  | "grain"
  | "sacred_animals"
  | "bread"
  | "olives"
  | "papyrus"
  | "scrolls";

export type BuildingDefId =
  | "sacred_way"
  | "priest_quarters"
  | "storehouse"
  | "castalian_spring"
  | "inner_sanctum"
  | "eternal_flame_brazier"
  | "sacrificial_altar"
  | "animal_pen"
  | "granary"
  | "kitchen"
  | "olive_press"
  | "incense_store"
  | "agora_market"
  | "xenon"
  | "grain_field"
  | "olive_grove"
  | "incense_workshop"
  | "papyrus_reed_bed"
  | "scriptorium"
  | "library";

export type PriestRole =
  | "attendant"
  | "spring_warden"
  | "flame_keeper"
  | "augur"
  | "sacrificial_priest"
  | "dream_priest"
  | "astronomer"
  | "scholar";

export type FactionId =
  | "athens"
  | "sparta"
  | "corinth"
  | "thebes"
  | "argos"
  | "miletus"
  | "syracuse"
  | "macedon";

export type FactionAgenda = "war" | "trade" | "faith" | "succession";

export type DomainTag = "military" | "economic" | "spiritual";

export type FactionProfile = "martial" | "mercantile" | "devout" | "scheming";

export type PhilosopherId = "anaxagoras" | "democritus" | "gorgias" | "heraclitus" | "pythagoras";

export type PhilosopherSchool = "court-naturalist" | "atomist" | "sophist" | "strife-doctrine" | "mystic-order";

export type PhilosopherThreatStage = "rumor" | "circle" | "sect" | "crisis";

export type RivalOracleId = "oak-seers" | "isthmus-ledger" | "wolf-cup" | "mist-sibyls";

export type RivalOracleOperationId = "poach-supplicants" | "plant-whispers" | "court-patron" | "counter-rite";

export type RivalOracleOperationKind = "pressure" | "espionage" | "patronage";

export type NamedCharacterRole = "merchant" | "general" | "envoy" | "philosopher" | "priest" | "legendary";

export type NamedCharacterArchetypeId =
  | "merchant-patron"
  | "campaign-general"
  | "faction-envoy"
  | "wandering-philosopher"
  | "delphic-priest"
  | "legendary-visitor";

export type NamedCharacterCadence = "seasonal" | "campaign" | "diplomatic" | "wandering" | "ritual" | "legendary";

export type PhilosopherEffectVector = {
  credibility: number;
  favour: number;
  debt: number;
  dependence: number;
};

export type PhilosopherPressureBias = {
  debt: number;
  dependence: number;
  lowCredibility: number;
  conflicts: number;
  embargoes: number;
  agendaAlignment: number;
  worldviewAlignment: number;
};

export type PythiaTraitId = "visionary" | "calculating" | "diplomatic" | "fragile";

export type ScenarioId =
  | "rising-oracle"
  | "sandbox"
  | "young-oracle"
  | "crossroads-of-power"
  | "skeptics-challenge"
  | "golden-age"
  | "twilight-of-the-gods"
  | "spys-web"
  | "sacred-excavation";

export type OriginId =
  | "ancient-spring"
  | "upstart-shrine"
  | "cursed-oracle"
  | "war-oracle"
  | "gods-favourite"
  | "merchant-oracle"
  | "exiles-oracle";

export type ReputationTierId = "obscure" | "recognized" | "revered" | "panhellenic";

export type BuildingCategory = "processional" | "housing" | "storage" | "ritual" | "production" | "trade" | "hospitality";

export type TileSemantics = {
  target: "army" | "fleet" | "king" | "city" | "oracle" | "harvest" | "treasury" | "alliance";
  action: "triumph" | "fall" | "fracture" | "endure" | "prosper" | "withhold";
  polarity: "favorable" | "warning" | "double";
  ambiguity: "cryptic" | "balanced" | "specific";
  timeHorizon: "immediate" | "seasonal" | "yearly";
  domain: DomainTag;
};

export type BuildingRecipe = {
  id: string;
  consumes: Partial<Record<ResourceId, number>>;
  produces: Partial<Record<ResourceId, number>>;
  dailyRate: number;
  requiresRoles?: PriestRole[];
  notes?: string;
};

export type BuildingPassiveEffect = {
  id: string;
  kind: "prestige" | "donation" | "omen_quality" | "trade_income" | "pilgrim_capacity" | "storage_buffer";
  value: number;
  resourceId?: ResourceId;
  notes?: string;
};

export type BuildingStaffing = {
  priests?: Partial<Record<PriestRole, number>>;
  custodians?: number;
  carriers?: number;
  visitors?: number;
};

export type BuildingDef = {
  id: BuildingDefId;
  name: string;
  description: string;
  category: BuildingCategory;
  costGold: number;
  requiresPriest: boolean;
  maxCondition: number;
  color: number;
  unlockTier?: ReputationTierId;
  upkeep: Partial<Record<ResourceId, number>>;
  startingResources: Partial<Record<ResourceId, number>>;
  staffing?: BuildingStaffing;
  storageCaps?: Partial<Record<ResourceId, number>>;
  recipes?: BuildingRecipe[];
  passiveEffects?: BuildingPassiveEffect[];
};

export type SeasonName = "Spring" | "Summer" | "Autumn" | "Winter";

export type ResourceDef = {
  id: ResourceId;
  label: string;
  category: "currency" | "ritual" | "food" | "trade";
  seasonalMultipliers?: Partial<Record<SeasonName, number>>;
};

export type OmenDef = {
  id: string;
  priestRole: string;
  family?: DomainTag | "ritual";
  templates: string[];
};

export type WordTileDef = {
  id: string;
  text: string;
  category: "subject" | "action" | "condition" | "modifier" | "seal";
  semantics: TileSemantics;
};

export type FactionDef = {
  id: FactionId;
  name: string;
  defaultAgenda: FactionAgenda;
  tradeAccess: boolean;
  profile: FactionProfile;
  favoredResource: Exclude<ResourceId, "gold">;
};

export type TraitDef = {
  id: PythiaTraitId;
  label: string;
  description: string;
};

export type AdvisorDef = {
  id: string;
  name: string;
  voice: string;
};

export type ScenarioStartConditions = {
  gold?: number;
  reputation?: string;
  factionMod?: Record<string, Partial<{ credibility: number; favour: number; hostility: number }>>;
  burdens?: string[];
  startAge?: string;
};

export type ScenarioDef = {
  id: ScenarioId;
  label: string;
  summary: string;
  description?: string;
  difficulty?: 1 | 2 | 3 | 4 | 5;
  startingConditions?: ScenarioStartConditions;
  specialRules?: string[];
  victoryCondition?: string;
  recommendedStartingTier?: ReputationTierId;
  dedicationMilestones?: number[];
  requiredResolvedCrises?: number;
  winningTier?: ReputationTierId;
};

export type OriginDef = {
  id: OriginId;
  label: string;
  title?: string;
  subtitle?: string;
  summary: string;
  advisorIntro: string;
  uniqueMechanic: string;
  disabledSystems?: string[];
  challengeTags?: string[];
  scoreModifier?: number;
  startingResources?: Partial<Record<ResourceId, number>>;
  pythiaModifiers?: {
    attunement?: number;
    physicalHealth?: number;
    mentalClarity?: number;
    tranceDepth?: number;
    prestige?: number;
    rest?: number;
    purification?: number;
    pilgrimageCooldown?: number;
    addTraits?: PythiaTraitId[];
  };
  campaignModifiers?: {
    reputation?: number;
    treasuryProgress?: number;
  };
  worldBias?: {
    climate?: number;
    economy?: number;
    divineMood?: number;
    oracleDensity?: number;
    war?: number;
    trade?: number;
    faith?: number;
    intrigue?: number;
    unrest?: number;
    pressure?: number;
  };
};

export type PoliticalEventDef = {
  id: string;
  label: string;
  domain: DomainTag;
  agenda: FactionAgenda;
  summaries: string[];
};

export type ConsultationQuestionDef = {
  id: string;
  text: string;
  domain: DomainTag;
  tags?: string[];
};

export type PhilosopherThreatDef = {
  id: PhilosopherId;
  name: string;
  school: PhilosopherSchool;
  doctrine: string;
  domain: DomainTag;
  favoredAgenda: FactionAgenda;
  preferredProfiles: FactionProfile[];
  preferredWorldviews: string[];
  consultationTags: string[];
  stageThresholds: {
    circle: number;
    sect: number;
    crisis: number;
  };
  pressureBias: PhilosopherPressureBias;
  stageEffects: Record<PhilosopherThreatStage, PhilosopherEffectVector>;
};

export type RivalOracleOperationDef = {
  id: RivalOracleOperationId;
  label: string;
  kind: RivalOracleOperationKind;
  domain: DomainTag;
  summary: string;
  basePressure: number;
  baseVisibility: number;
  baseIntel: number;
  patronageShift: number;
};

export type RivalOracleDef = {
  id: RivalOracleId;
  name: string;
  title: string;
  summary: string;
  homeRegionId: string;
  favoredDomain: DomainTag;
  baselinePressure: number;
  secrecy: number;
  pressureCap: number;
  patronPool: FactionId[];
  preferredPatronProfiles: FactionProfile[];
  preferredPatronAgendas: FactionAgenda[];
  operationIds: RivalOracleOperationId[];
};

export type NamedCharacterDef = {
  id: NamedCharacterArchetypeId;
  role: NamedCharacterRole;
  label: string;
  cadence: NamedCharacterCadence;
  minInitialCount: number;
  maxInitialCount: number;
  baseInfluence: number;
  baseProminence: number;
  titlePool: string[];
  firstNamePool: string[];
  epithetPool: string[];
  preferredAgendas?: FactionAgenda[];
  preferredProfiles?: FactionProfile[];
  preferredDomains?: DomainTag[];
  initialTags: string[];
};
