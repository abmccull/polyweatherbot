import {
  createInitialState,
  normalizeCampaignState,
  normalizeCarrierWalker,
  normalizeWorldGenerationState,
  type GameSnapshot
} from "@the-oracle/core";

export const SNAPSHOT_VERSION = 1;
export const SAVE_FORMAT_VERSION = 1;
export const MAX_SNAPSHOT_CHARS = 1_000_000;
export const MAX_SAVE_BYTES = 5 * 1024 * 1024; // 5 MB

const MAX_COLLECTION_ITEMS = 4_096;
const BASELINE_STATE = createInitialState();

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.length <= MAX_COLLECTION_ITEMS && value.every(isString);
}

function isRecordArray(value: unknown): value is Record<string, unknown>[] {
  return Array.isArray(value) && value.length <= MAX_COLLECTION_ITEMS && value.every(isRecord);
}

function isCoord(value: unknown): value is { x: number; y: number } {
  return isRecord(value) && isFiniteNumber(value.x) && isFiniteNumber(value.y);
}

function isCoordArray(value: unknown): value is { x: number; y: number }[] {
  return Array.isArray(value) && value.length <= MAX_COLLECTION_ITEMS && value.every(isCoord);
}

function isGameEventLike(value: unknown): boolean {
  return isRecord(value) && isString(value.type);
}

function isResourceStateRecord(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.amount)
    && isFiniteNumber(value.capacity)
    && isFiniteNumber(value.trend);
}

function isStoredResourceRecord(value: unknown): boolean {
  return isRecord(value) && Object.values(value).every((entry) => entry === undefined || isFiniteNumber(entry));
}

function isBuildingInstanceLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.defId)
    && isCoord(value.position)
    && isFiniteNumber(value.condition)
    && isFiniteNumber(value.maxCondition)
    && isBoolean(value.requiresPriest)
    && isStringArray(value.assignedPriestIds)
    && isStoredResourceRecord(value.storedResources)
    && isBoolean(value.connectedToRoad);
}

function isWalkerInstanceLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.role)
    && isString(value.name)
    && isCoord(value.tile)
    && isString(value.state)
    && isCoordArray(value.path)
    && isFiniteNumber(value.moveCooldown)
    && (value.homeBuildingId === undefined || isString(value.homeBuildingId))
    && (value.assignmentBuildingId === undefined || isString(value.assignmentBuildingId))
    && (value.destination === undefined || isCoord(value.destination))
    && (value.carrying === undefined || isString(value.carrying))
    && (value.carryingAmount === undefined || isFiniteNumber(value.carryingAmount))
    && (value.assignedJobId === undefined || isString(value.assignedJobId))
    && (value.fatigue === undefined || isFiniteNumber(value.fatigue))
    && (value.haulingSkill === undefined || isFiniteNumber(value.haulingSkill))
    && (value.supplyRadius === undefined || isFiniteNumber(value.supplyRadius));
}

function isPriestStateLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.walkerId)
    && isString(value.role)
    && isFiniteNumber(value.skill)
    && isFiniteNumber(value.morale)
    && isFiniteNumber(value.range)
    && (value.currentAssignmentBuildingId === undefined || isString(value.currentAssignmentBuildingId))
    && (value.homeBuildingId === undefined || isString(value.homeBuildingId));
}

function isPriestPoliticalProfileLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.priestId)
    && isString(value.temperament)
    && isString(value.ambition)
    && isString(value.stance)
    && isFiniteNumber(value.influence)
    && isFiniteNumber(value.loyalty)
    && isFiniteNumber(value.dissent)
    && (value.favoredFactionId === undefined || isString(value.favoredFactionId))
    && (value.anchorCharacterId === undefined || isString(value.anchorCharacterId))
    && isString(value.note);
}

function isPriestCouncilBlocLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.label)
    && isFiniteNumber(value.support)
    && isFiniteNumber(value.tension)
    && isString(value.note);
}

function isPriestPoliticsStateLike(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.overallPressure)
    && isFiniteNumber(value.unity)
    && isString(value.dominantBlocId)
    && isString(value.status)
    && isString(value.currentIssue)
    && isString(value.rumor)
    && isStringArray(value.featuredCharacterIds)
    && isFiniteNumber(value.lastUpdatedDay)
    && Array.isArray(value.blocs)
    && value.blocs.length <= MAX_COLLECTION_ITEMS
    && value.blocs.every(isPriestCouncilBlocLike)
    && isRecord(value.priests)
    && Object.values(value.priests).every(isPriestPoliticalProfileLike);
}

function isPythiaStateLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.name)
    && isFiniteNumber(value.attunement)
    && isFiniteNumber(value.physicalHealth)
    && isFiniteNumber(value.mentalClarity)
    && isFiniteNumber(value.tranceDepth)
    && isFiniteNumber(value.prestige)
    && isRecord(value.needs)
    && isFiniteNumber(value.needs.purification)
    && isFiniteNumber(value.needs.rest)
    && isFiniteNumber(value.needs.pilgrimageCooldown)
    && isStringArray(value.traits);
}

function isFactionStateLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.name)
    && isString(value.profile)
    && isString(value.favoredResource)
    && (value.relations === undefined || isRecord(value.relations))
    && (value.treaties === undefined || isStringArray(value.treaties))
    && (value.embargoes === undefined || isStringArray(value.embargoes))
    && isFiniteNumber(value.credibility)
    && isFiniteNumber(value.favour)
    && isFiniteNumber(value.dependence)
    && isFiniteNumber(value.debt)
    && isString(value.currentAgenda)
    && isStringArray(value.activeConflicts)
    && isBoolean(value.tradeAccess)
    && (value.lastOutcome === undefined || isString(value.lastOutcome))
    && (value.history === undefined || isStringArray(value.history));
}

function isCharacterRelationshipLike(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.trust)
    && isFiniteNumber(value.fear)
    && isFiniteNumber(value.hostility)
    && isFiniteNumber(value.familiarity);
}

function isCharacterMemoryLike(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.knownSinceDay)
    && isFiniteNumber(value.visitCount)
    && isFiniteNumber(value.successfulVisits)
    && isFiniteNumber(value.rebuffedVisits)
    && (value.lastSeenDay === undefined || isFiniteNumber(value.lastSeenDay))
    && (value.lastInteractionDay === undefined || isFiniteNumber(value.lastInteractionDay))
    && (value.lastLocationId === undefined || isString(value.lastLocationId))
    && (value.lastImpression === undefined || isString(value.lastImpression))
    && isStringArray(value.notableMoments);
}

function isNamedCharacterLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.defId)
    && isString(value.role)
    && isString(value.cadence)
    && isString(value.name)
    && isString(value.title)
    && isString(value.epithet)
    && isString(value.displayName)
    && (value.homeFactionId === undefined || isString(value.homeFactionId))
    && (value.anchorRegionId === undefined || isString(value.anchorRegionId))
    && isFiniteNumber(value.influence)
    && isFiniteNumber(value.prominence)
    && isString(value.status)
    && isStringArray(value.tags)
    && isCharacterRelationshipLike(value.relationship)
    && isCharacterMemoryLike(value.memory);
}

function isCharactersStateLike(value: unknown): boolean {
  return isRecord(value)
    && Array.isArray(value.roster)
    && value.roster.length <= MAX_COLLECTION_ITEMS
    && value.roster.every(isNamedCharacterLike)
    && isStringArray(value.spotlightCharacterIds);
}

function isRivalOracleOperationLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isFiniteNumber(value.successCount)
    && (value.lastExecutedDay === undefined || isFiniteNumber(value.lastExecutedDay));
}

function isRivalOracleIncidentLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isFiniteNumber(value.day)
    && isString(value.rivalId)
    && isString(value.operationId)
    && isString(value.targetRegionId)
    && (value.targetFactionId === undefined || isString(value.targetFactionId))
    && isString(value.discovery)
    && isFiniteNumber(value.pressureDelta)
    && isFiniteNumber(value.visibility)
    && isFiniteNumber(value.intel)
    && isString(value.summary);
}

function isRivalOracleStateLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.name)
    && isString(value.title)
    && isString(value.homeRegionId)
    && isString(value.favoredDomain)
    && isBoolean(value.active)
    && isString(value.patronFactionId)
    && isFiniteNumber(value.patronage)
    && isFiniteNumber(value.pressure)
    && isFiniteNumber(value.visibility)
    && isFiniteNumber(value.intel)
    && isFiniteNumber(value.intrigue)
    && isFiniteNumber(value.pressureCap)
    && (value.lastOperationDay === undefined || isFiniteNumber(value.lastOperationDay))
    && (value.lastKnownOperationId === undefined || isString(value.lastKnownOperationId))
    && (value.lastTargetRegionId === undefined || isString(value.lastTargetRegionId))
    && Array.isArray(value.operations)
    && value.operations.length <= MAX_COLLECTION_ITEMS
    && value.operations.every(isRivalOracleOperationLike);
}

function isRivalOraclesStateLike(value: unknown): boolean {
  return isRecord(value)
    && Array.isArray(value.roster)
    && value.roster.length <= MAX_COLLECTION_ITEMS
    && value.roster.every(isRivalOracleStateLike)
    && isStringArray(value.spotlightRivalIds)
    && Array.isArray(value.incidents)
    && value.incidents.length <= MAX_COLLECTION_ITEMS
    && value.incidents.every(isRivalOracleIncidentLike)
    && isFiniteNumber(value.totalPressure)
    && (value.lastPressureDay === undefined || isFiniteNumber(value.lastPressureDay));
}

function isOmenReportLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.sourceRole)
    && isString(value.text)
    && isRecord(value.semantics)
    && isFiniteNumber(value.reliability);
}

function isWordTileLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.text)
    && isString(value.category)
    && isRecord(value.semantics);
}

function isProphecyRecordLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.factionId)
    && isFiniteNumber(value.dayIssued)
    && isString(value.text)
    && isStringArray(value.tileIds)
    && Array.isArray(value.semantics)
    && value.semantics.length <= MAX_COLLECTION_ITEMS
    && value.semantics.every(isRecord)
    && isFiniteNumber(value.clarity)
    && isFiniteNumber(value.value)
    && isFiniteNumber(value.risk)
    && isFiniteNumber(value.dueDay)
    && isBoolean(value.resolved)
    && (value.resolvedDay === undefined || isFiniteNumber(value.resolvedDay))
    && (value.resolutionReport === undefined || isString(value.resolutionReport))
    && (value.credibilityDelta === undefined || isFiniteNumber(value.credibilityDelta));
}

function isConsultationCurrentLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.factionId)
    && isString(value.envoyName)
    && isString(value.mood)
    && isFiniteNumber(value.paymentOffered)
    && isString(value.question)
    && isString(value.domain)
    && Array.isArray(value.omenReports)
    && value.omenReports.length <= MAX_COLLECTION_ITEMS
    && value.omenReports.every(isOmenReportLike)
    && Array.isArray(value.tilePool)
    && value.tilePool.length <= MAX_COLLECTION_ITEMS
    && value.tilePool.every(isWordTileLike)
    && isStringArray(value.placedTileIds)
    && isRecord(value.scorePreview)
    && isFiniteNumber(value.scorePreview.clarity)
    && isFiniteNumber(value.scorePreview.value)
    && isFiniteNumber(value.scorePreview.risk);
}

function isConsultationStateLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.mode)
    && (value.current === undefined || isConsultationCurrentLike(value.current))
    && Array.isArray(value.history)
    && value.history.length <= MAX_COLLECTION_ITEMS
    && value.history.every(isProphecyRecordLike);
}

function isConsequenceCaseLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.prophecyId)
    && isString(value.factionId)
    && isFiniteNumber(value.dueDay)
    && isRecord(value.outcome)
    && isBoolean(value.resolved)
    && (value.report === undefined || isString(value.report))
    && (value.credibilityDelta === undefined || isFiniteNumber(value.credibilityDelta));
}

function isAdvisorMessageLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.advisorId)
    && isString(value.text)
    && isString(value.severity);
}

function isEventFeedItemLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isFiniteNumber(value.day)
    && isString(value.text);
}

function isTradeOfferLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.factionId)
    && isString(value.resourceId)
    && isFiniteNumber(value.amount)
    && isFiniteNumber(value.price);
}

function isResourceTransferJobLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.resourceId)
    && isFiniteNumber(value.amount)
    && isString(value.sourceBuildingId)
    && isString(value.targetBuildingId)
    && isString(value.priority)
    && (value.assignedWalkerId === undefined || isString(value.assignedWalkerId))
    && isString(value.phase);
}

function isClockLike(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.tick)
    && isFiniteNumber(value.day)
    && isFiniteNumber(value.month)
    && isFiniteNumber(value.year)
    && isString(value.season)
    && isFiniteNumber(value.tickOfDay)
    && isFiniteNumber(value.ticksPerDay)
    && isFiniteNumber(value.speed)
    && isBoolean(value.paused);
}

function isGridLike(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.width)
    && isFiniteNumber(value.height)
    && isCoordArray(value.roads);
}

function isUiStateLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.activeTool)
    && (value.selectedEntityId === undefined || isString(value.selectedEntityId))
    && (value.selectedEntityKind === undefined || isString(value.selectedEntityKind))
    && (value.hoveredTile === undefined || isCoord(value.hoveredTile));
}

function isWorldMetricLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.label)
    && isString(value.value)
    && (value.detail === undefined || isString(value.detail))
    && (value.meter === undefined || isFiniteNumber(value.meter))
    && (value.tone === undefined || isString(value.tone));
}

function isWorldShareLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.label)
    && (value.value === undefined || isString(value.value))
    && (value.detail === undefined || isString(value.detail))
    && (value.tone === undefined || isString(value.tone));
}

function isWorldPressureLike(value: unknown): boolean {
  return isRecord(value)
    && isWorldShareLike(value)
    && (value.severity === undefined || isString(value.severity))
    && (value.factionLabel === undefined || isString(value.factionLabel))
    && (value.nodeId === undefined || isString(value.nodeId));
}

function isGeneratedFactionLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.profile)
    && isString(value.agenda)
    && isBoolean(value.tradeAccess)
    && isFiniteNumber(value.credibilityDelta)
    && isFiniteNumber(value.favourDelta)
    && isFiniteNumber(value.dependenceDelta)
    && isFiniteNumber(value.debtDelta)
    && isFiniteNumber(value.influence)
    && isString(value.note);
}

function isGeneratedRegionLike(value: unknown): boolean {
  return isRecord(value)
    && isString(value.id)
    && isString(value.label)
    && isString(value.summary)
    && isCoord(value.position)
    && isStringArray(value.connectedNodeIds)
    && (value.controllingFactionId === undefined || isString(value.controllingFactionId))
    && isString(value.climate)
    && isString(value.hegemon)
    && isString(value.philosophy)
    && isString(value.divineMood)
    && isString(value.oracleDensity)
    && isFiniteNumber(value.unrest)
    && isFiniteNumber(value.pressure)
    && Array.isArray(value.tags)
    && value.tags.length <= MAX_COLLECTION_ITEMS
    && value.tags.every(isString)
    && Array.isArray(value.factionMix)
    && value.factionMix.length <= MAX_COLLECTION_ITEMS
    && value.factionMix.every(isWorldShareLike)
    && Array.isArray(value.pressures)
    && value.pressures.length <= MAX_COLLECTION_ITEMS
    && value.pressures.every(isWorldPressureLike)
    && isRecordArray(value.history)
    && value.history.every(isWorldShareLike);
}

function isWorldGenerationLike(value: unknown): boolean {
  return isRecord(value)
    && isFiniteNumber(value.seed)
    && isString(value.seedText)
    && isString(value.originId)
    && isString(value.originTitle)
    && isFiniteNumber(value.cityStateCount)
    && isString(value.politicalClimate)
    && isString(value.economicClimate)
    && isWorldMetricLike(value.economic)
    && isWorldMetricLike(value.climate)
    && isWorldMetricLike(value.divineMood)
    && isWorldMetricLike(value.oracleDensity)
    && Array.isArray(value.factionMix)
    && value.factionMix.length <= MAX_COLLECTION_ITEMS
    && value.factionMix.every(isWorldShareLike)
    && Array.isArray(value.pressures)
    && value.pressures.length <= MAX_COLLECTION_ITEMS
    && value.pressures.every(isWorldPressureLike)
    && isRecordArray(value.history)
    && value.history.every(isWorldShareLike)
    && isStringArray(value.challengeTags)
    && isString(value.note)
    && isString(value.summary)
    && isFiniteNumber(value.scoreModifier)
    && isString(value.advisorIntro)
    && isString(value.uniqueMechanic)
    && isStringArray(value.disabledSystems)
    && Array.isArray(value.factions)
    && value.factions.length <= MAX_COLLECTION_ITEMS
    && value.factions.every(isGeneratedFactionLike)
    && Array.isArray(value.regions)
    && value.regions.length <= MAX_COLLECTION_ITEMS
    && value.regions.every(isGeneratedRegionLike);
}

function validateKeyedRecord(
  value: unknown,
  expectedKeys: string[],
  itemGuard: (entry: unknown) => boolean
): boolean {
  if (!isRecord(value)) {
    return false;
  }

  return expectedKeys.every((key) => itemGuard(value[key]));
}

export function isGameSnapshot(value: unknown): value is GameSnapshot {
  if (!isRecord(value) || value.version !== SNAPSHOT_VERSION || !Array.isArray(value.recentEvents) || !isRecord(value.state)) {
    return false;
  }

  const { state, recentEvents } = value;
  if (
    recentEvents.length > MAX_COLLECTION_ITEMS
    || !recentEvents.every(isGameEventLike)
    || !isFiniteNumber(state.worldSeed)
    || (state.worldSeedText !== undefined && !isString(state.worldSeedText))
    || (state.originId !== undefined && !isString(state.originId))
    || (state.worldGeneration !== undefined && !isWorldGenerationLike(state.worldGeneration))
    || !isClockLike(state.clock)
    || !isGridLike(state.grid)
    || !validateKeyedRecord(state.resources, Object.keys(BASELINE_STATE.resources), isResourceStateRecord)
    || !Array.isArray(state.buildings)
    || state.buildings.length > MAX_COLLECTION_ITEMS
    || !state.buildings.every(isBuildingInstanceLike)
    || !Array.isArray(state.walkers)
    || state.walkers.length > MAX_COLLECTION_ITEMS
    || !state.walkers.every(isWalkerInstanceLike)
    || !Array.isArray(state.priests)
    || state.priests.length > MAX_COLLECTION_ITEMS
    || !state.priests.every(isPriestStateLike)
    || (state.priestPolitics !== undefined && !isPriestPoliticsStateLike(state.priestPolitics))
    || !isPythiaStateLike(state.pythia)
    || !validateKeyedRecord(state.factions, Object.keys(BASELINE_STATE.factions), isFactionStateLike)
    || (state.characters !== undefined && !isCharactersStateLike(state.characters))
    || (state.rivalOracles !== undefined && !isRivalOraclesStateLike(state.rivalOracles))
    || !isConsultationStateLike(state.consultation)
    || !Array.isArray(state.consequences)
    || state.consequences.length > MAX_COLLECTION_ITEMS
    || !state.consequences.every(isConsequenceCaseLike)
    || !Array.isArray(state.advisorMessages)
    || state.advisorMessages.length > MAX_COLLECTION_ITEMS
    || !state.advisorMessages.every(isAdvisorMessageLike)
    || !Array.isArray(state.eventFeed)
    || state.eventFeed.length > MAX_COLLECTION_ITEMS
    || !state.eventFeed.every(isEventFeedItemLike)
    || !Array.isArray(state.tradeOffers)
    || state.tradeOffers.length > MAX_COLLECTION_ITEMS
    || !state.tradeOffers.every(isTradeOfferLike)
    || !Array.isArray(state.resourceJobs)
    || state.resourceJobs.length > MAX_COLLECTION_ITEMS
    || !state.resourceJobs.every(isResourceTransferJobLike)
    || !isUiStateLike(state.ui)
    || !isFiniteNumber(state.lastAutosaveDay)
    || !isFiniteNumber(state.nextId)
    || (state.campaign !== undefined && !isRecord(state.campaign))
  ) {
    return false;
  }

  return true;
}

/**
 * Computes a simple FNV-1a 32-bit checksum of the given string.
 * Not cryptographic — intended for integrity checking only.
 */
export function computeChecksum(data: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < data.length; i++) {
    hash ^= data.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export type SaveEnvelope = {
  formatVersion: number;
  checksum: number;
  payload: string;
};

export function serializeSnapshot(snapshot: GameSnapshot): string {
  return JSON.stringify(snapshot);
}

/**
 * Wraps a serialized snapshot in an envelope with format version and checksum.
 */
export function serializeWithIntegrity(snapshot: GameSnapshot): string {
  const payload = JSON.stringify(snapshot);
  const checksum = computeChecksum(payload);
  const envelope: SaveEnvelope = {
    formatVersion: SAVE_FORMAT_VERSION,
    checksum,
    payload
  };
  const result = JSON.stringify(envelope);
  if (result.length > MAX_SAVE_BYTES) {
    throw new Error(`Save file exceeds maximum size of ${MAX_SAVE_BYTES} bytes`);
  }
  return result;
}

/**
 * Unwraps an envelope, verifying checksum and size, then deserializes the snapshot.
 * Falls back to legacy (envelope-less) format for backwards compatibility.
 */
export function deserializeWithIntegrity(raw: string): GameSnapshot {
  if (raw.length > MAX_SAVE_BYTES) {
    throw new Error(`Save file exceeds maximum size of ${MAX_SAVE_BYTES} bytes`);
  }

  // Try envelope format first
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (
      parsed !== null &&
      typeof parsed === "object" &&
      "formatVersion" in parsed &&
      "checksum" in parsed &&
      "payload" in parsed
    ) {
      const envelope = parsed as SaveEnvelope;
      const expectedChecksum = computeChecksum(envelope.payload);
      if (envelope.checksum !== expectedChecksum) {
        throw new Error("Save file integrity check failed: checksum mismatch");
      }
      return deserializeSnapshot(envelope.payload);
    }
  } catch (error) {
    // If the error is from our own validation, re-throw it
    if (error instanceof Error && (error.message.includes("checksum") || error.message.includes("maximum size"))) {
      throw error;
    }
    // Otherwise fall through to legacy format
  }

  // Legacy format: raw snapshot JSON
  return deserializeSnapshot(raw);
}

function normalizeSnapshot(snapshot: GameSnapshot): GameSnapshot {
  const generated = createInitialState({
    seed: (snapshot.state as GameSnapshot["state"] & { worldSeedText?: string }).worldSeedText ?? snapshot.state.worldSeed,
    originId: (snapshot.state as GameSnapshot["state"] & { originId?: string }).originId
  });
  let carrierIndex = 0;
  return {
    ...snapshot,
    state: {
      ...snapshot.state,
      worldSeedText: (snapshot.state as GameSnapshot["state"] & { worldSeedText?: string }).worldSeedText ?? generated.worldSeedText,
      originId: (snapshot.state as GameSnapshot["state"] & { originId?: string }).originId ?? generated.originId,
      worldGeneration: (snapshot.state as GameSnapshot["state"] & { worldGeneration?: GameSnapshot["state"]["worldGeneration"] }).worldGeneration
        ? normalizeWorldGenerationState((snapshot.state as GameSnapshot["state"] & { worldGeneration?: GameSnapshot["state"]["worldGeneration"] }).worldGeneration)
        : generated.worldGeneration,
      walkers: snapshot.state.walkers.map((walker) => {
        if (walker.role !== "carrier") {
          return walker;
        }
        const normalized = normalizeCarrierWalker(walker, carrierIndex);
        carrierIndex += 1;
        return normalized;
      }),
      factions: Object.fromEntries(
        Object.entries(snapshot.state.factions).map(([factionId, faction]) => [
          factionId,
          {
            ...faction,
            relations: { ...(faction.relations ?? {}) },
            treaties: [...(faction.treaties ?? [])],
            embargoes: [...(faction.embargoes ?? [])],
            history: [...(faction.history ?? [])]
          }
        ])
      ) as GameSnapshot["state"]["factions"],
      priestPolitics: snapshot.state.priestPolitics
        ? {
            ...generated.priestPolitics!,
            ...snapshot.state.priestPolitics,
            featuredCharacterIds: snapshot.state.priestPolitics.featuredCharacterIds?.filter((characterId) =>
              (snapshot.state.characters ?? generated.characters)?.roster.some((character) => character.id === characterId)
            ) ?? generated.priestPolitics!.featuredCharacterIds,
            blocs: generated.priestPolitics!.blocs.map((bloc) => {
              const existing = snapshot.state.priestPolitics?.blocs.find((entry) => entry.id === bloc.id);
              return existing ? { ...bloc, ...existing } : bloc;
            }),
            priests: Object.fromEntries(
              snapshot.state.priests.map((priest) => {
                const existing = snapshot.state.priestPolitics?.priests?.[priest.id];
                return [
                  priest.id,
                  existing
                    ? {
                        ...generated.priestPolitics!.priests[priest.id],
                        ...existing
                      }
                    : generated.priestPolitics!.priests[priest.id]
                ];
              })
            )
          }
        : generated.priestPolitics,
      characters: snapshot.state.characters ?? generated.characters,
      rivalOracles: snapshot.state.rivalOracles
        ? {
            roster: generated.rivalOracles!.roster.map((generatedRival) => {
              const saved = snapshot.state.rivalOracles?.roster.find((entry) => entry.id === generatedRival.id);
              if (!saved) {
                return generatedRival;
              }

              return {
                ...generatedRival,
                ...saved,
                patronFactionId: snapshot.state.factions[saved.patronFactionId] ? saved.patronFactionId : generatedRival.patronFactionId,
                operations: generatedRival.operations.map((generatedOperation) => ({
                  ...generatedOperation,
                  ...(saved.operations.find((entry) => entry.id === generatedOperation.id) ?? {})
                }))
              };
            }),
            spotlightRivalIds: snapshot.state.rivalOracles.spotlightRivalIds?.length
              ? snapshot.state.rivalOracles.spotlightRivalIds.filter((rivalId) =>
                generated.rivalOracles!.roster.some((entry) => entry.id === rivalId)
              )
              : generated.rivalOracles!.spotlightRivalIds,
            incidents: [...(snapshot.state.rivalOracles.incidents ?? [])].slice(0, MAX_COLLECTION_ITEMS),
            totalPressure: snapshot.state.rivalOracles.totalPressure,
            lastPressureDay: snapshot.state.rivalOracles.lastPressureDay
          }
        : generated.rivalOracles,
      campaign: snapshot.state.campaign
        ? normalizeCampaignState(snapshot.state.campaign)
        : generated.campaign
    }
  };
}

export function deserializeSnapshot(raw: string): GameSnapshot {
  if (raw.length > MAX_SNAPSHOT_CHARS) {
    throw new Error("Oracle snapshot exceeds maximum allowed size");
  }

  const parsed = JSON.parse(raw) as unknown;
  if (!isGameSnapshot(parsed)) {
    throw new Error("Invalid Oracle snapshot");
  }
  return normalizeSnapshot(parsed);
}
