import type {
  FactionId,
  FactionState,
  GameState,
  NamedCharacterState,
  PriestAmbition,
  PriestCouncilBlocId,
  PriestCouncilBlocState,
  PriestPoliticalProfile,
  PriestPoliticalStance,
  PriestPoliticsState,
  PriestState,
  PriestTemperament,
  WalkerInstance
} from "./gameState";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function hash(seed: number, text: string): number {
  let value = seed ^ 0x9e3779b9;
  for (let index = 0; index < text.length; index += 1) {
    value = Math.imul(value ^ text.charCodeAt(index), 0x45d9f3b);
    value ^= value >>> 15;
  }
  return (value >>> 0) / 4294967296;
}

function pickOne<T>(items: readonly T[], roll: number): T {
  return items[Math.floor(roll * items.length) % items.length] ?? items[0]!;
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function findWalker(state: Pick<GameState, "walkers">, priest: PriestState): WalkerInstance | undefined {
  return state.walkers.find((walker) => walker.id === priest.walkerId);
}

function rankedPriestCharacters(state: Pick<GameState, "characters">): NamedCharacterState[] {
  return [...(state.characters?.roster ?? [])]
    .filter((character) => character.role === "priest")
    .sort((left, right) => right.prominence - left.prominence || right.influence - left.influence || left.id.localeCompare(right.id));
}

function rankedPhilosopherCharacters(state: Pick<GameState, "characters">): NamedCharacterState[] {
  return [...(state.characters?.roster ?? [])]
    .filter((character) => character.role === "philosopher")
    .sort((left, right) => right.influence - left.influence || right.prominence - left.prominence || left.id.localeCompare(right.id));
}

function preferredFactionId(
  state: Pick<GameState, "factions" | "worldSeed">,
  priest: PriestState,
  anchor: NamedCharacterState | undefined
): FactionId | undefined {
  if (anchor?.homeFactionId) {
    return anchor.homeFactionId;
  }

  const ranked = Object.values(state.factions)
    .map((faction) => ({
      id: faction.id,
      score: (faction.currentAgenda === "faith" ? 35 : 0)
        + (faction.profile === "devout" ? 18 : 0)
        + faction.favour * 0.45
        + faction.credibility * 0.2
        - faction.debt * 0.18
        + hash(state.worldSeed, `${priest.id}:${faction.id}`) * 4
    }))
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id));

  return ranked[0]?.id;
}

function describeProfile(
  walkerName: string,
  temperament: PriestTemperament,
  ambition: PriestAmbition,
  stance: PriestPoliticalStance,
  favoredFaction: FactionState | undefined,
  anchor: NamedCharacterState | undefined
): string {
  const factionText = favoredFaction ? `leans toward ${favoredFaction.name}` : "keeps factional ties hidden";
  const anchorText = anchor ? `and measures every rite against ${anchor.displayName}` : "and takes cues from the older temple houses";

  return `${walkerName} is ${temperament}, acts as a ${ambition}, ${factionText}, ${anchorText}. ${stance} instincts dominate when pressure rises.`;
}

function createPriestProfile(
  state: Pick<GameState, "worldSeed" | "factions" | "pythia" | "characters" | "walkers">,
  priest: PriestState,
  slot: number,
  day: number
): PriestPoliticalProfile {
  const walker = findWalker(state, priest);
  const priestCharacters = rankedPriestCharacters(state);
  const anchor = priestCharacters[slot] ?? priestCharacters[0];
  const seedKey = `${priest.id}:${walker?.name ?? priest.walkerId}:${day}`;
  const temperament = pickOne<PriestTemperament>(
    ["steady", "zealous", "cunning", "scholarly"],
    hash(state.worldSeed + slot * 17, `${seedKey}:temperament`)
  );
  const ambition = pickOne<PriestAmbition>(
    ["guardian", "broker", "reformer", "mystic"],
    hash(state.worldSeed + slot * 29, `${seedKey}:ambition`)
  );
  const favoredFactionId = preferredFactionId(state, priest, anchor);
  const favoredFaction = favoredFactionId ? state.factions[favoredFactionId] : undefined;
  const loyalty = clamp(
    Math.round(
      priest.morale * 0.58
      + state.pythia.attunement * 0.18
      + state.pythia.tranceDepth * 0.1
      + (anchor?.relationship.trust ?? 28) * 0.2
      - (ambition === "reformer" ? 8 : 0)
      + (temperament === "steady" ? 5 : temperament === "zealous" ? 2 : 0)
    ),
    0,
    100
  );
  const dissent = clamp(
    Math.round(
      (100 - priest.morale) * 0.36
      + state.pythia.needs.purification * 0.18
      + state.pythia.needs.rest * 0.12
      + (anchor?.relationship.hostility ?? 12) * 0.15
      + (ambition === "reformer" ? 14 : ambition === "broker" ? 6 : 0)
      + (temperament === "cunning" ? 6 : temperament === "scholarly" ? 4 : 0)
      + hash(state.worldSeed + day, `${priest.id}:dissent`) * 8
    ),
    0,
    100
  );
  const influence = clamp(
    Math.round(
      priest.skill * 0.52
      + priest.morale * 0.22
      + (anchor?.influence ?? 42) * 0.24
      + (favoredFaction?.credibility ?? 50) * 0.08
    ),
    18,
    98
  );
  const stance: PriestPoliticalStance = loyalty - dissent >= 18
    ? "loyalist"
    : temperament === "zealous"
      ? "traditionalist"
      : ambition === "reformer" || dissent >= 52
        ? "reformer"
        : "broker";

  return {
    priestId: priest.id,
    temperament,
    ambition,
    stance,
    influence,
    loyalty,
    dissent,
    favoredFactionId,
    anchorCharacterId: anchor?.id,
    note: describeProfile(walker?.name ?? priest.id, temperament, ambition, stance, favoredFaction, anchor)
  };
}

function factionConflictPressure(factions: Record<FactionId, FactionState>): number {
  const pressures = Object.values(factions).map((faction) =>
    faction.activeConflicts.length * 12
    + faction.embargoes.length * 8
    + Math.max(0, faction.debt - 35) * 0.35
    + (faction.currentAgenda === "faith" ? 6 : 0)
  );

  return average(pressures);
}

function missingAssignments(state: Pick<GameState, "buildings">): number {
  return state.buildings.filter((building) => building.requiresPriest && building.assignedPriestIds.length === 0).length;
}

function createBlocStates(
  state: Pick<GameState, "factions" | "pythia" | "characters" | "buildings">,
  profiles: PriestPoliticalProfile[],
  day: number
): PriestCouncilBlocState[] {
  const loyalties = profiles.map((profile) => profile.loyalty);
  const dissents = profiles.map((profile) => profile.dissent);
  const influences = profiles.map((profile) => profile.influence);
  const patrons = profiles
    .map((profile) => profile.favoredFactionId ? state.factions[profile.favoredFactionId] : undefined)
    .filter((faction): faction is FactionState => Boolean(faction));
  const philosophers = rankedPhilosopherCharacters(state);
  const conflictPressure = factionConflictPressure(state.factions);
  const uncoveredRites = missingAssignments(state);
  const traditionalistCount = profiles.filter((profile) => profile.stance === "traditionalist").length;
  const reformerCount = profiles.filter((profile) => profile.stance === "reformer").length;
  const brokerCount = profiles.filter((profile) => profile.stance === "broker").length;

  const blocs: PriestCouncilBlocState[] = [
    {
      id: "pythia",
      label: "Pythia Loyalists",
      support: clamp(Math.round(average(loyalties) * 0.72 + state.pythia.attunement * 0.28), 12, 100),
      tension: clamp(Math.round(state.pythia.needs.rest * 0.34 + state.pythia.needs.purification * 0.42 + conflictPressure * 0.12), 0, 100),
      note: "They want the college closed around the Pythia until the air around Delphi steadies."
    },
    {
      id: "rites",
      label: "Ritual Traditionalists",
      support: clamp(Math.round(average(influences) * 0.22 + traditionalistCount * 18 + state.pythia.tranceDepth * 0.24), 10, 100),
      tension: clamp(Math.round(uncoveredRites * 16 + state.pythia.needs.purification * 0.28 + hash(day, "rites") * 10), 0, 100),
      note: "They insist every rite, offering, and purification cycle be restored to stricter form."
    },
    {
      id: "patrons",
      label: "Patron Brokers",
      support: clamp(Math.round(average(patrons.map((faction) => faction.favour)) * 0.46 + brokerCount * 15 + average(influences) * 0.18), 8, 100),
      tension: clamp(Math.round(average(patrons.map((faction) => faction.debt)) * 0.34 + conflictPressure * 0.24 + uncoveredRites * 8), 0, 100),
      note: "They keep one eye on foreign gold and another on which house gets access to Apollo first."
    },
    {
      id: "reformers",
      label: "Reform Voices",
      support: clamp(Math.round(average(dissents) * 0.44 + reformerCount * 18 + average(philosophers.map((character) => character.influence)) * 0.16), 6, 100),
      tension: clamp(Math.round(average(dissents) * 0.52 + average(philosophers.map((character) => character.prominence)) * 0.18 + hash(day, "reformers") * 12), 0, 100),
      note: "They argue the temple must adapt before outside schools redefine Delphi for them."
    }
  ];

  return blocs;
}

function politicsStatus(overallPressure: number): PriestPoliticsState["status"] {
  if (overallPressure >= 74) {
    return "crisis";
  }
  if (overallPressure >= 58) {
    return "fractured";
  }
  if (overallPressure >= 38) {
    return "restless";
  }
  return "calm";
}

function dominantBlocId(blocs: PriestCouncilBlocState[]): PriestCouncilBlocId {
  return [...blocs]
    .sort((left, right) => (right.support + right.tension * 0.6) - (left.support + left.tension * 0.6) || left.id.localeCompare(right.id))[0]?.id
    ?? "pythia";
}

function currentIssue(
  state: Pick<GameState, "buildings" | "characters">,
  dominantBloc: PriestCouncilBlocState | undefined,
  status: PriestPoliticsState["status"]
): string {
  const uncovered = missingAssignments(state);
  const anchor = rankedPriestCharacters(state)[0];

  if (uncovered > 0) {
    return uncovered === 1
      ? "One active rite stands without a resident priest, and everyone can feel the gap."
      : `${uncovered} ritual sites are uncovered, which has become the loudest accusation in the chamber.`;
  }
  if (dominantBloc?.id === "patrons") {
    return "Outside patrons are pushing gifts and obligations into the same conversation.";
  }
  if (dominantBloc?.id === "reformers") {
    return "The younger voices want new discipline before the philosophers seize the narrative.";
  }
  if (dominantBloc?.id === "rites") {
    return "Arguments over purification and rite order are slowing consensus inside the precinct.";
  }
  if (status === "calm") {
    return "The college is outwardly composed, but every ceremony is still being quietly scored.";
  }
  return anchor
    ? `${anchor.displayName} has become the measure each faction invokes when temple discipline is discussed.`
    : "Temple discipline is holding, but no one trusts the quiet to last.";
}

function rumor(state: Pick<GameState, "characters" | "factions">, dominantBloc: PriestCouncilBlocState | undefined): string {
  const anchor = rankedPriestCharacters(state)[0];
  const faction = anchor?.homeFactionId ? state.factions[anchor.homeFactionId] : undefined;

  if (anchor && faction && dominantBloc?.id === "patrons") {
    return `${anchor.displayName} is rumored to be taking questions from ${faction.name} before they ever reach the inner court.`;
  }
  if (anchor && dominantBloc?.id === "reformers") {
    return `${anchor.displayName} is said to be reading philosopher tracts by lamplight after the evening fire is banked.`;
  }
  if (anchor && dominantBloc?.id === "rites") {
    return `${anchor.displayName} has begun counting every missed purification turn and repeating the number aloud.`;
  }
  if (anchor) {
    return `${anchor.displayName} is being named in corridor whispers as the figure who could steady or split the college.`;
  }
  return "The temple servants say the incense smoke has started carrying gossip farther than prayer.";
}

function buildPriestPoliticsState(
  state: Pick<GameState, "worldSeed" | "clock" | "factions" | "priests" | "pythia" | "characters" | "walkers" | "buildings">,
  day: number
): PriestPoliticsState {
  const profiles = state.priests.map((priest, index) => createPriestProfile(state, priest, index, day));
  const blocs = createBlocStates(state, profiles, day);
  const dominantId = dominantBlocId(blocs);
  const dominantBloc = blocs.find((bloc) => bloc.id === dominantId);
  const overallPressure = clamp(
    Math.round(
      average(blocs.map((bloc) => bloc.tension)) * 0.42
      + average(profiles.map((profile) => profile.dissent)) * 0.26
      + factionConflictPressure(state.factions) * 0.16
      + state.pythia.needs.rest * 0.12
      + state.pythia.needs.purification * 0.18
      + missingAssignments(state) * 14
      + hash(state.worldSeed + day, "overall-pressure") * 8
    ),
    0,
    100
  );
  const unity = clamp(
    Math.round(
      average(profiles.map((profile) => profile.loyalty)) * 0.58
      + average(blocs.map((bloc) => bloc.support)) * 0.18
      + state.pythia.attunement * 0.14
      - overallPressure * 0.22
    ),
    0,
    100
  );
  const status = politicsStatus(overallPressure);

  return {
    overallPressure,
    unity,
    dominantBlocId: dominantId,
    status,
    currentIssue: currentIssue(state, dominantBloc, status),
    rumor: rumor(state, dominantBloc),
    featuredCharacterIds: rankedPriestCharacters(state).slice(0, 2).map((character) => character.id),
    lastUpdatedDay: day,
    blocs,
    priests: Object.fromEntries(profiles.map((profile) => [profile.priestId, profile]))
  };
}

export function createInitialPriestPoliticsState(
  state: Pick<GameState, "worldSeed" | "clock" | "factions" | "priests" | "pythia" | "characters" | "walkers" | "buildings">
): PriestPoliticsState {
  return buildPriestPoliticsState(state, state.clock.day);
}

export function normalizePriestPoliticsState(
  state: Pick<GameState, "worldSeed" | "clock" | "factions" | "priests" | "pythia" | "characters" | "walkers" | "buildings" | "priestPolitics">
): PriestPoliticsState {
  const fallback = createInitialPriestPoliticsState(state);
  if (!state.priestPolitics) {
    return fallback;
  }

  const priests = Object.fromEntries(
    state.priests.map((priest) => {
      const existing = state.priestPolitics?.priests?.[priest.id];
      return [
        priest.id,
        existing
          ? {
              ...fallback.priests[priest.id],
              ...existing
            }
          : fallback.priests[priest.id]
      ];
    })
  ) as PriestPoliticsState["priests"];
  const blocs = fallback.blocs.map((bloc) => {
    const existing = state.priestPolitics?.blocs?.find((entry) => entry.id === bloc.id);
    return existing ? { ...bloc, ...existing } : bloc;
  });

  return {
    ...fallback,
    ...state.priestPolitics,
    featuredCharacterIds: state.priestPolitics.featuredCharacterIds?.filter((characterId) =>
      (state.characters?.roster ?? []).some((character) => character.id === characterId)
    ) ?? fallback.featuredCharacterIds,
    blocs,
    priests
  };
}

export function advancePriestPoliticsState(
  state: Pick<GameState, "worldSeed" | "clock" | "factions" | "priests" | "pythia" | "characters" | "walkers" | "buildings" | "priestPolitics">
): PriestPoliticsState {
  const computed = buildPriestPoliticsState(state, state.clock.day);
  const previous = state.priestPolitics ? normalizePriestPoliticsState(state) : undefined;

  if (!previous || previous.lastUpdatedDay === state.clock.day) {
    return computed;
  }

  const priests = Object.fromEntries(
    state.priests.map((priest) => {
      const last = previous.priests[priest.id];
      const next = computed.priests[priest.id];
      return [
        priest.id,
        {
          ...next,
          influence: clamp(Math.round((last?.influence ?? next.influence) * 0.4 + next.influence * 0.6), 0, 100),
          loyalty: clamp(Math.round((last?.loyalty ?? next.loyalty) * 0.35 + next.loyalty * 0.65), 0, 100),
          dissent: clamp(Math.round((last?.dissent ?? next.dissent) * 0.35 + next.dissent * 0.65), 0, 100)
        }
      ];
    })
  ) as PriestPoliticsState["priests"];
  const blocs = computed.blocs.map((bloc) => {
    const last = previous.blocs.find((entry) => entry.id === bloc.id);
    return {
      ...bloc,
      support: clamp(Math.round((last?.support ?? bloc.support) * 0.38 + bloc.support * 0.62), 0, 100),
      tension: clamp(Math.round((last?.tension ?? bloc.tension) * 0.28 + bloc.tension * 0.72), 0, 100)
    };
  });
  const dominantId = dominantBlocId(blocs);
  const dominantBloc = blocs.find((bloc) => bloc.id === dominantId);
  const overallPressure = clamp(Math.round(previous.overallPressure * 0.32 + computed.overallPressure * 0.68), 0, 100);
  const unity = clamp(Math.round(previous.unity * 0.28 + computed.unity * 0.72), 0, 100);
  const status = politicsStatus(overallPressure);

  return {
    ...computed,
    overallPressure,
    unity,
    dominantBlocId: dominantId,
    status,
    currentIssue: currentIssue(state, dominantBloc, status),
    rumor: rumor(state, dominantBloc),
    lastUpdatedDay: state.clock.day,
    blocs,
    priests
  };
}
