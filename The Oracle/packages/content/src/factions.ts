import type { FactionDef } from "./schema";

export const factionDefs: FactionDef[] = [
  { id: "athens", name: "Athens", defaultAgenda: "war", tradeAccess: true, profile: "martial", favoredResource: "grain" },
  { id: "sparta", name: "Sparta", defaultAgenda: "war", tradeAccess: true, profile: "martial", favoredResource: "olive_oil" },
  { id: "corinth", name: "Corinth", defaultAgenda: "trade", tradeAccess: true, profile: "mercantile", favoredResource: "incense" },
  { id: "thebes", name: "Thebes", defaultAgenda: "faith", tradeAccess: true, profile: "devout", favoredResource: "incense" },
  { id: "argos", name: "Argos", defaultAgenda: "trade", tradeAccess: false, profile: "mercantile", favoredResource: "grain" },
  { id: "miletus", name: "Miletus", defaultAgenda: "trade", tradeAccess: true, profile: "mercantile", favoredResource: "sacred_water" },
  { id: "syracuse", name: "Syracuse", defaultAgenda: "war", tradeAccess: true, profile: "martial", favoredResource: "grain" },
  { id: "macedon", name: "Macedon", defaultAgenda: "succession", tradeAccess: false, profile: "scheming", favoredResource: "olive_oil" }
];

const factionRelations = {
  athens: { sparta: -65, corinth: -15, thebes: 8, argos: 14, miletus: 28, syracuse: -10, macedon: -22 },
  sparta: { athens: -65, corinth: 10, thebes: -18, argos: 18, miletus: -6, syracuse: 12, macedon: -14 },
  corinth: { athens: -15, sparta: 10, thebes: 6, argos: 22, miletus: 30, syracuse: 24, macedon: 4 },
  thebes: { athens: 8, sparta: -18, corinth: 6, argos: 10, miletus: 14, syracuse: 2, macedon: -12 },
  argos: { athens: 14, sparta: 18, corinth: 22, thebes: 10, miletus: 12, syracuse: 6, macedon: -4 },
  miletus: { athens: 28, sparta: -6, corinth: 30, thebes: 14, argos: 12, syracuse: 16, macedon: 2 },
  syracuse: { athens: -10, sparta: 12, corinth: 24, thebes: 2, argos: 6, miletus: 16, macedon: 6 },
  macedon: { athens: -22, sparta: -14, corinth: 4, thebes: -12, argos: -4, miletus: 2, syracuse: 6 }
} as const;

export const defaultFactionStates = factionDefs.map((faction, index) => ({
  id: faction.id,
  name: faction.name,
  profile: faction.profile,
  favoredResource: faction.favoredResource,
  relations: { ...(factionRelations[faction.id] ?? {}) },
  treaties: [],
  embargoes: [],
  credibility: 48 + index * 3,
  favour: 52 - index,
  dependence: 24 + index * 2,
  debt: 0,
  currentAgenda: faction.defaultAgenda,
  activeConflicts: faction.id === "athens" ? ["sparta"] : faction.id === "sparta" ? ["athens"] : [],
  tradeAccess: faction.tradeAccess,
  history: []
}));
