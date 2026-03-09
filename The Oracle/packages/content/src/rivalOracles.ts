import type { RivalOracleDef, RivalOracleOperationDef } from "./schema";

export const rivalOracleOperationDefs: RivalOracleOperationDef[] = [
  {
    id: "poach-supplicants",
    label: "Poach Supplicants",
    kind: "pressure",
    domain: "economic",
    summary: "Draws pilgrims, gifts, and rumor away from Delphi through aggressive hospitality and timed promises.",
    basePressure: 8,
    baseVisibility: 16,
    baseIntel: 12,
    patronageShift: 3
  },
  {
    id: "plant-whispers",
    label: "Plant Whispers",
    kind: "espionage",
    domain: "spiritual",
    summary: "Seeds half-true rumors through caravan agents, scribes, and temple attendants.",
    basePressure: 6,
    baseVisibility: 7,
    baseIntel: 4,
    patronageShift: 4
  },
  {
    id: "court-patron",
    label: "Court Patron",
    kind: "patronage",
    domain: "economic",
    summary: "Secures fresh backing from a city-state patron and turns that purse into sustained leverage.",
    basePressure: 7,
    baseVisibility: 10,
    baseIntel: 8,
    patronageShift: 7
  },
  {
    id: "counter-rite",
    label: "Stage Counter-Rite",
    kind: "pressure",
    domain: "spiritual",
    summary: "Publicly answers Delphi with a competing rite, omen, or interpretation designed to steal authority.",
    basePressure: 12,
    baseVisibility: 18,
    baseIntel: 14,
    patronageShift: 2
  }
];

export const rivalOracleOperationDefById: Record<RivalOracleOperationDef["id"], RivalOracleOperationDef> = Object.fromEntries(
  rivalOracleOperationDefs.map((definition) => [definition.id, definition])
) as Record<RivalOracleOperationDef["id"], RivalOracleOperationDef>;

export const rivalOracleDefs: RivalOracleDef[] = [
  {
    id: "oak-seers",
    name: "Oak Seers",
    title: "Brazen Oaks of Thebes",
    summary: "A doctrinal shrine network that weaponizes piety, lineage, and carefully staged omens.",
    homeRegionId: "thebes",
    favoredDomain: "spiritual",
    baselinePressure: 30,
    secrecy: 64,
    pressureCap: 84,
    patronPool: ["thebes", "macedon", "athens"],
    preferredPatronProfiles: ["devout", "scheming"],
    preferredPatronAgendas: ["faith", "succession"],
    operationIds: ["counter-rite", "plant-whispers", "court-patron"]
  },
  {
    id: "isthmus-ledger",
    name: "Isthmian Ledger-House",
    title: "Ledger-House of the Isthmus",
    summary: "A commercial oracle that treats prophecy as a market edge and pilgrim flow as a balance sheet.",
    homeRegionId: "corinth",
    favoredDomain: "economic",
    baselinePressure: 33,
    secrecy: 56,
    pressureCap: 88,
    patronPool: ["corinth", "miletus", "argos"],
    preferredPatronProfiles: ["mercantile", "scheming"],
    preferredPatronAgendas: ["trade", "succession"],
    operationIds: ["poach-supplicants", "court-patron", "plant-whispers"]
  },
  {
    id: "wolf-cup",
    name: "Wolf-Cup Prophets",
    title: "Wolf-Cup of Sparta",
    summary: "An austere martial oracle that promises cleaner answers for commanders, scouts, and governors under strain.",
    homeRegionId: "sparta",
    favoredDomain: "military",
    baselinePressure: 31,
    secrecy: 49,
    pressureCap: 86,
    patronPool: ["sparta", "syracuse", "athens"],
    preferredPatronProfiles: ["martial", "scheming"],
    preferredPatronAgendas: ["war", "succession"],
    operationIds: ["counter-rite", "poach-supplicants", "plant-whispers"]
  },
  {
    id: "mist-sibyls",
    name: "Mist Sibyls",
    title: "Mist Sibyls of Miletus",
    summary: "A veiled harbor cult that traffics in soft intelligence, patronage brokerage, and deniable sacred rumor.",
    homeRegionId: "miletus",
    favoredDomain: "spiritual",
    baselinePressure: 28,
    secrecy: 70,
    pressureCap: 82,
    patronPool: ["miletus", "athens", "corinth"],
    preferredPatronProfiles: ["mercantile", "scheming"],
    preferredPatronAgendas: ["trade", "faith"],
    operationIds: ["plant-whispers", "court-patron", "poach-supplicants"]
  }
];

export const rivalOracleDefById: Record<RivalOracleDef["id"], RivalOracleDef> = Object.fromEntries(
  rivalOracleDefs.map((definition) => [definition.id, definition])
) as Record<RivalOracleDef["id"], RivalOracleDef>;
