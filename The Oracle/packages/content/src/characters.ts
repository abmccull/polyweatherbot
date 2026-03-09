import type { NamedCharacterArchetypeId, NamedCharacterDef } from "./schema";

export const namedCharacterDefs: NamedCharacterDef[] = [
  {
    id: "merchant-patron",
    role: "merchant",
    label: "Merchant Patron",
    cadence: "seasonal",
    minInitialCount: 1,
    maxInitialCount: 2,
    baseInfluence: 52,
    baseProminence: 48,
    titlePool: ["Merchant", "Broker", "Caravan-Master", "Shipmaster"],
    firstNamePool: ["Melitta", "Sostratos", "Nikarete", "Philon", "Damon", "Thyra"],
    epithetPool: ["of the Isthmus", "of Two Harbors", "Silver Scales", "of Cedar Ledgers", "the Saffron Convoy"],
    preferredAgendas: ["trade", "succession"],
    preferredProfiles: ["mercantile", "scheming"],
    preferredDomains: ["economic"],
    initialTags: ["trade", "patronage", "routes"]
  },
  {
    id: "campaign-general",
    role: "general",
    label: "Campaign General",
    cadence: "campaign",
    minInitialCount: 1,
    maxInitialCount: 1,
    baseInfluence: 58,
    baseProminence: 54,
    titlePool: ["General", "Strategos", "Shield-Captain", "Horse-Lord"],
    firstNamePool: ["Lykon", "Timandra", "Brasos", "Euphemos", "Myrine", "Kallias"],
    epithetPool: ["of the Bronze Spear", "the Hill-Breaker", "of Ashen Standards", "the Red Mantle", "of the Vanguard"],
    preferredAgendas: ["war", "succession"],
    preferredProfiles: ["martial", "scheming"],
    preferredDomains: ["military"],
    initialTags: ["campaign", "levies", "pressure"]
  },
  {
    id: "faction-envoy",
    role: "envoy",
    label: "Faction Envoy",
    cadence: "diplomatic",
    minInitialCount: 1,
    maxInitialCount: 2,
    baseInfluence: 50,
    baseProminence: 50,
    titlePool: ["Envoy", "Seal-Bearer", "Speaker", "Delegate"],
    firstNamePool: ["Iason", "Doriea", "Charon", "Phaedra", "Nikon", "Acantha"],
    epithetPool: ["of the Laurel Seal", "of Quiet Oaths", "the Silver Tongue", "of the Outer Court", "the Soft Sandal"],
    preferredAgendas: ["trade", "faith", "succession"],
    preferredProfiles: ["mercantile", "devout", "scheming"],
    preferredDomains: ["economic", "spiritual"],
    initialTags: ["diplomacy", "petitions", "court"]
  },
  {
    id: "wandering-philosopher",
    role: "philosopher",
    label: "Wandering Philosopher",
    cadence: "wandering",
    minInitialCount: 1,
    maxInitialCount: 1,
    baseInfluence: 47,
    baseProminence: 51,
    titlePool: ["Philosopher", "Teacher", "Dialectician", "Wanderer"],
    firstNamePool: ["Thespis", "Axiothea", "Menon", "Leontis", "Diokles", "Perialla"],
    epithetPool: ["of the Open Portico", "the Questioner", "of Hollow Stars", "the Lamp-Bearer", "of Seven Knots"],
    preferredAgendas: ["faith", "succession", "trade"],
    preferredProfiles: ["devout", "scheming", "mercantile"],
    preferredDomains: ["spiritual", "economic"],
    initialTags: ["argument", "schools", "omens"]
  },
  {
    id: "delphic-priest",
    role: "priest",
    label: "Delphic Priest",
    cadence: "ritual",
    minInitialCount: 1,
    maxInitialCount: 2,
    baseInfluence: 55,
    baseProminence: 57,
    titlePool: ["Priest", "Rite-Keeper", "Laurel Warden", "Fire-Keeper"],
    firstNamePool: ["Ione", "Timon", "Kleio", "Aletes", "Rhodope", "Nestor"],
    epithetPool: ["of the Sacred Way", "of the Laurel Court", "the Clean-Handed", "of the Inner Fire", "the Spring Listener"],
    preferredAgendas: ["faith"],
    preferredProfiles: ["devout"],
    preferredDomains: ["spiritual"],
    initialTags: ["rites", "precinct", "purification"]
  },
  {
    id: "legendary-visitor",
    role: "legendary",
    label: "Legendary Visitor",
    cadence: "legendary",
    minInitialCount: 0,
    maxInitialCount: 1,
    baseInfluence: 76,
    baseProminence: 80,
    titlePool: ["Stranger", "Wonder-Guest", "Lion of No City", "Veiled Hero"],
    firstNamePool: ["Orsen", "Praxilla", "Medeon", "Thyestes"],
    epithetPool: ["the God-Touched", "of the Ninth Fire", "the Storm-Witness", "of No Recorded House"],
    preferredDomains: ["military", "economic", "spiritual"],
    initialTags: ["legend", "omens", "disturbance"]
  }
];

export const namedCharacterDefById: Record<NamedCharacterArchetypeId, NamedCharacterDef> = Object.fromEntries(
  namedCharacterDefs.map((definition) => [definition.id, definition])
) as Record<NamedCharacterArchetypeId, NamedCharacterDef>;
