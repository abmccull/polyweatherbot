export type AgeId = "archaic" | "classical" | "hellenic" | "hellenistic" | "roman_shadow";

export type AgeDef = {
  id: AgeId;
  name: string;
  description: string;
  yearThreshold: number;
  factionModifiers: { prominenceShift: Record<string, number> };
  consultationWeightMod: number;
  philosopherPrevalence: number;
  tradeActivityMod: number;
  rivalAggressionMod: number;
};

export const ageDefs: AgeDef[] = [
  {
    id: "archaic",
    name: "Archaic Age",
    description: "The oracle is young and the world is raw. Traditions are forming, philosophers are rare, and trade routes remain sparse.",
    yearThreshold: 0,
    factionModifiers: { prominenceShift: {} },
    consultationWeightMod: 1.0,
    philosopherPrevalence: 0.5,
    tradeActivityMod: 0.6,
    rivalAggressionMod: 0.7
  },
  {
    id: "classical",
    name: "Classical Age",
    description: "The great city-states rise in power. Philosophers wander the agora and trade caravans grow bolder.",
    yearThreshold: 3,
    factionModifiers: { prominenceShift: { athens: 5, sparta: 5 } },
    consultationWeightMod: 1.1,
    philosopherPrevalence: 1.0,
    tradeActivityMod: 1.0,
    rivalAggressionMod: 1.0
  },
  {
    id: "hellenic",
    name: "Hellenic Age",
    description: "Culture reaches its zenith. Oracles are sought by kings and generals alike. Philosophy and prophecy intertwine.",
    yearThreshold: 6,
    factionModifiers: { prominenceShift: { athens: 8, corinth: 5, miletus: 5 } },
    consultationWeightMod: 1.3,
    philosopherPrevalence: 1.4,
    tradeActivityMod: 1.3,
    rivalAggressionMod: 1.1
  },
  {
    id: "hellenistic",
    name: "Hellenistic Age",
    description: "Empires stretch across the known world. Cosmopolitan courts seek exotic prophecy, but rival oracles multiply.",
    yearThreshold: 10,
    factionModifiers: { prominenceShift: { macedon: 10, syracuse: 5 } },
    consultationWeightMod: 1.2,
    philosopherPrevalence: 1.2,
    tradeActivityMod: 1.5,
    rivalAggressionMod: 1.4
  },
  {
    id: "roman_shadow",
    name: "Roman Shadow",
    description: "A great power looms from the west. Autonomy wanes, consultation requests decline, and rivals grow desperate.",
    yearThreshold: 15,
    factionModifiers: { prominenceShift: { macedon: -5, sparta: -5 } },
    consultationWeightMod: 0.8,
    philosopherPrevalence: 0.9,
    tradeActivityMod: 1.2,
    rivalAggressionMod: 1.8
  }
];

export const ageDefById: Record<AgeId, AgeDef> = Object.fromEntries(
  ageDefs.map((age) => [age.id, age])
) as Record<AgeId, AgeDef>;
