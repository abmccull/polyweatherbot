import type { PhilosopherId, PhilosopherThreatDef } from "./schema";

export const philosopherThreatDefs: PhilosopherThreatDef[] = [
  {
    id: "gorgias",
    name: "Gorgias",
    school: "sophist",
    doctrine: "persuasion outruns inherited authority",
    domain: "economic",
    favoredAgenda: "trade",
    preferredProfiles: ["mercantile", "scheming"],
    preferredWorldviews: ["Skeptical courts", "Mercantile pragmatism", "Festival populism"],
    consultationTags: ["legitimacy", "court", "trade", "civic-order"],
    stageThresholds: {
      circle: 44,
      sect: 62,
      crisis: 80
    },
    pressureBias: {
      debt: 0.46,
      dependence: 0.22,
      lowCredibility: 0.28,
      conflicts: 3,
      embargoes: 4,
      agendaAlignment: 7,
      worldviewAlignment: 8
    },
    stageEffects: {
      rumor: { credibility: -1, favour: 0, debt: 0, dependence: 1 },
      circle: { credibility: -2, favour: -1, debt: 1, dependence: 2 },
      sect: { credibility: -4, favour: -2, debt: 2, dependence: 3 },
      crisis: { credibility: -6, favour: -4, debt: 3, dependence: 5 }
    }
  },
  {
    id: "heraclitus",
    name: "Heraclitus",
    school: "strife-doctrine",
    doctrine: "conflict is the hidden law of civic order",
    domain: "military",
    favoredAgenda: "war",
    preferredProfiles: ["martial", "scheming"],
    preferredWorldviews: ["Heroic cult", "Military austerity", "Ancestor law"],
    consultationTags: ["army", "fleet", "order", "strife"],
    stageThresholds: {
      circle: 46,
      sect: 64,
      crisis: 82
    },
    pressureBias: {
      debt: 0.34,
      dependence: 0.3,
      lowCredibility: 0.24,
      conflicts: 5,
      embargoes: 2,
      agendaAlignment: 8,
      worldviewAlignment: 7
    },
    stageEffects: {
      rumor: { credibility: -1, favour: 1, debt: 1, dependence: 1 },
      circle: { credibility: -2, favour: 0, debt: 2, dependence: 2 },
      sect: { credibility: -4, favour: -2, debt: 3, dependence: 3 },
      crisis: { credibility: -7, favour: -4, debt: 4, dependence: 4 }
    }
  },
  {
    id: "pythagoras",
    name: "Pythagoras",
    school: "mystic-order",
    doctrine: "purity, number, and hidden discipline should govern the city",
    domain: "spiritual",
    favoredAgenda: "faith",
    preferredProfiles: ["devout", "mercantile"],
    preferredWorldviews: ["Mystic reform", "Civic piety", "Ancestor law"],
    consultationTags: ["rites", "purification", "order", "soul"],
    stageThresholds: {
      circle: 43,
      sect: 61,
      crisis: 78
    },
    pressureBias: {
      debt: 0.28,
      dependence: 0.34,
      lowCredibility: 0.32,
      conflicts: 2,
      embargoes: 2,
      agendaAlignment: 6,
      worldviewAlignment: 9
    },
    stageEffects: {
      rumor: { credibility: 0, favour: 1, debt: 0, dependence: 1 },
      circle: { credibility: -1, favour: 0, debt: 0, dependence: 2 },
      sect: { credibility: -3, favour: -1, debt: 1, dependence: 3 },
      crisis: { credibility: -5, favour: -3, debt: 1, dependence: 5 }
    }
  },
  {
    id: "democritus",
    name: "Democritus",
    school: "atomist",
    doctrine: "material causes and appetite can explain what priests call fate",
    domain: "economic",
    favoredAgenda: "trade",
    preferredProfiles: ["mercantile", "martial"],
    preferredWorldviews: ["Mercantile pragmatism", "Skeptical courts", "Civic piety"],
    consultationTags: ["trade", "harvest", "treasury", "omens"],
    stageThresholds: {
      circle: 45,
      sect: 63,
      crisis: 81
    },
    pressureBias: {
      debt: 0.42,
      dependence: 0.2,
      lowCredibility: 0.3,
      conflicts: 3,
      embargoes: 3,
      agendaAlignment: 6,
      worldviewAlignment: 7
    },
    stageEffects: {
      rumor: { credibility: -1, favour: 0, debt: -1, dependence: 1 },
      circle: { credibility: -2, favour: -1, debt: 0, dependence: 2 },
      sect: { credibility: -4, favour: -2, debt: 1, dependence: 3 },
      crisis: { credibility: -6, favour: -4, debt: 2, dependence: 4 }
    }
  },
  {
    id: "anaxagoras",
    name: "Anaxagoras",
    school: "court-naturalist",
    doctrine: "mind and nature can order kingship without inherited priestly sanction",
    domain: "spiritual",
    favoredAgenda: "succession",
    preferredProfiles: ["scheming", "devout"],
    preferredWorldviews: ["Skeptical courts", "Civic piety", "Mystic reform"],
    consultationTags: ["king", "court", "legitimacy", "omens"],
    stageThresholds: {
      circle: 44,
      sect: 60,
      crisis: 77
    },
    pressureBias: {
      debt: 0.26,
      dependence: 0.38,
      lowCredibility: 0.4,
      conflicts: 3,
      embargoes: 2,
      agendaAlignment: 8,
      worldviewAlignment: 8
    },
    stageEffects: {
      rumor: { credibility: -1, favour: 0, debt: 0, dependence: 2 },
      circle: { credibility: -3, favour: -1, debt: 1, dependence: 3 },
      sect: { credibility: -5, favour: -2, debt: 1, dependence: 4 },
      crisis: { credibility: -7, favour: -4, debt: 2, dependence: 6 }
    }
  }
];

export const philosopherThreatDefById: Record<PhilosopherId, PhilosopherThreatDef> = Object.fromEntries(
  philosopherThreatDefs.map((definition) => [definition.id, definition])
) as Record<PhilosopherId, PhilosopherThreatDef>;
