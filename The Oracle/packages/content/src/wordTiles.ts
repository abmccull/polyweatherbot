import type { WordTileDef } from "./schema";

export const wordTiles: WordTileDef[] = [
  {
    id: "subject-army",
    text: "the army",
    category: "subject",
    semantics: {
      target: "army",
      action: "triumph",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    }
  },
  {
    id: "subject-fleet",
    text: "the fleet",
    category: "subject",
    semantics: {
      target: "fleet",
      action: "triumph",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    }
  },
  {
    id: "subject-king",
    text: "the king",
    category: "subject",
    semantics: {
      target: "king",
      action: "endure",
      polarity: "double",
      ambiguity: "specific",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "subject-city",
    text: "the city",
    category: "subject",
    semantics: {
      target: "city",
      action: "endure",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "subject-oracle",
    text: "the oracle",
    category: "subject",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "favorable",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "subject-harvest",
    text: "the harvest",
    category: "subject",
    semantics: {
      target: "harvest",
      action: "prosper",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "subject-treasury",
    text: "the treasury",
    category: "subject",
    semantics: {
      target: "treasury",
      action: "prosper",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "subject-alliance",
    text: "the alliance",
    category: "subject",
    semantics: {
      target: "alliance",
      action: "endure",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "action-triumph",
    text: "shall triumph",
    category: "action",
    semantics: {
      target: "army",
      action: "triumph",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    }
  },
  {
    id: "action-fall",
    text: "shall fall",
    category: "action",
    semantics: {
      target: "city",
      action: "fall",
      polarity: "warning",
      ambiguity: "specific",
      timeHorizon: "seasonal",
      domain: "military"
    }
  },
  {
    id: "action-endure",
    text: "shall endure",
    category: "action",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "action-fracture",
    text: "shall fracture",
    category: "action",
    semantics: {
      target: "alliance",
      action: "fracture",
      polarity: "warning",
      ambiguity: "balanced",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "action-prosper",
    text: "shall prosper",
    category: "action",
    semantics: {
      target: "harvest",
      action: "prosper",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "action-withhold",
    text: "shall withhold",
    category: "action",
    semantics: {
      target: "treasury",
      action: "withhold",
      polarity: "warning",
      ambiguity: "specific",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "action-hold-shore",
    text: "shall hold the shore",
    category: "action",
    semantics: {
      target: "fleet",
      action: "endure",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    }
  },
  {
    id: "action-answer-silence",
    text: "shall answer with silence",
    category: "action",
    semantics: {
      target: "oracle",
      action: "withhold",
      polarity: "double",
      ambiguity: "cryptic",
      timeHorizon: "immediate",
      domain: "spiritual"
    }
  },
  {
    id: "condition-harvest",
    text: "before the next harvest",
    category: "condition",
    semantics: {
      target: "harvest",
      action: "prosper",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "condition-river",
    text: "if the river is crossed",
    category: "condition",
    semantics: {
      target: "army",
      action: "fall",
      polarity: "warning",
      ambiguity: "specific",
      timeHorizon: "immediate",
      domain: "military"
    }
  },
  {
    id: "condition-altars",
    text: "when the altars smoke",
    category: "condition",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "double",
      ambiguity: "cryptic",
      timeHorizon: "immediate",
      domain: "spiritual"
    }
  },
  {
    id: "condition-sun",
    text: "when the sun stands high",
    category: "condition",
    semantics: {
      target: "city",
      action: "prosper",
      polarity: "favorable",
      ambiguity: "balanced",
      timeHorizon: "immediate",
      domain: "economic"
    }
  },
  {
    id: "condition-new-moon",
    text: "at the next new moon",
    category: "condition",
    semantics: {
      target: "king",
      action: "endure",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "immediate",
      domain: "spiritual"
    }
  },
  {
    id: "condition-dog-star",
    text: "under the Dog Star",
    category: "condition",
    semantics: {
      target: "harvest",
      action: "withhold",
      polarity: "warning",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "condition-third-watch",
    text: "in the third watch",
    category: "condition",
    semantics: {
      target: "fleet",
      action: "triumph",
      polarity: "favorable",
      ambiguity: "cryptic",
      timeHorizon: "immediate",
      domain: "military"
    }
  },
  {
    id: "condition-envoy-return",
    text: "after the envoy returns",
    category: "condition",
    semantics: {
      target: "alliance",
      action: "endure",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "spiritual"
    }
  },
  {
    id: "modifier-shadow",
    text: "in shadow",
    category: "modifier",
    semantics: {
      target: "alliance",
      action: "fracture",
      polarity: "double",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "modifier-bronze",
    text: "at a cost in bronze",
    category: "modifier",
    semantics: {
      target: "treasury",
      action: "withhold",
      polarity: "warning",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "modifier-sea",
    text: "by the wine-dark sea",
    category: "modifier",
    semantics: {
      target: "fleet",
      action: "triumph",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "military"
    }
  },
  {
    id: "modifier-laurel",
    text: "beneath the laurel crown",
    category: "modifier",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "favorable",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "modifier-stranger-hands",
    text: "through stranger hands",
    category: "modifier",
    semantics: {
      target: "alliance",
      action: "fracture",
      polarity: "warning",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "modifier-red-moon",
    text: "beneath a blood-red moon",
    category: "modifier",
    semantics: {
      target: "king",
      action: "fall",
      polarity: "warning",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "modifier-hidden-tribute",
    text: "with hidden tribute",
    category: "modifier",
    semantics: {
      target: "treasury",
      action: "prosper",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "modifier-smoke",
    text: "through sacred smoke",
    category: "modifier",
    semantics: {
      target: "oracle",
      action: "withhold",
      polarity: "double",
      ambiguity: "cryptic",
      timeHorizon: "immediate",
      domain: "spiritual"
    }
  },
  {
    id: "modifier-crosswinds",
    text: "against crosswinds",
    category: "modifier",
    semantics: {
      target: "fleet",
      action: "fall",
      polarity: "warning",
      ambiguity: "balanced",
      timeHorizon: "immediate",
      domain: "military"
    }
  },
  {
    id: "seal-god",
    text: "thus speaks the god",
    category: "seal",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "favorable",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "seal-phoebus",
    text: "so shines Phoebus",
    category: "seal",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "favorable",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "seal-measure",
    text: "Apollo keeps the measure",
    category: "seal",
    semantics: {
      target: "king",
      action: "endure",
      polarity: "double",
      ambiguity: "balanced",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "seal-hour",
    text: "let no mortal force the hour",
    category: "seal",
    semantics: {
      target: "city",
      action: "withhold",
      polarity: "warning",
      ambiguity: "balanced",
      timeHorizon: "seasonal",
      domain: "economic"
    }
  },
  {
    id: "seal-tripod",
    text: "the tripod remembers",
    category: "seal",
    semantics: {
      target: "oracle",
      action: "endure",
      polarity: "double",
      ambiguity: "cryptic",
      timeHorizon: "yearly",
      domain: "spiritual"
    }
  },
  {
    id: "seal-ridge",
    text: "the god sees beyond the ridge",
    category: "seal",
    semantics: {
      target: "army",
      action: "triumph",
      polarity: "favorable",
      ambiguity: "cryptic",
      timeHorizon: "seasonal",
      domain: "military"
    }
  }
];
