export type LegendaryConsultationId =
  | "croesus"
  | "themistocles"
  | "spartan-king"
  | "philip-of-macedon"
  | "alexander"
  | "roman-general";

export type LegendaryStage = {
  id: string;
  prompt: string;
  hint: string;
  domainFocus: string;
  depthRequirement?: string;
};

export type LegendaryReward = {
  credibilityBonus: number;
  goldBonus: number;
  reputationBonus: number;
  specialEffect?: string;
};

export type LegendaryConsultationDef = {
  id: LegendaryConsultationId;
  name: string;
  figure: string;
  description: string;
  minYear: number;
  requiredAgeId?: string;
  stages: LegendaryStage[];
  reward: LegendaryReward;
};

export const legendaryConsultationDefs: Record<LegendaryConsultationId, LegendaryConsultationDef> = {
  croesus: {
    id: "croesus",
    name: "The Lydian Question",
    figure: "Croesus of Lydia",
    description:
      "The fabulously wealthy King of Lydia arrives at your sanctuary, flanked by gift-laden servants. He asks whether he should wage war against the growing Persian Empire.",
    minYear: 2,
    stages: [
      {
        id: "croesus-1",
        prompt:
          "Croesus lays golden offerings before the tripod and asks: 'If I march against Persia, what fate awaits my kingdom?'",
        hint: "A double-edged prophecy is historically appropriate. Speak of a great empire falling — but whose?",
        domainFocus: "military"
      },
      {
        id: "croesus-2",
        prompt:
          "The king presses further: 'Will the gods protect Lydia if I cross the Halys River?'",
        hint: "The river crossing is the point of no return. Ambiguity serves the oracle's credibility regardless of outcome.",
        domainFocus: "military",
        depthRequirement: "grounded"
      }
    ],
    reward: {
      credibilityBonus: 18,
      goldBonus: 200,
      reputationBonus: 12
    }
  },

  themistocles: {
    id: "themistocles",
    name: "The Wooden Wall",
    figure: "Themistocles of Athens",
    description:
      "The Athenian strategos arrives in haste. Xerxes' vast army approaches Greece, and Athens must decide whether to fight on land or trust the fleet.",
    minYear: 4,
    stages: [
      {
        id: "themistocles-1",
        prompt:
          "Themistocles speaks urgently: 'The Great King comes with a million spears. How shall Athens survive the Persian flood?'",
        hint: "The famous prophecy of the 'wooden wall' pointed to the Athenian fleet. Guide toward the sea without naming it directly.",
        domainFocus: "military"
      },
      {
        id: "themistocles-2",
        prompt:
          "He asks a second question: 'If we abandon our city to the flames, will the gods grant us victory at sea?'",
        hint: "Salamis is the key. A prophecy mentioning a narrow strait or divine favor at sea cements the oracle's legend.",
        domainFocus: "military",
        depthRequirement: "deep"
      }
    ],
    reward: {
      credibilityBonus: 14,
      goldBonus: 300,
      reputationBonus: 18
    }
  },

  "spartan-king": {
    id: "spartan-king",
    name: "The Hot Gates",
    figure: "Leonidas of Sparta",
    description:
      "A laconic Spartan king arrives with a small retinue. He asks about the battle that awaits his Three Hundred at the narrow pass.",
    minYear: 5,
    stages: [
      {
        id: "spartan-king-1",
        prompt:
          "Leonidas stands before the tripod, unadorned. 'Oracle, speak plainly: will Sparta endure if I march to Thermopylae?'",
        hint: "The oracle foretold that either Sparta would fall or a king must die. A prophecy of noble sacrifice resonates through the ages.",
        domainFocus: "military",
        depthRequirement: "grounded"
      }
    ],
    reward: {
      credibilityBonus: 10,
      goldBonus: 80,
      reputationBonus: 20
    }
  },

  "philip-of-macedon": {
    id: "philip-of-macedon",
    name: "The Macedonian Ambition",
    figure: "Philip II of Macedon",
    description:
      "The one-eyed King of Macedon, fresh from victories at Chaeronea, seeks divine endorsement for his plan to unite Greece under a single banner and march against Persia.",
    minYear: 8,
    stages: [
      {
        id: "philip-1",
        prompt:
          "Philip leans forward on his good leg: 'The city-states bicker like children. Shall Macedon bring them to heel and forge a Hellenic league?'",
        hint: "Philip's League of Corinth unified Greece. A prophecy about unity through strength — or a warning about hubris — shapes the era.",
        domainFocus: "military"
      },
      {
        id: "philip-2",
        prompt:
          "He continues: 'Once Greece is mine, shall I carry the war across the sea to the Great King's own lands?'",
        hint: "Philip was assassinated before he could invade Persia. A warning about crowns and daggers is historically resonant.",
        domainFocus: "military",
        depthRequirement: "grounded"
      },
      {
        id: "philip-3",
        prompt:
          "The king asks a final, private question: 'My son Alexander — does he have the favor of the gods?'",
        hint: "Alexander would indeed be favored. A prophecy of a son surpassing the father ties this consultation to the next legendary visitor.",
        domainFocus: "spiritual",
        depthRequirement: "deep"
      }
    ],
    reward: {
      credibilityBonus: 22,
      goldBonus: 400,
      reputationBonus: 15
    }
  },

  alexander: {
    id: "alexander",
    name: "The World Conqueror",
    figure: "Alexander the Great",
    description:
      "The young Macedonian king, already called 'the Great,' storms into the sanctuary. He demands to know whether the gods will grant him dominion over the known world.",
    minYear: 10,
    stages: [
      {
        id: "alexander-1",
        prompt:
          "Alexander seizes the Pythia's arm: 'Prophetess! Tell me — am I truly the son of Zeus-Ammon?'",
        hint: "Alexander believed himself divine. Confirming divine parentage fuels his conquests; denying it risks his wrath.",
        domainFocus: "spiritual"
      },
      {
        id: "alexander-2",
        prompt:
          "He demands more: 'Will Darius fall before me? Will Persepolis burn?'",
        hint: "Both happened. A prophecy of fire and fallen thrones echoes the historical destruction of the Persian Empire.",
        domainFocus: "military",
        depthRequirement: "grounded"
      },
      {
        id: "alexander-3",
        prompt:
          "At last, quietly: 'How far will my empire stretch? Will I reach the edge of the world?'",
        hint: "Alexander reached India but never the ocean. A prophecy about limits — even for the favored of gods — is the most oracular answer.",
        domainFocus: "military",
        depthRequirement: "oracular"
      }
    ],
    reward: {
      credibilityBonus: 30,
      goldBonus: 600,
      reputationBonus: 25
    }
  },

  "roman-general": {
    id: "roman-general",
    name: "The Eagle's Shadow",
    figure: "Lucius Aemilius Paullus",
    description:
      "A Roman consul arrives — an unprecedented visitor. Rome's legions eye Greece hungrily, and this general asks whether the Republic will subjugate the Hellenic world.",
    minYear: 14,
    stages: [
      {
        id: "roman-1",
        prompt:
          "The Roman speaks through an interpreter: 'The Senate asks: will Rome bring order to these quarrelsome Greeks?'",
        hint: "Rome did conquer Greece. A prophecy acknowledging inevitability while preserving Greek dignity shows oracular wisdom.",
        domainFocus: "military"
      },
      {
        id: "roman-2",
        prompt:
          "He presses: 'And what becomes of this oracle — of Delphi itself — under Roman dominion?'",
        hint: "Delphi survived under Rome for centuries, though diminished. A prophecy about endurance through transformation is both honest and hopeful.",
        domainFocus: "spiritual",
        depthRequirement: "deep"
      }
    ],
    reward: {
      credibilityBonus: 20,
      goldBonus: 350,
      reputationBonus: 16,
      specialEffect: "roman_patronage"
    }
  }
};

export const legendaryConsultationIds: LegendaryConsultationId[] = Object.keys(
  legendaryConsultationDefs
) as LegendaryConsultationId[];
