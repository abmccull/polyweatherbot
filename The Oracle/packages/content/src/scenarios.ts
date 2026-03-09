import type { ScenarioDef } from "./schema";

export const scenarioDefs: ScenarioDef[] = [
  {
    id: "rising-oracle",
    label: "The Rising Oracle",
    summary: "A young sanctuary begins to attract the notice of powerful Greek city-states.",
    recommendedStartingTier: "obscure",
    dedicationMilestones: [150, 320],
    requiredResolvedCrises: 1,
    winningTier: "revered"
  },
  {
    id: "sandbox",
    label: "Sandbox",
    summary: "Open-ended precinct management without a campaign endpoint.",
    recommendedStartingTier: "obscure"
  },
  {
    id: "young-oracle",
    label: "The Young Oracle",
    summary: "A guided introduction to the ways of Delphi.",
    description:
      "You inherit a small shrine with a single spring and a handful of faithful. The gods are patient, the factions distant, and the world forgiving. Learn the rhythms of prophecy before the great powers come calling.",
    difficulty: 1,
    startingConditions: {
      gold: 120,
      reputation: "obscure"
    },
    specialRules: [
      "Philosopher threats disabled for the first 3 years",
      "Rival oracles remain dormant until reputation reaches recognized",
      "Guided advisor messages introduce each system as it unlocks"
    ],
    victoryCondition: "Reach recognized reputation and complete 5 consultations.",
    recommendedStartingTier: "obscure",
    winningTier: "recognized"
  },
  {
    id: "crossroads-of-power",
    label: "Crossroads of Power",
    summary: "Two dominant factions vie for control of the Greek world, and your prophecies tip the balance.",
    description:
      "Athens and Sparta stand at the brink of war, each sending envoys to secure divine endorsement. Every prophecy you deliver strengthens one side and enrages the other. Navigate the politics of parity or be crushed between them.",
    difficulty: 3,
    startingConditions: {
      gold: 80,
      reputation: "recognized",
      factionMod: {
        athens: { credibility: 15, favour: 10 },
        sparta: { credibility: 15, favour: 10 }
      }
    },
    specialRules: [
      "Athens and Sparta begin with heightened aggression toward each other",
      "Delivering a prophecy to one dominant faction reduces favour with the other",
      "Alliance formation between dominant factions ends the scenario in a draw"
    ],
    victoryCondition: "Reach revered reputation without any faction dropping below 20 credibility.",
    recommendedStartingTier: "recognized",
    winningTier: "revered"
  },
  {
    id: "skeptics-challenge",
    label: "The Skeptic's Challenge",
    summary: "Philosophers question everything. Prove the oracle's worth against relentless doubt.",
    description:
      "Rationalist philosophers have spread across the Greek world, eroding faith in oracles everywhere. Your credibility starts low, philosopher threats are amplified, and every failed prophecy is magnified. Only undeniable accuracy can silence the skeptics.",
    difficulty: 4,
    startingConditions: {
      gold: 60,
      reputation: "obscure",
      factionMod: {
        athens: { credibility: -15 },
        corinth: { credibility: -10 },
        miletus: { credibility: -20 }
      },
      burdens: ["philosopher_pressure_doubled"]
    },
    specialRules: [
      "All philosopher threat thresholds reduced by 30%",
      "Credibility gains from prophecies halved until first crisis resolved",
      "Philosopher sect crises occur 50% more frequently"
    ],
    victoryCondition: "Reach revered reputation and resolve 3 philosopher crises.",
    recommendedStartingTier: "obscure",
    requiredResolvedCrises: 3,
    winningTier: "revered"
  },
  {
    id: "golden-age",
    label: "Golden Age",
    summary: "Wealth flows, culture blooms, and the oracle stands at the centre of a golden era.",
    description:
      "The Greek world enters a period of unprecedented prosperity. Trade caravans arrive laden with goods, factions are generous, and the oracle basks in abundant resources. But abundance breeds complacency -- can you build a legacy that outlasts the good times?",
    difficulty: 2,
    startingConditions: {
      gold: 200,
      reputation: "recognized",
      factionMod: {
        corinth: { favour: 15 },
        miletus: { favour: 15 },
        argos: { favour: 10 }
      }
    },
    specialRules: [
      "Trade income doubled for the first 5 years",
      "Pilgrim arrival rates increased by 50%",
      "After year 8, a recession event halves trade income permanently"
    ],
    victoryCondition: "Reach panhellenic reputation and complete 3 treasury dedications.",
    recommendedStartingTier: "recognized",
    dedicationMilestones: [200, 400, 600],
    winningTier: "panhellenic"
  },
  {
    id: "twilight-of-the-gods",
    label: "Twilight of the Gods",
    summary: "The old world fades. Keep the flame burning as faith itself declines.",
    description:
      "You begin in the Hellenistic Age. The great city-states have spent their vigour, rival oracles multiply, and a shadowy power looms from the west. Resources dwindle, consultations grow rare, and survival itself is victory.",
    difficulty: 5,
    startingConditions: {
      gold: 40,
      reputation: "recognized",
      factionMod: {
        athens: { credibility: -10, favour: -5 },
        sparta: { credibility: -10, favour: -5 },
        macedon: { credibility: 10, favour: 5, hostility: 10 }
      },
      startAge: "hellenistic"
    },
    specialRules: [
      "Consultation frequency reduced by 40%",
      "Rival oracle aggression starts at maximum",
      "Building upkeep costs increased by 25%",
      "No new buildings unlocked after start"
    ],
    victoryCondition: "Survive 10 years without reputation dropping below recognized.",
    recommendedStartingTier: "recognized",
    winningTier: "recognized"
  },
  {
    id: "spys-web",
    label: "The Spy's Web",
    summary: "Rival oracles infiltrate from every direction. Trust no one.",
    description:
      "Four rival oracles have established networks throughout the Greek world, each scheming to steal your supplicants and undermine your prophecies. The espionage system is your primary weapon -- uncover their plots before they dismantle yours.",
    difficulty: 4,
    startingConditions: {
      gold: 90,
      reputation: "recognized",
      burdens: ["rival_oracles_all_active"]
    },
    specialRules: [
      "All rival oracles begin active from day one",
      "Espionage operations cost 30% less",
      "Rival oracle pressure cap increased by 50%",
      "Counter-intelligence missions grant bonus credibility on success"
    ],
    victoryCondition: "Neutralize 3 rival oracle networks and reach revered reputation.",
    recommendedStartingTier: "recognized",
    winningTier: "revered"
  },
  {
    id: "sacred-excavation",
    label: "Sacred Excavation",
    summary: "Ancient relics lie buried beneath the sanctuary. Dig deep and claim divine power.",
    description:
      "Your sanctuary sits atop layers of ancient ruins. Excavation sites dot the landscape, each promising powerful relics that can reshape your oracle's capabilities. Focus your resources on unearthing the past to secure the future.",
    difficulty: 3,
    startingConditions: {
      gold: 100,
      reputation: "obscure",
      burdens: ["excavation_rich_map"]
    },
    specialRules: [
      "Triple the normal number of excavation sites available",
      "Relic effects are 50% stronger",
      "Excavation costs reduced by 25%",
      "Victory requires collecting specific legendary relics"
    ],
    victoryCondition: "Collect 5 relics including at least one legendary relic, and reach revered reputation.",
    recommendedStartingTier: "obscure",
    winningTier: "revered"
  }
];
