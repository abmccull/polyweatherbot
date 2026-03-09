import type { OriginDef, OriginId } from "./schema";

export const DEFAULT_ORIGIN_ID: OriginId = "upstart-shrine";

export const originDefs: OriginDef[] = [
  {
    id: "ancient-spring",
    label: "Ancient Spring",
    title: "Ancient Spring",
    subtitle: "Memory runs deeper than masonry.",
    summary: "Delphi begins around an older sacred spring whose rites are trusted even before the precinct is wealthy.",
    advisorIntro: "The old spring still speaks with authority. Protect the rites, and the cities will come to Delphi already half-convinced.",
    uniqueMechanic: "Starts with ritual depth, devout attention, and higher sacred-water security.",
    challengeTags: ["ritual-depth", "devout-world", "pilgrim-pressure"],
    scoreModifier: 4,
    startingResources: {
      sacred_water: 18,
      incense: 4,
      gold: 10
    },
    pythiaModifiers: {
      attunement: 6,
      mentalClarity: 4,
      prestige: 7
    },
    campaignModifiers: {
      reputation: 5
    },
    worldBias: {
      climate: 1,
      divineMood: 1,
      oracleDensity: 1,
      faith: 2,
      trade: -1,
      pressure: -1
    }
  },
  {
    id: "upstart-shrine",
    label: "Upstart Shrine",
    title: "Upstart Shrine",
    subtitle: "A shrine with ambition and very little margin.",
    summary: "A young sanctuary has momentum but few guarantees. The opening years reward sharp logistics and disciplined prophecy.",
    advisorIntro: "You have attention but not yet legitimacy. Every road stone and every answer must prove that this shrine deserves to rise.",
    uniqueMechanic: "Balanced baseline start designed to surface the full precinct loop quickly.",
    challengeTags: ["baseline", "growth-run"],
    scoreModifier: 0,
    worldBias: {
      climate: 0,
      economy: 0,
      divineMood: 0,
      oracleDensity: 0,
      war: 0,
      trade: 0,
      faith: 0,
      intrigue: 0,
      unrest: 0,
      pressure: 0
    }
  },
  {
    id: "cursed-oracle",
    label: "Cursed Oracle",
    title: "Cursed Oracle",
    subtitle: "The tripod speaks, but it never speaks cleanly.",
    summary: "The Pythia is feared as much as she is revered. The signs run hot, costly, and politically dangerous.",
    advisorIntro: "Your gift comes wrapped in dread. If you cannot discipline the god's violence, every true prophecy will still feel like a threat.",
    uniqueMechanic: "Stronger trance and attunement, but worse strain, credibility drag, and harsher world pressure.",
    disabledSystems: ["gentle-opening"],
    challengeTags: ["high-risk", "credibility-fragile", "ominous-world"],
    scoreModifier: 9,
    startingResources: {
      gold: -25,
      incense: 3
    },
    pythiaModifiers: {
      attunement: 9,
      tranceDepth: 10,
      mentalClarity: -7,
      physicalHealth: -9,
      purification: 12,
      addTraits: ["fragile"]
    },
    worldBias: {
      climate: -1,
      divineMood: -2,
      oracleDensity: 1,
      faith: 1,
      intrigue: 2,
      unrest: 2,
      pressure: 2
    }
  },
  {
    id: "war-oracle",
    label: "War Oracle",
    title: "War Oracle",
    subtitle: "Generals arrive before pilgrims.",
    summary: "Delphi is already bound to campaigns, musters, and rival leagues. Rich patronage comes with dangerous expectations.",
    advisorIntro: "The warlords already listen. That is an honor only until their armies begin treating Delphi as part of the campaign plan.",
    uniqueMechanic: "Higher martial stakes, more conflict pressure, and stronger early military patronage.",
    challengeTags: ["martial-world", "conflict-heavy", "fast-stakes"],
    scoreModifier: 6,
    startingResources: {
      gold: 30,
      grain: 14,
      sacred_animals: 2
    },
    pythiaModifiers: {
      physicalHealth: 5,
      tranceDepth: 3,
      prestige: 4
    },
    campaignModifiers: {
      reputation: 4
    },
    worldBias: {
      climate: -1,
      economy: 1,
      divineMood: -1,
      war: 3,
      trade: -1,
      faith: -1,
      unrest: 2,
      pressure: 2
    }
  },
  {
    id: "gods-favourite",
    label: "God's Favourite",
    title: "God's Favourite",
    subtitle: "Grace invites devotion and scrutiny alike.",
    summary: "The sanctuary begins with unusual favor. The world expects signs of greatness, and disappointment will travel quickly.",
    advisorIntro: "Apollo has smiled on Delphi in public. Use that favor before the world turns it into a standard you cannot meet.",
    uniqueMechanic: "High divine favor, better opening reputation, and denser oracle traffic around the sanctuary.",
    challengeTags: ["blessed-start", "high-expectation", "reputation-run"],
    scoreModifier: 5,
    startingResources: {
      incense: 5,
      sacred_water: 6,
      gold: 20
    },
    pythiaModifiers: {
      attunement: 8,
      mentalClarity: 6,
      prestige: 10
    },
    campaignModifiers: {
      reputation: 8,
      treasuryProgress: 40
    },
    worldBias: {
      climate: 1,
      divineMood: 3,
      oracleDensity: 2,
      faith: 2,
      trade: 1,
      pressure: 1
    }
  },
  {
    id: "merchant-oracle",
    label: "Merchant Oracle",
    title: "Merchant Oracle",
    subtitle: "Caravans, contracts, and a god who understands margins.",
    summary: "The sanctuary is tied into caravan networks and donor houses from the outset, making supply and diplomacy central.",
    advisorIntro: "The ledgers already know your name. Keep the caravans open and Delphi can turn commerce into legitimacy faster than piety alone.",
    uniqueMechanic: "Stronger opening treasury, trade access, and mercantile faction weighting.",
    challengeTags: ["trade-world", "wealthy-start", "caravan-politics"],
    scoreModifier: 3,
    startingResources: {
      gold: 70,
      olive_oil: 8,
      grain: 8,
      papyrus: 3
    },
    pythiaModifiers: {
      mentalClarity: 4,
      prestige: 3
    },
    campaignModifiers: {
      reputation: 3,
      treasuryProgress: 25
    },
    worldBias: {
      economy: 3,
      divineMood: 1,
      oracleDensity: 1,
      war: -1,
      trade: 3,
      faith: -1,
      pressure: -1
    }
  },
  {
    id: "exiles-oracle",
    label: "Exile's Oracle",
    title: "Exile's Oracle",
    subtitle: "Built by displaced hands and long memory.",
    summary: "Delphi survives on refugees, defectors, and patrons with unfinished grudges. The run opens politically unstable but full of leverage.",
    advisorIntro: "Exiles remember everything. If you can turn their grievances into alliances, Delphi will become a sanctuary with teeth.",
    uniqueMechanic: "Harsher legitimacy curve, stronger intrigue, and more volatile faction memories.",
    challengeTags: ["intrigue-world", "unstable-start", "memory-politics"],
    scoreModifier: 7,
    startingResources: {
      gold: -15,
      grain: 10,
      olives: 8
    },
    pythiaModifiers: {
      attunement: 3,
      mentalClarity: -2,
      tranceDepth: 5,
      prestige: -4,
      addTraits: ["calculating"]
    },
    worldBias: {
      climate: 0,
      economy: -1,
      divineMood: -1,
      oracleDensity: 0,
      war: 1,
      trade: -1,
      faith: 0,
      intrigue: 3,
      unrest: 2,
      pressure: 1
    }
  }
];

export const originDefById = Object.fromEntries(
  originDefs.map((origin) => [origin.id, origin])
) as Record<OriginId, OriginDef>;
