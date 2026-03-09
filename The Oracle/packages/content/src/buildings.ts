import type { BuildingDef } from "./schema";

export const buildingDefs: Record<BuildingDef["id"], BuildingDef> = {
  sacred_way: {
    id: "sacred_way",
    name: "Sacred Way",
    description: "Processional road for pilgrims and envoys.",
    category: "processional",
    costGold: 1,
    requiresPriest: false,
    maxCondition: 100,
    color: 0xcaa66b,
    upkeep: {},
    startingResources: {},
    passiveEffects: [{ id: "pilgrim-flow", kind: "pilgrim_capacity", value: 2, notes: "Raises precinct throughput for arrivals." }]
  },
  priest_quarters: {
    id: "priest_quarters",
    name: "Priest Quarters",
    description: "Houses the priesthood and anchors daily assignments.",
    category: "housing",
    costGold: 22,
    requiresPriest: false,
    maxCondition: 100,
    color: 0xdfd2b0,
    upkeep: { grain: 0.01 },
    startingResources: {},
    staffing: {
      priests: { attendant: 2, spring_warden: 1, flame_keeper: 1 },
      custodians: 1
    }
  },
  storehouse: {
    id: "storehouse",
    name: "Storehouse",
    description: "Buffers food, incense, and oil reserves.",
    category: "storage",
    costGold: 26,
    requiresPriest: false,
    maxCondition: 100,
    color: 0x9d7658,
    upkeep: {},
    startingResources: { olive_oil: 4, grain: 8, incense: 3 },
    storageCaps: {
      grain: 60,
      incense: 24,
      olive_oil: 24,
      bread: 30,
      scrolls: 18,
      papyrus: 18,
      sacred_water: 40,
      sacred_animals: 4
    },
    passiveEffects: [{ id: "storehouse-buffer", kind: "storage_buffer", value: 1, notes: "Stabilizes delivery targets for core ritual goods." }]
  },
  castalian_spring: {
    id: "castalian_spring",
    name: "Castalian Spring",
    description: "Purification waters for the Pythia and pilgrims.",
    category: "ritual",
    costGold: 18,
    requiresPriest: true,
    maxCondition: 100,
    color: 0x60a5d8,
    upkeep: {},
    startingResources: { sacred_water: 6 },
    staffing: {
      priests: { spring_warden: 1 }
    },
    recipes: [
      {
        id: "draw-sacred-water",
        consumes: {},
        produces: { sacred_water: 0.15 },
        dailyRate: 1,
        requiresRoles: ["spring_warden"],
        notes: "Provides steady ritual purification stock."
      }
    ]
  },
  inner_sanctum: {
    id: "inner_sanctum",
    name: "Inner Sanctum",
    description: "Seat of the Pythia and heart of consultations.",
    category: "ritual",
    costGold: 42,
    requiresPriest: true,
    maxCondition: 120,
    color: 0xd7c8a0,
    upkeep: { incense: 0.01, sacred_water: 0.01 },
    startingResources: {},
    unlockTier: "obscure",
    staffing: {
      priests: { attendant: 1, augur: 1 }
    },
    passiveEffects: [{ id: "consultation-prestige", kind: "prestige", value: 3, notes: "The sanctum anchors campaign progression and envoy confidence." }]
  },
  eternal_flame_brazier: {
    id: "eternal_flame_brazier",
    name: "Eternal Flame Brazier",
    description: "Consumes olive oil and symbolizes divine favor.",
    category: "ritual",
    costGold: 16,
    requiresPriest: true,
    maxCondition: 100,
    color: 0xf08a24,
    upkeep: { olive_oil: 0.015 },
    startingResources: { olive_oil: 4 },
    staffing: {
      priests: { flame_keeper: 1 }
    },
    passiveEffects: [{ id: "flame-prestige", kind: "prestige", value: 1, resourceId: "olive_oil", notes: "Well-fed flame improves visible sanctity." }]
  },
  sacrificial_altar: {
    id: "sacrificial_altar",
    name: "Sacrificial Altar",
    description: "Processes sacred animals into stronger ritual legitimacy and omen weight.",
    category: "ritual",
    costGold: 34,
    requiresPriest: true,
    maxCondition: 110,
    color: 0xb35f3f,
    unlockTier: "recognized",
    upkeep: { incense: 0.01 },
    startingResources: {},
    staffing: {
      priests: { sacrificial_priest: 1, attendant: 1 }
    },
    recipes: [
      {
        id: "sacrifice-animal",
        consumes: { sacred_animals: 0.08, incense: 0.01 },
        produces: {},
        dailyRate: 1,
        requiresRoles: ["sacrificial_priest"],
        notes: "Raises omen quality input for higher-risk consultation tracks."
      }
    ],
    passiveEffects: [{ id: "omen-quality", kind: "omen_quality", value: 2, notes: "Prepared sacrifice improves consultation reliability ceilings." }]
  },
  animal_pen: {
    id: "animal_pen",
    name: "Animal Pen",
    description: "Keeps sacrificial stock healthy and near the precinct.",
    category: "production",
    costGold: 24,
    requiresPriest: false,
    maxCondition: 95,
    color: 0x8d6a45,
    unlockTier: "recognized",
    upkeep: { grain: 0.02 },
    startingResources: { sacred_animals: 1 },
    staffing: {
      custodians: 1
    },
    storageCaps: {
      sacred_animals: 8,
      grain: 20
    },
    recipes: [
      {
        id: "raise-sacred-animals",
        consumes: { grain: 0.08 },
        produces: { sacred_animals: 0.09 },
        dailyRate: 1,
        notes: "Converts feed into a steady altar supply."
      }
    ]
  },
  granary: {
    id: "granary",
    name: "Granary",
    description: "Bulk grain reserve that smooths seasonal hunger and workshop supply.",
    category: "storage",
    costGold: 28,
    requiresPriest: false,
    maxCondition: 105,
    color: 0xba9158,
    unlockTier: "recognized",
    upkeep: {},
    startingResources: { grain: 12 },
    storageCaps: {
      grain: 120,
      bread: 36
    },
    passiveEffects: [{ id: "granary-buffer", kind: "storage_buffer", value: 2, resourceId: "grain", notes: "Absorbs caravan variability for food chains." }]
  },
  kitchen: {
    id: "kitchen",
    name: "Temple Kitchen",
    description: "Turns stored grain into bread for priests, pilgrims, and resident workers.",
    category: "production",
    costGold: 20,
    requiresPriest: false,
    maxCondition: 95,
    color: 0xc7834e,
    unlockTier: "recognized",
    upkeep: { grain: 0.015 },
    startingResources: {},
    staffing: {
      custodians: 1
    },
    storageCaps: {
      bread: 24,
      grain: 24
    },
    recipes: [
      {
        id: "bake-bread",
        consumes: { grain: 0.12 },
        produces: { bread: 0.1 },
        dailyRate: 1,
        notes: "Foundational food chain for growing staffing and hospitality."
      }
    ]
  },
  olive_press: {
    id: "olive_press",
    name: "Olive Press",
    description: "Presses raw olives into the oil that keeps the precinct's rituals alive.",
    category: "production",
    costGold: 23,
    requiresPriest: false,
    maxCondition: 95,
    color: 0x768a42,
    unlockTier: "recognized",
    upkeep: { olives: 0.02 },
    startingResources: { olives: 2 },
    storageCaps: {
      olives: 24,
      olive_oil: 24
    },
    recipes: [
      {
        id: "press-olives",
        consumes: { olives: 0.12 },
        produces: { olive_oil: 0.1 },
        dailyRate: 1,
        notes: "Feeds braziers and high-status rites."
      }
    ]
  },
  incense_store: {
    id: "incense_store",
    name: "Incense Store",
    description: "Dedicated reserve for ritual incense and papyrus imports.",
    category: "storage",
    costGold: 21,
    requiresPriest: false,
    maxCondition: 95,
    color: 0x7f5c62,
    unlockTier: "recognized",
    upkeep: {},
    startingResources: { incense: 4, papyrus: 2 },
    storageCaps: {
      incense: 30,
      papyrus: 20,
      scrolls: 20
    },
    passiveEffects: [{ id: "ritual-buffer", kind: "storage_buffer", value: 1, resourceId: "incense", notes: "Protects sanctum rituals from convoy gaps." }]
  },
  agora_market: {
    id: "agora_market",
    name: "Agora Market",
    description: "A commercial forecourt that converts prestige and traffic into income.",
    category: "trade",
    costGold: 30,
    requiresPriest: false,
    maxCondition: 100,
    color: 0x5d7f7f,
    unlockTier: "revered",
    upkeep: { bread: 0.01, incense: 0.005 },
    startingResources: {},
    staffing: {
      custodians: 1,
      carriers: 1,
      visitors: 6
    },
    passiveEffects: [{ id: "market-income", kind: "trade_income", value: 4, notes: "High-traffic precincts convert goods and reputation into treasury flow." }]
  },
  xenon: {
    id: "xenon",
    name: "Xenon",
    description: "Guest lodging for envoys, pilgrims, and wealthy benefactors.",
    category: "hospitality",
    costGold: 32,
    requiresPriest: false,
    maxCondition: 105,
    color: 0x8a8fb0,
    unlockTier: "revered",
    upkeep: { bread: 0.015, sacred_water: 0.01 },
    startingResources: {},
    staffing: {
      custodians: 1,
      visitors: 10
    },
    passiveEffects: [
      { id: "guest-capacity", kind: "pilgrim_capacity", value: 6, notes: "More visitors can be hosted during campaign peaks." },
      { id: "guest-donations", kind: "donation", value: 3, notes: "Comfortable guests donate more reliably." }
    ]
  },
  grain_field: {
    id: "grain_field",
    name: "Grain Field",
    description: "Terraced plots that produce a steady grain harvest for the precinct.",
    category: "production",
    costGold: 15,
    requiresPriest: false,
    maxCondition: 95,
    color: 0xc8b44e,
    upkeep: {},
    startingResources: { grain: 2 },
    staffing: {
      custodians: 1
    },
    storageCaps: {
      grain: 30
    },
    recipes: [
      {
        id: "harvest-grain",
        consumes: {},
        produces: { grain: 0.15 },
        dailyRate: 1,
        notes: "Base grain production. Seasonal climate affects yield."
      }
    ]
  },
  olive_grove: {
    id: "olive_grove",
    name: "Olive Grove",
    description: "Sacred olive trees planted on the temple slopes, yielding fruit for the press.",
    category: "production",
    costGold: 18,
    requiresPriest: false,
    maxCondition: 95,
    color: 0x6b8e3a,
    upkeep: {},
    startingResources: { olives: 2 },
    staffing: {
      custodians: 1
    },
    storageCaps: {
      olives: 30
    },
    recipes: [
      {
        id: "harvest-olives",
        consumes: {},
        produces: { olives: 0.12 },
        dailyRate: 1,
        notes: "Olives peak in autumn. One grove nearly sustains one press."
      }
    ]
  },
  incense_workshop: {
    id: "incense_workshop",
    name: "Incense Workshop",
    description: "Blends imported resins into the sacred incense used throughout the precinct.",
    category: "production",
    costGold: 25,
    requiresPriest: true,
    maxCondition: 100,
    color: 0x8a6b5e,
    unlockTier: "recognized",
    upkeep: {},
    startingResources: {},
    staffing: {
      priests: { attendant: 1 },
      custodians: 1
    },
    storageCaps: {
      incense: 20
    },
    recipes: [
      {
        id: "blend-incense",
        consumes: {},
        produces: { incense: 0.08 },
        dailyRate: 1,
        requiresRoles: ["attendant"],
        notes: "Provides comfortable incense surplus once recognized tier is reached."
      }
    ]
  },
  papyrus_reed_bed: {
    id: "papyrus_reed_bed",
    name: "Papyrus Reed Bed",
    description: "Irrigated beds that convert sacred water into writing material for the scholars.",
    category: "production",
    costGold: 20,
    requiresPriest: false,
    maxCondition: 95,
    color: 0x5a9e6b,
    unlockTier: "recognized",
    upkeep: {},
    startingResources: {},
    staffing: {
      custodians: 1
    },
    storageCaps: {
      papyrus: 20,
      sacred_water: 10
    },
    recipes: [
      {
        id: "press-papyrus",
        consumes: { sacred_water: 0.03 },
        produces: { papyrus: 0.06 },
        dailyRate: 1,
        notes: "Creates tension: sacred water feeds both purification and knowledge."
      }
    ]
  },
  scriptorium: {
    id: "scriptorium",
    name: "Scriptorium",
    description: "Scholar-priests transcribe prophecies and lore onto papyrus scrolls.",
    category: "production",
    costGold: 30,
    requiresPriest: true,
    maxCondition: 100,
    color: 0x7a6e8a,
    unlockTier: "revered",
    upkeep: {},
    startingResources: {},
    staffing: {
      priests: { scholar: 1 }
    },
    storageCaps: {
      papyrus: 12,
      scrolls: 18
    },
    recipes: [
      {
        id: "transcribe-scrolls",
        consumes: { papyrus: 0.04 },
        produces: { scrolls: 0.03 },
        dailyRate: 1,
        requiresRoles: ["scholar"],
        notes: "Slow prestige production. Scrolls enhance consultations."
      }
    ],
    passiveEffects: [{ id: "scriptorium-prestige", kind: "prestige", value: 2, notes: "The written word elevates Delphi's scholarly reputation." }]
  },
  library: {
    id: "library",
    name: "Library",
    description: "A repository of sacred scrolls and prophecy records that deepens Delphi's authority.",
    category: "storage",
    costGold: 38,
    requiresPriest: true,
    maxCondition: 110,
    color: 0x6a5c8a,
    unlockTier: "revered",
    upkeep: { scrolls: 0.01 },
    startingResources: {},
    staffing: {
      priests: { scholar: 1 }
    },
    storageCaps: {
      scrolls: 40,
      papyrus: 20
    },
    passiveEffects: [
      { id: "library-prestige", kind: "prestige", value: 4, notes: "A well-stocked library cements Delphi's intellectual primacy." },
      { id: "library-omen", kind: "omen_quality", value: 1, notes: "Recorded precedents improve prophecy calibration." }
    ]
  }
};
