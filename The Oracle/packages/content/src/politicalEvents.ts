import type { PoliticalEventDef } from "./schema";

export const politicalEventDefs: PoliticalEventDef[] = [
  {
    id: "war-muster",
    label: "War Muster",
    domain: "military",
    agenda: "war",
    summaries: [
      "draws hoplites into a fresh border muster",
      "tests its captains with an aggressive field exercise",
      "orders bronze and grain toward a tense frontier"
    ]
  },
  {
    id: "naval-raid",
    label: "Naval Raid",
    domain: "military",
    agenda: "war",
    summaries: [
      "whispers of a coastal raid and a hurried fleet levy",
      "arms triremes after pirates strike a sacred convoy",
      "answers a rival harbor with sharpened naval patrols"
    ]
  },
  {
    id: "grain-shortage",
    label: "Grain Shortage",
    domain: "economic",
    agenda: "trade",
    summaries: [
      "haggles fiercely after a thin grain market",
      "leans on merchants to stretch a fragile harvest",
      "reopens treaty tables to steady its markets"
    ]
  },
  {
    id: "harbor-compacts",
    label: "Harbor Compacts",
    domain: "economic",
    agenda: "trade",
    summaries: [
      "courts harbor guilds with generous caravan terms",
      "extends warehouse credit to keep caravans moving",
      "brokers a customs pact that favors patient traders"
    ]
  },
  {
    id: "sacred-festival",
    label: "Sacred Festival",
    domain: "spiritual",
    agenda: "faith",
    summaries: [
      "raises hymns and offerings through a crowded sacred festival",
      "purifies its shrines to calm a restless populace",
      "seeks legitimacy through public rites and lavish dedications"
    ]
  },
  {
    id: "omens-disputed",
    label: "Omens Disputed",
    domain: "spiritual",
    agenda: "faith",
    summaries: [
      "argues over troubled omens in the temples",
      "sends priests to quiet rumors of divine displeasure",
      "leans harder on sanctuaries after a season of bad signs"
    ]
  },
  {
    id: "succession-rumor",
    label: "Succession Rumor",
    domain: "military",
    agenda: "succession",
    summaries: [
      "staggers under whispered claims around the ruling house",
      "tests loyalties as heirs and generals trade accusations",
      "guards the palace while succession talk grows dangerous"
    ]
  },
  {
    id: "hostage-bargain",
    label: "Hostage Bargain",
    domain: "economic",
    agenda: "succession",
    summaries: [
      "trades hostages and gifts to steady a brittle succession",
      "buys noble silence with hurried patronage and debt",
      "offers market concessions to keep rival claimants still"
    ]
  }
];
