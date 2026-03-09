import type { ConsultationQuestionDef } from "./schema";

export const consultationQuestions: ConsultationQuestionDef[] = [
  {
    id: "fleet-campaign",
    text: "Should our fleet sail before the summer winds settle?",
    domain: "military",
    tags: ["fleet", "timing", "campaign"]
  },
  {
    id: "frontier-march",
    text: "Will the army gain more by marching now or by waiting behind the river?",
    domain: "military",
    tags: ["army", "timing", "river", "campaign"]
  },
  {
    id: "garrison-oath",
    text: "Can a garrison oath hold a divided frontier city?",
    domain: "military",
    tags: ["city", "alliance", "oath", "frontier"]
  },
  {
    id: "grain-treaty",
    text: "Will a treaty of grain strengthen the city or invite weakness?",
    domain: "economic",
    tags: ["trade", "grain", "treaty", "city"]
  },
  {
    id: "harvest-levy",
    text: "Should we press the harvest levy before winter stores are sealed?",
    domain: "economic",
    tags: ["harvest", "treasury", "timing", "levy"]
  },
  {
    id: "harbor-tolls",
    text: "Will new harbor tolls enrich the treasury or fracture our allies?",
    domain: "economic",
    tags: ["treasury", "alliance", "trade", "harbor"]
  },
  {
    id: "colony-relief",
    text: "Should we send relief to the colony now, or keep the grain at home?",
    domain: "economic",
    tags: ["grain", "city", "trade", "timing"]
  },
  {
    id: "sacred-marriage",
    text: "Does the god bless this union of houses?",
    domain: "spiritual",
    tags: ["marriage", "legitimacy", "alliance", "rites"]
  },
  {
    id: "succession-portent",
    text: "Is the king's heir guarded by the god, or shadowed by rival blood?",
    domain: "spiritual",
    tags: ["king", "succession", "legitimacy", "court"]
  },
  {
    id: "plague-rites",
    text: "Will stricter rites turn the sickness from the city gates?",
    domain: "spiritual",
    tags: ["city", "purification", "rites", "health"]
  },
  {
    id: "pilgrim-season",
    text: "Should Delphi open the sacred road to more pilgrims this season?",
    domain: "spiritual",
    tags: ["oracle", "pilgrims", "timing", "prestige"]
  },
  {
    id: "embassy-truce",
    text: "Can a shared sacrifice turn this embassy toward peace?",
    domain: "spiritual",
    tags: ["alliance", "sacrifice", "peace", "diplomacy"]
  }
];
