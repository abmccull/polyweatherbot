import type { AdvisorDef, TraitDef } from "./schema";

export const traitDefs: TraitDef[] = [
  { id: "visionary", label: "Visionary", description: "Sharper omen accuracy, weaker physical resilience." },
  { id: "calculating", label: "Calculating", description: "Balances clarity and ambiguity with unusual precision." },
  { id: "diplomatic", label: "Diplomatic", description: "Smooths over faction fallout at delivery." },
  { id: "fragile", label: "Fragile", description: "High peaks of attunement, but rest needs spike quickly." }
];

export const advisorDefs: AdvisorDef[] = [
  { id: "hierophant", name: "Hierophant", voice: "grave" },
  { id: "treasurer", name: "Treasurer", voice: "anxious" },
  { id: "builder", name: "Master Builder", voice: "gruff" },
  { id: "diplomat", name: "Diplomat", voice: "smooth" },
  { id: "shadow", name: "Shadow", voice: "quiet" }
];
