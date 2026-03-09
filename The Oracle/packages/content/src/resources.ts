import type { ResourceDef } from "./schema";

export const resourceDefs: ResourceDef[] = [
  { id: "gold", label: "Gold", category: "currency" },
  { id: "sacred_water", label: "Sacred Water", category: "ritual", seasonalMultipliers: { Spring: 1.3, Summer: 0.7, Autumn: 1.0, Winter: 1.2 } },
  { id: "olive_oil", label: "Olive Oil", category: "ritual" },
  { id: "incense", label: "Incense", category: "ritual" },
  { id: "grain", label: "Grain", category: "food", seasonalMultipliers: { Spring: 1.3, Summer: 1.0, Autumn: 0.6, Winter: 0.3 } },
  { id: "sacred_animals", label: "Sacred Animals", category: "ritual" },
  { id: "bread", label: "Bread", category: "food" },
  { id: "olives", label: "Olives", category: "food", seasonalMultipliers: { Spring: 0.5, Summer: 0.7, Autumn: 1.4, Winter: 0.8 } },
  { id: "papyrus", label: "Papyrus", category: "trade" },
  { id: "scrolls", label: "Scrolls", category: "trade" }
];
