import type { WalkerInstance } from "./gameState";

export type CarrierProfile = Required<Pick<WalkerInstance, "fatigue" | "haulingSkill" | "supplyRadius">>;

const carrierProfiles: CarrierProfile[] = [
  { fatigue: 10, haulingSkill: 72, supplyRadius: 9 },
  { fatigue: 16, haulingSkill: 58, supplyRadius: 7 },
  { fatigue: 8, haulingSkill: 66, supplyRadius: 11 }
];

export function carrierProfileForIndex(index: number): CarrierProfile {
  const profile = carrierProfiles[index % carrierProfiles.length];
  const cycle = Math.floor(index / carrierProfiles.length);

  return {
    fatigue: Math.min(36, profile.fatigue + cycle * 4),
    haulingSkill: Math.max(40, profile.haulingSkill - cycle * 2),
    supplyRadius: profile.supplyRadius + (cycle % 2)
  };
}

export function normalizeCarrierWalker(walker: WalkerInstance, carrierIndex = 0): WalkerInstance {
  if (walker.role !== "carrier") {
    return walker;
  }

  if (walker.fatigue !== undefined && walker.haulingSkill !== undefined && walker.supplyRadius !== undefined) {
    return walker;
  }

  const profile = carrierProfileForIndex(carrierIndex);
  return {
    ...walker,
    fatigue: walker.fatigue ?? profile.fatigue,
    haulingSkill: walker.haulingSkill ?? profile.haulingSkill,
    supplyRadius: walker.supplyRadius ?? profile.supplyRadius
  };
}
