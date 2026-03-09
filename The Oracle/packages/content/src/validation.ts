import type { BuildingDef, ResourceId, ScenarioDef } from "./schema";
import type { LegendaryConsultationDef } from "./legendaryConsultations";

/**
 * Validates building definitions for internal consistency.
 * Checks: valid resource references, cost sanity, production chain DAG (no cycles).
 */
export function validateBuildingDefs(
  defs: Record<string, BuildingDef>,
  validResourceIds: readonly ResourceId[]
): string[] {
  const errors: string[] = [];
  const resourceSet = new Set<string>(validResourceIds);

  for (const [key, def] of Object.entries(defs)) {
    if (def.id !== key) {
      errors.push(`Building "${key}" has mismatched id "${def.id}".`);
    }
    if (def.costGold < 0) {
      errors.push(`Building "${def.id}" has negative costGold (${def.costGold}).`);
    }
    if (def.maxCondition <= 0) {
      errors.push(`Building "${def.id}" has non-positive maxCondition (${def.maxCondition}).`);
    }

    // Validate upkeep resource references
    for (const resId of Object.keys(def.upkeep)) {
      if (!resourceSet.has(resId)) {
        errors.push(`Building "${def.id}" upkeep references unknown resource "${resId}".`);
      }
    }

    // Validate startingResources references
    for (const resId of Object.keys(def.startingResources)) {
      if (!resourceSet.has(resId)) {
        errors.push(`Building "${def.id}" startingResources references unknown resource "${resId}".`);
      }
    }

    // Validate storageCaps references
    if (def.storageCaps) {
      for (const resId of Object.keys(def.storageCaps)) {
        if (!resourceSet.has(resId)) {
          errors.push(`Building "${def.id}" storageCaps references unknown resource "${resId}".`);
        }
      }
    }

    // Validate recipe resource references
    if (def.recipes) {
      for (const recipe of def.recipes) {
        for (const resId of Object.keys(recipe.consumes)) {
          if (!resourceSet.has(resId)) {
            errors.push(`Building "${def.id}" recipe "${recipe.id}" consumes unknown resource "${resId}".`);
          }
        }
        for (const resId of Object.keys(recipe.produces)) {
          if (!resourceSet.has(resId)) {
            errors.push(`Building "${def.id}" recipe "${recipe.id}" produces unknown resource "${resId}".`);
          }
        }
        if (recipe.dailyRate <= 0) {
          errors.push(`Building "${def.id}" recipe "${recipe.id}" has non-positive dailyRate.`);
        }
      }
    }
  }

  // Check production chain forms a DAG (no cycles)
  // Build a graph: resource -> resources it depends on (consumes -> produces)
  const producedBy = new Map<string, Set<string>>(); // resource -> set of resources consumed to produce it
  for (const def of Object.values(defs)) {
    if (!def.recipes) continue;
    for (const recipe of def.recipes) {
      const produced = Object.keys(recipe.produces);
      const consumed = Object.keys(recipe.consumes);
      for (const p of produced) {
        if (!producedBy.has(p)) producedBy.set(p, new Set());
        for (const c of consumed) {
          producedBy.get(p)!.add(c);
        }
      }
    }
  }

  // DFS cycle detection
  const visited = new Set<string>();
  const inStack = new Set<string>();

  function hasCycle(node: string): boolean {
    if (inStack.has(node)) return true;
    if (visited.has(node)) return false;
    visited.add(node);
    inStack.add(node);
    const deps = producedBy.get(node);
    if (deps) {
      for (const dep of deps) {
        if (hasCycle(dep)) return true;
      }
    }
    inStack.delete(node);
    return false;
  }

  for (const resource of producedBy.keys()) {
    if (hasCycle(resource)) {
      errors.push(`Production chain cycle detected involving resource "${resource}".`);
      break; // one cycle error is enough
    }
  }

  return errors;
}

/**
 * Validates scenario definitions for internal consistency.
 */
export function validateScenarioDefs(
  scenarios: ScenarioDef[],
  validAgeIds: readonly string[]
): string[] {
  const errors: string[] = [];
  const seenIds = new Set<string>();
  const ageSet = new Set(validAgeIds);

  for (const scenario of scenarios) {
    if (seenIds.has(scenario.id)) {
      errors.push(`Duplicate scenario id "${scenario.id}".`);
    }
    seenIds.add(scenario.id);

    if (!scenario.label || scenario.label.trim().length === 0) {
      errors.push(`Scenario "${scenario.id}" has empty label.`);
    }
    if (!scenario.summary || scenario.summary.trim().length === 0) {
      errors.push(`Scenario "${scenario.id}" has empty summary.`);
    }

    if (scenario.difficulty !== undefined && (scenario.difficulty < 1 || scenario.difficulty > 5)) {
      errors.push(`Scenario "${scenario.id}" has invalid difficulty ${scenario.difficulty} (must be 1-5).`);
    }

    if (scenario.startingConditions?.startAge && !ageSet.has(scenario.startingConditions.startAge)) {
      errors.push(`Scenario "${scenario.id}" references unknown startAge "${scenario.startingConditions.startAge}".`);
    }

    if (scenario.dedicationMilestones) {
      for (let i = 1; i < scenario.dedicationMilestones.length; i++) {
        if (scenario.dedicationMilestones[i]! <= scenario.dedicationMilestones[i - 1]!) {
          errors.push(`Scenario "${scenario.id}" dedicationMilestones are not strictly ascending.`);
          break;
        }
      }
    }
  }

  return errors;
}

/**
 * Validates legendary consultation definitions.
 */
export function validateLegendaryDefs(
  legendaries: Record<string, LegendaryConsultationDef>
): string[] {
  const errors: string[] = [];
  const minYears: number[] = [];

  for (const [key, def] of Object.entries(legendaries)) {
    if (def.id !== key) {
      errors.push(`Legendary "${key}" has mismatched id "${def.id}".`);
    }
    if (!def.name || def.name.trim().length === 0) {
      errors.push(`Legendary "${def.id}" has empty name.`);
    }
    if (def.stages.length === 0) {
      errors.push(`Legendary "${def.id}" has no stages.`);
    }
    for (const stage of def.stages) {
      if (!stage.prompt || stage.prompt.trim().length === 0) {
        errors.push(`Legendary "${def.id}" stage "${stage.id}" has empty prompt.`);
      }
      if (!stage.domainFocus) {
        errors.push(`Legendary "${def.id}" stage "${stage.id}" has no domainFocus.`);
      }
    }
    if (def.minYear < 0) {
      errors.push(`Legendary "${def.id}" has negative minYear (${def.minYear}).`);
    }
    minYears.push(def.minYear);
  }

  // Check year thresholds are ascending across the collection
  for (let i = 1; i < minYears.length; i++) {
    if (minYears[i]! < minYears[i - 1]!) {
      errors.push(`Legendary consultation minYear values are not ascending (${minYears[i - 1]} -> ${minYears[i]}).`);
      break;
    }
  }

  return errors;
}

/**
 * Runs all content validators and returns a combined result.
 * Import and call on-demand -- not at startup.
 */
export function validateContentIntegrity(content: {
  buildingDefs: Record<string, BuildingDef>;
  resourceIds: readonly ResourceId[];
  scenarioDefs: ScenarioDef[];
  ageIds: readonly string[];
  legendaryDefs: Record<string, LegendaryConsultationDef>;
}): { valid: boolean; errors: string[] } {
  const errors: string[] = [
    ...validateBuildingDefs(content.buildingDefs, content.resourceIds),
    ...validateScenarioDefs(content.scenarioDefs, content.ageIds),
    ...validateLegendaryDefs(content.legendaryDefs)
  ];

  return { valid: errors.length === 0, errors };
}
