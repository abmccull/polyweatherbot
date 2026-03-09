/**
 * Seed-deterministic terrain generation.
 * Pure function of (worldSeed, x, y) → TerrainType.
 * No external dependencies.
 */

export type TerrainType =
  | "grass"
  | "dry_earth"
  | "limestone"
  | "cliff"
  | "water"
  | "scrub"
  | "forest"
  | "sacred_ground"
  | "road";

/**
 * Simple integer hash for deterministic noise.
 * Returns a float in [0, 1).
 */
function hashNoise(seed: number, x: number, y: number): number {
  let h = seed ^ 0x68e31da4;
  h = Math.imul(h ^ (x * 374761393), 0x85ebca6b);
  h = Math.imul(h ^ (y * 668265263), 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 13), 0x27d4eb2d);
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 0x100000000;
}

/**
 * Smoother noise by averaging 4 hash samples (value noise).
 */
function smoothNoise(seed: number, x: number, y: number, scale: number): number {
  const sx = x / scale;
  const sy = y / scale;
  const ix = Math.floor(sx);
  const iy = Math.floor(sy);
  const fx = sx - ix;
  const fy = sy - iy;

  const n00 = hashNoise(seed, ix, iy);
  const n10 = hashNoise(seed, ix + 1, iy);
  const n01 = hashNoise(seed, ix, iy + 1);
  const n11 = hashNoise(seed, ix + 1, iy + 1);

  const nx0 = n00 + (n10 - n00) * fx;
  const nx1 = n01 + (n11 - n01) * fx;
  return nx0 + (nx1 - nx0) * fy;
}

/**
 * Generate base terrain type for a tile given the world seed.
 *
 * Zone layout for a 60x60 grid:
 *   - Outer 2-tile ring: cliff (map boundary)
 *   - North edge (y < 6): elevated, more rock/forest
 *   - Central band (y 10-50): buildable, grass/dry_earth/limestone mix
 *   - South edge (y > 54): lower terrain, water features
 *   - Scattered features: rock outcrops, tree clusters, water pools
 */
export function generateTileTerrain(
  seed: number,
  x: number,
  y: number,
  gridWidth = 60,
  gridHeight = 60
): TerrainType {
  // Outer boundary ring → cliff
  if (x <= 1 || y <= 1 || x >= gridWidth - 2 || y >= gridHeight - 2) {
    return "cliff";
  }

  // Second ring → forest edge
  if (x === 2 || y === 2 || x === gridWidth - 3 || y === gridHeight - 3) {
    const edgeNoise = hashNoise(seed + 1, x, y);
    return edgeNoise < 0.55 ? "forest" : "cliff";
  }

  // Multi-scale noise for feature distribution
  const n1 = smoothNoise(seed, x, y, 8);      // Large-scale zones
  const n2 = smoothNoise(seed + 100, x, y, 4); // Medium features
  const n3 = hashNoise(seed + 200, x, y);       // Fine detail

  // Distance from center (normalized 0-1)
  const cx = gridWidth / 2;
  const cy = gridHeight / 2;
  const dx = (x - cx) / cx;
  const dy = (y - cy) / cy;
  const distFromCenter = Math.sqrt(dx * dx + dy * dy);

  // Elevation proxy (higher in north, lower in south)
  const elevation = (1 - y / gridHeight) * 0.6 + n1 * 0.4;

  // Water: south-facing, low elevation, clustered
  if (elevation < 0.22 && n2 > 0.55 && y > gridHeight * 0.6) {
    return "water";
  }

  // Also small spring pools scattered
  if (n2 > 0.82 && n3 > 0.7 && distFromCenter > 0.3) {
    return "water";
  }

  // Rocky areas: high elevation or noise clusters
  if (elevation > 0.72 && n2 > 0.5) {
    return n3 > 0.4 ? "cliff" : "limestone";
  }

  // Limestone terraces: moderate-high elevation
  if (elevation > 0.58 && n2 > 0.45) {
    return "limestone";
  }

  // Forest clusters: medium noise, away from center
  if (n1 > 0.62 && n2 > 0.48 && distFromCenter > 0.25) {
    return "forest";
  }

  // Scrub: scattered bushes
  if (n2 > 0.65 && n3 > 0.55 && distFromCenter > 0.15) {
    return "scrub";
  }

  // Dry earth patches
  if (n1 < 0.38 && n3 < 0.45) {
    return "dry_earth";
  }

  // Default: grass
  return "grass";
}

/**
 * Resolve terrain for a tile, checking overrides first.
 */
export function resolveTileTerrain(
  seed: number,
  x: number,
  y: number,
  terrainOverrides?: Record<string, TerrainType>,
  gridWidth = 60,
  gridHeight = 60
): TerrainType {
  const key = `${x},${y}`;
  if (terrainOverrides?.[key]) {
    return terrainOverrides[key];
  }
  return generateTileTerrain(seed, x, y, gridWidth, gridHeight);
}

/**
 * Check if a terrain type is buildable (can place buildings/roads on it).
 */
export function isBuildableTerrain(terrain: TerrainType): boolean {
  return terrain !== "cliff" && terrain !== "water";
}
