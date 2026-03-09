/**
 * Building footprint calculation and collision detection.
 */

import type { BuildingDefId } from "@the-oracle/content";
import { getBuildingArt } from "@the-oracle/content";
import type { GameState } from "../state/gameState";
import { isBuildableTerrain, resolveTileTerrain } from "./generate";

export type Footprint = { width: number; height: number };
export type Coord = { x: number; y: number };

/**
 * Parse footprint string ("2x2", "3x2", "landmark", "tile_kit") into dimensions.
 */
export function parseFootprint(footprintStr: string): Footprint {
  if (footprintStr === "landmark") return { width: 4, height: 3 };
  if (footprintStr === "tile_kit") return { width: 1, height: 1 };

  const match = footprintStr.match(/^(\d+)x(\d+)$/);
  if (match) {
    return { width: Number(match[1]), height: Number(match[2]) };
  }
  return { width: 1, height: 1 };
}

// Cache parsed footprints
const footprintCache = new Map<string, Footprint>();

/**
 * Get the footprint for a building definition.
 */
export function getBuildingFootprint(defId: BuildingDefId): Footprint {
  if (footprintCache.has(defId)) {
    return footprintCache.get(defId)!;
  }
  const art = getBuildingArt(defId);
  const fp = art ? parseFootprint(art.footprint) : { width: 1, height: 1 };
  footprintCache.set(defId, fp);
  return fp;
}

/**
 * Returns all tiles occupied by a building placed at origin.
 * Origin is the top-left (min x, min y) tile.
 */
export function getOccupiedTiles(defId: BuildingDefId, origin: Coord): Coord[] {
  const fp = getBuildingFootprint(defId);
  const tiles: Coord[] = [];
  for (let dy = 0; dy < fp.height; dy++) {
    for (let dx = 0; dx < fp.width; dx++) {
      tiles.push({ x: origin.x + dx, y: origin.y + dy });
    }
  }
  return tiles;
}

/**
 * Build a set of all currently occupied tile keys from buildings.
 */
function buildOccupiedSet(state: GameState): Set<string> {
  const occupied = new Set<string>();
  for (const building of state.buildings) {
    for (const tile of getOccupiedTiles(building.defId as BuildingDefId, building.position)) {
      occupied.add(`${tile.x},${tile.y}`);
    }
  }
  return occupied;
}

/**
 * Check if all tiles for a proposed building are free.
 */
export function canPlaceBuilding(
  state: GameState,
  defId: BuildingDefId,
  origin: Coord
): boolean {
  const tiles = getOccupiedTiles(defId, origin);
  const roadSet = new Set(state.grid.roads.map((r) => `${r.x},${r.y}`));
  const occupiedSet = buildOccupiedSet(state);

  return tiles.every((tile) => {
    const key = `${tile.x},${tile.y}`;
    if (tile.x < 0 || tile.y < 0 || tile.x >= state.grid.width || tile.y >= state.grid.height) return false;
    if (roadSet.has(key)) return false;
    if (occupiedSet.has(key)) return false;
    const terrain = resolveTileTerrain(
      state.worldSeed,
      tile.x,
      tile.y,
      state.grid.terrainOverrides,
      state.grid.width,
      state.grid.height
    );
    if (!isBuildableTerrain(terrain)) return false;
    return true;
  });
}

/**
 * Check if at least one tile of the footprint is cardinally adjacent to a road.
 */
export function isFootprintAdjacentToRoad(
  state: GameState,
  defId: BuildingDefId,
  origin: Coord
): boolean {
  const tiles = getOccupiedTiles(defId, origin);
  const roadSet = new Set(state.grid.roads.map((r) => `${r.x},${r.y}`));
  return tiles.some((tile) =>
    roadSet.has(`${tile.x},${tile.y - 1}`) ||
    roadSet.has(`${tile.x + 1},${tile.y}`) ||
    roadSet.has(`${tile.x},${tile.y + 1}`) ||
    roadSet.has(`${tile.x - 1},${tile.y}`)
  );
}

/**
 * Find a building that occupies the given tile (checks full footprints).
 */
export function findBuildingAtTile(state: GameState, tile: Coord): string | undefined {
  for (const building of state.buildings) {
    const occupied = getOccupiedTiles(building.defId as BuildingDefId, building.position);
    if (occupied.some((t) => t.x === tile.x && t.y === tile.y)) {
      return building.id;
    }
  }
  return undefined;
}
