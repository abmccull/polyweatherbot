import type { Coord } from "@the-oracle/core";

export const TILE_WIDTH = 64;
export const TILE_HEIGHT = 32;
export const ORIGIN_X = 980;
export const ORIGIN_Y = 120;

export function tileToScreen(tile: Coord): Coord {
  return {
    x: ORIGIN_X + (tile.x - tile.y) * (TILE_WIDTH / 2),
    y: ORIGIN_Y + (tile.x + tile.y) * (TILE_HEIGHT / 2)
  };
}

export function screenToTile(x: number, y: number): Coord {
  const localX = x - ORIGIN_X;
  const localY = y - ORIGIN_Y;
  return {
    x: Math.floor((localY / (TILE_HEIGHT / 2) + localX / (TILE_WIDTH / 2)) / 2),
    y: Math.floor((localY / (TILE_HEIGHT / 2) - localX / (TILE_WIDTH / 2)) / 2)
  };
}
