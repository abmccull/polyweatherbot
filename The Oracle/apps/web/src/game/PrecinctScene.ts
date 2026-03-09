import Phaser from "phaser";
import { buildingDefs, getBuildingArt, getFloraArt, getPropArt, getTerrainArt, getTerrainArtVariants, getWalkerArt, listFloraArtAssets, listPropArtAssets, listTerrainArtAssets, listWalkerArtAssets } from "@the-oracle/content";
import type { TerrainArtAsset } from "@the-oracle/content";
import type { GameState } from "@the-oracle/core";

import { getRuntime } from "../runtime";
import { TILE_HEIGHT, TILE_WIDTH, screenToTile, tileToScreen } from "./iso";

type Tile = { x: number; y: number };

type InspectionTone = "consecrated" | "attuned" | "watchful" | "frayed";

type InspectionLinkTone = "ritual" | "approach" | "support" | "strain";

type InspectionReading = {
  integrity: number;
  tone: InspectionTone;
  toneLabel: string;
  approach: number;
  sanctity: number;
  strain: number;
  roadContacts: number;
  ritualAnchors: number;
  links: {
    id: string;
    tile: Tile;
    tone: InspectionLinkTone;
  }[];
};

type DecorPlacement = {
  id: string;
  assetId: string;
  tile: Tile;
  alpha?: number;
  flipX?: boolean;
};

type FloraPlacement = {
  id: string;
  assetId: string;
  tile: Tile;
  alpha?: number;
  flipX?: boolean;
};

type PrecinctTerrainContext = {
  occupiedKeys: Set<string>;
  buildingKeys: Set<string>;
  roadKeys: Set<string>;
  springPoolKeys: Set<string>;
  springPoolTiles: Tile[];
  ritualCourtKeys: Set<string>;
  ritualCourtTiles: Tile[];
  approachShoulderKeys: Set<string>;
  approachShoulderTiles: Tile[];
  cliffKeys: Set<string>;
  cliffTiles: Tile[];
  terraceApronKeys: Set<string>;
  terraceApronTiles: Tile[];
  dryEarthKeys: Set<string>;
  dryEarthTiles: Tile[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  centerX: number;
};

const INSPECTION_RADIUS = 4;
const BUILDING_TEXTURE_PREFIX = "precinct-building:";
const FLORA_TEXTURE_PREFIX = "precinct-flora:";
const PROP_TEXTURE_PREFIX = "precinct-prop:";
const TERRAIN_TEXTURE_PREFIX = "precinct-terrain:";
const BUILDING_SPRITE_BASELINE_Y = 20;
const FLORA_SPRITE_BASELINE_Y = 18;
const PROP_SPRITE_BASELINE_Y = 19;
const TERRAIN_SPRITE_BASELINE_Y = 18;
const WALKER_SPRITE_BASELINE_Y = 20;
const ROAD_ART_DEF_ID = "sacred_way";
const ROAD_SPRITE_BASELINE_Y = 18;

function clamp(value: number, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

function manhattan(left: Tile, right: Tile) {
  return Math.abs(left.x - right.x) + Math.abs(left.y - right.y);
}

function average(values: number[]) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function uniqueTiles(tiles: Tile[]) {
  const seen = new Set<string>();
  return tiles.filter((tile) => {
    const key = `${tile.x},${tile.y}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function toneForIntegrity(value: number): { tone: InspectionTone; toneLabel: string } {
  if (value >= 76) {
    return { tone: "consecrated", toneLabel: "Consecrated Ground" };
  }
  if (value >= 62) {
    return { tone: "attuned", toneLabel: "Attuned Ground" };
  }
  if (value >= 48) {
    return { tone: "watchful", toneLabel: "Watchful Ground" };
  }
  return { tone: "frayed", toneLabel: "Frayed Ground" };
}

function toneColor(tone: InspectionTone) {
  if (tone === "consecrated") return 0xe8c060;
  if (tone === "attuned") return 0x8db57b;
  if (tone === "watchful") return 0x6ec2e8;
  return 0xc0513a;
}

function linkColor(tone: InspectionLinkTone) {
  if (tone === "ritual") return 0xe8c060;
  if (tone === "approach") return 0xe6d5aa;
  if (tone === "support") return 0x6ec2e8;
  return 0xc0513a;
}

function inspectTile(state: GameState, tile: Tile): InspectionReading {
  const localBuildings = [...state.buildings]
    .map((building) => ({
      building,
      def: buildingDefs[building.defId],
      distance: manhattan(tile, building.position)
    }))
    .filter((entry) => entry.distance <= INSPECTION_RADIUS)
    .sort((left, right) => left.distance - right.distance || left.building.id.localeCompare(right.building.id));
  const localWalkers = state.walkers
    .map((walker) => ({ walker, distance: manhattan(tile, walker.tile) }))
    .filter((entry) => entry.distance <= INSPECTION_RADIUS);
  const nearbyRoads = state.grid.roads.filter((road) => manhattan(tile, road) <= 2);
  const roadContacts = state.grid.roads.filter((road) => manhattan(tile, road) === 1).length;
  const ritualAnchors = localBuildings.filter((entry) => entry.def.category === "ritual").length;
  const supportAnchors = localBuildings.filter((entry) => entry.def.category === "housing" || entry.def.category === "storage" || entry.def.category === "hospitality").length;
  const localWear = average(localBuildings.map((entry) => 100 - (entry.building.condition / Math.max(1, entry.building.maxCondition)) * 100));
  const localFatigue = average(localWalkers.filter((entry) => entry.walker.role === "carrier").map((entry) => entry.walker.fatigue ?? 0));
  const worldPressure = Math.max(
    state.campaign.worldMap.nodes.find((node) => node.id === state.campaign.worldMap.selectedNodeId)?.pressure ?? 0,
    state.worldGeneration.regions[0]?.pressure ?? 0
  );
  const divineMeter = state.worldGeneration.divineMood.meter ?? 50;

  const approach = clamp(20 + roadContacts * 15 + nearbyRoads.length * 4 + localBuildings.filter((entry) => entry.def.category === "processional").length * 12 - localFatigue * 0.16);
  const sanctity = clamp(16 + ritualAnchors * 18 + divineMeter * 0.24 + (state.resources.sacred_water?.amount ?? 0) * 0.42 + (state.resources.incense?.amount ?? 0) * 0.3 - state.pythia.needs.purification * 0.18);
  const shelter = clamp(18 + supportAnchors * 14 + localBuildings.filter((entry) => entry.def.category === "production").length * 8 + (state.resources.grain?.amount ?? 0) * 0.1 - localWear * 0.35);
  const strain = clamp(
    8
    + localFatigue * 0.42
    + state.pythia.needs.rest * 0.14
    + state.pythia.needs.purification * 0.18
    + localWear * 0.54
    + worldPressure * 0.2
    + localWalkers.filter((entry) => entry.walker.state === "hauling" || entry.walker.state === "delivering").length * 5
  );
  const integrity = clamp(Math.round(sanctity * 0.42 + approach * 0.24 + shelter * 0.18 + (100 - strain) * 0.24));
  const { tone, toneLabel } = toneForIntegrity(integrity);

  return {
    integrity,
    tone,
    toneLabel,
    approach: Math.round(approach),
    sanctity: Math.round(sanctity),
    strain: Math.round(strain),
    roadContacts,
    ritualAnchors,
    links: localBuildings.slice(0, 5).map(({ building, def }) => ({
      id: building.id,
      tile: building.position,
      tone: def.category === "ritual"
        ? "ritual"
        : def.category === "processional"
          ? "approach"
          : (building.condition / Math.max(1, building.maxCondition)) < 0.65
            ? "strain"
            : "support"
    }))
  };
}

function siteCellAlpha(reading: InspectionReading, focusTile: Tile, sample: Tile) {
  const distance = manhattan(focusTile, sample);
  if (distance > 3) {
    return 0;
  }
  const nearbyLink = reading.links.some((link) => manhattan(link.tile, sample) <= 1) ? 0.12 : 0;
  return Phaser.Math.Clamp((reading.integrity / 100) * 0.1 - distance * 0.018 + nearbyLink * 0.45, 0.025, 0.13);
}

function spriteBoundsForFootprint(footprint: string) {
  if (footprint === "tile_kit") {
    return { maxWidth: 92, maxHeight: 74 };
  }
  if (footprint === "landmark") {
    return { maxWidth: 276, maxHeight: 238 };
  }

  const match = footprint.match(/^(\d+)x(\d+)$/);
  if (!match) {
    return { maxWidth: 122, maxHeight: 114 };
  }

  const widthTiles = Number(match[1]);
  const heightTiles = Number(match[2]);
  const footprintSpan = Math.max(widthTiles, heightTiles);

  return {
    maxWidth: 118 + (widthTiles - 1) * 44 + (heightTiles - 1) * 30,
    maxHeight: 128 + (footprintSpan - 1) * 42
  };
}

export class PrecinctScene extends Phaser.Scene {
  private terrainGraphics!: Phaser.GameObjects.Graphics;

  private roadGraphics!: Phaser.GameObjects.Graphics;

  private floraGraphics!: Phaser.GameObjects.Graphics;

  private propGraphics!: Phaser.GameObjects.Graphics;

  private buildingGraphics!: Phaser.GameObjects.Graphics;

  private walkerGraphics!: Phaser.GameObjects.Graphics;

  private overlayGraphics!: Phaser.GameObjects.Graphics;

  private inspectionLabel!: Phaser.GameObjects.Text;

  private inspectionDetail!: Phaser.GameObjects.Text;

  private keys!: Record<string, Phaser.Input.Keyboard.Key>;

  private terrainSprites = new Map<string, Phaser.GameObjects.Image>();

  private buildingSprites = new Map<string, Phaser.GameObjects.Image>();

  private roadSprites = new Map<string, Phaser.GameObjects.Image>();

  private floraSprites = new Map<string, Phaser.GameObjects.Image>();

  private propSprites = new Map<string, Phaser.GameObjects.Image>();

  private walkerSprites = new Map<string, Phaser.GameObjects.Image>();

  private autoFrameActive = true;

  constructor() {
    super("PrecinctScene");
  }

  preload() {
    const loadedTextureKeys = new Set<string>();
    for (const art of listTerrainArtAssets()) {
      const textureKey = `${TERRAIN_TEXTURE_PREFIX}${art.terrainAssetId}`;
      if (loadedTextureKeys.has(textureKey) || this.textures.exists(textureKey)) {
        continue;
      }
      loadedTextureKeys.add(textureKey);
      this.load.image(textureKey, art.publicPath);
    }

    for (const defId of Object.keys(buildingDefs)) {
      const art = getBuildingArt(defId);
      if (!art) {
        continue;
      }
      const textureKey = `${BUILDING_TEXTURE_PREFIX}${art.assetId}`;
      if (loadedTextureKeys.has(textureKey) || this.textures.exists(textureKey)) {
        continue;
      }
      loadedTextureKeys.add(textureKey);
      this.load.image(textureKey, art.publicPath);
    }

    for (const art of listFloraArtAssets()) {
      const textureKey = `${FLORA_TEXTURE_PREFIX}${art.assetId}`;
      if (loadedTextureKeys.has(textureKey) || this.textures.exists(textureKey)) {
        continue;
      }
      loadedTextureKeys.add(textureKey);
      this.load.image(textureKey, art.publicPath);
    }

    for (const art of listPropArtAssets()) {
      const textureKey = `${PROP_TEXTURE_PREFIX}${art.assetId}`;
      if (loadedTextureKeys.has(textureKey) || this.textures.exists(textureKey)) {
        continue;
      }
      loadedTextureKeys.add(textureKey);
      this.load.image(textureKey, art.publicPath);
    }

    for (const art of listWalkerArtAssets()) {
      const textureKey = `precinct-walker:${art.walkerRole}`;
      if (loadedTextureKeys.has(textureKey) || this.textures.exists(textureKey)) {
        continue;
      }
      loadedTextureKeys.add(textureKey);
      this.load.image(textureKey, art.publicPath);
    }
  }

  create() {
    this.cameras.main.setBackgroundColor("#d9c79c");
    const focus = tileToScreen({ x: 29, y: 48 });
    this.cameras.main.centerOn(focus.x, focus.y);
    this.cameras.main.setZoom(2.25);
    window.__oracleDebug.viewportForTile = (tile) => {
      const camera = this.cameras.main;
      camera.preRender();
      const world = tileToScreen(tile);
      const guess = {
        x: (world.x - camera.worldView.x) * camera.zoom,
        y: (world.y - camera.worldView.y) * camera.zoom
      };

      const projectTileAtViewport = (x: number, y: number) => {
        const worldPoint = camera.getWorldPoint(x, y);
        return screenToTile(worldPoint.x, worldPoint.y);
      };

      let best: { x: number; y: number; distance: number } | undefined;
      for (let radius = 0; radius <= 320; radius += 2) {
        for (let offsetY = -radius; offsetY <= radius; offsetY += 2) {
          for (let offsetX = -radius; offsetX <= radius; offsetX += 2) {
            if (radius > 0 && Math.max(Math.abs(offsetX), Math.abs(offsetY)) !== radius) {
              continue;
            }

            const candidateX = guess.x + offsetX;
            const candidateY = guess.y + offsetY;
            const projectedTile = projectTileAtViewport(candidateX, candidateY);
            if (projectedTile.x !== tile.x || projectedTile.y !== tile.y) {
              continue;
            }

            const distance = Math.abs(offsetX) + Math.abs(offsetY);
            if (!best || distance < best.distance) {
              best = { x: candidateX, y: candidateY, distance };
            }
          }
        }

        if (best) {
          return {
            x: best.x,
            y: best.y
          };
        }
      }

      return guess;
    };

    this.terrainGraphics = this.add.graphics().setDepth(0);
    this.roadGraphics = this.add.graphics().setDepth(10);
    this.floraGraphics = this.add.graphics().setDepth(14);
    this.propGraphics = this.add.graphics().setDepth(18);
    this.buildingGraphics = this.add.graphics().setDepth(20);
    this.walkerGraphics = this.add.graphics().setDepth(80);
    this.overlayGraphics = this.add.graphics().setDepth(90);
    this.inspectionLabel = this.add.text(0, 0, "", {
      fontFamily: "Georgia, serif",
      fontSize: "13px",
      fontStyle: "bold",
      color: "#f7e7c0",
      backgroundColor: "rgba(34, 26, 12, 0.72)",
      padding: { left: 7, right: 7, top: 3, bottom: 3 }
    }).setOrigin(0.5, 1).setDepth(110).setVisible(false);
    this.inspectionDetail = this.add.text(0, 0, "", {
      fontFamily: "Georgia, serif",
      fontSize: "10px",
      color: "#ddcda6",
      backgroundColor: "rgba(34, 26, 12, 0.58)",
      padding: { left: 7, right: 7, top: 2, bottom: 2 }
    }).setOrigin(0.5, 1).setDepth(110).setVisible(false);

    this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => {
      delete window.__oracleDebug.getPrecinctArtDebug;
      for (const sprite of this.terrainSprites.values()) {
        sprite.destroy();
      }
      this.terrainSprites.clear();
      for (const sprite of this.roadSprites.values()) {
        sprite.destroy();
      }
      this.roadSprites.clear();
      for (const sprite of this.floraSprites.values()) {
        sprite.destroy();
      }
      this.floraSprites.clear();
      for (const sprite of this.propSprites.values()) {
        sprite.destroy();
      }
      this.propSprites.clear();
      for (const sprite of this.buildingSprites.values()) {
        sprite.destroy();
      }
      this.buildingSprites.clear();
      for (const sprite of this.walkerSprites.values()) {
        sprite.destroy();
      }
      this.walkerSprites.clear();
    });

    window.__oracleDebug.getPrecinctArtDebug = () => ({
      terrainSpriteCount: this.terrainSprites.size,
      spriteCount: this.buildingSprites.size,
      roadSpriteCount: this.roadSprites.size,
      floraSpriteCount: this.floraSprites.size,
      propSpriteCount: this.propSprites.size,
      walkerSpriteCount: this.walkerSprites.size,
      terrain: [...this.terrainSprites.entries()].map(([terrainId, sprite]) => ({
        terrainId,
        textureKey: sprite.texture.key,
        x: sprite.x,
        y: sprite.y,
        depth: sprite.depth,
        visible: sprite.visible,
        scaleX: sprite.scaleX,
        scaleY: sprite.scaleY,
        displayWidth: sprite.displayWidth,
        displayHeight: sprite.displayHeight
      })),
      flora: [...this.floraSprites.entries()].map(([floraId, sprite]) => ({
        floraId,
        textureKey: sprite.texture.key,
        x: sprite.x,
        y: sprite.y,
        depth: sprite.depth,
        visible: sprite.visible,
        scaleX: sprite.scaleX,
        scaleY: sprite.scaleY,
        displayWidth: sprite.displayWidth,
        displayHeight: sprite.displayHeight
      })),
      props: [...this.propSprites.entries()].map(([propId, sprite]) => ({
        propId,
        textureKey: sprite.texture.key,
        x: sprite.x,
        y: sprite.y,
        depth: sprite.depth,
        visible: sprite.visible,
        scaleX: sprite.scaleX,
        scaleY: sprite.scaleY,
        displayWidth: sprite.displayWidth,
        displayHeight: sprite.displayHeight
      })),
      sprites: [...this.buildingSprites.entries()].map(([buildingId, sprite]) => {
        const currentState = getRuntime().getState();
        const building = currentState.buildings.find((entry) => entry.id === buildingId);
        return ({
          buildingId,
          defId: building?.defId ?? null,
          textureKey: sprite.texture.key,
          x: sprite.x,
          y: sprite.y,
          depth: sprite.depth,
          visible: sprite.visible,
          scaleX: sprite.scaleX,
          scaleY: sprite.scaleY,
          displayWidth: sprite.displayWidth,
          displayHeight: sprite.displayHeight
        });
      })
    });

    this.keys = this.input.keyboard?.addKeys("W,A,S,D,LEFT,RIGHT,UP,DOWN") as Record<string, Phaser.Input.Keyboard.Key>;

    this.input.on("pointermove", (pointer: Phaser.Input.Pointer) => {
      const worldPoint = pointer.positionToCamera(this.cameras.main) as Phaser.Math.Vector2;
      const tile = screenToTile(worldPoint.x, worldPoint.y);
      const runtime = getRuntime();
      runtime.dispatch({ type: "HoverTileCommand", tile });
    });

    this.input.on("wheel", (_pointer: Phaser.Input.Pointer, _objects: unknown[], _deltaX: number, deltaY: number) => {
      const camera = this.cameras.main;
      this.autoFrameActive = false;
      camera.setZoom(Phaser.Math.Clamp(camera.zoom - deltaY * 0.001, 0.75, 2.2));
    });

    this.input.on("pointerdown", (pointer: Phaser.Input.Pointer) => {
      if (pointer.button !== 0) {
        return;
      }
      this.autoFrameActive = false;
      const runtime = getRuntime();
      const worldPoint = pointer.positionToCamera(this.cameras.main) as Phaser.Math.Vector2;
      const tile = screenToTile(worldPoint.x, worldPoint.y);
      const state = runtime.getState();
      const tool = state.ui.activeTool;

      if (tool === "select") {
        const building = state.buildings.find((entry) => entry.position.x === tile.x && entry.position.y === tile.y);
        if (building) {
          runtime.dispatch({ type: "SelectEntityCommand", entityId: building.id, entityKind: "building" });
          return;
        }
        const walker = state.walkers.find((entry) => entry.tile.x === tile.x && entry.tile.y === tile.y);
        runtime.dispatch(
          walker
            ? { type: "SelectEntityCommand", entityId: walker.id, entityKind: "walker" }
            : { type: "SelectEntityCommand", entityId: undefined, entityKind: undefined }
        );
        return;
      }

      if (tool === "sacred_way") {
        runtime.dispatch({ type: "PlaceRoadCommand", tile });
        return;
      }

      runtime.dispatch({ type: "PlaceBuildingCommand", defId: tool, tile });
    });
  }

  update() {
    const camera = this.cameras.main;
    const speed = 18 / camera.zoom;
    const movedCamera = this.keys.W?.isDown || this.keys.UP?.isDown || this.keys.S?.isDown || this.keys.DOWN?.isDown || this.keys.A?.isDown || this.keys.LEFT?.isDown || this.keys.D?.isDown || this.keys.RIGHT?.isDown;

    if (this.keys.W?.isDown || this.keys.UP?.isDown) {
      camera.scrollY -= speed;
    }
    if (this.keys.S?.isDown || this.keys.DOWN?.isDown) {
      camera.scrollY += speed;
    }
    if (this.keys.A?.isDown || this.keys.LEFT?.isDown) {
      camera.scrollX -= speed;
    }
    if (this.keys.D?.isDown || this.keys.RIGHT?.isDown) {
      camera.scrollX += speed;
    }

    if (movedCamera) {
      this.autoFrameActive = false;
    }

    this.drawState(getRuntime().getState());
  }

  private drawDiamond(graphics: Phaser.GameObjects.Graphics, screenX: number, screenY: number, fillColor: number, alpha = 1) {
    graphics.fillStyle(fillColor, alpha);
    graphics.beginPath();
    graphics.moveTo(screenX, screenY);
    graphics.lineTo(screenX + TILE_WIDTH / 2, screenY + TILE_HEIGHT / 2);
    graphics.lineTo(screenX, screenY + TILE_HEIGHT);
    graphics.lineTo(screenX - TILE_WIDTH / 2, screenY + TILE_HEIGHT / 2);
    graphics.closePath();
    graphics.fillPath();
    graphics.lineStyle(1, 0x604d2d, 0.25);
    graphics.strokePath();
  }

  private drawTerrainDiamond(screenX: number, screenY: number, fillColor: number, outlineAlpha = 0.08) {
    this.terrainGraphics.fillStyle(fillColor, 0.92);
    this.terrainGraphics.beginPath();
    this.terrainGraphics.moveTo(screenX, screenY);
    this.terrainGraphics.lineTo(screenX + TILE_WIDTH / 2, screenY + TILE_HEIGHT / 2);
    this.terrainGraphics.lineTo(screenX, screenY + TILE_HEIGHT);
    this.terrainGraphics.lineTo(screenX - TILE_WIDTH / 2, screenY + TILE_HEIGHT / 2);
    this.terrainGraphics.closePath();
    this.terrainGraphics.fillPath();
    this.terrainGraphics.lineStyle(1, 0x8c7652, outlineAlpha);
    this.terrainGraphics.strokePath();
  }

  private drawSelectionHalo(graphics: Phaser.GameObjects.Graphics, screenX: number, screenY: number, color: number) {
    graphics.lineStyle(3, color, 0.9);
    graphics.beginPath();
    graphics.moveTo(screenX, screenY - 4);
    graphics.lineTo(screenX + TILE_WIDTH / 2 + 3, screenY + TILE_HEIGHT / 2);
    graphics.lineTo(screenX, screenY + TILE_HEIGHT + 4);
    graphics.lineTo(screenX - TILE_WIDTH / 2 - 3, screenY + TILE_HEIGHT / 2);
    graphics.closePath();
    graphics.strokePath();
  }

  private getTerrainTextureKey(terrainId: string) {
    const art = getTerrainArt(terrainId);
    if (!art) {
      return null;
    }
    const textureKey = `${TERRAIN_TEXTURE_PREFIX}${art.terrainAssetId}`;
    return this.textures.exists(textureKey) ? textureKey : null;
  }

  private terrainVariantIndex(terrainId: string, tile: Tile, variantCount: number) {
    let terrainBias = 0;
    for (const character of terrainId) {
      terrainBias += character.charCodeAt(0);
    }
    const value = (
      tile.x * 6
      + tile.y * 7
      + tile.x * tile.y * 2
      + tile.x * tile.x * 3
      + tile.y * tile.y * 5
      + terrainBias
    );
    return ((value % variantCount) + variantCount) % variantCount;
  }

  private getTerrainArtForTile(terrainId: string, tile: Tile): TerrainArtAsset | null {
    const variants = getTerrainArtVariants(terrainId);
    if (variants.length === 0) {
      return null;
    }
    if (variants.length === 1) {
      return variants[0] ?? null;
    }
    return variants[this.terrainVariantIndex(terrainId, tile, variants.length)] ?? variants[0] ?? null;
  }

  private getTerrainTextureSelection(terrainId: string, tile: Tile) {
    const art = this.getTerrainArtForTile(terrainId, tile);
    if (!art) {
      return null;
    }
    const textureKey = `${TERRAIN_TEXTURE_PREFIX}${art.terrainAssetId}`;
    if (!this.textures.exists(textureKey)) {
      return null;
    }
    return { art, textureKey };
  }

  private getBuildingTextureKey(defId: string) {
    const art = getBuildingArt(defId);
    if (!art) {
      return null;
    }
    const textureKey = `${BUILDING_TEXTURE_PREFIX}${art.assetId}`;
    return this.textures.exists(textureKey) ? textureKey : null;
  }

  private getFloraTextureKey(assetId: string) {
    const art = getFloraArt(assetId);
    if (!art) {
      return null;
    }
    const textureKey = `${FLORA_TEXTURE_PREFIX}${art.assetId}`;
    return this.textures.exists(textureKey) ? textureKey : null;
  }

  private getPropTextureKey(assetId: string) {
    const art = getPropArt(assetId);
    if (!art) {
      return null;
    }
    const textureKey = `${PROP_TEXTURE_PREFIX}${art.assetId}`;
    return this.textures.exists(textureKey) ? textureKey : null;
  }

  private getWalkerTextureKey(walkerRole: string) {
    const art = getWalkerArt(walkerRole);
    if (!art) {
      return null;
    }
    const textureKey = `precinct-walker:${art.walkerRole}`;
    return this.textures.exists(textureKey) ? textureKey : null;
  }

  private buildingSpriteDepth(tile: Tile) {
    return 40 + tile.y * 0.1 + tile.x * 0.001;
  }

  private roadSpriteDepth(tile: Tile) {
    return 12 + tile.y * 0.06 + tile.x * 0.0005;
  }

  private propSpriteDepth(tile: Tile) {
    return 26 + tile.y * 0.07 + tile.x * 0.0006;
  }

  private floraSpriteDepth(tile: Tile) {
    return 16 + tile.y * 0.065 + tile.x * 0.00055;
  }

  private walkerSpriteDepth(tile: Tile) {
    return 82 + tile.y * 0.09 + tile.x * 0.0008;
  }

  private roadSpriteKey(tile: Tile) {
    return `${tile.x},${tile.y}`;
  }

  private terrainTileKey(tile: Tile) {
    return `${tile.x},${tile.y}`;
  }

  private buildTerrainContext(state: GameState): PrecinctTerrainContext | null {
    const occupiedTiles = [
      ...state.grid.roads,
      ...state.buildings.map((building) => building.position)
    ];
    if (occupiedTiles.length === 0) {
      return null;
    }

    const occupiedKeys = new Set<string>();
    const roadKeys = new Set<string>();
    const buildingKeys = new Set<string>();
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (const road of state.grid.roads) {
      roadKeys.add(this.terrainTileKey(road));
    }
    for (const building of state.buildings) {
      buildingKeys.add(this.terrainTileKey(building.position));
    }
    for (const tile of occupiedTiles) {
      occupiedKeys.add(this.terrainTileKey(tile));
      minX = Math.min(minX, tile.x);
      maxX = Math.max(maxX, tile.x);
      minY = Math.min(minY, tile.y);
      maxY = Math.max(maxY, tile.y);
    }

    const springPoolTiles: Tile[] = [];
    const springPoolKeys = new Set<string>();
    for (const building of state.buildings) {
      if (building.defId !== "castalian_spring") {
        continue;
      }

      const candidates: Tile[] = [
        { x: building.position.x + 1, y: building.position.y + 1 },
        { x: building.position.x + 1, y: building.position.y },
        { x: building.position.x, y: building.position.y + 1 },
        { x: building.position.x - 1, y: building.position.y + 1 }
      ];
      const openTile = candidates.find((tile) => {
        if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
          return false;
        }
        return !occupiedKeys.has(this.terrainTileKey(tile));
      });
      if (!openTile) {
        continue;
      }
      springPoolTiles.push(openTile);
      springPoolKeys.add(this.terrainTileKey(openTile));
    }

    const ritualCourtTiles: Tile[] = [];
    const ritualCourtKeys = new Set<string>();
    const addRitualCourtTile = (tile: Tile) => {
      if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
        return;
      }
      const tileKey = this.terrainTileKey(tile);
      if (occupiedKeys.has(tileKey) || springPoolKeys.has(tileKey) || ritualCourtKeys.has(tileKey)) {
        return;
      }
      ritualCourtTiles.push(tile);
      ritualCourtKeys.add(tileKey);
    };

    state.buildings
      .filter((building) => buildingDefs[building.defId].category === "ritual")
      .forEach((building) => {
        [
          { x: building.position.x - 1, y: building.position.y },
          { x: building.position.x + 1, y: building.position.y },
          { x: building.position.x, y: building.position.y - 1 },
          { x: building.position.x - 1, y: building.position.y + 1 },
          { x: building.position.x + 1, y: building.position.y + 1 },
          { x: building.position.x, y: building.position.y + 1 }
        ].forEach(addRitualCourtTile);
      });

    const cliffTiles: Tile[] = [];
    const cliffKeys = new Set<string>();
    for (let x = minX - 2; x <= maxX + 1; x += 1) {
      const tile = { x, y: minY - 2 };
      if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
        continue;
      }
      cliffTiles.push(tile);
      cliffKeys.add(this.terrainTileKey(tile));
    }

    const terraceApronTiles: Tile[] = [];
    const terraceApronKeys = new Set<string>();
    for (let x = minX - 1; x <= maxX + 1; x += 1) {
      for (let y = minY; y <= maxY + 1; y += 1) {
        const tile = { x, y };
        if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
          continue;
        }
        const tileKey = this.terrainTileKey(tile);
        if (cliffKeys.has(tileKey) || springPoolKeys.has(tileKey)) {
          continue;
        }
        terraceApronTiles.push(tile);
        terraceApronKeys.add(tileKey);
      }
    }

    const dryEarthTiles: Tile[] = [];
    const dryEarthKeys = new Set<string>();
    const addDryEarthTile = (tile: Tile) => {
      if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
        return;
      }
      const tileKey = this.terrainTileKey(tile);
      if (occupiedKeys.has(tileKey) || terraceApronKeys.has(tileKey) || springPoolKeys.has(tileKey) || cliffKeys.has(tileKey) || dryEarthKeys.has(tileKey)) {
        return;
      }
      dryEarthTiles.push(tile);
      dryEarthKeys.add(tileKey);
    };
    for (let x = minX - 2; x <= maxX + 2; x += 1) {
      addDryEarthTile({ x, y: maxY + 2 });
    }
    for (let x = Math.max(0, Math.min(state.grid.width - 1, Math.round((minX + maxX) / 2) - 1)); x <= Math.min(state.grid.width - 1, Math.round((minX + maxX) / 2) + 1); x += 1) {
      addDryEarthTile({ x, y: maxY + 3 });
    }
    for (let y = minY + 1; y <= maxY; y += 2) {
      addDryEarthTile({ x: minX - 2, y });
      addDryEarthTile({ x: maxX + 2, y });
    }

    const centerX = Math.round((minX + maxX) / 2);
    const approachShoulderTiles: Tile[] = [];
    const approachShoulderKeys = new Set<string>();
    const addApproachShoulderTile = (tile: Tile) => {
      if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
        return;
      }
      const tileKey = this.terrainTileKey(tile);
      if (occupiedKeys.has(tileKey) || springPoolKeys.has(tileKey) || cliffKeys.has(tileKey) || approachShoulderKeys.has(tileKey)) {
        return;
      }
      approachShoulderTiles.push(tile);
      approachShoulderKeys.add(tileKey);
    };
    const processionalAxis = uniqueTiles([
      ...state.grid.roads,
      { x: centerX, y: maxY + 1 },
      { x: centerX, y: maxY + 2 }
    ]);
    processionalAxis.forEach((tile) => {
      addApproachShoulderTile({ x: tile.x - 1, y: tile.y });
      addApproachShoulderTile({ x: tile.x + 1, y: tile.y });
      if (tile.y >= maxY + 1) {
        addApproachShoulderTile({ x: tile.x - 2, y: tile.y + 1 });
        addApproachShoulderTile({ x: tile.x + 2, y: tile.y + 1 });
      }
    });
    for (let x = centerX - 2; x <= centerX + 2; x += 1) {
      addApproachShoulderTile({ x, y: maxY + 2 });
    }

    return {
      occupiedKeys,
      buildingKeys,
      roadKeys,
      springPoolKeys,
      springPoolTiles,
      ritualCourtKeys,
      ritualCourtTiles,
      approachShoulderKeys,
      approachShoulderTiles,
      cliffKeys,
      cliffTiles,
      terraceApronKeys,
      terraceApronTiles,
      dryEarthKeys,
      dryEarthTiles,
      minX,
      maxX,
      minY,
      maxY,
      centerX
    };
  }

  private precinctDistance(context: PrecinctTerrainContext, tile: Tile) {
    const minX = context.minX - 2;
    const maxX = context.maxX + 2;
    const minY = context.minY - 2;
    const maxY = context.maxY + 3;
    const dx = tile.x < minX ? minX - tile.x : tile.x > maxX ? tile.x - maxX : 0;
    const dy = tile.y < minY ? minY - tile.y : tile.y > maxY ? tile.y - maxY : 0;
    return dx + dy;
  }

  private isAdjacentToOccupied(context: PrecinctTerrainContext, tile: Tile) {
    return context.occupiedKeys.has(this.terrainTileKey({ x: tile.x + 1, y: tile.y }))
      || context.occupiedKeys.has(this.terrainTileKey({ x: tile.x - 1, y: tile.y }))
      || context.occupiedKeys.has(this.terrainTileKey({ x: tile.x, y: tile.y + 1 }))
      || context.occupiedKeys.has(this.terrainTileKey({ x: tile.x, y: tile.y - 1 }));
  }

  private terrainTextureIdForTile(context: PrecinctTerrainContext, tile: Tile) {
    const tileKey = this.terrainTileKey(tile);
    if (context.roadKeys.has(tileKey)) {
      return "sacred_paving";
    }

    if (context.springPoolKeys.has(tileKey)) {
      return "spring_pool";
    }

    if (tile.x === context.centerX && tile.y === context.maxY + 1) {
      return "stone_stairs";
    }

    if (context.cliffKeys.has(tileKey)) {
      return "cliff_edge";
    }

    const onNorthRetainingWall = tile.y === context.minY - 1 && tile.x >= context.minX - 1 && tile.x <= context.maxX + 1;
    const onWestRetainingWall = tile.x === context.minX - 1 && tile.y >= context.minY - 1 && tile.y <= context.maxY;
    if (onNorthRetainingWall || onWestRetainingWall) {
      return "retaining_wall";
    }

    if (context.ritualCourtKeys.has(tileKey)) {
      return "sacred_paving";
    }

    if (context.buildingKeys.has(tileKey)) {
      return "limestone_terrace";
    }

    if (context.approachShoulderKeys.has(tileKey)) {
      return tile.y >= context.maxY + 2 ? "dry_earth" : "limestone_terrace";
    }

    if (context.terraceApronKeys.has(tileKey)) {
      if (tile.y <= context.maxY || Math.abs(tile.x - context.centerX) <= 1 || (tile.x + tile.y) % 2 === 0) {
        return "limestone_terrace";
      }
      return null;
    }

    if (this.isAdjacentToOccupied(context, tile)) {
      return "limestone_terrace";
    }

    if (context.dryEarthKeys.has(tileKey)) {
      if (tile.y === context.maxY + 3) {
        return Math.abs(tile.x - context.centerX) <= 1 ? "dry_earth" : null;
      }
      const sideAccent = (tile.x === context.minX - 2 || tile.x === context.maxX + 2) && tile.y <= context.maxY && tile.y % 2 === 0;
      const shoulderAccent = tile.y === context.maxY + 2 && Math.abs(tile.x - context.centerX) >= 2 && (tile.x + tile.y) % 3 === 0;
      return sideAccent || shoulderAccent ? "dry_earth" : null;
    }

    return null;
  }

  private defaultTerrainBoardStyle(tile: Tile) {
    const palette = tile.y < 16
      ? [0xd8ccaa, 0xdbcfb0, 0xddd2b5]
      : tile.y < 35
        ? [0xe3d8b4, 0xe7ddbd, 0xe9e1c4]
        : [0xe9e0c3, 0xeee6cd, 0xf2ebd5];
    return {
      color: palette[(tile.x + tile.y * 2) % palette.length],
      outlineAlpha: tile.y < 16 ? 0.045 : 0.018
    };
  }

  private terrainBoardStyle(context: PrecinctTerrainContext | null, tile: Tile) {
    const fallback = this.defaultTerrainBoardStyle(tile);
    if (!context) {
      return fallback;
    }

    const tileKey = this.terrainTileKey(tile);
    if (context.cliffKeys.has(tileKey)) {
      return { color: 0xc8b589, outlineAlpha: 0.08 };
    }
    if (context.springPoolKeys.has(tileKey)) {
      return { color: 0xcad7cf, outlineAlpha: 0.04 };
    }
    if (context.roadKeys.has(tileKey)) {
      return { color: 0xd7c496, outlineAlpha: 0.04 };
    }
    if (context.ritualCourtKeys.has(tileKey)) {
      const palette = [0xddcfaa, 0xe2d6b2, 0xd8c79f];
      return {
        color: palette[(tile.x + tile.y) % palette.length],
        outlineAlpha: 0.04
      };
    }
    if (context.approachShoulderKeys.has(tileKey)) {
      const palette = tile.y >= context.maxY + 2
        ? [0xd2c194, 0xd7c99d, 0xccb887]
        : [0xdcca9e, 0xe0d0a8, 0xd7c394];
      return {
        color: palette[(tile.x * 2 + tile.y) % palette.length],
        outlineAlpha: 0.04
      };
    }
    if (context.buildingKeys.has(tileKey) || context.terraceApronKeys.has(tileKey)) {
      const palette = [0xd8c9a4, 0xd3c29b, 0xcebb91];
      return {
        color: palette[(tile.x + tile.y) % palette.length],
        outlineAlpha: 0.045
      };
    }
    if (context.dryEarthKeys.has(tileKey)) {
      const palette = tile.y === context.maxY + 3
        ? [0xc6ad7c, 0xc0a26f, 0xc8b387]
        : [0xcfbc92, 0xc7b081, 0xd4c39c];
      return {
        color: palette[(tile.x * 2 + tile.y) % palette.length],
        outlineAlpha: 0.05
      };
    }

    const precinctDistance = this.precinctDistance(context, tile);
    if (precinctDistance >= 8) {
      return {
        color: tile.y < context.minY ? 0xe7ddbe : 0xefead2,
        outlineAlpha: 0.006
      };
    }
    if (precinctDistance >= 4) {
      const palette = tile.y <= context.maxY ? [0xe2d5b2, 0xe6dbc0] : [0xe8debf, 0xece4c9];
      return {
        color: palette[(tile.x + tile.y) % palette.length],
        outlineAlpha: 0.012
      };
    }

    const nearPrecinct = tile.x >= context.minX - 2 && tile.x <= context.maxX + 2 && tile.y >= context.minY - 1 && tile.y <= context.maxY + 3;
    if (nearPrecinct) {
      return {
        color: tile.y <= context.maxY + 1 ? 0xddcd9f : 0xe3d5ab,
        outlineAlpha: 0.05
      };
    }

    return fallback;
  }

  private terrainSpriteKey(tile: Tile) {
    return `${tile.x},${tile.y}`;
  }

  private floraSpriteKey(id: string) {
    return id;
  }

  private floraSpriteLayout(assetId: string, tile: Tile) {
    const art = getFloraArt(assetId);
    const screen = tileToScreen(tile);
    const visualWidth = art?.trimWidth || art?.width || 128;
    const visualHeight = art?.trimHeight || art?.height || 128;
    const scale = Math.min(
      (art?.worldWidth ?? 36) / visualWidth,
      (art?.worldHeight ?? 60) / visualHeight,
      assetId === "cypress_tree" ? 1.6 : 1.3
    );
    const offsetX = (art?.trimOffsetX ?? 0) * scale;
    const offsetY = (art?.trimBottomInset ?? 0) * scale;
    return {
      x: Math.round(screen.x + offsetX),
      y: Math.round(screen.y + FLORA_SPRITE_BASELINE_Y + offsetY),
      scale
    };
  }

  private propSpriteLayout(assetId: string, tile: Tile) {
    const art = getPropArt(assetId);
    const screen = tileToScreen(tile);
    const visualWidth = art?.trimWidth || art?.width || 128;
    const visualHeight = art?.trimHeight || art?.height || 128;
    const scale = Math.min(
      (art?.worldWidth ?? 36) / visualWidth,
      (art?.worldHeight ?? 32) / visualHeight,
      1.75
    );
    const offsetX = (art?.trimOffsetX ?? 0) * scale;
    const offsetY = (art?.trimBottomInset ?? 0) * scale;
    return {
      x: Math.round(screen.x + offsetX),
      y: Math.round(screen.y + PROP_SPRITE_BASELINE_Y + offsetY),
      scale
    };
  }

  private isAvailableDecorTile(state: GameState, occupiedKeys: Set<string>, usedKeys: Set<string>, tile: Tile) {
    if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
      return false;
    }
    const tileKey = this.terrainTileKey(tile);
    return !occupiedKeys.has(tileKey) && !usedKeys.has(tileKey);
  }

  private buildDecorPlacements(state: GameState): DecorPlacement[] {
    const occupiedKeys = new Set<string>();
    const usedKeys = new Set<string>();
    const placements: DecorPlacement[] = [];
    const buildingsByDefId = new Map<string, GameState["buildings"]>();
    const terrainContext = this.buildTerrainContext(state);

    for (const road of state.grid.roads) {
      occupiedKeys.add(this.terrainTileKey(road));
    }
    for (const building of state.buildings) {
      occupiedKeys.add(this.terrainTileKey(building.position));
      const existing = buildingsByDefId.get(building.defId) ?? [];
      existing.push(building);
      buildingsByDefId.set(building.defId, existing);
    }

    const reserveAt = (id: string, assetId: string, tile: Tile, alpha = 0.96, flipX = false) => {
      if (!this.isAvailableDecorTile(state, occupiedKeys, usedKeys, tile)) {
        return false;
      }
      usedKeys.add(this.terrainTileKey(tile));
      placements.push({ id, assetId, tile, alpha, flipX });
      return true;
    };

    const reserveNear = (id: string, assetId: string, origin: Tile, offsets: Tile[], alpha = 0.96, flipX = false) => {
      for (const offset of offsets) {
        if (reserveAt(id, assetId, { x: origin.x + offset.x, y: origin.y + offset.y }, alpha, flipX)) {
          return true;
        }
      }
      return false;
    };

    if (terrainContext) {
      reserveNear("approach:font:left", "purification_font", { x: terrainContext.centerX, y: terrainContext.maxY + 1 }, [
        { x: -2, y: 0 },
        { x: -3, y: 0 },
        { x: -2, y: 1 }
      ], 0.95, true);
      reserveNear("approach:font:right", "purification_font", { x: terrainContext.centerX, y: terrainContext.maxY + 1 }, [
        { x: 2, y: 0 },
        { x: 3, y: 0 },
        { x: 2, y: 1 }
      ], 0.95);
      reserveNear("approach:statue:left", "votive_statue_small", { x: terrainContext.centerX, y: terrainContext.maxY + 1 }, [
        { x: -3, y: 1 },
        { x: -4, y: 1 },
        { x: -3, y: 0 }
      ], 0.91, true);
      reserveNear("approach:statue:right", "votive_statue_small", { x: terrainContext.centerX, y: terrainContext.maxY + 1 }, [
        { x: 3, y: 1 },
        { x: 4, y: 1 },
        { x: 3, y: 0 }
      ], 0.91);
    }

    for (const spring of buildingsByDefId.get("castalian_spring") ?? []) {
      reserveNear(`${spring.id}:ritual_basin`, "ritual_basin", spring.position, [
        { x: 1, y: 1 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 1 }
      ], 0.98);
      reserveNear(`${spring.id}:votive_offering_rack`, "votive_offering_rack", spring.position, [
        { x: 1, y: 0 },
        { x: 1, y: -1 },
        { x: 0, y: -1 },
        { x: -1, y: 0 }
      ], 0.94, true);
      reserveNear(`${spring.id}:offering_bowl`, "offering_bowl", spring.position, [
        { x: -1, y: 1 },
        { x: 0, y: -1 },
        { x: -1, y: 0 },
        { x: 1, y: -1 }
      ], 0.96);
      reserveNear(`${spring.id}:votive_statue_small`, "votive_statue_small", spring.position, [
        { x: -2, y: 0 },
        { x: 0, y: -2 },
        { x: 2, y: 0 },
        { x: -2, y: 1 }
      ], 0.9, true);
    }

    for (const storehouse of buildingsByDefId.get("storehouse") ?? []) {
      reserveNear(`${storehouse.id}:amphora_stack`, "amphora_stack", storehouse.position, [
        { x: 0, y: -1 },
        { x: -1, y: -1 },
        { x: 1, y: -1 },
        { x: -1, y: 0 },
        { x: 1, y: 0 }
      ], 0.97);
      reserveNear(`${storehouse.id}:grain_sacks`, "grain_sacks", storehouse.position, [
        { x: -1, y: -1 },
        { x: 1, y: -1 },
        { x: 0, y: -2 },
        { x: -2, y: 0 }
      ], 0.97, true);
      reserveNear(`${storehouse.id}:oil_jars`, "oil_jars", storehouse.position, [
        { x: 2, y: 0 },
        { x: 2, y: -1 },
        { x: 1, y: -1 },
        { x: -1, y: -1 }
      ], 0.95);
    }

    for (const granary of buildingsByDefId.get("granary") ?? []) {
      reserveNear(`${granary.id}:amphora_stack`, "amphora_stack", granary.position, [
        { x: 1, y: -1 },
        { x: 0, y: -1 },
        { x: -1, y: -1 },
        { x: 1, y: 0 }
      ], 0.95);
      reserveNear(`${granary.id}:grain_sacks`, "grain_sacks", granary.position, [
        { x: -1, y: -1 },
        { x: 1, y: 0 },
        { x: -1, y: 0 },
        { x: 0, y: 1 }
      ], 0.98);
    }

    for (const sanctum of buildingsByDefId.get("inner_sanctum") ?? []) {
      reserveNear(`${sanctum.id}:omphalos_stone`, "omphalos_stone", sanctum.position, [
        { x: 0, y: 1 },
        { x: -1, y: 1 },
        { x: 1, y: 1 },
        { x: -1, y: 0 }
      ], 0.98);
      reserveNear(`${sanctum.id}:bronze_tripod`, "bronze_tripod", sanctum.position, [
        { x: 1, y: 1 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 1 }
      ], 0.96);
      reserveNear(`${sanctum.id}:votive_offering_rack`, "votive_offering_rack", sanctum.position, [
        { x: -1, y: 1 },
        { x: -1, y: 0 },
        { x: 0, y: -1 },
        { x: 1, y: -1 }
      ], 0.94, true);
      reserveNear(`${sanctum.id}:incense_censer`, "incense_censer", sanctum.position, [
        { x: 1, y: -1 },
        { x: -1, y: 1 },
        { x: 0, y: 1 },
        { x: 1, y: 0 }
      ], 0.95);
      reserveNear(`${sanctum.id}:votive_statue_small`, "votive_statue_small", sanctum.position, [
        { x: 0, y: -1 },
        { x: -1, y: 0 },
        { x: 1, y: -1 },
        { x: -1, y: 1 }
      ], 0.94);
    }

    for (const altar of buildingsByDefId.get("sacrificial_altar") ?? []) {
      reserveNear(`${altar.id}:bronze_tripod`, "bronze_tripod", altar.position, [
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
        { x: -1, y: 1 }
      ], 0.94);
      reserveNear(`${altar.id}:offering_bowl`, "offering_bowl", altar.position, [
        { x: -1, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 1 },
        { x: 1, y: 1 }
      ], 0.97);
      reserveNear(`${altar.id}:incense_censer`, "incense_censer", altar.position, [
        { x: 1, y: -1 },
        { x: 1, y: 0 },
        { x: 0, y: -1 },
        { x: -1, y: 0 }
      ], 0.95);
    }

    for (const incenseStore of buildingsByDefId.get("incense_store") ?? []) {
      reserveNear(`${incenseStore.id}:incense_censer`, "incense_censer", incenseStore.position, [
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 0 },
        { x: 1, y: 1 }
      ], 0.94);
    }

    for (const olivePress of buildingsByDefId.get("olive_press") ?? []) {
      reserveNear(`${olivePress.id}:oil_jars`, "oil_jars", olivePress.position, [
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
        { x: -1, y: 0 }
      ], 0.96);
    }

    for (const quarters of buildingsByDefId.get("priest_quarters") ?? []) {
      reserveNear(`${quarters.id}:stone_bench`, "stone_bench", quarters.position, [
        { x: -1, y: 1 },
        { x: -1, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 1 }
      ], 0.93, true);
      reserveNear(`${quarters.id}:purification_font`, "purification_font", quarters.position, [
        { x: 1, y: 1 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 1 }
      ], 0.94);
    }

    for (const kitchen of buildingsByDefId.get("kitchen") ?? []) {
      reserveNear(`${kitchen.id}:grain_sacks`, "grain_sacks", kitchen.position, [
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
        { x: -1, y: 0 }
      ], 0.95);
      reserveNear(`${kitchen.id}:amphora_stack`, "amphora_stack", kitchen.position, [
        { x: -1, y: 0 },
        { x: -1, y: 1 },
        { x: 0, y: 1 },
        { x: 1, y: 0 }
      ], 0.94, true);
    }

    for (const xenon of buildingsByDefId.get("xenon") ?? []) {
      reserveNear(`${xenon.id}:stone_bench`, "stone_bench", xenon.position, [
        { x: -1, y: 1 },
        { x: 1, y: 1 },
        { x: -1, y: 0 },
        { x: 1, y: 0 }
      ], 0.92, xenon.position.x % 2 === 0);
      reserveNear(`${xenon.id}:amphora_stack`, "amphora_stack", xenon.position, [
        { x: 1, y: 0 },
        { x: -1, y: 0 },
        { x: 1, y: 1 },
        { x: -1, y: 1 }
      ], 0.93);
    }

    for (const market of buildingsByDefId.get("agora_market") ?? []) {
      reserveNear(`${market.id}:market_stall_set`, "market_stall_set", market.position, [
        { x: 1, y: 1 },
        { x: 0, y: 1 },
        { x: 1, y: 0 },
        { x: -1, y: 1 }
      ], 0.97, (market.position.x + market.position.y) % 2 === 0);
    }

    for (const gatehouse of buildingsByDefId.get("gatehouse_entrance") ?? []) {
      reserveNear(`${gatehouse.id}:votive_statue_small`, "votive_statue_small", gatehouse.position, [
        { x: -1, y: 1 },
        { x: 1, y: 1 },
        { x: -1, y: 0 },
        { x: 1, y: 0 }
      ], 0.92, gatehouse.position.x % 2 === 0);
    }

    [...state.grid.roads]
      .sort((left, right) => left.x - right.x || left.y - right.y)
      .forEach((road, index) => {
        if (index % 4 === 0) {
          reserveNear(`road:${road.x},${road.y}:votive_statue_small`, "votive_statue_small", road, [
            { x: 1, y: -1 },
            { x: -1, y: 1 },
            { x: -1, y: -1 },
            { x: 1, y: 1 }
          ], 0.9, index % 8 === 0);
        }
        if (index % 3 !== 1) {
          return;
        }
        const preferSouth = index % 2 === 0;
        const offsets = preferSouth
          ? [{ x: 0, y: 1 }, { x: 0, y: -1 }, { x: 1, y: 0 }, { x: -1, y: 0 }]
          : [{ x: 0, y: -1 }, { x: 0, y: 1 }, { x: -1, y: 0 }, { x: 1, y: 0 }];
        reserveNear(`road:${road.x},${road.y}:stone_bench`, "stone_bench", road, offsets, 0.93, !preferSouth);
      });

    return placements;
  }

  private buildFloraPlacements(state: GameState, decorPlacements: DecorPlacement[]): FloraPlacement[] {
    const context = this.buildTerrainContext(state);
    if (!context) {
      return [];
    }

    const placements: FloraPlacement[] = [];
    const blockedKeys = new Set<string>([
      ...context.occupiedKeys,
      ...decorPlacements.map((placement) => this.terrainTileKey(placement.tile))
    ]);
    for (let x = context.centerX - 1; x <= context.centerX + 1; x += 1) {
      for (let y = context.maxY + 1; y <= context.maxY + 3; y += 1) {
        blockedKeys.add(this.terrainTileKey({ x, y }));
      }
    }
    const usedKeys = new Set<string>();
    const reserveAt = (id: string, assetId: string, tile: Tile, alpha = 0.94, flipX = false) => {
      if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
        return false;
      }
      const tileKey = this.terrainTileKey(tile);
      if (blockedKeys.has(tileKey) || usedKeys.has(tileKey)) {
        return false;
      }
      usedKeys.add(tileKey);
      placements.push({ id, assetId, tile, alpha, flipX });
      return true;
    };
    const reserveFromCandidates = (id: string, assetId: string, candidates: Tile[], alpha = 0.94, flipX = false) => {
      for (const candidate of candidates) {
        if (reserveAt(id, assetId, candidate, alpha, flipX)) {
          return true;
        }
      }
      return false;
    };
    const uniqueCandidates = (tiles: Tile[]) => {
      const seen = new Set<string>();
      return tiles.filter((tile) => {
        const tileKey = this.terrainTileKey(tile);
        if (seen.has(tileKey)) {
          return false;
        }
        seen.add(tileKey);
        return true;
      });
    };

    const perimeterTiles = uniqueCandidates([
      ...context.cliffTiles,
      ...Array.from({ length: context.maxX - context.minX + 5 }, (_unused, index) => ({
        x: context.minX - 2 + index,
        y: context.maxY + 2
      })),
      ...Array.from({ length: context.maxY - context.minY + 3 }, (_unused, index) => ({
        x: context.minX - 2,
        y: context.minY + index
      })),
      ...Array.from({ length: context.maxY - context.minY + 3 }, (_unused, index) => ({
        x: context.maxX + 2,
        y: context.minY + index
      }))
    ]);
    const nearPerimeter = uniqueCandidates(
      perimeterTiles.flatMap((tile) => [
        tile,
        { x: tile.x, y: tile.y + 1 },
        { x: tile.x + 1, y: tile.y },
        { x: tile.x - 1, y: tile.y }
      ])
    );
    const cliffAnchors = uniqueCandidates([
      { x: context.minX - 1, y: context.minY - 2 },
      { x: context.centerX - 2, y: context.minY - 2 },
      { x: context.centerX + 2, y: context.minY - 2 },
      { x: context.maxX + 1, y: context.minY - 2 }
    ]);
    const approachCypressAnchors = uniqueCandidates([
      { x: context.centerX - 4, y: context.maxY + 2 },
      { x: context.centerX + 4, y: context.maxY + 2 }
    ]);
    const ritualCypressAnchors = uniqueCandidates([
      ...state.buildings
        .filter((building) => buildingDefs[building.defId].category === "ritual")
        .flatMap((building) => [
          { x: building.position.x - 3, y: building.position.y },
          { x: building.position.x + 3, y: building.position.y + 1 }
        ])
    ]).slice(0, 4);

    approachCypressAnchors.forEach((tile, index) => {
      reserveFromCandidates(
        `flora:cypress:approach:${index}`,
        "cypress_tree",
        [
          tile,
          { x: tile.x, y: tile.y + 1 },
          { x: tile.x + (index === 0 ? -1 : 1), y: tile.y }
        ],
        0.95,
        index === 1
      );
    });

    cliffAnchors.forEach((tile, index) => {
      reserveFromCandidates(
        `flora:cypress:${index}`,
        "cypress_tree",
        [
          tile,
          { x: tile.x, y: tile.y + 1 },
          { x: tile.x + (index % 2 === 0 ? 1 : -1), y: tile.y + 1 }
        ],
        0.94,
        index % 2 === 1
      );
    });

    ritualCypressAnchors.forEach((tile, index) => {
      reserveFromCandidates(
        `flora:cypress:ritual:${index}`,
        "cypress_tree",
        [
          tile,
          { x: tile.x, y: tile.y + 1 },
          { x: tile.x + (index % 2 === 0 ? -1 : 1), y: tile.y + 1 }
        ],
        0.9,
        index % 2 === 1
      );
    });

    [
      { id: "flora:rock:left", tile: { x: context.minX - 2, y: context.minY - 1 } },
      { id: "flora:rock:right", tile: { x: context.maxX + 2, y: context.minY - 1 } },
      { id: "flora:rock:apron:left", tile: { x: context.centerX - 4, y: context.maxY + 3 } },
      { id: "flora:rock:apron:right", tile: { x: context.centerX + 4, y: context.maxY + 3 } }
    ].forEach(({ id, tile }, index) => {
      reserveFromCandidates(
        id,
        "rocky_outcrop",
        [
          tile,
          { x: tile.x, y: tile.y + 1 },
          { x: tile.x + (index === 1 ? -1 : 1), y: tile.y }
        ],
        0.9,
        index === 1
      );
    });

    const shrineShrubAnchors = uniqueCandidates([
      ...state.buildings
        .filter((building) => building.defId === "castalian_spring" || building.defId === "inner_sanctum")
        .flatMap((building) => [
          { x: building.position.x - 2, y: building.position.y + 1 },
          { x: building.position.x + 2, y: building.position.y + 1 },
          { x: building.position.x - 1, y: building.position.y + 2 },
          { x: building.position.x + 1, y: building.position.y + 2 }
        ])
    ]).slice(0, 4);
    const perimeterShrubAnchors = nearPerimeter.filter((_tile, index) => index % 7 === 2).slice(0, 6);
    const shrubAnchors = uniqueCandidates([...shrineShrubAnchors, ...perimeterShrubAnchors]);
    shrubAnchors.forEach((tile, index) => {
      reserveFromCandidates(
        `flora:shrub:${index}`,
        "laurel_shrub",
        [
          tile,
          { x: tile.x, y: tile.y + 1 },
          { x: tile.x + (index % 2 === 0 ? 1 : -1), y: tile.y }
        ],
        0.92,
        index % 2 === 1
      );
    });

    uniqueCandidates(nearPerimeter.filter((_tile, index) => index % 4 === 0)).slice(0, 8).forEach((tile, index) => {
      reserveFromCandidates(
        `flora:grass:${index}`,
        "dry_grass_cluster",
        [
          tile,
          { x: tile.x + 1, y: tile.y },
          { x: tile.x, y: tile.y - 1 }
        ],
        0.86,
        index % 3 === 0
      );
    });

    return placements;
  }

  private terrainSpriteLayout(terrainId: string, tile: Tile, art: TerrainArtAsset | null) {
    const screen = tileToScreen(tile);
    const visualWidth = art?.trimWidth || art?.width || 256;
    const visualHeight = art?.trimHeight || art?.height || 192;
    const bounds = terrainId === "retaining_wall"
      ? { maxWidth: 92, maxHeight: 32, baselineY: 10 }
      : terrainId === "stone_stairs"
        ? { maxWidth: 122, maxHeight: 72, baselineY: 22 }
        : terrainId === "cliff_edge"
          ? { maxWidth: 128, maxHeight: 44, baselineY: 8 }
          : terrainId === "spring_pool"
            ? { maxWidth: 92, maxHeight: 56, baselineY: 18 }
        : terrainId === "dry_earth"
          ? { maxWidth: 104, maxHeight: 58, baselineY: 18 }
        : terrainId === "sacred_paving"
          ? { maxWidth: 74, maxHeight: 42, baselineY: 18 }
          : { maxWidth: 86, maxHeight: 48, baselineY: TERRAIN_SPRITE_BASELINE_Y };
    const scale = Math.min(bounds.maxWidth / visualWidth, bounds.maxHeight / visualHeight, 1.15);
    const offsetX = (art?.trimOffsetX ?? 0) * scale;
    const offsetY = (art?.trimBottomInset ?? 0) * scale;
    return {
      x: Math.round(screen.x + offsetX),
      y: Math.round(screen.y + bounds.baselineY + offsetY),
      scale
    };
  }

  private terrainSpriteDepth(terrainId: string) {
    if (terrainId === "cliff_edge") {
      return 1;
    }
    if (terrainId === "retaining_wall" || terrainId === "stone_stairs") {
      return 4;
    }
    if (terrainId === "spring_pool") {
      return 2;
    }
    return 2;
  }

  private terrainSpriteAlpha(terrainId: string) {
    if (terrainId === "dry_earth") {
      return 0.62;
    }
    if (terrainId === "sacred_paving") {
      return 0.74;
    }
    if (terrainId === "limestone_terrace") {
      return 0.84;
    }
    if (terrainId === "cliff_edge") {
      return 0.94;
    }
    if (terrainId === "spring_pool") {
      return 0.96;
    }
    return 0.9;
  }

  private roadSpriteLayout(tile: Tile) {
    const art = getBuildingArt(ROAD_ART_DEF_ID);
    const screen = tileToScreen(tile);
    const visualWidth = art?.trimWidth || art?.width || 96;
    const visualHeight = art?.trimHeight || art?.height || 96;
    const scale = Math.min(58 / visualWidth, 40 / visualHeight, 1.5);
    const offsetX = (art?.trimOffsetX ?? 0) * scale;
    const offsetY = (art?.trimBottomInset ?? 0) * scale;
    return {
      x: Math.round(screen.x + offsetX),
      y: Math.round(screen.y + ROAD_SPRITE_BASELINE_Y + offsetY),
      scale
    };
  }

  private buildingSpriteLayout(defId: string, tile: Tile) {
    const art = getBuildingArt(defId);
    const screen = tileToScreen(tile);
    const scale = this.buildingSpriteScale(defId);
    const offsetX = (art?.trimOffsetX ?? 0) * scale;
    const offsetY = (art?.trimBottomInset ?? 0) * scale;
    return {
      x: Math.round(screen.x + offsetX),
      y: Math.round(screen.y + BUILDING_SPRITE_BASELINE_Y + offsetY),
      scale
    };
  }

  private buildingSpriteScale(defId: string) {
    const art = getBuildingArt(defId);
    if (!art) {
      return 1;
    }
    const bounds = spriteBoundsForFootprint(art.footprint);
    const visualWidth = art.trimWidth || art.width;
    const visualHeight = art.trimHeight || art.height;
    return Math.min(bounds.maxWidth / visualWidth, bounds.maxHeight / visualHeight, 4.1);
  }

  private walkerSpriteLayout(walkerRole: string, tile: Tile) {
    const art = getWalkerArt(walkerRole);
    const screen = tileToScreen(tile);
    const visualWidth = art?.trimWidth || art?.width || 220;
    const visualHeight = art?.trimHeight || art?.height || 760;
    const bounds = walkerRole === "pilgrim"
      ? { maxWidth: 40, maxHeight: 76 }
      : walkerRole === "carrier"
        ? { maxWidth: 50, maxHeight: 94 }
        : { maxWidth: 44, maxHeight: 88 };
    const scale = Math.min(bounds.maxWidth / visualWidth, bounds.maxHeight / visualHeight, 0.2);
    const offsetX = (art?.trimOffsetX ?? 0) * scale;
    const offsetY = (art?.trimBottomInset ?? 0) * scale;
    return {
      x: Math.round(screen.x + offsetX),
      y: Math.round(screen.y + WALKER_SPRITE_BASELINE_Y + offsetY),
      scale
    };
  }

  private reconcileTerrainSprites(state: GameState) {
    const context = this.buildTerrainContext(state);
    if (!context) {
      for (const [spriteId, sprite] of this.terrainSprites.entries()) {
        sprite.destroy();
        this.terrainSprites.delete(spriteId);
      }
      return;
    }

    const candidateTiles = new Map<string, Tile>();
    const considerTile = (tile: Tile) => {
      if (tile.x < 0 || tile.x >= state.grid.width || tile.y < 0 || tile.y >= state.grid.height) {
        return;
      }
      candidateTiles.set(this.terrainSpriteKey(tile), tile);
    };

    for (const road of state.grid.roads) {
      considerTile(road);
      considerTile({ x: road.x + 1, y: road.y });
      considerTile({ x: road.x - 1, y: road.y });
      considerTile({ x: road.x, y: road.y + 1 });
      considerTile({ x: road.x, y: road.y - 1 });
    }

    for (const building of state.buildings) {
      considerTile(building.position);
      considerTile({ x: building.position.x + 1, y: building.position.y });
      considerTile({ x: building.position.x - 1, y: building.position.y });
      considerTile({ x: building.position.x, y: building.position.y + 1 });
      considerTile({ x: building.position.x, y: building.position.y - 1 });
    }

    for (let x = context.minX - 1; x <= context.maxX + 1; x += 1) {
      considerTile({ x, y: context.minY - 1 });
    }
    for (let y = context.minY - 1; y <= context.maxY; y += 1) {
      considerTile({ x: context.minX - 1, y });
    }
    considerTile({ x: context.centerX, y: context.maxY + 1 });
    for (const tile of context.springPoolTiles) {
      considerTile(tile);
    }
    for (const tile of context.ritualCourtTiles) {
      considerTile(tile);
    }
    for (const tile of context.approachShoulderTiles) {
      considerTile(tile);
    }
    for (const tile of context.cliffTiles) {
      considerTile(tile);
    }
    for (const tile of context.terraceApronTiles) {
      considerTile(tile);
    }
    for (const tile of context.dryEarthTiles) {
      considerTile(tile);
    }

    const activeTerrainIds = new Set<string>();
    for (const tile of candidateTiles.values()) {
      const terrainId = this.terrainTextureIdForTile(context, tile);
      if (!terrainId) {
        continue;
      }

      const terrainSelection = this.getTerrainTextureSelection(terrainId, tile);
      if (!terrainSelection) {
        continue;
      }
      const { art, textureKey } = terrainSelection;

      const spriteId = this.terrainSpriteKey(tile);
      const { x, y, scale } = this.terrainSpriteLayout(terrainId, tile, art);
      activeTerrainIds.add(spriteId);
      let sprite = this.terrainSprites.get(spriteId);

      if (!sprite || sprite.texture.key !== textureKey) {
        sprite?.destroy();
        sprite = this.add.image(x, y, textureKey)
          .setOrigin(0.5, 1)
          .setDepth(this.terrainSpriteDepth(terrainId))
          .setAlpha(this.terrainSpriteAlpha(terrainId));
        this.terrainSprites.set(spriteId, sprite);
      }

      sprite
        .setPosition(x, y)
        .setDepth(this.terrainSpriteDepth(terrainId))
        .setAlpha(this.terrainSpriteAlpha(terrainId))
        .setScale(scale)
        .setVisible(true);
    }

    for (const [spriteId, sprite] of this.terrainSprites.entries()) {
      if (activeTerrainIds.has(spriteId)) {
        continue;
      }
      sprite.destroy();
      this.terrainSprites.delete(spriteId);
    }
  }

  private reconcileBuildingSprites(state: GameState) {
    const activeBuildingIds = new Set<string>();

    [...state.buildings]
      .sort((left, right) => left.position.y - right.position.y || left.position.x - right.position.x)
      .forEach((building) => {
        const textureKey = this.getBuildingTextureKey(building.defId);
        if (!textureKey) {
          return;
        }

        activeBuildingIds.add(building.id);
        const { x, y, scale } = this.buildingSpriteLayout(building.defId, building.position);
        const depth = this.buildingSpriteDepth(building.position);
        let sprite = this.buildingSprites.get(building.id);

        if (!sprite || sprite.texture.key !== textureKey) {
          sprite?.destroy();
          sprite = this.add.image(x, y, textureKey)
            .setOrigin(0.5, 1)
            .setDepth(depth);
          this.buildingSprites.set(building.id, sprite);
        }

        sprite
          .setPosition(x, y)
          .setDepth(depth)
          .setScale(scale)
          .setVisible(true);
      });

    for (const [buildingId, sprite] of this.buildingSprites.entries()) {
      if (activeBuildingIds.has(buildingId)) {
        continue;
      }
      sprite.destroy();
      this.buildingSprites.delete(buildingId);
    }
  }

  private reconcileFloraSprites(placements: FloraPlacement[]) {
    const activeFloraIds = new Set<string>();

    [...placements]
      .sort((left, right) => left.tile.y - right.tile.y || left.tile.x - right.tile.x || left.id.localeCompare(right.id))
      .forEach((placement) => {
        const textureKey = this.getFloraTextureKey(placement.assetId);
        if (!textureKey) {
          return;
        }

        activeFloraIds.add(this.floraSpriteKey(placement.id));
        const { x, y, scale } = this.floraSpriteLayout(placement.assetId, placement.tile);
        const depth = this.floraSpriteDepth(placement.tile);
        let sprite = this.floraSprites.get(placement.id);

        if (!sprite || sprite.texture.key !== textureKey) {
          sprite?.destroy();
          sprite = this.add.image(x, y, textureKey)
            .setOrigin(0.5, 1)
            .setDepth(depth);
          this.floraSprites.set(placement.id, sprite);
        }

        sprite
          .setPosition(x, y)
          .setDepth(depth)
          .setScale(placement.flipX ? -scale : scale, scale)
          .setAlpha(placement.alpha ?? 0.94)
          .setVisible(true);
      });

    for (const [floraId, sprite] of this.floraSprites.entries()) {
      if (activeFloraIds.has(floraId)) {
        continue;
      }
      sprite.destroy();
      this.floraSprites.delete(floraId);
    }
  }

  private reconcilePropSprites(placements: DecorPlacement[]) {
    const activePropIds = new Set<string>();

    [...placements]
      .sort((left, right) => left.tile.y - right.tile.y || left.tile.x - right.tile.x || left.id.localeCompare(right.id))
      .forEach((placement) => {
        const textureKey = this.getPropTextureKey(placement.assetId);
        if (!textureKey) {
          return;
        }

        activePropIds.add(placement.id);
        const { x, y, scale } = this.propSpriteLayout(placement.assetId, placement.tile);
        const depth = this.propSpriteDepth(placement.tile);
        let sprite = this.propSprites.get(placement.id);

        if (!sprite || sprite.texture.key !== textureKey) {
          sprite?.destroy();
          sprite = this.add.image(x, y, textureKey)
            .setOrigin(0.5, 1)
            .setDepth(depth);
          this.propSprites.set(placement.id, sprite);
        }

        sprite
          .setPosition(x, y)
          .setDepth(depth)
          .setScale(placement.flipX ? -scale : scale, scale)
          .setAlpha(placement.alpha ?? 0.96)
          .setVisible(true);
      });

    for (const [propId, sprite] of this.propSprites.entries()) {
      if (activePropIds.has(propId)) {
        continue;
      }
      sprite.destroy();
      this.propSprites.delete(propId);
    }
  }

  private reconcileRoadSprites(state: GameState) {
    const textureKey = this.getBuildingTextureKey(ROAD_ART_DEF_ID);
    if (!textureKey) {
      for (const [roadId, sprite] of this.roadSprites.entries()) {
        sprite.destroy();
        this.roadSprites.delete(roadId);
      }
      return;
    }

    const activeRoadIds = new Set<string>();
    for (const road of state.grid.roads) {
      const roadId = this.roadSpriteKey(road);
      const { x, y, scale } = this.roadSpriteLayout(road);
      const depth = this.roadSpriteDepth(road);
      activeRoadIds.add(roadId);
      let sprite = this.roadSprites.get(roadId);

      if (!sprite || sprite.texture.key !== textureKey) {
        sprite?.destroy();
        sprite = this.add.image(x, y, textureKey)
          .setOrigin(0.5, 1)
          .setDepth(depth)
          .setAlpha(0.96);
        this.roadSprites.set(roadId, sprite);
      }

      sprite
        .setPosition(x, y)
        .setDepth(depth)
        .setScale(scale)
        .setVisible(true);
    }

    for (const [roadId, sprite] of this.roadSprites.entries()) {
      if (activeRoadIds.has(roadId)) {
        continue;
      }
      sprite.destroy();
      this.roadSprites.delete(roadId);
    }
  }

  private reconcileWalkerSprites(state: GameState) {
    const activeWalkerIds = new Set<string>();

    [...state.walkers]
      .sort((left, right) => left.tile.y - right.tile.y || left.tile.x - right.tile.x)
      .forEach((walker) => {
        const textureKey = this.getWalkerTextureKey(walker.role);
        if (!textureKey) {
          return;
        }

        activeWalkerIds.add(walker.id);
        const { x, y, scale } = this.walkerSpriteLayout(walker.role, walker.tile);
        const depth = this.walkerSpriteDepth(walker.tile);
        let sprite = this.walkerSprites.get(walker.id);

        if (!sprite || sprite.texture.key !== textureKey) {
          sprite?.destroy();
          sprite = this.add.image(x, y, textureKey)
            .setOrigin(0.5, 1)
            .setDepth(depth);
          this.walkerSprites.set(walker.id, sprite);
        }

        sprite
          .setPosition(x, y)
          .setDepth(depth)
          .setScale(scale)
          .setAlpha(walker.state === "hauling" || walker.state === "delivering" ? 1 : 0.96)
          .setVisible(true);
      });

    for (const [walkerId, sprite] of this.walkerSprites.entries()) {
      if (activeWalkerIds.has(walkerId)) {
        continue;
      }
      sprite.destroy();
      this.walkerSprites.delete(walkerId);
    }
  }

  private framePrecinct(state: GameState, extraTiles: Tile[] = []) {
    if (!this.autoFrameActive) {
      return;
    }

    const camera = this.cameras.main;
    const terrainContext = this.buildTerrainContext(state);
    const occupiedTiles = terrainContext
      ? [
          ...state.grid.roads,
          ...state.buildings.map((building) => building.position),
          { x: terrainContext.minX - 2, y: terrainContext.minY - 2 },
          { x: terrainContext.maxX + 2, y: terrainContext.minY - 1 },
          { x: terrainContext.minX - 3, y: terrainContext.maxY + 2 },
          { x: terrainContext.maxX + 3, y: terrainContext.maxY + 3 },
          { x: terrainContext.centerX, y: terrainContext.maxY + 3 },
          ...extraTiles
        ]
      : [
          ...state.grid.roads,
          ...state.buildings.map((building) => building.position),
          ...extraTiles
        ];

    if (occupiedTiles.length === 0) {
      const focus = tileToScreen({ x: 29, y: 48 });
      camera.setZoom(2.25);
      camera.centerOn(focus.x, focus.y);
      return;
    }

    let minScreenX = Number.POSITIVE_INFINITY;
    let maxScreenX = Number.NEGATIVE_INFINITY;
    let minScreenY = Number.POSITIVE_INFINITY;
    let maxScreenY = Number.NEGATIVE_INFINITY;

    for (const tile of occupiedTiles) {
      const screen = tileToScreen(tile);
      minScreenX = Math.min(minScreenX, screen.x - TILE_WIDTH / 2);
      maxScreenX = Math.max(maxScreenX, screen.x + TILE_WIDTH / 2);
      minScreenY = Math.min(minScreenY, screen.y - 64);
      maxScreenY = Math.max(maxScreenY, screen.y + TILE_HEIGHT + 92);
    }

    const spanX = Math.max(220, maxScreenX - minScreenX);
    const spanY = Math.max(180, maxScreenY - minScreenY);
    const availableWidth = Math.max(760, camera.width - 40);
    const availableHeight = Math.max(560, camera.height - 28);
    const targetZoom = Phaser.Math.Clamp(
      Math.min(availableWidth / spanX, availableHeight / spanY),
      2.85,
      4.4
    );
    const focusX = (minScreenX + maxScreenX) / 2 - 88 / targetZoom;
    const focusY = (minScreenY + maxScreenY) / 2 + 18 / targetZoom;

    camera.setZoom(targetZoom);
    camera.centerOn(focusX, focusY);
  }

  private drawState(state: GameState) {
    this.terrainGraphics.clear();
    this.roadGraphics.clear();
    this.floraGraphics.clear();
    this.propGraphics.clear();
    this.buildingGraphics.clear();
    this.walkerGraphics.clear();
    this.overlayGraphics.clear();
    const decorPlacements = this.buildDecorPlacements(state);
    const floraPlacements = this.buildFloraPlacements(state, decorPlacements);
    const shouldFrameScenery = state.grid.roads.length > 0 || state.buildings.length > 0;
    this.framePrecinct(
      state,
      shouldFrameScenery
        ? [
            ...decorPlacements.map((placement) => placement.tile),
            ...floraPlacements
              .filter((placement) => placement.id.includes("approach") || placement.id.includes("rock:apron"))
              .map((placement) => placement.tile)
          ]
        : []
    );
    this.reconcileTerrainSprites(state);
    this.reconcileRoadSprites(state);
    this.reconcileFloraSprites(floraPlacements);
    this.reconcilePropSprites(decorPlacements);
    this.reconcileBuildingSprites(state);
    this.reconcileWalkerSprites(state);
    const terrainContext = this.buildTerrainContext(state);
    const selectedBuilding = state.ui.selectedEntityKind === "building"
      ? state.buildings.find((building) => building.id === state.ui.selectedEntityId)
      : undefined;
    const selectedWalker = state.ui.selectedEntityKind === "walker"
      ? state.walkers.find((walker) => walker.id === state.ui.selectedEntityId)
      : undefined;
    const focusTile = selectedBuilding?.position ?? selectedWalker?.tile ?? state.ui.hoveredTile;
    const time = this.time.now;
    const springPulse = 0.5 + 0.5 * Math.sin(time * 0.004);
    const emberPulse = 0.5 + 0.5 * Math.sin(time * 0.006 + 0.8);
    const ritualPulse = 0.5 + 0.5 * Math.sin(time * 0.0032 + 1.4);
    const ritualBuildings = state.buildings.filter((building) => buildingDefs[building.defId].category === "ritual");

    if (terrainContext) {
      const apronCenter = tileToScreen({ x: terrainContext.centerX, y: terrainContext.maxY + 1 });
      const shadowWidth = Math.max(240, (terrainContext.maxX - terrainContext.minX + 6) * 42);
      const shadowHeight = Math.max(110, (terrainContext.maxY - terrainContext.minY + 5) * 24);
      this.terrainGraphics.fillStyle(0x6f5732, 0.06);
      this.terrainGraphics.fillEllipse(apronCenter.x, apronCenter.y + 22, shadowWidth, shadowHeight);
    }

    if (ritualBuildings.length > 0) {
      const ritualCenter = {
        x: average(ritualBuildings.map((building) => building.position.x)),
        y: average(ritualBuildings.map((building) => building.position.y))
      };
      const ritualCenterScreen = tileToScreen(ritualCenter);
      this.roadGraphics.fillStyle(0xf0da95, 0.018 + ritualPulse * 0.018);
      this.roadGraphics.fillEllipse(ritualCenterScreen.x, ritualCenterScreen.y - 12, 182, 82);
      for (let index = 0; index < 6; index += 1) {
        const drift = ((time * 0.00022) + index * 0.16) % 1;
        const moteX = ritualCenterScreen.x + Math.sin(time * 0.0016 + index * 1.3) * (18 + index * 4);
        const moteY = ritualCenterScreen.y - 52 - drift * 30 + Math.cos(time * 0.001 + index) * 6;
        this.overlayGraphics.fillStyle(0xf6e8bc, 0.02 + (1 - drift) * 0.06);
        this.overlayGraphics.fillCircle(moteX, moteY, 1.1 + (1 - drift) * 0.9);
      }
    }

    for (let y = 0; y < state.grid.height; y += 1) {
      for (let x = 0; x < state.grid.width; x += 1) {
        const tile = { x, y };
        const screen = tileToScreen(tile);
        const boardStyle = this.terrainBoardStyle(terrainContext, tile);
        this.drawTerrainDiamond(screen.x, screen.y, boardStyle.color, boardStyle.outlineAlpha);
      }
    }

    state.grid.roads.forEach((road) => {
      if (this.roadSprites.has(this.roadSpriteKey(road))) {
        return;
      }
      const screen = tileToScreen(road);
      this.drawDiamond(this.roadGraphics, screen.x, screen.y, 0xb18b5c, 0.95);
    });

    floraPlacements.forEach((placement) => {
      if (!this.floraSprites.has(placement.id)) {
        return;
      }
      const screen = tileToScreen(placement.tile);
      const shadowWidth = placement.assetId === "cypress_tree"
        ? 28
        : placement.assetId === "rocky_outcrop"
          ? 26
          : placement.assetId === "laurel_shrub"
            ? 24
            : 16;
      const shadowHeight = placement.assetId === "cypress_tree" ? 10 : 8;
      this.floraGraphics.fillStyle(0x24180c, placement.assetId === "dry_grass_cluster" ? 0.06 : 0.11);
      this.floraGraphics.fillEllipse(screen.x, screen.y + 15, shadowWidth, shadowHeight);
    });

    decorPlacements.forEach((placement) => {
      if (!this.propSprites.has(placement.id)) {
        return;
      }
      const screen = tileToScreen(placement.tile);
      const shadowWidth = placement.assetId === "market_stall_set"
        ? 28
        : placement.assetId === "stone_bench"
          ? 24
          : 20;
      const shadowHeight = placement.assetId === "market_stall_set" ? 10 : 8;
      this.propGraphics.fillStyle(0x24180c, 0.1);
      this.propGraphics.fillEllipse(screen.x, screen.y + 16, shadowWidth, shadowHeight);
    });

    [...state.buildings]
      .sort((left, right) => left.position.y - right.position.y || left.position.x - right.position.x)
      .forEach((building) => {
        const screen = tileToScreen(building.position);
        const hasSprite = this.buildingSprites.has(building.id);
        const color = buildingDefs[building.defId].color;
        const hovered = state.ui.hoveredTile?.x === building.position.x && state.ui.hoveredTile?.y === building.position.y;
        const selected = selectedBuilding?.id === building.id;
        const conditionRatio = Math.max(0, Math.min(1, building.condition / Math.max(1, building.maxCondition)));
        const conditionColor = conditionRatio > 0.72 ? 0x4da06e : conditionRatio > 0.4 ? 0xc59e45 : 0xc45540;
        const springPoolTile = building.defId === "castalian_spring"
          ? terrainContext?.springPoolTiles.find((tile) => manhattan(tile, building.position) <= 2)
          : undefined;

        if (hasSprite) {
          this.buildingGraphics.fillStyle(0x24180c, 0.08);
          this.buildingGraphics.fillEllipse(screen.x, screen.y + 18, 34, 12);
        } else {
          this.buildingGraphics.fillStyle(0x24180c, 0.16);
          this.buildingGraphics.fillEllipse(screen.x, screen.y + 18, 30, 10);
          this.drawDiamond(this.buildingGraphics, screen.x, screen.y - 12, color, 1);
          this.buildingGraphics.fillStyle(0x2b2417, 0.18);
          this.buildingGraphics.fillRect(screen.x - 18, screen.y - 18, 36, 14);
        }

        if (springPoolTile) {
          const springScreen = tileToScreen(springPoolTile);
          this.roadGraphics.fillStyle(0x77d8ee, 0.045 + springPulse * 0.055);
          this.roadGraphics.fillEllipse(springScreen.x, springScreen.y + 15, 34 + springPulse * 10, 16 + springPulse * 5);
          this.roadGraphics.lineStyle(1.2, 0xe7ffff, 0.1 + springPulse * 0.09);
          this.roadGraphics.strokeEllipse(springScreen.x + Math.sin(time * 0.003) * 2.5, springScreen.y + 15, 24 + springPulse * 7, 9 + springPulse * 3);
          this.overlayGraphics.fillStyle(0xf2ffff, 0.06 + springPulse * 0.05);
          this.overlayGraphics.fillCircle(springScreen.x - 6 + Math.sin(time * 0.004) * 2, springScreen.y + 10, 1.8 + springPulse * 0.9);
          this.overlayGraphics.fillCircle(springScreen.x + 5 + Math.cos(time * 0.0035) * 2, springScreen.y + 13, 1.4 + springPulse * 0.7);
        }

        if (building.defId === "eternal_flame_brazier") {
          this.roadGraphics.fillStyle(0xf08a24, 0.05 + emberPulse * 0.07);
          this.roadGraphics.fillCircle(screen.x, screen.y - 4, 18 + emberPulse * 4);
          this.roadGraphics.fillStyle(0xffdc91, 0.07 + emberPulse * 0.08);
          this.roadGraphics.fillCircle(screen.x, screen.y - 8, 8 + emberPulse * 2.4);
          for (let index = 0; index < 3; index += 1) {
            const rise = ((time * 0.00075) + index * 0.24) % 1;
            this.overlayGraphics.fillStyle(0xffcf75, 0.05 + (1 - rise) * 0.12);
            this.overlayGraphics.fillCircle(
              screen.x + Math.sin(time * 0.003 + index) * (3 + index * 1.5),
              screen.y - 18 - rise * 18,
              1.2 + (1 - rise) * 1.1
            );
          }
        } else if (building.defId === "inner_sanctum" || building.defId === "sacrificial_altar") {
          this.roadGraphics.fillStyle(0xe6c066, 0.025 + ritualPulse * 0.035);
          this.roadGraphics.fillEllipse(screen.x, screen.y + 4, 54 + ritualPulse * 12, 24 + ritualPulse * 5);
        }

        const showStatusBar = selected || hovered || conditionRatio < 0.999 || building.assignedPriestIds.length > 0;
        if (showStatusBar) {
          const barWidth = selected || hovered ? 40 : 32;
          const barX = screen.x - barWidth / 2;
          const barY = selected || hovered ? screen.y - 30 : screen.y - 24;
          this.overlayGraphics.fillStyle(0xf6ecd0, selected || hovered ? 0.96 : 0.72);
          this.overlayGraphics.fillRect(barX, barY, barWidth, 8);
          this.overlayGraphics.lineStyle(1.5, 0x3a2d1a, selected || hovered ? 0.72 : 0.5);
          this.overlayGraphics.strokeRect(barX, barY, barWidth, 8);
          this.overlayGraphics.fillStyle(0x281d12, 0.34);
          this.overlayGraphics.fillRect(barX + 2, barY + 2, barWidth - 4, 3);
          this.overlayGraphics.fillStyle(conditionColor, selected || hovered ? 0.98 : 0.84);
          this.overlayGraphics.fillRect(barX + 2, barY + 2, (barWidth - 4) * conditionRatio, 3);
        }

        if (building.assignedPriestIds.length > 0) {
          this.overlayGraphics.fillStyle(0xffefbf, 0.95);
          building.assignedPriestIds.slice(0, 3).forEach((_, index) => {
            this.overlayGraphics.fillCircle(screen.x - 9 + index * 9, screen.y - 13, 2);
          });
        }

        if (selected) {
          this.drawSelectionHalo(this.overlayGraphics, screen.x, screen.y - 16, 0xe8cd78);
        } else if (hovered) {
          this.overlayGraphics.lineStyle(2, 0x8ad6ef, 0.62);
          this.overlayGraphics.strokeEllipse(screen.x, screen.y + 8, 42, 18);
        }
      });

    [...state.walkers]
      .sort((left, right) => left.tile.y - right.tile.y || left.tile.x - right.tile.x)
      .forEach((walker) => {
        const screen = tileToScreen(walker.tile);
        const hasSprite = this.walkerSprites.has(walker.id);
        const color = walker.role === "priest" ? 0xf1f1ec : walker.role === "custodian" ? 0x8f643e : walker.role === "carrier" ? 0x5f85b8 : 0x9d7d5a;
        if (selectedWalker?.id === walker.id) {
          this.walkerGraphics.fillStyle(0xf0d27a, 0.2);
          this.walkerGraphics.fillCircle(screen.x, screen.y + 10, 12);
          this.walkerGraphics.lineStyle(2, 0xf0d27a, 0.9);
          this.walkerGraphics.strokeCircle(screen.x, screen.y + 10, 11);
        } else if (state.ui.hoveredTile?.x === walker.tile.x && state.ui.hoveredTile?.y === walker.tile.y) {
          this.walkerGraphics.lineStyle(2, 0x6ec2e8, 0.9);
          this.walkerGraphics.strokeCircle(screen.x, screen.y + 10, 10);
        }
        this.walkerGraphics.fillStyle(0x24180c, hasSprite ? 0.1 : 0.16);
        this.walkerGraphics.fillEllipse(screen.x, screen.y + 14, hasSprite ? 18 : 14, hasSprite ? 7 : 10);
        if (!hasSprite) {
          this.walkerGraphics.fillStyle(color, 1);
          this.walkerGraphics.fillCircle(screen.x, screen.y + 10, walker.role === "pilgrim" ? 5 : 7);
          this.walkerGraphics.lineStyle(2, 0x2f2113, 0.9);
          this.walkerGraphics.strokeCircle(screen.x, screen.y + 10, walker.role === "pilgrim" ? 5 : 7);
        }
      });

    if (state.ui.hoveredTile) {
      const screen = tileToScreen(state.ui.hoveredTile);
      const tool = state.ui.activeTool;
      const occupied = state.grid.roads.some((road) => road.x === state.ui.hoveredTile?.x && road.y === state.ui.hoveredTile?.y)
        || state.buildings.some((building) => building.position.x === state.ui.hoveredTile?.x && building.position.y === state.ui.hoveredTile?.y);
      const valid = tool === "select"
        ? false
        : tool === "sacred_way"
          ? !occupied
          : !occupied && state.grid.roads.some((road) => Math.abs(road.x - state.ui.hoveredTile!.x) + Math.abs(road.y - state.ui.hoveredTile!.y) === 1);
      const ghostColor = tool === "select"
        ? 0x6ec2e8
        : tool === "sacred_way"
          ? (valid ? 0x39a96b : 0xc65050)
          : (valid ? buildingDefs[tool].color : 0xc65050);

      this.drawDiamond(this.overlayGraphics, screen.x, screen.y, ghostColor, tool === "select" ? 0.08 : 0.14);
      this.overlayGraphics.lineStyle(1.5, valid ? 0xf6f0d9 : 0x461b15, 0.72);
      this.overlayGraphics.strokeEllipse(screen.x, screen.y + 9, 28, 12);

      if (tool !== "select" && tool !== "sacred_way") {
        this.overlayGraphics.fillStyle(valid ? 0xf5ead2 : 0x4f1f1a, 0.68);
        this.overlayGraphics.fillRect(screen.x - 10, screen.y - 15, 20, 3);
      }

      if (selectedBuilding) {
        const selectedScreen = tileToScreen(selectedBuilding.position);
        this.overlayGraphics.lineStyle(1, 0xf0d27a, 0.45);
        this.overlayGraphics.lineBetween(selectedScreen.x, selectedScreen.y + 12, screen.x, screen.y + 12);
      }
    }

    if (!focusTile) {
      this.inspectionLabel.setVisible(false);
      this.inspectionDetail.setVisible(false);
      return;
    }

    const reading = inspectTile(state, focusTile);
    const focusScreen = tileToScreen(focusTile);
    const glowColor = toneColor(reading.tone);

    for (let offsetY = -3; offsetY <= 3; offsetY += 1) {
      for (let offsetX = -3; offsetX <= 3; offsetX += 1) {
        const sample = { x: focusTile.x + offsetX, y: focusTile.y + offsetY };
        if (sample.x < 0 || sample.x >= state.grid.width || sample.y < 0 || sample.y >= state.grid.height) {
          continue;
        }
        const alpha = siteCellAlpha(reading, focusTile, sample);
        if (alpha <= 0) {
          continue;
        }
        const sampleScreen = tileToScreen(sample);
        this.drawDiamond(this.overlayGraphics, sampleScreen.x, sampleScreen.y, glowColor, alpha);
      }
    }

    state.grid.roads
      .filter((road) => manhattan(road, focusTile) <= 2)
      .forEach((road) => {
        const roadScreen = tileToScreen(road);
        this.overlayGraphics.lineStyle(1.5, 0xe8c060, 0.38);
        this.overlayGraphics.strokeEllipse(roadScreen.x, roadScreen.y + TILE_HEIGHT / 2, 18, 10);
      });

    reading.links.forEach((link) => {
      const linkScreen = tileToScreen(link.tile);
      const color = linkColor(link.tone);
      this.overlayGraphics.lineStyle(1.5, color, 0.34);
      this.overlayGraphics.lineBetween(focusScreen.x, focusScreen.y + 12, linkScreen.x, linkScreen.y + 12);
      this.overlayGraphics.lineStyle(1.5, color, 0.54);
      this.overlayGraphics.strokeCircle(linkScreen.x, linkScreen.y - 2, 8);
    });

    this.overlayGraphics.lineStyle(1.5, glowColor, 0.52);
    this.overlayGraphics.strokeEllipse(focusScreen.x, focusScreen.y + 8, 46, 22);
    this.overlayGraphics.lineStyle(2, glowColor, 0.64);
    this.overlayGraphics.strokeCircle(focusScreen.x, focusScreen.y - 2, 11);

    const labelText = `${reading.toneLabel} · ${reading.integrity}`;
    const detailText = `Approach ${reading.approach} · Sanctity ${reading.sanctity} · Strain ${reading.strain} · Roads ${reading.roadContacts} · Anchors ${reading.ritualAnchors}`;
    this.inspectionLabel.setText(labelText);
    this.inspectionDetail.setText(detailText);

    const cameraLeft = this.cameras.main.worldView.x;
    const cameraRight = cameraLeft + this.cameras.main.worldView.width;
    const prefersRight = focusScreen.x < cameraLeft + this.cameras.main.worldView.width * 0.38;
    const labelHalfSpan = Math.max(this.inspectionLabel.width, this.inspectionDetail.width) / 2 + 16;
    const labelX = Phaser.Math.Clamp(
      focusScreen.x + (prefersRight ? 176 : -176),
      cameraLeft + labelHalfSpan,
      cameraRight - labelHalfSpan
    );
    const labelY = focusScreen.y - 94;

    this.inspectionLabel
      .setOrigin(0.5, 1)
      .setVisible(true)
      .setAlpha(0.95)
      .setPosition(labelX, labelY);
    this.inspectionDetail
      .setOrigin(0.5, 1)
      .setVisible(true)
      .setAlpha(0.88)
      .setPosition(labelX, labelY + 21);
  }
}
