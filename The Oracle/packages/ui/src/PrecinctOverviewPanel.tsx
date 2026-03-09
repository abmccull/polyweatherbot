import { buildingDefs } from "@the-oracle/content";
import type { BuildingInstance, GameState, WalkerInstance } from "@the-oracle/core";

type PrecinctOverviewPanelProps = {
  state: GameState;
  selectedBuilding?: BuildingInstance;
  selectedWalker?: WalkerInstance;
};

type Tile = { x: number; y: number };

type SoulTone = "consecrated" | "attuned" | "watchful" | "frayed";

type SiteLinkTone = "ritual" | "approach" | "support" | "strain";

export type PrecinctSoulReading = {
  integrity: number;
  tone: SoulTone;
  toneLabel: string;
  consecration: number;
  procession: number;
  shelter: number;
  strain: number;
  anchorCount: number;
  watchword: string;
  horizonText: string;
  regionalPressureLabel: string;
};

export type SiteInspectionReading = {
  tile: Tile;
  tone: SoulTone;
  toneLabel: string;
  integrity: number;
  approach: number;
  sanctity: number;
  shelter: number;
  strain: number;
  roadContacts: number;
  ritualAnchors: number;
  pressureLoad: number;
  note: string;
  links: {
    id: string;
    tile: Tile;
    label: string;
    tone: SiteLinkTone;
  }[];
};

const INSPECTION_RADIUS = 4;

function colorForBuilding(defId: keyof typeof buildingDefs): string {
  return `#${buildingDefs[defId].color.toString(16).padStart(6, "0")}`;
}

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

function toneForIntegrity(value: number): { tone: SoulTone; toneLabel: string } {
  if (value >= 76) {
    return { tone: "consecrated", toneLabel: "Consecrated" };
  }
  if (value >= 62) {
    return { tone: "attuned", toneLabel: "Attuned" };
  }
  if (value >= 48) {
    return { tone: "watchful", toneLabel: "Watchful" };
  }
  return { tone: "frayed", toneLabel: "Frayed" };
}

function meterClassForTone(tone: SoulTone) {
  if (tone === "consecrated") return "tone-consecrated";
  if (tone === "attuned") return "tone-attuned";
  if (tone === "watchful") return "tone-watchful";
  return "tone-critical";
}

function impliedPrecinctHeart(state: GameState): Tile {
  const focusPool = state.buildings.map((building) => building.position);
  if (focusPool.length > 0) {
    const total = focusPool.reduce((sum, tile) => ({ x: sum.x + tile.x, y: sum.y + tile.y }), { x: 0, y: 0 });
    return {
      x: Math.round(total.x / focusPool.length),
      y: Math.round(total.y / focusPool.length)
    };
  }

  if (state.grid.roads.length > 0) {
    const total = state.grid.roads.reduce((sum, tile) => ({ x: sum.x + tile.x, y: sum.y + tile.y }), { x: 0, y: 0 });
    return {
      x: Math.round(total.x / state.grid.roads.length),
      y: Math.round(total.y / state.grid.roads.length)
    };
  }

  return state.walkers[0]?.tile ?? { x: Math.round(state.grid.width / 2), y: Math.round(state.grid.height / 2) };
}

export function resolvePrecinctFocusTile(
  state: GameState,
  selectedBuilding?: BuildingInstance,
  selectedWalker?: WalkerInstance
): Tile {
  return selectedBuilding?.position ?? selectedWalker?.tile ?? state.ui.hoveredTile ?? impliedPrecinctHeart(state);
}

export function derivePrecinctSoul(state: GameState): PrecinctSoulReading {
  const categories = state.buildings.map((building) => buildingDefs[building.defId].category);
  const buildingHealth = state.buildings.length > 0
    ? average(state.buildings.map((building) => (building.condition / Math.max(1, building.maxCondition)) * 100))
    : 100;
  const ritualCount = categories.filter((category) => category === "ritual").length;
  const processionalCount = categories.filter((category) => category === "processional").length;
  const supportCount = categories.filter((category) => category === "housing" || category === "storage" || category === "hospitality").length;
  const foodBuffer = (state.resources.grain?.amount ?? 0) * 0.18 + (state.resources.bread?.amount ?? 0) * 0.28;
  const incenseCover = (state.resources.incense?.amount ?? 0) * 0.42 + (state.resources.sacred_water?.amount ?? 0) * 0.38;
  const divineMeter = state.worldGeneration.divineMood.meter ?? 50;
  const pressureRegion = [...state.worldGeneration.regions].sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))[0];
  const regionalPressure = pressureRegion?.pressure ?? 42;
  const averageFatigue = average(state.walkers.filter((walker) => walker.role === "carrier").map((walker) => walker.fatigue ?? 0));

  const consecration = clamp(24 + ritualCount * 16 + incenseCover + divineMeter * 0.24 + buildingHealth * 0.16 - state.pythia.needs.purification * 0.26);
  const procession = clamp(18 + processionalCount * 14 + state.grid.roads.length * 2.4 + state.walkers.filter((walker) => walker.role === "pilgrim").length * 5 - regionalPressure * 0.08);
  const shelter = clamp(18 + supportCount * 13 + foodBuffer + buildingHealth * 0.24 - averageFatigue * 0.18);
  const strain = clamp(
    12
    + state.pythia.needs.rest * 0.24
    + state.pythia.needs.purification * 0.28
    + averageFatigue * 0.38
    + state.resourceJobs.length * 2.4
    + regionalPressure * 0.24
    + (100 - buildingHealth) * 0.62
  );
  const integrity = clamp(Math.round(consecration * 0.4 + procession * 0.22 + shelter * 0.18 + (100 - strain) * 0.28));
  const { tone, toneLabel } = toneForIntegrity(integrity);

  let watchword = "The precinct is still proving it can hold sacred attention over time.";
  if (tone === "consecrated") {
    watchword = "Delphi is accumulating memory. New rites will land on already-sanctified ground.";
  } else if (tone === "attuned") {
    watchword = "The precinct carries a stable sacred cadence, but gaps in approach or care will still echo.";
  } else if (tone === "watchful") {
    watchword = "The ground answers, though it is not yet secure against fatigue and wider pressure.";
  }

  const horizonText = pressureRegion
    ? `${pressureRegion.label} remains the loudest regional strain. Its ${pressureRegion.hegemon.toLowerCase()} pressure will reward precincts that keep rites legible and durable.`
    : `${state.worldGeneration.originTitle} opened under ${state.worldGeneration.divineMood.value.toLowerCase()} skies. Long-horizon integrity still matters more than short logistics wins.`;

  return {
    integrity,
    tone,
    toneLabel,
    consecration: Math.round(consecration),
    procession: Math.round(procession),
    shelter: Math.round(shelter),
    strain: Math.round(strain),
    anchorCount: ritualCount + processionalCount,
    watchword,
    horizonText,
    regionalPressureLabel: pressureRegion ? `${pressureRegion.label} ${pressureRegion.pressure}` : "Regional pressure 0"
  };
}

export function deriveSiteInspection(
  state: GameState,
  tile: Tile
): SiteInspectionReading {
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
  const localCarrierFatigue = average(localWalkers.filter((entry) => entry.walker.role === "carrier").map((entry) => entry.walker.fatigue ?? 0));
  const pressureRegion = state.campaign.worldMap.nodes.find((node) => node.id === state.campaign.worldMap.selectedNodeId)
    ?? [...state.campaign.worldMap.nodes].sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))[0];
  const divineMeter = state.worldGeneration.divineMood.meter ?? 50;
  const approach = clamp(20 + roadContacts * 15 + nearbyRoads.length * 4 + localBuildings.filter((entry) => entry.def.category === "processional").length * 12 - localCarrierFatigue * 0.16);
  const sanctity = clamp(16 + ritualAnchors * 18 + divineMeter * 0.24 + (state.resources.sacred_water?.amount ?? 0) * 0.42 + (state.resources.incense?.amount ?? 0) * 0.3 - state.pythia.needs.purification * 0.18);
  const shelter = clamp(18 + supportAnchors * 14 + localBuildings.filter((entry) => entry.def.category === "production").length * 8 + (state.resources.grain?.amount ?? 0) * 0.1 - localWear * 0.35);
  const strain = clamp(
    8
    + localCarrierFatigue * 0.42
    + state.pythia.needs.rest * 0.14
    + state.pythia.needs.purification * 0.18
    + localWear * 0.54
    + (pressureRegion?.pressure ?? 42) * 0.2
    + localWalkers.filter((entry) => entry.walker.state === "hauling" || entry.walker.state === "delivering").length * 5
  );
  const integrity = clamp(Math.round(sanctity * 0.42 + approach * 0.24 + shelter * 0.18 + (100 - strain) * 0.24));
  const { tone, toneLabel } = toneForIntegrity(integrity);

  let note = "The ground is waiting for a clearer sacred pattern.";
  if (ritualAnchors === 0) {
    note = "No ritual anchor sanctifies this tile yet. It reads as approach ground, not holy ground.";
  } else if (roadContacts === 0 && localBuildings.some((entry) => entry.def.category === "ritual")) {
    note = "Rites can accumulate here, but the approach is cut off from the Sacred Way.";
  } else if (strain >= 64) {
    note = "This patch of ground is carrying too much fatigue and deferred care to keep its sanctity stable.";
  } else if (tone === "consecrated") {
    note = "Ritual anchors, clean approaches, and low strain make this one of the precinct's enduring sacred pockets.";
  } else if (tone === "attuned") {
    note = "The site holds a usable sacred charge. More processional access or ritual support would deepen it.";
  }

  const links = localBuildings.slice(0, 5).map(({ building, def }) => {
    const tone: SiteLinkTone = def.category === "ritual"
      ? "ritual"
      : def.category === "processional"
        ? "approach"
        : (building.condition / Math.max(1, building.maxCondition)) < 0.65
          ? "strain"
          : "support";

    return {
      id: building.id,
      tile: building.position,
      label: def.name,
      tone
    };
  });

  return {
    tile,
    tone,
    toneLabel,
    integrity,
    approach: Math.round(approach),
    sanctity: Math.round(sanctity),
    shelter: Math.round(shelter),
    strain: Math.round(strain),
    roadContacts,
    ritualAnchors,
    pressureLoad: pressureRegion?.pressure ?? 0,
    note,
    links
  };
}

function siteCellOpacity(reading: SiteInspectionReading, sample: Tile) {
  const distance = manhattan(reading.tile, sample);
  if (distance > INSPECTION_RADIUS) {
    return 0;
  }
  const localPresence = reading.links.some((link) => manhattan(link.tile, sample) <= 1) ? 0.18 : 0;
  return Math.max(0.06, (reading.integrity / 100) * 0.24 - distance * 0.035 + localPresence);
}

export function PrecinctOverviewPanel({
  state,
  selectedBuilding,
  selectedWalker
}: PrecinctOverviewPanelProps) {
  const selectedTile = resolvePrecinctFocusTile(state, selectedBuilding, selectedWalker);
  const siteReading = deriveSiteInspection(state, selectedTile);
  const soulReading = derivePrecinctSoul(state);
  const occupiedTiles = [
    ...state.grid.roads,
    ...state.buildings.map((building) => building.position),
    ...state.walkers.map((walker) => walker.tile),
    selectedTile
  ];
  const mapBounds = (() => {
    const padding = 3;
    const minSpan = 16;
    const xs = occupiedTiles.map((tile) => tile.x);
    const ys = occupiedTiles.map((tile) => tile.y);
    let minX = Math.max(0, Math.min(...xs) - padding);
    let maxX = Math.min(state.grid.width - 1, Math.max(...xs) + padding);
    let minY = Math.max(0, Math.min(...ys) - padding);
    let maxY = Math.min(state.grid.height - 1, Math.max(...ys) + padding);

    while (maxX - minX + 1 < minSpan && (minX > 0 || maxX < state.grid.width - 1)) {
      if (minX > 0) minX -= 1;
      if (maxX < state.grid.width - 1 && maxX - minX + 1 < minSpan) maxX += 1;
    }

    while (maxY - minY + 1 < minSpan && (minY > 0 || maxY < state.grid.height - 1)) {
      if (minY > 0) minY -= 1;
      if (maxY < state.grid.height - 1 && maxY - minY + 1 < minSpan) maxY += 1;
    }

    return {
      minX,
      minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1
    };
  })();
  const meterClass = meterClassForTone(soulReading.tone);
  const toneBadgeClass = soulReading.tone === "consecrated" || soulReading.tone === "attuned"
    ? "good"
    : soulReading.tone === "watchful"
      ? "warn"
      : "bad";

  return (
    <section className="mini-map-panel">
      <div className="section-title">Precinct Atlas</div>
      <svg
        className="precinct-mini-map"
        viewBox={`${mapBounds.minX} ${mapBounds.minY} ${mapBounds.width} ${mapBounds.height}`}
        role="img"
        aria-label="Precinct overview map"
      >
        <rect x={mapBounds.minX} y={mapBounds.minY} width={mapBounds.width} height={mapBounds.height} className="mini-map-background" />
        {Array.from({ length: INSPECTION_RADIUS * 2 + 1 }, (_, rowIndex) => rowIndex - INSPECTION_RADIUS).flatMap((offsetY) =>
          Array.from({ length: INSPECTION_RADIUS * 2 + 1 }, (_, columnIndex) => columnIndex - INSPECTION_RADIUS).map((offsetX) => {
            const sample = { x: selectedTile.x + offsetX, y: selectedTile.y + offsetY };
            if (sample.x < 0 || sample.x >= state.grid.width || sample.y < 0 || sample.y >= state.grid.height) {
              return null;
            }
            const opacity = siteCellOpacity(siteReading, sample);
            return (
              <rect
                key={`cell-${sample.x}-${sample.y}`}
                x={sample.x - 0.48}
                y={sample.y - 0.48}
                width={0.96}
                height={0.96}
                rx={0.18}
                className={`mini-map-soul-cell ${siteReading.tone}`}
                style={{ opacity }}
              />
            );
          })
        )}
        {state.grid.roads.map((road) => (
          <rect
            key={`road-${road.x}-${road.y}`}
            x={road.x - 0.45}
            y={road.y - 0.45}
            width={0.9}
            height={0.9}
            className={`mini-map-road ${manhattan(road, selectedTile) <= 2 ? "focused" : ""}`}
          />
        ))}
        {siteReading.links.map((link) => (
          <line
            key={`link-${link.id}`}
            x1={selectedTile.x}
            y1={selectedTile.y}
            x2={link.tile.x}
            y2={link.tile.y}
            className={`mini-map-inspection-link ${link.tone}`}
          />
        ))}
        {state.buildings.map((building) => (
          <g key={building.id}>
            <rect
              x={building.position.x - 0.8}
              y={building.position.y - 0.8}
              width={1.6}
              height={1.6}
              rx={0.18}
              fill={colorForBuilding(building.defId)}
              className={selectedBuilding?.id === building.id ? "mini-map-building selected" : "mini-map-building"}
            />
            {selectedBuilding?.id === building.id ? (
              <circle cx={building.position.x} cy={building.position.y} r={1.75} className="mini-map-selected-ring" />
            ) : null}
          </g>
        ))}
        {siteReading.links.map((link) => (
          <circle
            key={`anchor-${link.id}`}
            cx={link.tile.x}
            cy={link.tile.y}
            r={1.05}
            className={`mini-map-site-anchor ${link.tone}`}
          />
        ))}
        {state.walkers.map((walker) => (
          <g key={walker.id}>
            <circle
              cx={walker.tile.x}
              cy={walker.tile.y}
              r={walker.role === "pilgrim" ? 0.42 : 0.58}
              className={`mini-map-walker ${selectedWalker?.id === walker.id ? "selected" : ""} ${walker.role}`}
            />
            {selectedWalker?.id === walker.id ? (
              <circle cx={walker.tile.x} cy={walker.tile.y} r={1.2} className="mini-map-selected-ring walker" />
            ) : null}
          </g>
        ))}
        {state.ui.hoveredTile ? (
          <rect
            x={state.ui.hoveredTile.x - 0.7}
            y={state.ui.hoveredTile.y - 0.7}
            width={1.4}
            height={1.4}
            rx={0.12}
            className="mini-map-hover"
          />
        ) : null}
        {selectedTile ? (
          <rect
            x={selectedTile.x - 1}
            y={selectedTile.y - 1}
            width={2}
            height={2}
            rx={0.2}
            className="mini-map-focus-frame"
          />
        ) : null}
        <circle cx={selectedTile.x} cy={selectedTile.y} r={2.65} className={`mini-map-soul-ring ${siteReading.tone}`} />
      </svg>
      <div className="mini-map-summary">
        <span>Buildings {state.buildings.length}</span>
        <span>Roads {state.grid.roads.length}</span>
        <span>Walkers {state.walkers.length}</span>
        <span>Focus {selectedTile.x},{selectedTile.y}</span>
      </div>
      <div className="mini-map-caption" id="precinct-site-note">
        {selectedBuilding
          ? `Selected ${buildingDefs[selectedBuilding.defId].name}. ${siteReading.note}`
          : selectedWalker
            ? `Selected ${selectedWalker.name}. ${siteReading.note}`
            : state.ui.hoveredTile
              ? `Hovered tile ${state.ui.hoveredTile.x},${state.ui.hoveredTile.y}. ${siteReading.note}`
              : `Precinct heart ${selectedTile.x},${selectedTile.y}. ${siteReading.note}`}
      </div>
      <div className="precinct-soul-card" id="precinct-soul-panel">
        <div className="precinct-soul-header">
          <div>
            <div className="section-title">Precinct Soul</div>
            <div className="headline">{soulReading.toneLabel}</div>
          </div>
          <span className={`condition-badge ${toneBadgeClass}`}>{soulReading.integrity}</span>
        </div>
        <div className="campaign-meter precinct-soul-meter">
          <div className={`campaign-meter-fill ${meterClass}`} style={{ width: `${soulReading.integrity}%` }} />
        </div>
        <div className="precinct-soul-grid">
          <div className="precinct-soul-stat">
            <span className="precinct-soul-stat-label">Consecration</span>
            <strong>{soulReading.consecration}</strong>
          </div>
          <div className="precinct-soul-stat">
            <span className="precinct-soul-stat-label">Procession</span>
            <strong>{soulReading.procession}</strong>
          </div>
          <div className="precinct-soul-stat">
            <span className="precinct-soul-stat-label">Shelter</span>
            <strong>{soulReading.shelter}</strong>
          </div>
          <div className="precinct-soul-stat strain">
            <span className="precinct-soul-stat-label">Strain</span>
            <strong>{soulReading.strain}</strong>
          </div>
        </div>
        <div className="mini-map-summary precinct-soul-meta">
          <span>Anchors {soulReading.anchorCount}</span>
          <span>{soulReading.regionalPressureLabel}</span>
        </div>
        <div className="campaign-copy" id="precinct-soul-summary">{soulReading.watchword}</div>
        <div className="campaign-copy">{soulReading.horizonText}</div>
      </div>
    </section>
  );
}
