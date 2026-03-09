import { buildingDefs } from "@the-oracle/content";
import type { BuildingDefId, PlacementTool } from "@the-oracle/core";
import React, { useState } from "react";

import { PrecinctArtThumb } from "./PrecinctArtThumb";

type BuildPaletteProps = {
  activeTool: PlacementTool;
  unlockedBuildingIds: BuildingDefId[];
  onSetTool: (tool: PlacementTool) => void;
};

const CATEGORY_META: Record<string, { label: string; icon: string }> = {
  processional: { label: "Roads & Paths", icon: "\u{1F6E4}" },
  ritual: { label: "Sacred Structures", icon: "\u{1F525}" },
  housing: { label: "Quarters", icon: "\u{1F6CF}" },
  storage: { label: "Storage", icon: "\u{1F4E6}" },
  production: { label: "Production", icon: "\u{2699}" },
  trade: { label: "Trade", icon: "\u{1F3EA}" },
  hospitality: { label: "Visitor Facilities", icon: "\u{1F3E0}" }
};

const CATEGORY_ORDER = ["processional", "ritual", "housing", "storage", "production", "trade", "hospitality"];

function formatResourceName(id: string): string {
  return id.replace(/_/g, " ");
}

function BuildingDetail({ defId }: { defId: BuildingDefId }) {
  const def = buildingDefs[defId];
  const colorHex = `#${def.color.toString(16).padStart(6, "0")}`;
  const upkeepEntries = Object.entries(def.upkeep).filter(([, v]) => v > 0);
  const hasRecipes = def.recipes && def.recipes.length > 0;
  const hasEffects = def.passiveEffects && def.passiveEffects.length > 0;
  const hasStorage = def.storageCaps && Object.keys(def.storageCaps).length > 0;
  const hasStaffing = def.staffing && (def.staffing.priests || def.staffing.custodians || def.staffing.carriers || def.staffing.visitors);

  return (
    <div className="building-detail">
      {/* Color swatch + name */}
      <div className="building-detail-header">
        <div className="building-detail-visual">
          <PrecinctArtThumb
            defId={defId}
            alt={def.name}
            className="building-detail-art"
          />
          <div className="building-swatch" style={{ background: colorHex }} />
        </div>
        <div>
          <div className="building-detail-name">{def.name}</div>
          <div className="building-detail-category">{def.category}</div>
        </div>
      </div>

      {/* Description */}
      <p className="building-detail-desc">{def.description}</p>

      {/* Cost & upkeep */}
      <div className="building-detail-section">
        <div className="building-detail-row">
          <span className="building-detail-label">Build Cost</span>
          <span className="building-detail-value">{def.costGold} gold</span>
        </div>
        <div className="building-detail-row">
          <span className="building-detail-label">Durability</span>
          <span className="building-detail-value">{def.maxCondition} HP</span>
        </div>
        {def.requiresPriest ? (
          <div className="building-detail-row">
            <span className="building-detail-label">Requires</span>
            <span className="building-detail-value building-detail-priest">Priest</span>
          </div>
        ) : null}
        {upkeepEntries.length > 0 ? (
          <div className="building-detail-row">
            <span className="building-detail-label">Upkeep</span>
            <span className="building-detail-value">
              {upkeepEntries.map(([id, amount]) => `${Number(amount).toFixed(3)} ${formatResourceName(id)}`).join(", ")}/day
            </span>
          </div>
        ) : (
          <div className="building-detail-row">
            <span className="building-detail-label">Upkeep</span>
            <span className="building-detail-value" style={{ opacity: 0.5 }}>None</span>
          </div>
        )}
      </div>

      {/* Production recipes */}
      {hasRecipes ? (
        <div className="building-detail-section">
          <div className="building-detail-section-title">Production</div>
          {def.recipes!.map((recipe) => {
            const consumes = Object.entries(recipe.consumes).filter(([, v]) => v > 0);
            const produces = Object.entries(recipe.produces).filter(([, v]) => v > 0);
            return (
              <div key={recipe.id} className="building-detail-recipe">
                {consumes.length > 0 ? (
                  <div className="building-detail-recipe-line">
                    <span className="recipe-label">Uses</span>
                    {consumes.map(([id, amount]) => (
                      <span key={id} className="recipe-chip consumes">{Number(amount).toFixed(2)} {formatResourceName(id)}</span>
                    ))}
                  </div>
                ) : null}
                {produces.length > 0 ? (
                  <div className="building-detail-recipe-line">
                    <span className="recipe-label">Makes</span>
                    {produces.map(([id, amount]) => (
                      <span key={id} className="recipe-chip produces">{Number(amount).toFixed(2)} {formatResourceName(id)}</span>
                    ))}
                  </div>
                ) : null}
                {recipe.notes ? <div className="building-detail-note">{recipe.notes}</div> : null}
              </div>
            );
          })}
        </div>
      ) : null}

      {/* Passive effects */}
      {hasEffects ? (
        <div className="building-detail-section">
          <div className="building-detail-section-title">Effects</div>
          {def.passiveEffects!.map((effect) => (
            <div key={effect.id} className="building-detail-effect">
              <span className="effect-kind">{effect.kind.replace(/_/g, " ")}</span>
              <span className="effect-value">+{effect.value}</span>
              {effect.notes ? <div className="building-detail-note">{effect.notes}</div> : null}
            </div>
          ))}
        </div>
      ) : null}

      {/* Storage */}
      {hasStorage ? (
        <div className="building-detail-section">
          <div className="building-detail-section-title">Storage Capacity</div>
          <div className="building-detail-storage">
            {Object.entries(def.storageCaps!).map(([id, cap]) => (
              <span key={id} className="storage-chip">{formatResourceName(id)} {cap}</span>
            ))}
          </div>
        </div>
      ) : null}

      {/* Staffing */}
      {hasStaffing ? (
        <div className="building-detail-section">
          <div className="building-detail-section-title">Staffing</div>
          {def.staffing!.priests ? (
            <div className="building-detail-row">
              <span className="building-detail-label">Priests</span>
              <span className="building-detail-value">
                {Object.entries(def.staffing!.priests).map(([role, count]) => `${count} ${role.replace(/_/g, " ")}`).join(", ")}
              </span>
            </div>
          ) : null}
          {def.staffing!.custodians ? (
            <div className="building-detail-row">
              <span className="building-detail-label">Custodians</span>
              <span className="building-detail-value">{def.staffing!.custodians}</span>
            </div>
          ) : null}
          {def.staffing!.carriers ? (
            <div className="building-detail-row">
              <span className="building-detail-label">Carriers</span>
              <span className="building-detail-value">{def.staffing!.carriers}</span>
            </div>
          ) : null}
          {def.staffing!.visitors ? (
            <div className="building-detail-row">
              <span className="building-detail-label">Visitor Slots</span>
              <span className="building-detail-value">{def.staffing!.visitors}</span>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function BuildPalette({ activeTool, unlockedBuildingIds, onSetTool }: BuildPaletteProps) {
  const [openCategories, setOpenCategories] = useState<Set<string>>(() => new Set(CATEGORY_ORDER));
  const [inspectedBuilding, setInspectedBuilding] = useState<BuildingDefId | null>(null);
  const unlockedSet = new Set<string>(unlockedBuildingIds);

  const grouped = new Map<string, (typeof buildingDefs)[BuildingDefId][]>();
  for (const def of Object.values(buildingDefs)) {
    const cat = def.category;
    if (!grouped.has(cat)) {
      grouped.set(cat, []);
    }
    grouped.get(cat)!.push(def);
  }

  function toggleCategory(cat: string) {
    setOpenCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) {
        next.delete(cat);
      } else {
        next.add(cat);
      }
      return next;
    });
  }

  function handleItemClick(defId: BuildingDefId, locked: boolean) {
    if (locked) {
      // Show details even for locked buildings
      setInspectedBuilding((prev) => (prev === defId ? null : defId));
      return;
    }
    if (activeTool === defId) {
      // Already active — toggle detail panel
      setInspectedBuilding((prev) => (prev === defId ? null : defId));
    } else {
      // Select the tool and show details
      onSetTool(defId);
      setInspectedBuilding(defId);
    }
  }

  return (
    <div className="build-palette">
      <div style={{ padding: "8px 12px" }}>
        <button
          className={`build-item ${activeTool === "select" ? "active" : ""}`}
          style={{ width: "100%" }}
          id="tool-select"
          onClick={() => { onSetTool("select"); setInspectedBuilding(null); }}
          type="button"
        >
          Select
        </button>
      </div>

      {/* Building detail popup */}
      {inspectedBuilding ? (
        <div className="building-detail-container">
          <button
            className="building-detail-close"
            onClick={() => setInspectedBuilding(null)}
            type="button"
          >
            {"\u2715"}
          </button>
          <BuildingDetail defId={inspectedBuilding} />
        </div>
      ) : null}

      {CATEGORY_ORDER.map((cat) => {
        const items = grouped.get(cat);
        if (!items || items.length === 0) return null;
        const meta = CATEGORY_META[cat] ?? { label: cat, icon: "" };
        const isOpen = openCategories.has(cat);

        return (
          <div key={cat} className="build-category">
            <button className="build-category-header" onClick={() => toggleCategory(cat)} type="button">
              <span>
                <span className="cat-icon">{meta.icon}</span>
                {meta.label}
              </span>
              <span className={`chevron ${isOpen ? "open" : ""}`}>&#9654;</span>
            </button>
            {isOpen ? (
              <div className="build-category-items">
                {items.map((def) => {
                  const locked = !unlockedSet.has(def.id);
                  const isActive = activeTool === def.id;
                  const isInspected = inspectedBuilding === def.id;
                  const colorHex = `#${def.color.toString(16).padStart(6, "0")}`;
                  return (
                    <button
                      key={def.id}
                      className={`build-item ${isActive ? "active" : ""} ${locked ? "locked" : ""} ${isInspected && !isActive ? "inspected" : ""}`}
                      id={`tool-${def.id}`}
                      onClick={() => handleItemClick(def.id, locked)}
                      title={locked && def.unlockTier ? `Requires ${def.unlockTier} reputation` : def.description}
                      type="button"
                    >
                      <span className="build-item-main">
                        <PrecinctArtThumb
                          defId={def.id}
                          alt=""
                          className="build-item-thumb"
                        />
                        <span className="build-item-fallback" style={{ background: colorHex }} aria-hidden="true" />
                        <span>{locked ? "\u{1F512} " : ""}{def.name}</span>
                      </span>
                      <span className="build-item-cost">
                        {locked && def.unlockTier ? def.unlockTier : `${def.costGold}g`}
                      </span>
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
