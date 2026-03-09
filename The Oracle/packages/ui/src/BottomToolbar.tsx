import { buildingDefs } from "@the-oracle/content";
import type { BuildingDefId, GameState, PlacementTool } from "@the-oracle/core";
import React, { useState } from "react";

type BottomToolbarProps = {
  activeTool: PlacementTool;
  unlockedBuildingIds: BuildingDefId[];
  resources: GameState["resources"];
  onSetTool: (tool: PlacementTool) => void;
};

const CATEGORY_META: Record<string, { label: string; icon: string }> = {
  processional: { label: "Roads & Paths", icon: "\u{1F6E4}" },
  ritual: { label: "Sacred Structures", icon: "\u{1F525}" },
  housing: { label: "Quarters", icon: "\u{1F6CF}" },
  storage: { label: "Storage", icon: "\u{1F4E6}" },
  production: { label: "Production", icon: "\u{2699}" },
  trade: { label: "Trade", icon: "\u{1F3EA}" },
  hospitality: { label: "Visitor Facilities", icon: "\u{1F3E0}" },
};

const CATEGORY_ORDER = [
  "processional",
  "ritual",
  "housing",
  "storage",
  "production",
  "trade",
  "hospitality",
] as const;

function groupBuildingsByCategory() {
  const grouped = new Map<string, (typeof buildingDefs)[BuildingDefId][]>();
  for (const def of Object.values(buildingDefs)) {
    const cat = def.category;
    if (!grouped.has(cat)) {
      grouped.set(cat, []);
    }
    grouped.get(cat)!.push(def);
  }
  return grouped;
}

const GROUPED = groupBuildingsByCategory();

export function BottomToolbar({
  activeTool,
  unlockedBuildingIds,
  resources,
  onSetTool,
}: BottomToolbarProps) {
  const [openCategory, setOpenCategory] = useState<string | null>(null);
  const unlockedSet = new Set<string>(unlockedBuildingIds);

  const goldAmount = resources.gold?.amount ?? 0;

  function handleCategoryClick(cat: string) {
    setOpenCategory((prev) => (prev === cat ? null : cat));
  }

  function handleItemClick(defId: BuildingDefId, locked: boolean) {
    if (locked) return;
    onSetTool(defId);
    setOpenCategory(null);
  }

  function handleSelectClick() {
    onSetTool("select");
    setOpenCategory(null);
  }

  const activeToolLabel =
    activeTool === "select"
      ? "Select"
      : buildingDefs[activeTool]?.name ?? activeTool;

  const expansionItems = openCategory ? GROUPED.get(openCategory) : null;

  return (
    <div className="bottom-toolbar" role="toolbar" aria-label="Build toolbar">
      {/* Select tool */}
      <button
        type="button"
        className={`bottom-toolbar-btn bottom-toolbar-select ${activeTool === "select" ? "active" : ""}`}
        id="tool-select"
        onClick={handleSelectClick}
        aria-pressed={activeTool === "select"}
      >
        Select
      </button>

      <div className="bottom-toolbar-divider" aria-hidden="true" />

      {/* Category tabs */}
      {CATEGORY_ORDER.map((cat) => {
        const meta = CATEGORY_META[cat] ?? { label: cat, icon: "" };
        const isOpen = openCategory === cat;
        return (
          <button
            key={cat}
            type="button"
            className={`bottom-toolbar-btn ${isOpen ? "active" : ""}`}
            id={`tool-category-${cat}`}
            onClick={() => handleCategoryClick(cat)}
            aria-expanded={isOpen}
            aria-controls={isOpen ? "bottom-toolbar-expansion" : undefined}
          >
            <span className="bottom-toolbar-btn-icon" aria-hidden="true">
              {meta.icon}
            </span>
            {meta.label}
          </button>
        );
      })}

      {/* Active tool mode indicator */}
      <span className="bottom-toolbar-mode" aria-live="polite">
        {activeToolLabel}
      </span>

      {/* Expansion panel */}
      {openCategory && expansionItems && expansionItems.length > 0 ? (
        <div
          id="bottom-toolbar-expansion"
          className="bottom-toolbar-expansion"
          role="listbox"
          aria-label={`${CATEGORY_META[openCategory]?.label ?? openCategory} buildings`}
        >
          {expansionItems.map((def) => {
            const locked = !unlockedSet.has(def.id);
            const isActive = activeTool === def.id;
            const affordable = goldAmount >= def.costGold;

            return (
              <button
                key={def.id}
                type="button"
                role="option"
                id={`tool-${def.id}`}
                aria-selected={isActive}
                aria-disabled={locked}
                className={`bottom-toolbar-item ${isActive ? "active" : ""} ${locked ? "locked" : ""}`}
                onClick={() => handleItemClick(def.id, locked)}
                title={
                  locked && def.unlockTier
                    ? `Requires ${def.unlockTier} reputation`
                    : def.description
                }
              >
                <span>
                  {locked ? "\u{1F512} " : ""}
                  {def.name}
                </span>
                <span
                  className={`bottom-toolbar-item-cost ${locked ? "" : affordable ? "affordable" : "expensive"}`}
                >
                  {locked && def.unlockTier ? def.unlockTier : `${def.costGold}g`}
                </span>
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
