import React from "react";
import { type GameState } from "@the-oracle/core";
import { resourceDefs, type ResourceDef, type SeasonName } from "@the-oracle/content";
import { CarrierDetailPanel } from "./CarrierDetailPanel";
import { trendArrow } from "./SharedComponents";

const RESOURCE_ICON_MAP: Record<string, string> = {
  gold: "\u25C9", olive_oil: "\uD83C\uDFFA", incense: "\uD83C\uDF3F",
  sacred_water: "\uD83D\uDCA7", grain: "\uD83C\uDF3E", sacred_animals: "\uD83D\uDC11",
  bread: "\uD83C\uDF5E", olives: "\uD83E\uDED2", papyrus: "\uD83D\uDCC3", scrolls: "\uD83D\uDCDC"
};

function seasonalBadge(def: ResourceDef, season: string) {
  const mult = def.seasonalMultipliers?.[season as SeasonName];
  if (mult === undefined || mult === 1.0) return null;
  const pct = Math.round((mult - 1) * 100);
  if (pct > 0) return <span className="seasonal-badge seasonal-up">+{pct}%</span>;
  return <span className="seasonal-badge seasonal-down">{pct}%</span>;
}

const PRODUCTION_CHAIN: Record<string, { from?: string; to?: string[] }> = {
  grain: { from: "Grain Field", to: ["Kitchen", "Animal Pen", "Quarters"] },
  olives: { from: "Olive Grove", to: ["Olive Press"] },
  olive_oil: { from: "Olive Press", to: ["Brazier", "Sanctum"] },
  incense: { from: "Incense Workshop", to: ["Sanctum", "Altar", "Agora"] },
  sacred_water: { from: "Castalian Spring", to: ["Sanctum", "Reed Bed"] },
  sacred_animals: { from: "Animal Pen", to: ["Altar"] },
  bread: { from: "Kitchen", to: ["Agora", "Xenon"] },
  papyrus: { from: "Reed Bed", to: ["Scriptorium"] },
  scrolls: { from: "Scriptorium", to: ["Library", "Consultations"] }
};

export function StoresOverlayPanel({ state }: { state: GameState }) {
  return (
    <>
      {["currency", "ritual", "food", "trade"].map((cat) => {
        const defs = resourceDefs.filter((d) => d.category === cat);
        if (defs.length === 0) return null;
        return (
          <div key={cat}>
            <div className="store-category-label">{cat}</div>
            {defs.map((def) => {
              const res = state.resources[def.id];
              if (!res) return null;
              const chain = PRODUCTION_CHAIN[def.id];
              return (
                <div key={def.id}>
                  <div className="store-row">
                    <span className={`store-row-icon ${def.id === "gold" ? "gold-icon" : ""}`}>{RESOURCE_ICON_MAP[def.id] ?? ""}</span>
                    <span className="store-row-name">{def.label}</span>
                    {seasonalBadge(def, state.clock.season)}
                    <span className="store-row-value">{res.amount.toFixed(0)}</span>
                    {trendArrow(res.trend)}
                  </div>
                  {chain ? (
                    <div className="store-chain-detail">
                      {chain.from ? <span className="chain-producer">From: {chain.from}</span> : null}
                      {chain.to ? <span className="chain-consumers"> {"\u2192"} {chain.to.join(", ")}</span> : null}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        );
      })}
      <CarrierDetailPanel state={state} />
    </>
  );
}
