import type { GameState } from "@the-oracle/core";
import {
  selectChronicleEntries,
  selectSacredRecordEntries
} from "@the-oracle/core";
import React, { useState } from "react";

import { ConsequenceTrackerPanel } from "./ConsequenceTrackerPanel";
import { useGameDispatch } from "./GameDispatchContext";
import { LegendaryConsultationPanel } from "./LegendaryConsultationPanel";
import { PortraitArt, resolveFactionPortraitId } from "./PortraitArt";

type OracleTab = "status" | "pythia" | "prophecies" | "trade";

const TAB_LABELS: Record<OracleTab, string> = {
  status: "Status",
  pythia: "Pythia",
  prophecies: "Prophecies",
  trade: "Trade"
};

const TABS: readonly OracleTab[] = ["status", "pythia", "prophecies", "trade"] as const;

function conditionBadge(restNeed: number) {
  if (restNeed < 30) return <span className="condition-badge good">Rested</span>;
  if (restNeed < 60) return <span className="condition-badge warn">Weary</span>;
  return <span className="condition-badge bad">Exhausted</span>;
}

function StatBar({ label, value, max = 100, variant }: { label: string; value: number; max?: number; variant?: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="stat-bar-row">
      <span className="stat-bar-label">{label}</span>
      <div className="stat-bar-track">
        <div className={`stat-bar-fill ${variant ?? ""}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="stat-bar-value">{Math.round(value)}</span>
    </div>
  );
}

function PopStat({ label, count, detail }: { label: string; count: number; detail?: string }) {
  return (
    <div className="pop-stat">
      <span className="pop-stat-count">{count}</span>
      <span className="pop-stat-label">{label}</span>
      {detail ? <span className="pop-stat-detail">{detail}</span> : null}
    </div>
  );
}

export function OracleOverlayPanel({ state }: { state: GameState }) {
  const [activeTab, setActiveTab] = useState<OracleTab>("status");

  const {
    onStartConsultation,
    onRestPythia,
    onPurifyPythia,
    onPurchaseTradeOffer
  } = useGameDispatch();

  const consultationReady = state.consultation.mode === "pending";
  const chronicle = selectChronicleEntries(state);
  const sacredRecords = selectSacredRecordEntries(state);
  const envoyPortraitId = state.consultation.current
    ? resolveFactionPortraitId(
      state.consultation.current.factionId,
      state.factions[state.consultation.current.factionId].profile
    )
    : null;
  const currentTierLabel = `${state.campaign.reputation.currentTier.charAt(0).toUpperCase()}${state.campaign.reputation.currentTier.slice(1)}`;
  const nextTierProgress = (() => {
    const thresholds = state.campaign.reputation.thresholds;
    const current = thresholds[state.campaign.reputation.currentTier];
    const next = state.campaign.reputation.nextTier
      ? thresholds[state.campaign.reputation.nextTier]
      : current + 1;
    return next > current ? Math.min(100, ((state.campaign.reputation.score - current) / (next - current)) * 100) : 100;
  })();

  return (
    <div className="oracle-surface-shell">
      <section className="oracle-surface-hero">
        <div className="oracle-surface-hero-copy">
          <span className="eyebrow">Delphic Chamber</span>
          <div className="headline oracle-surface-title">Seat of Apollo</div>
          <div className="campaign-copy oracle-surface-summary">
            Govern the Pythia, keep the temple solvent, and decide which voices are worthy of the tripod.
          </div>
          <div className="oracle-surface-metrics">
            <div className="oracle-surface-metric">
              <span className="campaign-pill-label">Tier</span>
              <strong>{currentTierLabel}</strong>
            </div>
            <div className="oracle-surface-metric">
              <span className="campaign-pill-label">Prophecies</span>
              <strong>{sacredRecords.length}</strong>
            </div>
            <div className="oracle-surface-metric">
              <span className="campaign-pill-label">Pending</span>
              <strong>{state.consultation.mode === "pending" ? 1 : 0}</strong>
            </div>
          </div>
        </div>
        <div className="oracle-surface-portrait-card">
          <PortraitArt
            portraitId="hierophant_portrait"
            alt={state.pythia.name}
            className="portrait-frame portrait-frame-tall oracle-surface-portrait"
            imgClassName="portrait-image"
          />
          <div className="oracle-surface-portrait-copy">
            <strong>{state.pythia.name}</strong>
            <span>Voice of the tripod</span>
          </div>
        </div>
      </section>

      {/* Tab Bar */}
      <div className="oracle-tab-bar oracle-tab-bar-hero" role="tablist" aria-label="Oracle sections">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={`oracle-tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => setActiveTab(tab)}
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls={`oracle-tabpanel-${tab}`}
            id={`oracle-tab-${tab}`}
            type="button"
          >
            {TAB_LABELS[tab]}
          </button>
        ))}
      </div>

      {/* Envoy Banner — visible across ALL tabs when consultation is pending */}
      {consultationReady && state.consultation.current ? (
        <div className="sidebar-block oracle-surface-section">
          <div className="envoy-card">
            <div className="envoy-card-shell">
              <PortraitArt
                portraitId={envoyPortraitId}
                alt={state.consultation.current.envoyName}
                className="portrait-frame portrait-frame-compact envoy-card-portrait"
                imgClassName="portrait-image"
              />
              <div className="envoy-card-copy">
                <div className="text-sm">
                  <span className="pulsing-dot" />
                  Envoy Approaching
                </div>
                <strong>{state.consultation.current.envoyName}</strong>
                <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                  {state.factions[state.consultation.current.factionId].name}
                </div>
                <div className="text-xs">{state.consultation.current.question}</div>
              </div>
            </div>
            <button className="oracle-button" onClick={onStartConsultation} type="button">
              Receive Envoy
            </button>
          </div>
        </div>
      ) : null}

      {/* ── Status Tab ── */}
      {activeTab === "status" ? (
        <div role="tabpanel" id="oracle-tabpanel-status" aria-labelledby="oracle-tab-status" className="oracle-tabpanel-shell">
          <div className="oracle-dashboard-grid">
          {/* Reputation */}
            <div className="sidebar-block oracle-surface-section oracle-panel-card oracle-panel-card-emphasis">
              <div className="oracle-card-header">
                <div>
                  <div className="section-title">Reputation</div>
                  <strong className="text-base oracle-card-accent">
                    {currentTierLabel}
                  </strong>
                </div>
                <span className="oracle-inline-chip">
                  Score {state.campaign.reputation.score}
                </span>
              </div>
              <div className="campaign-meter oracle-card-meter">
                <div
                  className="campaign-meter-fill"
                  style={{ width: `${nextTierProgress}%` }}
                />
              </div>
              {state.campaign.reputation.nextTier ? (
                <div className="text-xs oracle-muted-copy">
                  Next: {state.campaign.reputation.nextTier.charAt(0).toUpperCase() + state.campaign.reputation.nextTier.slice(1)}
                </div>
              ) : null}
            </div>

            {/* Treasury */}
            <div className="sidebar-block oracle-surface-section oracle-panel-card">
              <div className="oracle-card-header">
                <div className="section-title">Treasury</div>
                <span className="oracle-inline-chip">
                  {state.campaign.treasury.completed} dedications
                </span>
              </div>
              <div className="campaign-meter campaign-meter-secondary oracle-card-meter">
                <div
                  className="campaign-meter-fill campaign-meter-fill-secondary"
                  style={{
                    width: `${state.campaign.treasury.nextMilestoneGold > 0
                      ? Math.min(100, (state.campaign.treasury.totalGoldInvested / state.campaign.treasury.nextMilestoneGold) * 100)
                      : 100}%`
                  }}
                />
              </div>
              <div className="text-xs oracle-muted-copy">
                {state.campaign.treasury.totalGoldInvested} / {state.campaign.treasury.nextMilestoneGold} gold
              </div>
            </div>

            {/* Population */}
            <div className="sidebar-block oracle-surface-section oracle-panel-card">
              <div className="oracle-card-header">
                <div className="section-title">Population</div>
                <span className="oracle-inline-chip">Total {state.walkers.length}</span>
              </div>
              <div className="pop-grid">
                <PopStat label="Priests" count={state.walkers.filter((w) => w.role === "priest").length} detail={`${state.priests.length} ordained`} />
                <PopStat label="Pilgrims" count={state.walkers.filter((w) => w.role === "pilgrim").length} />
                <PopStat label="Custodians" count={state.walkers.filter((w) => w.role === "custodian").length} />
                <PopStat label="Carriers" count={state.walkers.filter((w) => w.role === "carrier").length} />
              </div>
              <div className="oracle-stat-footnote">
                <span>Buildings {state.buildings.length}</span>
                <span>Chronicle {chronicle.length}</span>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* ── Pythia Tab ── */}
      {activeTab === "pythia" ? (
        <div role="tabpanel" id="oracle-tabpanel-pythia" aria-labelledby="oracle-tab-pythia" className="oracle-tabpanel-shell">
          <div className="oracle-pythia-layout">
          {/* Pythia Card */}
            <div className="sidebar-block oracle-surface-section oracle-panel-card oracle-panel-card-portrait">
              <div className="pythia-card">
                <div className="pythia-card-header">
                  <PortraitArt
                    portraitId="hierophant_portrait"
                    alt={state.pythia.name}
                    className="portrait-frame portrait-frame-compact pythia-card-portrait"
                    imgClassName="portrait-image"
                  />
                  <div className="pythia-card-copy">
                    <div className="headline">{state.pythia.name}</div>
                    <div className="text-xs oracle-muted-copy">
                      Voice of the tripod
                    </div>
                  </div>
                  {conditionBadge(state.pythia.needs.rest)}
                </div>
                <div className="stat-bar-group">
                  <StatBar label="Attune" value={state.pythia.attunement} />
                  <StatBar label="Health" value={state.pythia.physicalHealth} variant="health" />
                  <StatBar label="Clarity" value={state.pythia.mentalClarity} variant="blue" />
                  <StatBar label="Trance" value={state.pythia.tranceDepth} />
                  <StatBar label="Rest" value={state.pythia.needs.rest} />
                  <StatBar label="Purify" value={state.pythia.needs.purification} />
                </div>
                <div className="pythia-actions">
                  <button className="oracle-button" id="rest-pythia-btn" onClick={onRestPythia} type="button">Rest</button>
                  <button className="oracle-button" id="purify-pythia-btn" onClick={onPurifyPythia} type="button">Purify</button>
                </div>
              </div>
            </div>

            {/* Priest Summary (advisor notes) */}
            {state.priests.length > 0 ? (
              <div className="sidebar-block oracle-surface-section oracle-panel-card">
                <div className="oracle-card-header">
                  <div className="section-title">Advisor Notes</div>
                  <span className="oracle-inline-chip">{state.priests.length} ordained</span>
                </div>
                <div className="priest-summary">
                  {state.priests.map((priest) => {
                    const walker = state.walkers.find((w) => w.id === priest.walkerId);
                    return (
                      <div key={priest.id} className="priest-summary-row">
                        <span className="priest-summary-name">{walker?.name ?? priest.id}</span>
                        <span className="priest-summary-role">{priest.role.replace(/_/g, " ")}</span>
                        <span className={`priest-summary-status ${priest.currentAssignmentBuildingId ? "assigned" : "idle"}`}>
                          {priest.currentAssignmentBuildingId ? "Assigned" : "Idle"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="sidebar-block oracle-surface-section oracle-panel-card">
                <div className="text-xs oracle-muted-copy" style={{ fontStyle: "italic" }}>
                  Build Priest Quarters to ordain priests
                </div>
              </div>
            )}
          </div>
        </div>
      ) : null}

      {/* ── Prophecies Tab ── */}
      {activeTab === "prophecies" ? (
        <div role="tabpanel" id="oracle-tabpanel-prophecies" aria-labelledby="oracle-tab-prophecies" className="oracle-tabpanel-shell oracle-prophecy-layout">
          {/* Sacred Record */}
          <div className="sidebar-block oracle-surface-section oracle-panel-card">
            <div className="oracle-card-header">
              <div className="section-title">Sacred Record</div>
              <span className="oracle-inline-chip">{sacredRecords.length} sealed</span>
            </div>
            <div className="history-list">
              {sacredRecords.length === 0 ? <div className="history-row">No prophecies recorded yet.</div> : null}
              {sacredRecords.slice(0, 3).map((entry) => (
                <div key={entry.id} className={`history-row ${entry.status}`}>
                  <strong>{entry.factionName}</strong>
                  <span>Day {entry.dayIssued} · {entry.depthBand} depth</span>
                  <span>{entry.interpretation.summary}</span>
                  <span>{entry.resolutionReport ?? `Due Day ${entry.dueDay}`}</span>
                </div>
              ))}
              {chronicle.slice(0, 2).map((entry) => (
                <div key={entry.id} className={`history-row ${entry.kind}`}>
                  <strong>{entry.title}</strong>
                  <span>Day {entry.day}</span>
                  <span>{entry.text}</span>
                  {entry.delta !== null ? <span>Credibility {entry.delta > 0 ? `+${entry.delta}` : entry.delta}</span> : null}
                </div>
              ))}
            </div>
          </div>

          {/* Consequences */}
          <div className="oracle-surface-section oracle-panel-card">
            <ConsequenceTrackerPanel state={state} />
          </div>

          {/* Legendary Consultations */}
          <div className="oracle-surface-section oracle-panel-card">
            <LegendaryConsultationPanel state={state} />
          </div>
        </div>
      ) : null}

      {/* ── Trade Tab ── */}
      {activeTab === "trade" ? (
        <div role="tabpanel" id="oracle-tabpanel-trade" aria-labelledby="oracle-tab-trade" className="oracle-tabpanel-shell">
          <div className="sidebar-block oracle-surface-section oracle-panel-card">
            <div className="oracle-card-header">
              <div>
                <div className="section-title">Trade Offers</div>
                <div className="campaign-copy">Pilgrim caravans and courtly exchanges.</div>
              </div>
              <span className="oracle-inline-chip">{state.tradeOffers.length} offers</span>
            </div>
            <div className="trade-list">
              {state.tradeOffers.length === 0 ? <div className="text-sm oracle-muted-copy">No caravans this month.</div> : null}
              {state.tradeOffers.map((offer) => (
                <div key={offer.id} className="trade-row">
                  <div className="trade-row-copy">
                    <span>{offer.amount} {offer.resourceId.replace("_", " ")}</span>
                    <span className="campaign-copy">Temple broker rate</span>
                  </div>
                  <span className="trade-row-price">{offer.price.toFixed(0)}g</span>
                  <button
                    className="oracle-button trade-buy"
                    id={`buy-trade-${offer.id}`}
                    onClick={() => onPurchaseTradeOffer(offer.id)}
                    type="button"
                  >
                    Buy
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
