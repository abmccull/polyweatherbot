import React, { useState } from "react";
import type { GameState, FactionId, EspionageOperationKind, EspionageAgentCover } from "@the-oracle/core";
import { selectEspionageOverview } from "@the-oracle/core";
import { useGameDispatch } from "./GameDispatchContext";

const OPERATION_KINDS: { kind: EspionageOperationKind; label: string }[] = [
  { kind: "intercept_prophecy", label: "Intercept Prophecy" },
  { kind: "plant_false_omen", label: "Plant False Omen" },
  { kind: "recruit_informant", label: "Recruit Informant" },
  { kind: "sabotage_rival", label: "Sabotage Rival" },
  { kind: "protect_oracle", label: "Protect Oracle" },
  { kind: "seed_philosopher", label: "Seed Philosopher" }
];

const COVER_OPTIONS: { cover: EspionageAgentCover; label: string }[] = [
  { cover: "merchant", label: "Merchant" },
  { cover: "pilgrim", label: "Pilgrim" },
  { cover: "priest", label: "Priest" },
  { cover: "scholar", label: "Scholar" }
];

export function EspionagePanel({ state }: { state: GameState }) {
  const overview = selectEspionageOverview(state);
  const { onLaunchEspionageOperation, onRecruitAgent } = useGameDispatch();
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedOpKind, setSelectedOpKind] = useState<EspionageOperationKind>("intercept_prophecy");
  const [recruitCover, setRecruitCover] = useState<EspionageAgentCover>("merchant");

  const factionIds = Object.keys(state.factions) as FactionId[];
  const [recruitFaction, setRecruitFaction] = useState<FactionId>(factionIds[0] ?? ("athens" as FactionId));

  const rivalIds = state.rivalOracles?.roster.map((r) => r.id) ?? [];

  function handleLaunch() {
    if (!selectedAgentId) return;
    const targetId = rivalIds[0] ?? "";
    onLaunchEspionageOperation(selectedOpKind, selectedAgentId, targetId);
  }

  return (
    <>
      <div className="sidebar-block">
        <div className="section-title">Intelligence Network</div>
        <div className="priest-politics-meters">
          <div className="priest-politics-meter">
            <span className="priest-politics-meter-label">Network Strength</span>
            <strong>{overview.networkStrength}</strong>
          </div>
          <div className="priest-politics-meter">
            <span className="priest-politics-meter-label">Active Agents</span>
            <strong>{overview.agents.filter((a) => !a.compromised).length}</strong>
          </div>
        </div>
      </div>

      <div className="sidebar-block">
        <div className="section-title">Agent Roster</div>
        {overview.agents.length === 0 ? (
          <div className="text-sm" style={{ color: "var(--text-dim)" }}>No agents recruited yet.</div>
        ) : null}
        {overview.agents.map((agent) => (
          <div
            key={agent.id}
            className={`priest-row ${agent.compromised ? "compromised" : ""}`}
            style={{ opacity: agent.compromised ? 0.6 : 1 }}
          >
            <div className="priest-row-header">
              <span className="priest-row-name">{agent.name}</span>
              <span className="text-xs" style={{ color: "var(--text-dim)" }}>
                {agent.cover} | {agent.targetFactionName}
              </span>
            </div>
            <div className="priest-row-details">
              Skill {agent.skill} | Loyalty {agent.loyalty}
              {agent.compromised ? " | COMPROMISED" : ""}
            </div>
            {!agent.compromised ? (
              <button
                className="oracle-button text-xs"
                onClick={() => setSelectedAgentId(agent.id === selectedAgentId ? null : agent.id)}
                type="button"
                style={{ marginTop: "4px" }}
              >
                {selectedAgentId === agent.id ? "Deselect" : "Select for Op"}
              </button>
            ) : null}
          </div>
        ))}
      </div>

      {selectedAgentId ? (
        <div className="sidebar-block">
          <div className="section-title">Launch Operation</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <select
              className="text-sm"
              value={selectedOpKind}
              onChange={(e) => setSelectedOpKind(e.target.value as EspionageOperationKind)}
              style={{ padding: "4px" }}
            >
              {OPERATION_KINDS.map((op) => (
                <option key={op.kind} value={op.kind}>{op.label}</option>
              ))}
            </select>
            <button className="oracle-button" onClick={handleLaunch} type="button">
              Launch Operation
            </button>
          </div>
        </div>
      ) : null}

      {overview.activeOperations.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Active Operations</div>
          {overview.activeOperations.map((op) => (
            <div key={op.id} className="priest-row">
              <div className="priest-row-header">
                <span className="priest-row-name">{op.kind.replace(/_/g, " ")}</span>
                <span className="text-xs" style={{ color: "var(--text-dim)" }}>{op.agentName}</span>
              </div>
              <div className="priest-row-details">
                Progress {op.progress}% | {op.daysRemaining} days remaining
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <div className="sidebar-block">
        <div className="section-title">Recruit Agent</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <select
            className="text-sm"
            value={recruitCover}
            onChange={(e) => setRecruitCover(e.target.value as EspionageAgentCover)}
            style={{ padding: "4px" }}
          >
            {COVER_OPTIONS.map((opt) => (
              <option key={opt.cover} value={opt.cover}>{opt.label}</option>
            ))}
          </select>
          <select
            className="text-sm"
            value={recruitFaction}
            onChange={(e) => setRecruitFaction(e.target.value as FactionId)}
            style={{ padding: "4px" }}
          >
            {factionIds.map((id) => (
              <option key={id} value={id}>{state.factions[id]?.name ?? id}</option>
            ))}
          </select>
          <button
            className="oracle-button"
            onClick={() => onRecruitAgent(recruitCover, recruitFaction)}
            type="button"
          >
            Recruit (25 gold)
          </button>
        </div>
      </div>

      {overview.recentCounterIntel.length > 0 ? (
        <div className="sidebar-block">
          <div className="section-title">Counter-Intelligence</div>
          {overview.recentCounterIntel.map((event) => (
            <div key={event.id} className="priest-row">
              <div className="priest-row-details" style={{ color: "var(--red, #c44)" }}>
                Day {event.day}: {event.description}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </>
  );
}
