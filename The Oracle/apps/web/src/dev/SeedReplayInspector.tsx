import type { GameState } from "@the-oracle/core";
import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";

import type { RunDebugContext } from "../runtime";

type SeedReplayInspectorProps = {
  currentContext: RunDebugContext;
  draftContext: RunDebugContext;
  runSetupOpen: boolean;
  state: GameState;
  onDraftOriginChange: (originId: GameState["originId"]) => void;
  onDraftSeedChange: (seed: string) => void;
  onOpenDraftInSetup: () => void;
  onRandomizeSeed: () => void;
  onResetDraftToLive: () => void;
  onStartDraftRun: () => void;
};

const toggleButtonStyle: CSSProperties = {
  position: "absolute",
  top: 56,
  right: 14,
  zIndex: 6,
  pointerEvents: "auto"
};

const inspectorStyle: CSSProperties = {
  position: "absolute",
  top: 96,
  right: 14,
  zIndex: 6,
  width: 380,
  maxWidth: "calc(100vw - 28px)",
  maxHeight: "calc(100vh - 112px)",
  overflow: "auto",
  padding: 14,
  pointerEvents: "auto",
  display: "grid",
  gap: 12
};

const sectionStyle: CSSProperties = {
  display: "grid",
  gap: 8,
  padding: 10,
  border: "1px solid rgba(200, 151, 42, 0.18)",
  borderRadius: 10,
  background: "rgba(0, 0, 0, 0.14)"
};

const fieldGridStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)",
  gap: 8
};

const inputStyle: CSSProperties = {
  width: "100%",
  borderRadius: 6,
  border: "1px solid rgba(200, 151, 42, 0.25)",
  background: "rgba(16, 12, 6, 0.92)",
  color: "var(--text)",
  padding: "8px 10px",
  font: "inherit"
};

const chipRowStyle: CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 6
};

const chipStyle: CSSProperties = {
  borderRadius: 999,
  border: "1px solid rgba(200, 151, 42, 0.24)",
  background: "rgba(200, 151, 42, 0.08)",
  color: "var(--text-mid)",
  padding: "3px 8px",
  fontSize: "0.7rem"
};

const linkBoxStyle: CSSProperties = {
  padding: "8px 10px",
  borderRadius: 8,
  background: "rgba(16, 12, 6, 0.7)",
  border: "1px solid rgba(200, 151, 42, 0.16)",
  color: "var(--text-mid)",
  fontSize: "0.7rem",
  wordBreak: "break-all"
};

const preStyle: CSSProperties = {
  margin: 0,
  padding: "8px 10px",
  borderRadius: 8,
  background: "rgba(16, 12, 6, 0.82)",
  border: "1px solid rgba(200, 151, 42, 0.16)",
  color: "var(--text-mid)",
  fontSize: "0.68rem",
  lineHeight: 1.45,
  whiteSpace: "pre-wrap",
  overflowX: "auto"
};

const actionRowStyle: CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 6
};

const secondaryTextStyle: CSSProperties = {
  color: "var(--text-dim)",
  fontSize: "0.72rem",
  lineHeight: 1.4
};

function compareLiveRunToPreview(state: GameState, currentContext: RunDebugContext) {
  const mismatches: string[] = [];
  const preview = currentContext.preview;

  if (preview.summary !== state.worldGeneration.summary) {
    mismatches.push("summary");
  }
  if (preview.climate.value !== state.worldGeneration.climate.value) {
    mismatches.push("climate");
  }
  if (preview.divineMood.value !== state.worldGeneration.divineMood.value) {
    mismatches.push("divine mood");
  }
  if (preview.oracleDensity.value !== state.worldGeneration.oracleDensity.value) {
    mismatches.push("oracle density");
  }

  return mismatches;
}

function summarizeDraftDelta(currentContext: RunDebugContext, draftContext: RunDebugContext) {
  const changes: string[] = [];

  if (draftContext.seed !== currentContext.seed) {
    changes.push("seed");
  }
  if (draftContext.originId !== currentContext.originId) {
    changes.push("origin");
  }
  if (draftContext.preview.summary !== currentContext.preview.summary) {
    changes.push("opening summary");
  }
  if (draftContext.preview.climate.value !== currentContext.preview.climate.value) {
    changes.push("climate");
  }
  if (draftContext.preview.divineMood.value !== currentContext.preview.divineMood.value) {
    changes.push("divine mood");
  }
  if (draftContext.preview.oracleDensity.value !== currentContext.preview.oracleDensity.value) {
    changes.push("oracle density");
  }

  return changes;
}

async function copyText(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.append(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div style={chipStyle}>
      <strong>{label}</strong> {value}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section style={sectionStyle}>
      <div className="section-title">{title}</div>
      {children}
    </section>
  );
}

export function SeedReplayInspector({
  currentContext,
  draftContext,
  runSetupOpen,
  state,
  onDraftOriginChange,
  onDraftSeedChange,
  onOpenDraftInSetup,
  onRandomizeSeed,
  onResetDraftToLive,
  onStartDraftRun
}: SeedReplayInspectorProps) {
  const [open, setOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const livePreviewMismatches = useMemo(() => compareLiveRunToPreview(state, currentContext), [currentContext, state]);
  const draftDelta = useMemo(() => summarizeDraftDelta(currentContext, draftContext), [currentContext, draftContext]);
  const replayContextJson = useMemo(
    () =>
      JSON.stringify(
        {
          run: {
            seed: state.worldSeedText,
            originId: state.originId,
            originTitle: state.worldGeneration.originTitle,
            summary: state.worldGeneration.summary
          },
          clock: state.clock,
          setup: {
            replayHref: currentContext.href,
            setupHref: `${window.location.origin}${window.location.pathname}?${currentContext.query}&setup=1`
          },
          worldGeneration: {
            challengeTags: state.worldGeneration.challengeTags,
            scoreModifier: state.worldGeneration.scoreModifier,
            disabledSystems: state.worldGeneration.disabledSystems
          }
        },
        null,
        2
      ),
    [currentContext.href, currentContext.query, state]
  );

  useEffect(() => {
    if (!copyStatus) {
      return;
    }

    const timer = window.setTimeout(() => {
      setCopyStatus(null);
    }, 1800);

    return () => {
      window.clearTimeout(timer);
    };
  }, [copyStatus]);

  async function handleCopy(label: string, value: string) {
    try {
      await copyText(value);
      setCopyStatus(`${label} copied`);
    } catch (error) {
      console.error(`Failed to copy ${label}`, error);
      setCopyStatus(`Copy failed for ${label}`);
    }
  }

  const currentSetupHref = `${window.location.origin}${window.location.pathname}?${currentContext.query}&setup=1`;
  const draftSetupHref = `${window.location.origin}${window.location.pathname}?${draftContext.query}&setup=1`;

  return (
    <>
      <button
        className="metal-button"
        data-testid="seed-replay-inspector-toggle"
        onClick={() => setOpen((current) => !current)}
        style={toggleButtonStyle}
        type="button"
      >
        {open ? "Hide Seed / Replay" : "Seed / Replay"}
      </button>
      {open ? (
        <aside className="panel" data-testid="seed-replay-inspector" style={inspectorStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
            <div>
              <div className="headline" style={{ fontSize: "1rem" }}>
                Seed / Replay Inspector
              </div>
              <div style={secondaryTextStyle}>{runSetupOpen ? "Run setup is open." : "Live shell mode."}</div>
            </div>
            {copyStatus ? (
              <div role="status" style={{ ...secondaryTextStyle, color: "var(--gold-bright)" }}>
                {copyStatus}
              </div>
            ) : null}
          </div>

          <Section title="Live Run">
            <div style={chipRowStyle}>
              <MetricChip label="Seed" value={currentContext.seed} />
              <MetricChip label="Origin" value={state.worldGeneration.originTitle} />
              <MetricChip label="Day" value={`${state.clock.day}/${state.clock.month}/${state.clock.year}`} />
            </div>
            <div style={secondaryTextStyle}>{state.worldGeneration.summary}</div>
            <div style={secondaryTextStyle}>
              {livePreviewMismatches.length === 0
                ? "Preview parity matches the live run."
                : `Preview parity drift: ${livePreviewMismatches.join(", ")}.`}
            </div>
            <div style={actionRowStyle}>
              <button className="metal-button" data-testid="seed-replay-copy-live-url-btn" onClick={() => void handleCopy("live replay URL", currentContext.href)} type="button">
                Copy replay URL
              </button>
              <button className="metal-button" data-testid="seed-replay-copy-live-setup-btn" onClick={() => void handleCopy("live setup URL", currentSetupHref)} type="button">
                Copy setup URL
              </button>
              <button className="metal-button" data-testid="seed-replay-copy-context-btn" onClick={() => void handleCopy("replay context", replayContextJson)} type="button">
                Copy context JSON
              </button>
            </div>
            <div style={linkBoxStyle}>{currentContext.href}</div>
            <pre style={preStyle}>{replayContextJson}</pre>
          </Section>

          <Section title="Origin Debug Draft">
            <div style={fieldGridStyle}>
              <label style={{ display: "grid", gap: 4 }}>
                <span className="section-title">Seed</span>
                <input
                  data-testid="seed-replay-inspector-seed-input"
                  onChange={(event) => onDraftSeedChange(event.target.value)}
                  style={inputStyle}
                  type="text"
                  value={draftContext.seed}
                />
              </label>
              <label style={{ display: "grid", gap: 4 }}>
                <span className="section-title">Origin</span>
                <select
                  data-testid="seed-replay-inspector-origin-select"
                  onChange={(event) => onDraftOriginChange(event.target.value as GameState["originId"])}
                  style={inputStyle}
                  value={draftContext.originId}
                >
                  {draftContext.origins.map((origin) => (
                    <option key={origin.id} value={origin.id}>
                      {origin.title ?? origin.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div style={chipRowStyle}>
              <MetricChip label="Climate" value={draftContext.preview.climate.value} />
              <MetricChip label="Divine" value={draftContext.preview.divineMood.value} />
              <MetricChip label="Density" value={draftContext.preview.oracleDensity.value} />
            </div>
            <div style={secondaryTextStyle}>{draftContext.preview.summary}</div>
            <div style={secondaryTextStyle}>
              {draftDelta.length === 0 ? "Draft matches the live run." : `Draft changes: ${draftDelta.join(", ")}.`}
            </div>
            <div style={actionRowStyle}>
              <button className="metal-button" data-testid="seed-replay-inspector-randomize-btn" onClick={onRandomizeSeed} type="button">
                Randomize
              </button>
              <button className="metal-button" data-testid="seed-replay-inspector-reset-btn" onClick={onResetDraftToLive} type="button">
                Use live run
              </button>
              <button className="metal-button" data-testid="seed-replay-inspector-open-setup-btn" onClick={onOpenDraftInSetup} type="button">
                Open in setup
              </button>
              <button className="oracle-button" data-testid="seed-replay-inspector-start-btn" onClick={onStartDraftRun} type="button">
                Start draft
              </button>
              <button className="metal-button" data-testid="seed-replay-copy-draft-url-btn" onClick={() => void handleCopy("draft replay URL", draftContext.href)} type="button">
                Copy draft replay URL
              </button>
              <button className="metal-button" data-testid="seed-replay-copy-draft-setup-btn" onClick={() => void handleCopy("draft setup URL", draftSetupHref)} type="button">
                Copy draft setup URL
              </button>
            </div>
            <div style={linkBoxStyle}>{draftContext.href}</div>
          </Section>
        </aside>
      ) : null}
    </>
  );
}
