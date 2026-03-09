import React from "react";

interface OverlayTriggerStripProps {
  activeOverlay: string | null;
  onToggleOverlay: (overlayId: string) => void;
  notifications?: Record<string, boolean>;
}

const OVERLAY_TABS = [
  { id: "oracle", label: "Oracle", icon: "\u2609", shortcutKey: "O" },
  { id: "world", label: "World", icon: "\u2641", shortcutKey: "W" },
  { id: "stores", label: "Stores", icon: "\u2696", shortcutKey: "S" },
  { id: "priests", label: "Priests", icon: "\u2628", shortcutKey: "P" },
  { id: "espionage", label: "Espionage", icon: "\u2694", shortcutKey: "E" },
  { id: "record", label: "Record", icon: "\u270D", shortcutKey: "R" },
  { id: "legacy", label: "Legacy", icon: "\u2726", shortcutKey: "L" },
  { id: "lineage", label: "Lineage", icon: "\u2042", shortcutKey: "I" },
] as const;

export function OverlayTriggerStrip({
  activeOverlay,
  onToggleOverlay,
  notifications,
}: OverlayTriggerStripProps) {
  return (
    <aside className="overlay-trigger-strip">
      {OVERLAY_TABS.map((tab) => (
        <button
          key={tab.id}
          id={`trigger-${tab.id}`}
          className={`trigger-strip-button${activeOverlay === tab.id ? " active" : ""}`}
          title={`${tab.label} (${tab.shortcutKey})`}
          aria-label={`${tab.label} overlay (${tab.shortcutKey})`}
          aria-pressed={activeOverlay === tab.id}
          onClick={() => onToggleOverlay(tab.id)}
          type="button"
        >
          {tab.icon}
          <span>{tab.label}</span>
          <kbd className="trigger-strip-kbd">{tab.shortcutKey}</kbd>
          {notifications?.[tab.id] ? <span className="trigger-strip-dot" /> : null}
        </button>
      ))}
    </aside>
  );
}
