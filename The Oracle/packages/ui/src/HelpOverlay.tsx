import React from "react";

const SHORTCUT_GROUPS = [
  {
    title: "Navigation",
    shortcuts: [
      { key: "Esc", desc: "Open menu / close overlay" },
      { key: "M", desc: "Toggle minimap" },
      { key: "?", desc: "This help screen" },
    ],
  },
  {
    title: "Overlays",
    shortcuts: [
      { key: "O", desc: "Oracle" },
      { key: "W", desc: "World" },
      { key: "S", desc: "Stores" },
      { key: "P", desc: "Priests" },
      { key: "E", desc: "Espionage" },
      { key: "R", desc: "Record" },
      { key: "L", desc: "Legacy" },
      { key: "I", desc: "Lineage" },
    ],
  },
  {
    title: "Speed",
    shortcuts: [
      { key: "1", desc: "Normal speed" },
      { key: "2", desc: "Fast" },
      { key: "3", desc: "Fastest" },
      { key: "Space", desc: "Pause / resume" },
    ],
  },
] as const;

interface HelpOverlayProps {
  onClose: () => void;
}

export function HelpOverlay({ onClose }: HelpOverlayProps) {
  return (
    <div className="escape-menu-backdrop" onClick={onClose}>
      <div
        className="help-overlay"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
      >
        <div className="escape-menu-title">Keyboard Shortcuts</div>
        <div className="help-groups">
          {SHORTCUT_GROUPS.map((group) => (
            <div key={group.title} className="help-group">
              <div className="section-title">{group.title}</div>
              {group.shortcuts.map((s) => (
                <div key={s.key} className="help-shortcut-row">
                  <kbd className="help-kbd">{s.key}</kbd>
                  <span>{s.desc}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <button className="oracle-button" onClick={onClose} type="button" style={{ marginTop: 12 }}>
          Close
        </button>
      </div>
    </div>
  );
}
