import React from "react";

type EscapeMenuProps = {
  onResume: () => void;
  onSave: () => void;
  onLoad: () => void;
  onNewRun: () => void;
};

export function EscapeMenu({ onResume, onSave, onLoad, onNewRun }: EscapeMenuProps) {
  return (
    <div className="escape-menu-backdrop" onClick={onResume}>
      <div className="escape-menu" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Game menu">
        <div className="escape-menu-title">Oracle of Delphi</div>
        <div className="escape-menu-buttons">
          <button className="oracle-button gold escape-menu-btn" id="escape-resume-btn" data-testid="escape-resume-btn" onClick={onResume} type="button">Resume</button>
          <button className="oracle-button escape-menu-btn" id="escape-save-btn" data-testid="escape-save-btn" onClick={onSave} type="button">Save</button>
          <button className="oracle-button escape-menu-btn" id="escape-load-btn" data-testid="escape-load-btn" onClick={onLoad} type="button">Load</button>
          <button className="oracle-button escape-menu-btn" id="escape-new-run-btn" data-testid="escape-new-run-btn" onClick={onNewRun} type="button">New Run</button>
        </div>
      </div>
    </div>
  );
}
