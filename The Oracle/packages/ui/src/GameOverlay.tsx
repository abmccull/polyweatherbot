import React, { useEffect } from "react";

type GameOverlayProps = {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  width?: "narrow" | "medium" | "wide";
};

export function GameOverlay({ title, onClose, children, width = "medium" }: GameOverlayProps) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="game-overlay-backdrop" onClick={onClose}>
      <div
        className={`game-overlay game-overlay-${width}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="game-overlay-header">
          <h2>{title}</h2>
          <button className="game-overlay-close" onClick={onClose} type="button" aria-label="Close overlay">
            {"\u2715"}
          </button>
        </div>
        <div className="game-overlay-content">{children}</div>
      </div>
    </div>
  );
}
