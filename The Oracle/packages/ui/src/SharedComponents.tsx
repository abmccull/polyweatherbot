import React from "react";

export function CredibilityPips({ value }: { value: number }) {
  const filled = Math.round(Math.min(5, (value / 20) * 5));
  return (
    <span className="credibility-pips">
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} className={`pip ${i < filled ? "filled" : "empty"}`} />
      ))}
    </span>
  );
}

export function trendArrow(trend: number) {
  if (trend > 0.01) return <span className="trend-arrow up" title={`+${trend.toFixed(2)}/tick`}>{"\u25B2"}</span>;
  if (trend < -0.01) return <span className="trend-arrow down" title={`${trend.toFixed(2)}/tick`}>{"\u25BC"}</span>;
  return <span className="trend-arrow flat">{"\u25C6"}</span>;
}
