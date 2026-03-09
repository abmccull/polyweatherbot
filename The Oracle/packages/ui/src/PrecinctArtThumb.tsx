import { getBuildingArt } from "@the-oracle/content";
import type { BuildingDefId } from "@the-oracle/core";
import React from "react";

export function getPrecinctBuildingArtPath(defId: BuildingDefId): string | null {
  return getBuildingArt(defId)?.publicPath ?? null;
}

type PrecinctArtThumbProps = {
  defId: BuildingDefId;
  alt: string;
  className?: string;
};

export function PrecinctArtThumb({ defId, alt, className }: PrecinctArtThumbProps) {
  const src = getPrecinctBuildingArtPath(defId);
  if (!src) {
    return null;
  }

  return <img src={src} alt={alt} className={className} loading="lazy" decoding="async" />;
}
