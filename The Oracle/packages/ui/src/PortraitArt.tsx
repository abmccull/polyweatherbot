import { getPortraitArt, type PortraitArtId } from "@the-oracle/content";
import React from "react";

type PortraitArtProps = {
  portraitId: PortraitArtId | string | null | undefined;
  alt: string;
  className?: string;
  imgClassName?: string;
};

export function resolveFactionPortraitId(factionId?: string, profile?: string): PortraitArtId {
  if (factionId === "athens") {
    return "roman_envoy_portrait";
  }
  if (factionId === "sparta") {
    return "alexander_portrait";
  }
  if (factionId === "corinth") {
    return "croesus_portrait";
  }
  if (factionId === "thebes") {
    return "hierophant_portrait";
  }
  if (factionId === "miletus") {
    return "diplomat_portrait";
  }
  if (factionId === "syracuse") {
    return "diplomat_portrait";
  }
  if (factionId === "macedon") {
    return "alexander_portrait";
  }
  if (factionId === "argos") {
    return "treasurer_portrait";
  }
  if (profile === "mercantile") {
    return "croesus_portrait";
  }
  if (profile === "martial") {
    return "alexander_portrait";
  }
  if (profile === "scheming") {
    return "shadow_portrait";
  }
  return "diplomat_portrait";
}

export function PortraitArt({ portraitId, alt, className, imgClassName }: PortraitArtProps) {
  const src = portraitId ? getPortraitArt(portraitId)?.publicPath ?? null : null;
  if (!src) {
    return null;
  }

  return (
    <div className={className}>
      <img
        src={src}
        alt={alt}
        className={imgClassName}
        loading="lazy"
        decoding="async"
      />
    </div>
  );
}
