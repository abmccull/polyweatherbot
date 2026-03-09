export const portraitArtCatalogIds = [
  "hierophant_portrait",
  "treasurer_portrait",
  "builder_portrait",
  "diplomat_portrait",
  "shadow_portrait",
  "croesus_portrait",
  "alexander_portrait",
  "roman_envoy_portrait"
] as const;

export type PortraitArtId = (typeof portraitArtCatalogIds)[number];

export type PortraitArtAsset = {
  portraitId: PortraitArtId;
  sourcePortraitId: PortraitArtId;
  isFallback: boolean;
  fallbackSourcePortraitId: PortraitArtId | null;
  publicPath: string;
  outputPath: string;
  publicOutputPath: string;
  sourceKind: "manual" | "meshy" | "imagegen";
  sourceRenderPath: string;
  width: number;
  height: number;
  trimWidth: number;
  trimHeight: number;
  trimOffsetX: number;
  trimBottomInset: number;
};

const portraitArtEntries: PortraitArtAsset[] = [
  {
    "portraitId": "alexander_portrait",
    "sourcePortraitId": "builder_portrait",
    "isFallback": true,
    "fallbackSourcePortraitId": "builder_portrait",
    "publicPath": "/assets/precinct/portraits/alexander_portrait.png",
    "outputPath": "output/art/pixel/portraits/alexander_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/alexander_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/003-builder_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "builder_portrait",
    "sourcePortraitId": "builder_portrait",
    "isFallback": false,
    "fallbackSourcePortraitId": null,
    "publicPath": "/assets/precinct/portraits/builder_portrait.png",
    "outputPath": "output/art/pixel/portraits/builder_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/builder_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/003-builder_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "croesus_portrait",
    "sourcePortraitId": "treasurer_portrait",
    "isFallback": true,
    "fallbackSourcePortraitId": "treasurer_portrait",
    "publicPath": "/assets/precinct/portraits/croesus_portrait.png",
    "outputPath": "output/art/pixel/portraits/croesus_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/croesus_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/002-treasurer_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "diplomat_portrait",
    "sourcePortraitId": "diplomat_portrait",
    "isFallback": false,
    "fallbackSourcePortraitId": null,
    "publicPath": "/assets/precinct/portraits/diplomat_portrait.png",
    "outputPath": "output/art/pixel/portraits/diplomat_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/diplomat_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/004-diplomat_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "hierophant_portrait",
    "sourcePortraitId": "hierophant_portrait",
    "isFallback": false,
    "fallbackSourcePortraitId": null,
    "publicPath": "/assets/precinct/portraits/hierophant_portrait.png",
    "outputPath": "output/art/pixel/portraits/hierophant_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/hierophant_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/001-hierophant_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "roman_envoy_portrait",
    "sourcePortraitId": "diplomat_portrait",
    "isFallback": true,
    "fallbackSourcePortraitId": "diplomat_portrait",
    "publicPath": "/assets/precinct/portraits/roman_envoy_portrait.png",
    "outputPath": "output/art/pixel/portraits/roman_envoy_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/roman_envoy_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/004-diplomat_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "shadow_portrait",
    "sourcePortraitId": "shadow_portrait",
    "isFallback": false,
    "fallbackSourcePortraitId": null,
    "publicPath": "/assets/precinct/portraits/shadow_portrait.png",
    "outputPath": "output/art/pixel/portraits/shadow_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/shadow_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/005-shadow_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  },
  {
    "portraitId": "treasurer_portrait",
    "sourcePortraitId": "treasurer_portrait",
    "isFallback": false,
    "fallbackSourcePortraitId": null,
    "publicPath": "/assets/precinct/portraits/treasurer_portrait.png",
    "outputPath": "output/art/pixel/portraits/treasurer_portrait.png",
    "publicOutputPath": "apps/web/public/assets/precinct/portraits/treasurer_portrait.png",
    "sourceKind": "imagegen",
    "sourceRenderPath": "output/art/imagegen/portraits-core/002-treasurer_portrait.png",
    "width": 512,
    "height": 512,
    "trimWidth": 512,
    "trimHeight": 512,
    "trimOffsetX": 0,
    "trimBottomInset": 0
  }
];

export const portraitArtById: Partial<Record<PortraitArtId, PortraitArtAsset>> = Object.fromEntries(
  portraitArtEntries.map((entry) => [entry.portraitId, entry])
) as Partial<Record<PortraitArtId, PortraitArtAsset>>;

export const portraitArtMissingIds: PortraitArtId[] = portraitArtCatalogIds.filter((portraitId) => !portraitArtById[portraitId]);

export function getPortraitArt(portraitId: PortraitArtId | string): PortraitArtAsset | null {
  return portraitArtById[portraitId as PortraitArtId] ?? null;
}

export function listPortraitArtAssets(): PortraitArtAsset[] {
  return portraitArtEntries;
}
