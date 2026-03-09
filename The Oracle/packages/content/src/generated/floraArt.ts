export type FloraArtAsset = {
  assetId: string;
  floraRole: string;
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
  worldWidth: number;
  worldHeight: number;
};

const floraArtEntries: FloraArtAsset[] = [
  {
    "assetId": "cypress_tree",
    "floraRole": "tree",
    "publicPath": "/assets/precinct/flora/cypress_tree.png",
    "outputPath": "output/art/pixel/flora/cypress_tree.png",
    "publicOutputPath": "apps/web/public/assets/precinct/flora/cypress_tree.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/cypress_tree.png",
    "width": 192,
    "height": 192,
    "trimWidth": 72,
    "trimHeight": 97,
    "trimOffsetX": -5,
    "trimBottomInset": 44,
    "worldWidth": 42,
    "worldHeight": 108
  },
  {
    "assetId": "dry_grass_cluster",
    "floraRole": "grass",
    "publicPath": "/assets/precinct/flora/dry_grass_cluster.png",
    "outputPath": "output/art/pixel/flora/dry_grass_cluster.png",
    "publicOutputPath": "apps/web/public/assets/precinct/flora/dry_grass_cluster.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/dry_grass_cluster.png",
    "width": 96,
    "height": 96,
    "trimWidth": 41,
    "trimHeight": 49,
    "trimOffsetX": -2.5,
    "trimBottomInset": 18,
    "worldWidth": 24,
    "worldHeight": 28
  },
  {
    "assetId": "laurel_shrub",
    "floraRole": "shrub",
    "publicPath": "/assets/precinct/flora/laurel_shrub.png",
    "outputPath": "output/art/pixel/flora/laurel_shrub.png",
    "publicOutputPath": "apps/web/public/assets/precinct/flora/laurel_shrub.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/laurel_shrub.png",
    "width": 144,
    "height": 144,
    "trimWidth": 83,
    "trimHeight": 76,
    "trimOffsetX": -13.5,
    "trimBottomInset": 33,
    "worldWidth": 42,
    "worldHeight": 42
  },
  {
    "assetId": "rocky_outcrop",
    "floraRole": "rock",
    "publicPath": "/assets/precinct/flora/rocky_outcrop.png",
    "outputPath": "output/art/pixel/flora/rocky_outcrop.png",
    "publicOutputPath": "apps/web/public/assets/precinct/flora/rocky_outcrop.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/rocky_outcrop.png",
    "width": 144,
    "height": 144,
    "trimWidth": 69,
    "trimHeight": 52,
    "trimOffsetX": -1.5,
    "trimBottomInset": 28,
    "worldWidth": 50,
    "worldHeight": 30
  }
];

export const floraArtByAssetId: Record<string, FloraArtAsset> = Object.fromEntries(
  floraArtEntries.map((entry) => [entry.assetId, entry])
);

export function getFloraArt(assetId: string): FloraArtAsset | null {
  return floraArtByAssetId[assetId] ?? null;
}

export function listFloraArtAssets(): FloraArtAsset[] {
  return floraArtEntries;
}
