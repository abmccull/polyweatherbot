export type PropArtAsset = {
  assetId: string;
  decorRole: string;
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

const propArtEntries: PropArtAsset[] = [
  {
    "assetId": "amphora_stack",
    "decorRole": "storage",
    "publicPath": "/assets/precinct/props/amphora_stack.png",
    "outputPath": "output/art/pixel/props/amphora_stack.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/amphora_stack.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/amphora_stack.png",
    "width": 128,
    "height": 128,
    "trimWidth": 73,
    "trimHeight": 65,
    "trimOffsetX": -10.5,
    "trimBottomInset": 32,
    "worldWidth": 34,
    "worldHeight": 34
  },
  {
    "assetId": "bronze_tripod",
    "decorRole": "ritual",
    "publicPath": "/assets/precinct/props/bronze_tripod.png",
    "outputPath": "output/art/pixel/props/bronze_tripod.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/bronze_tripod.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/bronze_tripod.png",
    "width": 128,
    "height": 128,
    "trimWidth": 71,
    "trimHeight": 72,
    "trimOffsetX": -11.5,
    "trimBottomInset": 27,
    "worldWidth": 30,
    "worldHeight": 40
  },
  {
    "assetId": "grain_sacks",
    "decorRole": "storage",
    "publicPath": "/assets/precinct/props/grain_sacks.png",
    "outputPath": "output/art/pixel/props/grain_sacks.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/grain_sacks.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/grain_sacks.png",
    "width": 160,
    "height": 160,
    "trimWidth": 90,
    "trimHeight": 69,
    "trimOffsetX": -7,
    "trimBottomInset": 25,
    "worldWidth": 48,
    "worldHeight": 34
  },
  {
    "assetId": "incense_censer",
    "decorRole": "ritual",
    "publicPath": "/assets/precinct/props/incense_censer.png",
    "outputPath": "output/art/pixel/props/incense_censer.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/incense_censer.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/incense_censer.png",
    "width": 128,
    "height": 128,
    "trimWidth": 51,
    "trimHeight": 61,
    "trimOffsetX": -7.5,
    "trimBottomInset": 31,
    "worldWidth": 28,
    "worldHeight": 40
  },
  {
    "assetId": "market_stall_set",
    "decorRole": "market",
    "publicPath": "/assets/precinct/props/market_stall_set.png",
    "outputPath": "output/art/pixel/props/market_stall_set.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/market_stall_set.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/market_stall_set.png",
    "width": 176,
    "height": 176,
    "trimWidth": 114,
    "trimHeight": 94,
    "trimOffsetX": -14,
    "trimBottomInset": 38,
    "worldWidth": 56,
    "worldHeight": 42
  },
  {
    "assetId": "offering_bowl",
    "decorRole": "offering",
    "publicPath": "/assets/precinct/props/offering_bowl.png",
    "outputPath": "output/art/pixel/props/offering_bowl.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/offering_bowl.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/offering_bowl.png",
    "width": 128,
    "height": 128,
    "trimWidth": 60,
    "trimHeight": 50,
    "trimOffsetX": -6,
    "trimBottomInset": 22,
    "worldWidth": 30,
    "worldHeight": 24
  },
  {
    "assetId": "oil_jars",
    "decorRole": "storage",
    "publicPath": "/assets/precinct/props/oil_jars.png",
    "outputPath": "output/art/pixel/props/oil_jars.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/oil_jars.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/oil_jars.png",
    "width": 144,
    "height": 144,
    "trimWidth": 73,
    "trimHeight": 63,
    "trimOffsetX": -2.5,
    "trimBottomInset": 24,
    "worldWidth": 44,
    "worldHeight": 30
  },
  {
    "assetId": "omphalos_stone",
    "decorRole": "ritual",
    "publicPath": "/assets/precinct/props/omphalos_stone.png",
    "outputPath": "output/art/pixel/props/omphalos_stone.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/omphalos_stone.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/omphalos_stone.png",
    "width": 128,
    "height": 128,
    "trimWidth": 128,
    "trimHeight": 92,
    "trimOffsetX": 0,
    "trimBottomInset": 6,
    "worldWidth": 34,
    "worldHeight": 34
  },
  {
    "assetId": "purification_font",
    "decorRole": "water",
    "publicPath": "/assets/precinct/props/purification_font.png",
    "outputPath": "output/art/pixel/props/purification_font.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/purification_font.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/purification_font.png",
    "width": 160,
    "height": 160,
    "trimWidth": 151,
    "trimHeight": 121,
    "trimOffsetX": -1.5,
    "trimBottomInset": 14,
    "worldWidth": 48,
    "worldHeight": 42
  },
  {
    "assetId": "ritual_basin",
    "decorRole": "water",
    "publicPath": "/assets/precinct/props/ritual_basin.png",
    "outputPath": "output/art/pixel/props/ritual_basin.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/ritual_basin.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/ritual_basin.png",
    "width": 128,
    "height": 128,
    "trimWidth": 80,
    "trimHeight": 65,
    "trimOffsetX": -7,
    "trimBottomInset": 22,
    "worldWidth": 40,
    "worldHeight": 28
  },
  {
    "assetId": "stone_bench",
    "decorRole": "seating",
    "publicPath": "/assets/precinct/props/stone_bench.png",
    "outputPath": "output/art/pixel/props/stone_bench.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/stone_bench.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/stone_bench.png",
    "width": 144,
    "height": 144,
    "trimWidth": 98,
    "trimHeight": 85,
    "trimOffsetX": -10,
    "trimBottomInset": 26,
    "worldWidth": 42,
    "worldHeight": 28
  },
  {
    "assetId": "votive_offering_rack",
    "decorRole": "offering",
    "publicPath": "/assets/precinct/props/votive_offering_rack.png",
    "outputPath": "output/art/pixel/props/votive_offering_rack.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/votive_offering_rack.png",
    "sourceKind": "manual",
    "sourceRenderPath": "output/art/renders/manual/votive_offering_rack.png",
    "width": 144,
    "height": 144,
    "trimWidth": 143,
    "trimHeight": 108,
    "trimOffsetX": 0.5,
    "trimBottomInset": 18,
    "worldWidth": 38,
    "worldHeight": 34
  },
  {
    "assetId": "votive_statue_small",
    "decorRole": "ritual",
    "publicPath": "/assets/precinct/props/votive_statue_small.png",
    "outputPath": "output/art/pixel/props/votive_statue_small.png",
    "publicOutputPath": "apps/web/public/assets/precinct/props/votive_statue_small.png",
    "sourceKind": "meshy",
    "sourceRenderPath": "output/art/renders/meshy/votive_statue_small.png",
    "width": 144,
    "height": 144,
    "trimWidth": 43,
    "trimHeight": 62,
    "trimOffsetX": -9.5,
    "trimBottomInset": 42,
    "worldWidth": 30,
    "worldHeight": 54
  }
];

export const propArtByAssetId: Record<string, PropArtAsset> = Object.fromEntries(
  propArtEntries.map((entry) => [entry.assetId, entry])
);

export function getPropArt(assetId: string): PropArtAsset | null {
  return propArtByAssetId[assetId] ?? null;
}

export function listPropArtAssets(): PropArtAsset[] {
  return propArtEntries;
}
