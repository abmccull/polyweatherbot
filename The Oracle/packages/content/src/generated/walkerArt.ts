export type WalkerArtAsset = {
  walkerRole: string;
  sourceAssetId: string;
  publicPath: string;
  outputPath: string;
  publicOutputPath: string;
  sourceRenderPath: string;
  width: number;
  height: number;
  trimWidth: number;
  trimHeight: number;
  trimOffsetX: number;
  trimBottomInset: number;
};

const walkerArtEntries: WalkerArtAsset[] = [
  {
    "walkerRole": "priest",
    "sourceAssetId": "priest_attendant",
    "publicPath": "/assets/precinct/walkers/priest.png",
    "outputPath": "output/art/pixel/walkers/priest.png",
    "publicOutputPath": "apps/web/public/assets/precinct/walkers/priest.png",
    "sourceRenderPath": "output/art/imagegen/core_slice_tuned/016-priest_attendant.png",
    "width": 248,
    "height": 812,
    "trimWidth": 248,
    "trimHeight": 639,
    "trimOffsetX": 0,
    "trimBottomInset": 120
  },
  {
    "walkerRole": "custodian",
    "sourceAssetId": "custodian",
    "publicPath": "/assets/precinct/walkers/custodian.png",
    "outputPath": "output/art/pixel/walkers/custodian.png",
    "publicOutputPath": "apps/web/public/assets/precinct/walkers/custodian.png",
    "sourceRenderPath": "output/art/imagegen/core_slice_tuned/020-custodian.png",
    "width": 244,
    "height": 824,
    "trimWidth": 244,
    "trimHeight": 804,
    "trimOffsetX": 0,
    "trimBottomInset": 20
  },
  {
    "walkerRole": "carrier",
    "sourceAssetId": "carrier",
    "publicPath": "/assets/precinct/walkers/carrier.png",
    "outputPath": "output/art/pixel/walkers/carrier.png",
    "publicOutputPath": "apps/web/public/assets/precinct/walkers/carrier.png",
    "sourceRenderPath": "output/art/imagegen/core_slice_tuned/021-carrier.png",
    "width": 282,
    "height": 814,
    "trimWidth": 282,
    "trimHeight": 592,
    "trimOffsetX": 0,
    "trimBottomInset": 118
  },
  {
    "walkerRole": "pilgrim",
    "sourceAssetId": "ordinary_pilgrim",
    "publicPath": "/assets/precinct/walkers/pilgrim.png",
    "outputPath": "output/art/pixel/walkers/pilgrim.png",
    "publicOutputPath": "apps/web/public/assets/precinct/walkers/pilgrim.png",
    "sourceRenderPath": "output/art/imagegen/core_slice_tuned/022-ordinary_pilgrim.png",
    "width": 316,
    "height": 894,
    "trimWidth": 316,
    "trimHeight": 890,
    "trimOffsetX": 0,
    "trimBottomInset": 4
  }
];

export const walkerArtByRole: Record<string, WalkerArtAsset> = Object.fromEntries(
  walkerArtEntries.map((entry) => [entry.walkerRole, entry])
);

export function getWalkerArt(walkerRole: string): WalkerArtAsset | null {
  return walkerArtByRole[walkerRole] ?? null;
}

export function listWalkerArtAssets(): WalkerArtAsset[] {
  return walkerArtEntries;
}
