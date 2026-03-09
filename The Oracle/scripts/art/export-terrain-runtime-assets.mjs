#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { deflateSync, inflateSync } from "node:zlib";

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUTPUT_PIXEL_DIR = path.join(ROOT_DIR, "output", "art", "pixel", "terrain");
const PUBLIC_TERRAIN_DIR = path.join(ROOT_DIR, "apps", "web", "public", "assets", "precinct", "terrain");
const GENERATED_CONTENT_DIR = path.join(ROOT_DIR, "packages", "content", "src", "generated");
const GENERATED_TS_PATH = path.join(GENERATED_CONTENT_DIR, "terrainArt.ts");
const OUTPUT_MANIFEST_PATH = path.join(OUTPUT_PIXEL_DIR, "manifest.json");
const PUBLIC_MANIFEST_PATH = path.join(ROOT_DIR, "apps", "web", "public", "assets", "precinct", "terrain-art-manifest.json");

const TERRAIN_EXPORTS = [
  {
    terrainId: "dry_earth",
    sourceAssetId: "dry_earth_tiles",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "009-dry_earth_tiles.png"),
    variants: [
      { variantId: "v1", crop: { left: 426, top: 26, width: 288, height: 228 } },
      { variantId: "v2", crop: { left: 732, top: 36, width: 288, height: 228 } },
      { variantId: "v3", crop: { left: 1036, top: 24, width: 288, height: 228 } },
      { variantId: "v4", crop: { left: 584, top: 260, width: 288, height: 228 } }
    ]
  },
  {
    terrainId: "limestone_terrace",
    sourceAssetId: "limestone_terrace_tiles",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "008-limestone_terrace_tiles.png"),
    variants: [
      { variantId: "v1", crop: { left: 528, top: 12, width: 372, height: 266 } },
      { variantId: "v2", crop: { left: 904, top: 18, width: 372, height: 266 } },
      { variantId: "v3", crop: { left: 144, top: 308, width: 372, height: 266 } },
      { variantId: "v4", crop: { left: 650, top: 314, width: 372, height: 266 } }
    ]
  },
  {
    terrainId: "sacred_paving",
    sourceAssetId: "sacred_paving_tiles",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "010-sacred_paving_tiles.png"),
    variants: [
      { variantId: "v1", crop: { left: 1048, top: 608, width: 370, height: 280 } },
      { variantId: "v2", crop: { left: 648, top: 598, width: 370, height: 280 } },
      { variantId: "v3", crop: { left: 240, top: 602, width: 370, height: 280 } }
    ]
  },
  {
    terrainId: "cliff_edge",
    sourceAssetId: "cliff_edge_set",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "011-cliff_edge_set.png"),
    crop: { left: 44, top: 118, width: 282, height: 154 }
  },
  {
    terrainId: "retaining_wall",
    sourceAssetId: "retaining_wall_set",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "013-retaining_wall_set.png"),
    crop: { left: 598, top: 112, width: 260, height: 122 }
  },
  {
    terrainId: "stone_stairs",
    sourceAssetId: "stone_stair_set",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "012-stone_stair_set.png"),
    crop: { left: 420, top: 292, width: 700, height: 246 }
  },
  {
    terrainId: "spring_pool",
    sourceAssetId: "spring_pool_tiles",
    sourcePath: path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned", "014-spring_pool_tiles.png"),
    crop: { left: 650, top: 32, width: 246, height: 188 }
  }
];

function expandTerrainVariants(entry) {
  const variants = entry.variants ?? [{ variantId: "v1", crop: entry.crop }];
  const isMultiVariant = variants.length > 1;
  return variants.map((variant, index) => {
    const variantId = variant.variantId ?? `v${index + 1}`;
    const terrainAssetId = `${entry.terrainId}:${variantId}`;
    const fileName = isMultiVariant ? `${entry.terrainId}-${variantId}.png` : `${entry.terrainId}.png`;
    return {
      terrainId: entry.terrainId,
      terrainAssetId,
      variantId,
      sourceAssetId: entry.sourceAssetId,
      sourcePath: entry.sourcePath,
      crop: variant.crop,
      fileName
    };
  });
}

function printHelp() {
  console.log(`Terrain runtime asset exporter

Usage:
  node scripts/art/export-terrain-runtime-assets.mjs

This script extracts runtime terrain tiles from the tuned imagegen sheets,
writes them under output/art/pixel/terrain and apps/web/public/assets/precinct/terrain,
and generates packages/content/src/generated/terrainArt.ts.
`);
}

function parseArgs(argv) {
  return {
    help: argv.includes("--help") || argv.includes("-h")
  };
}

function paethPredictor(left, up, upLeft) {
  const predictor = left + up - upLeft;
  const leftDistance = Math.abs(predictor - left);
  const upDistance = Math.abs(predictor - up);
  const upLeftDistance = Math.abs(predictor - upLeft);

  if (leftDistance <= upDistance && leftDistance <= upLeftDistance) {
    return left;
  }
  if (upDistance <= upLeftDistance) {
    return up;
  }
  return upLeft;
}

function decodePng(buffer) {
  const signature = buffer.subarray(0, 8).toString("hex");
  if (signature !== "89504e470d0a1a0a") {
    throw new Error("Expected a PNG file when decoding terrain art.");
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatChunks = [];

  while (offset < buffer.length) {
    const chunkLength = buffer.readUInt32BE(offset);
    const chunkType = buffer.subarray(offset + 4, offset + 8).toString("ascii");
    const chunkData = buffer.subarray(offset + 8, offset + 8 + chunkLength);
    offset += 12 + chunkLength;

    if (chunkType === "IHDR") {
      width = chunkData.readUInt32BE(0);
      height = chunkData.readUInt32BE(4);
      bitDepth = chunkData.readUInt8(8);
      colorType = chunkData.readUInt8(9);
      continue;
    }

    if (chunkType === "IDAT") {
      idatChunks.push(chunkData);
      continue;
    }

    if (chunkType === "IEND") {
      break;
    }
  }

  if (bitDepth !== 8) {
    throw new Error(`Unsupported PNG bit depth ${bitDepth}; expected 8-bit data.`);
  }

  const channels = colorType === 6
    ? 4
    : colorType === 4
      ? 2
      : colorType === 2
        ? 3
        : colorType === 0
          ? 1
          : 0;
  if (channels === 0) {
    throw new Error(`Unsupported PNG color type ${colorType}; expected grayscale, RGB, grayscale+alpha, or RGBA.`);
  }

  const stride = width * channels;
  const inflated = inflateSync(Buffer.concat(idatChunks));
  const pixels = Buffer.alloc(height * stride);
  let sourceOffset = 0;

  for (let row = 0; row < height; row += 1) {
    const filter = inflated[sourceOffset];
    sourceOffset += 1;
    const rowStart = row * stride;

    for (let column = 0; column < stride; column += 1) {
      const raw = inflated[sourceOffset + column];
      const left = column >= channels ? pixels[rowStart + column - channels] : 0;
      const up = row > 0 ? pixels[rowStart + column - stride] : 0;
      const upLeft = row > 0 && column >= channels ? pixels[rowStart + column - stride - channels] : 0;
      let value = raw;

      if (filter === 1) {
        value = (raw + left) & 0xff;
      } else if (filter === 2) {
        value = (raw + up) & 0xff;
      } else if (filter === 3) {
        value = (raw + Math.floor((left + up) / 2)) & 0xff;
      } else if (filter === 4) {
        value = (raw + paethPredictor(left, up, upLeft)) & 0xff;
      }

      pixels[rowStart + column] = value;
    }

    sourceOffset += stride;
  }

  return { width, height, colorType, channels, pixels };
}

function toRgbaPixels(image) {
  const rgba = Buffer.alloc(image.width * image.height * 4);
  const alphaChannelIndex = image.colorType === 6 ? 3 : image.colorType === 4 ? 1 : -1;
  const stride = image.width * image.channels;

  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      const sourceOffset = y * stride + x * image.channels;
      const targetOffset = (y * image.width + x) * 4;

      if (image.colorType === 0 || image.colorType === 4) {
        const value = image.pixels[sourceOffset];
        rgba[targetOffset] = value;
        rgba[targetOffset + 1] = value;
        rgba[targetOffset + 2] = value;
      } else {
        rgba[targetOffset] = image.pixels[sourceOffset];
        rgba[targetOffset + 1] = image.pixels[sourceOffset + 1];
        rgba[targetOffset + 2] = image.pixels[sourceOffset + 2];
      }

      rgba[targetOffset + 3] = alphaChannelIndex >= 0
        ? image.pixels[sourceOffset + alphaChannelIndex]
        : 255;
    }
  }

  return rgba;
}

function averageCornerColor(rgba, width, height) {
  const sampleSize = Math.min(18, Math.max(4, Math.floor(Math.min(width, height) * 0.06)));
  const patches = [
    { startX: 0, startY: 0 },
    { startX: width - sampleSize, startY: 0 },
    { startX: 0, startY: height - sampleSize },
    { startX: width - sampleSize, startY: height - sampleSize }
  ];

  let red = 0;
  let green = 0;
  let blue = 0;
  let count = 0;

  for (const patch of patches) {
    for (let y = 0; y < sampleSize; y += 1) {
      for (let x = 0; x < sampleSize; x += 1) {
        const offset = ((patch.startY + y) * width + (patch.startX + x)) * 4;
        red += rgba[offset];
        green += rgba[offset + 1];
        blue += rgba[offset + 2];
        count += 1;
      }
    }
  }

  return {
    red: red / count,
    green: green / count,
    blue: blue / count
  };
}

function applyBackgroundKey(rgba, width, height) {
  const background = averageCornerColor(rgba, width, height);
  const keyed = Buffer.from(rgba);
  const hardThreshold = 30;
  const softThreshold = 62;
  const borderThreshold = 40;
  const visited = new Uint8Array(width * height);
  const stack = [];
  const enqueue = (x, y) => {
    if (x < 0 || x >= width || y < 0 || y >= height) {
      return;
    }
    const index = y * width + x;
    if (visited[index] === 1) {
      return;
    }
    visited[index] = 1;
    stack.push(index);
  };
  const colorDistanceAtIndex = (index) => {
    const offset = index * 4;
    return Math.hypot(
      keyed[offset] - background.red,
      keyed[offset + 1] - background.green,
      keyed[offset + 2] - background.blue
    );
  };

  for (let x = 0; x < width; x += 1) {
    enqueue(x, 0);
    enqueue(x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    enqueue(0, y);
    enqueue(width - 1, y);
  }

  while (stack.length > 0) {
    const index = stack.pop();
    if (index === undefined) {
      continue;
    }
    if (colorDistanceAtIndex(index) > borderThreshold) {
      continue;
    }

    const offset = index * 4;
    keyed[offset + 3] = 0;
    const x = index % width;
    const y = Math.floor(index / width);
    enqueue(x + 1, y);
    enqueue(x - 1, y);
    enqueue(x, y + 1);
    enqueue(x, y - 1);
  }

  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    if (keyed[offset + 3] === 0) {
      continue;
    }

    const distance = colorDistanceAtIndex(index);
    if (distance <= hardThreshold) {
      keyed[offset + 3] = 0;
      continue;
    }

    if (distance < softThreshold) {
      const fade = (distance - hardThreshold) / (softThreshold - hardThreshold);
      keyed[offset + 3] = Math.round(keyed[offset + 3] * fade);
    }

    const x = index % width;
    const y = Math.floor(index / width);
    const neighbors = [
      [x + 1, y],
      [x - 1, y],
      [x, y + 1],
      [x, y - 1]
    ];
    const touchesTransparent = neighbors.some(([nx, ny]) => {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        return true;
      }
      return keyed[(ny * width + nx) * 4 + 3] === 0;
    });

    if (touchesTransparent && distance < softThreshold + 10) {
      const fade = Math.max(0.25, (distance - hardThreshold) / (softThreshold + 10 - hardThreshold));
      keyed[offset + 3] = Math.min(keyed[offset + 3], Math.round(255 * fade));
    }
  }

  return keyed;
}

function cropRgba(width, height, rgba, crop) {
  const cropWidth = Math.max(1, Math.min(width, crop.width));
  const cropHeight = Math.max(1, Math.min(height, crop.height));
  const left = Math.max(0, Math.min(width - cropWidth, crop.left));
  const top = Math.max(0, Math.min(height - cropHeight, crop.top));
  const right = Math.min(width, left + crop.width);
  const bottom = Math.min(height, top + crop.height);
  const croppedWidth = Math.max(1, right - left);
  const croppedHeight = Math.max(1, bottom - top);
  const cropped = Buffer.alloc(croppedWidth * croppedHeight * 4);

  for (let y = 0; y < croppedHeight; y += 1) {
    const sourceStart = ((top + y) * width + left) * 4;
    const targetStart = y * croppedWidth * 4;
    rgba.copy(cropped, targetStart, sourceStart, sourceStart + croppedWidth * 4);
  }

  return {
    width: croppedWidth,
    height: croppedHeight,
    rgba: cropped
  };
}

function scanVisibleBounds(width, height, rgba) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixelOffset = (y * width + x) * 4;
      const alpha = rgba[pixelOffset + 3];
      if (alpha < 8) {
        continue;
      }

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (maxX < minX || maxY < minY) {
    return {
      trimWidth: width,
      trimHeight: height,
      trimOffsetX: 0,
      trimBottomInset: 0
    };
  }

  const trimWidth = maxX - minX + 1;
  const trimHeight = maxY - minY + 1;
  const trimCenterX = minX + trimWidth / 2;
  const trimOffsetX = width / 2 - trimCenterX;
  const trimBottomInset = height - (maxY + 1);

  return {
    trimWidth,
    trimHeight,
    trimOffsetX,
    trimBottomInset
  };
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (let index = 0; index < buffer.length; index += 1) {
    crc ^= buffer[index];
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc & 1) === 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function createChunk(type, data) {
  const typeBuffer = Buffer.from(type, "ascii");
  const lengthBuffer = Buffer.alloc(4);
  lengthBuffer.writeUInt32BE(data.length, 0);
  const crcBuffer = Buffer.alloc(4);
  crcBuffer.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([lengthBuffer, typeBuffer, data, crcBuffer]);
}

function encodePngRgba(width, height, rgba) {
  const signature = Buffer.from("89504e470d0a1a0a", "hex");
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.writeUInt8(8, 8);
  ihdr.writeUInt8(6, 9);
  ihdr.writeUInt8(0, 10);
  ihdr.writeUInt8(0, 11);
  ihdr.writeUInt8(0, 12);

  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let row = 0; row < height; row += 1) {
    const targetOffset = row * (stride + 1);
    raw[targetOffset] = 0;
    rgba.copy(raw, targetOffset + 1, row * stride, row * stride + stride);
  }

  return Buffer.concat([
    signature,
    createChunk("IHDR", ihdr),
    createChunk("IDAT", deflateSync(raw)),
    createChunk("IEND", Buffer.alloc(0))
  ]);
}

function toPosix(targetPath) {
  return targetPath.split(path.sep).join("/");
}

async function convertTerrainAsset(entry) {
  const outputFilePath = path.join(OUTPUT_PIXEL_DIR, entry.fileName);
  const publicFilePath = path.join(PUBLIC_TERRAIN_DIR, entry.fileName);
  const sourceBuffer = await fs.readFile(entry.sourcePath);
  const decoded = decodePng(sourceBuffer);
  const rgba = toRgbaPixels(decoded);
  const cropped = cropRgba(decoded.width, decoded.height, rgba, entry.crop);
  const keyed = applyBackgroundKey(cropped.rgba, cropped.width, cropped.height);
  const bounds = scanVisibleBounds(cropped.width, cropped.height, keyed);
  const normalizedBuffer = encodePngRgba(cropped.width, cropped.height, keyed);

  await fs.writeFile(outputFilePath, normalizedBuffer);
  await fs.copyFile(outputFilePath, publicFilePath);

  return {
    terrainId: entry.terrainId,
    terrainAssetId: entry.terrainAssetId,
    variantId: entry.variantId,
    sourceAssetId: entry.sourceAssetId,
    publicPath: `/assets/precinct/terrain/${entry.fileName}`,
    outputPath: toPosix(path.relative(ROOT_DIR, outputFilePath)),
    publicOutputPath: toPosix(path.relative(ROOT_DIR, publicFilePath)),
    sourceRenderPath: toPosix(path.relative(ROOT_DIR, entry.sourcePath)),
    width: cropped.width,
    height: cropped.height,
    trimWidth: bounds.trimWidth,
    trimHeight: bounds.trimHeight,
    trimOffsetX: Number(bounds.trimOffsetX.toFixed(2)),
    trimBottomInset: bounds.trimBottomInset
  };
}

function generateTypeScriptManifest(entries) {
  const serializedEntries = JSON.stringify(entries, null, 2);

  return `export type TerrainArtAsset = {
  terrainId: string;
  terrainAssetId: string;
  variantId: string;
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

const terrainArtEntries: TerrainArtAsset[] = ${serializedEntries};

export const terrainArtVariantsByTerrainId: Record<string, TerrainArtAsset[]> = terrainArtEntries.reduce((acc, entry) => {
  const existing = acc[entry.terrainId];
  if (existing) {
    existing.push(entry);
  } else {
    acc[entry.terrainId] = [entry];
  }
  return acc;
}, {} as Record<string, TerrainArtAsset[]>);

export const terrainArtByAssetId: Record<string, TerrainArtAsset> = Object.fromEntries(
  terrainArtEntries.map((entry) => [entry.terrainAssetId, entry])
);

export function getTerrainArt(terrainId: string): TerrainArtAsset | null {
  return terrainArtVariantsByTerrainId[terrainId]?.[0] ?? null;
}

export function getTerrainArtVariants(terrainId: string): TerrainArtAsset[] {
  return terrainArtVariantsByTerrainId[terrainId] ?? [];
}

export function getTerrainArtVariant(terrainId: string, variantId: string): TerrainArtAsset | null {
  return terrainArtVariantsByTerrainId[terrainId]?.find((entry) => entry.variantId === variantId) ?? null;
}

export function listTerrainArtAssets(): TerrainArtAsset[] {
  return terrainArtEntries;
}
`;
}

async function main() {
  const { help } = parseArgs(process.argv.slice(2));
  if (help) {
    printHelp();
    return;
  }

  await fs.mkdir(OUTPUT_PIXEL_DIR, { recursive: true });
  await fs.mkdir(PUBLIC_TERRAIN_DIR, { recursive: true });
  await fs.mkdir(GENERATED_CONTENT_DIR, { recursive: true });

  const convertedEntries = [];
  for (const entry of TERRAIN_EXPORTS) {
    const variants = expandTerrainVariants(entry);
    for (const variant of variants) {
      convertedEntries.push(await convertTerrainAsset(variant));
    }
  }

  const json = `${JSON.stringify(convertedEntries, null, 2)}\n`;
  await fs.writeFile(OUTPUT_MANIFEST_PATH, json, "utf8");
  await fs.writeFile(PUBLIC_MANIFEST_PATH, json, "utf8");
  await fs.writeFile(GENERATED_TS_PATH, generateTypeScriptManifest(convertedEntries), "utf8");

  console.log(`Exported ${convertedEntries.length} terrain runtime assets.`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
