#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { deflateSync, inflateSync } from "node:zlib";

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUTPUT_PIXEL_DIR = path.join(ROOT_DIR, "output", "art", "pixel", "buildings");
const PUBLIC_BUILDING_DIR = path.join(ROOT_DIR, "apps", "web", "public", "assets", "precinct", "buildings");
const GENERATED_CONTENT_DIR = path.join(ROOT_DIR, "packages", "content", "src", "generated");
const GENERATED_TS_PATH = path.join(GENERATED_CONTENT_DIR, "buildingArt.ts");
const OUTPUT_MANIFEST_PATH = path.join(OUTPUT_PIXEL_DIR, "manifest.json");
const PUBLIC_MANIFEST_PATH = path.join(ROOT_DIR, "apps", "web", "public", "assets", "precinct", "building-art-manifest.json");
const ASSET_MANIFEST_PATH = path.join(ROOT_DIR, "art-library", "generated", "asset-manifest.json");
const IMAGEGEN_SOURCE_DIRS = [
  path.join(ROOT_DIR, "output", "art", "imagegen", "core_slice_tuned")
];

const DEF_ID_ALIASES = {
  sacred_way: "sacred_way_kit"
};

let imagegenSourceIndexPromise;

function printHelp() {
  console.log(`Building pixel asset exporter

Usage:
  node scripts/art/export-building-pixel-assets.mjs

This script exports built building renders into runtime PNGs, copies them
into apps/web/public/assets/precinct/buildings, and generates a code manifest under
packages/content/src/generated/buildingArt.ts.
`);
}

function parseArgs(argv) {
  return {
    help: argv.includes("--help") || argv.includes("-h")
  };
}

async function pathExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT_DIR,
      stdio: "inherit",
      env: process.env
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${command} exited with code ${code}`));
    });
  });
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function getImagegenSourceIndex() {
  if (imagegenSourceIndexPromise) {
    return imagegenSourceIndexPromise;
  }

  imagegenSourceIndexPromise = (async () => {
    const index = new Map();

    for (const sourceDir of IMAGEGEN_SOURCE_DIRS) {
      if (!await pathExists(sourceDir)) {
        continue;
      }

      const fileNames = await fs.readdir(sourceDir);
      for (const fileName of fileNames) {
        const match = fileName.match(/^\d+-(.+)\.png$/);
        if (!match) {
          continue;
        }

        const assetId = match[1];
        if (!index.has(assetId)) {
          index.set(assetId, path.join(sourceDir, fileName));
        }
      }
    }

    return index;
  })();

  return imagegenSourceIndexPromise;
}

function getPngDimensions(buffer) {
  const signature = buffer.subarray(0, 8).toString("hex");
  if (signature !== "89504e470d0a1a0a") {
    throw new Error("Expected a PNG file when reading generated pixel asset dimensions.");
  }
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20)
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
    throw new Error("Expected a PNG file when decoding runtime art bounds.");
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
    throw new Error(`Unsupported PNG bit depth ${bitDepth}; expected 8-bit RGBA/RGB data.`);
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

  const bytesPerPixel = channels;
  const stride = width * bytesPerPixel;
  const inflated = inflateSync(Buffer.concat(idatChunks));
  const pixels = Buffer.alloc(height * stride);
  let sourceOffset = 0;

  for (let row = 0; row < height; row += 1) {
    const filter = inflated[sourceOffset];
    sourceOffset += 1;
    const rowStart = row * stride;

    for (let column = 0; column < stride; column += 1) {
      const raw = inflated[sourceOffset + column];
      const left = column >= bytesPerPixel ? pixels[rowStart + column - bytesPerPixel] : 0;
      const up = row > 0 ? pixels[rowStart + column - stride] : 0;
      const upLeft = row > 0 && column >= bytesPerPixel ? pixels[rowStart + column - stride - bytesPerPixel] : 0;
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
  const sampleSize = Math.min(10, Math.max(2, Math.floor(Math.min(width, height) * 0.05)));
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
  const hardThreshold = 18;
  const softThreshold = 52;

  for (let offset = 0; offset < keyed.length; offset += 4) {
    const distance = Math.hypot(
      keyed[offset] - background.red,
      keyed[offset + 1] - background.green,
      keyed[offset + 2] - background.blue
    );

    if (distance <= hardThreshold) {
      keyed[offset + 3] = 0;
      continue;
    }

    if (distance < softThreshold) {
      const fade = (distance - hardThreshold) / (softThreshold - hardThreshold);
      keyed[offset + 3] = Math.round(keyed[offset + 3] * fade);
    }
  }

  return keyed;
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

function resolveOutputSize(footprint) {
  if (footprint === "tile_kit") {
    return 96;
  }
  if (footprint === "landmark") {
    return 320;
  }

  const match = footprint.match(/^(\d+)x(\d+)$/);
  if (!match) {
    return 224;
  }

  const widthTiles = Number(match[1]);
  const heightTiles = Number(match[2]);
  const footprintSpan = widthTiles + heightTiles;
  return Math.min(320, 128 + (footprintSpan - 2) * 32 + (Math.max(widthTiles, heightTiles) - 1) * 16);
}

async function resolveSourceRender(assetId) {
  const imagegenSourceIndex = await getImagegenSourceIndex();
  const imagegenPath = imagegenSourceIndex.get(assetId);
  if (imagegenPath) {
    return { sourceKind: "imagegen", sourceRenderPath: imagegenPath };
  }

  const manualPath = path.join(ROOT_DIR, "output", "art", "renders", "manual", `${assetId}.png`);
  if (await pathExists(manualPath)) {
    return { sourceKind: "manual", sourceRenderPath: manualPath };
  }

  const meshyPath = path.join(ROOT_DIR, "output", "art", "renders", "meshy", `${assetId}.png`);
  if (await pathExists(meshyPath)) {
    return { sourceKind: "meshy", sourceRenderPath: meshyPath };
  }

  throw new Error(`No source render found for ${assetId}`);
}

function toPosix(targetPath) {
  return targetPath.split(path.sep).join("/");
}

async function convertBuildingAsset(entry) {
  const { assetId, footprint } = entry;
  const { sourceKind, sourceRenderPath } = await resolveSourceRender(assetId);
  const outputFilePath = path.join(OUTPUT_PIXEL_DIR, `${assetId}.png`);
  const publicFilePath = path.join(PUBLIC_BUILDING_DIR, `${assetId}.png`);
  const outputSize = resolveOutputSize(footprint);

  await run("sips", [
    "-z",
    String(outputSize),
    String(outputSize),
    sourceRenderPath,
    "--out",
    outputFilePath,
  ]);

  const outputBuffer = await fs.readFile(outputFilePath);
  const decoded = decodePng(outputBuffer);
  let rgba = toRgbaPixels(decoded);
  if (sourceKind === "imagegen") {
    rgba = applyBackgroundKey(rgba, decoded.width, decoded.height);
  }

  const normalizedBuffer = encodePngRgba(decoded.width, decoded.height, rgba);
  await fs.writeFile(outputFilePath, normalizedBuffer);
  await fs.copyFile(outputFilePath, publicFilePath);

  const bounds = scanVisibleBounds(decoded.width, decoded.height, rgba);

  return {
    assetId,
    footprint,
    publicPath: `/assets/precinct/buildings/${assetId}.png`,
    outputPath: toPosix(path.relative(ROOT_DIR, outputFilePath)),
    publicOutputPath: toPosix(path.relative(ROOT_DIR, publicFilePath)),
    sourceKind,
    sourceRenderPath: toPosix(path.relative(ROOT_DIR, sourceRenderPath)),
    width: decoded.width,
    height: decoded.height,
    trimWidth: bounds.trimWidth,
    trimHeight: bounds.trimHeight,
    trimOffsetX: Number(bounds.trimOffsetX.toFixed(2)),
    trimBottomInset: bounds.trimBottomInset
  };
}

function generateTypeScriptManifest(entries) {
  const sortedEntries = [...entries].sort((left, right) => left.assetId.localeCompare(right.assetId));
  const serializedEntries = JSON.stringify(sortedEntries, null, 2);
  const serializedAliases = JSON.stringify(DEF_ID_ALIASES, null, 2);

  return `export type BuildingArtAsset = {
  assetId: string;
  footprint: string;
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

const buildingArtEntries: BuildingArtAsset[] = ${serializedEntries};

export const buildingArtByAssetId: Record<string, BuildingArtAsset> = Object.fromEntries(
  buildingArtEntries.map((entry) => [entry.assetId, entry])
);

export const buildingArtAliasesByDefId: Record<string, string> = ${serializedAliases};

export function getBuildingArt(defId: string): BuildingArtAsset | null {
  const assetId = buildingArtAliasesByDefId[defId] ?? defId;
  return buildingArtByAssetId[assetId] ?? null;
}

export function listBuildingArtAssets(): BuildingArtAsset[] {
  return buildingArtEntries;
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
  await fs.mkdir(PUBLIC_BUILDING_DIR, { recursive: true });
  await fs.mkdir(GENERATED_CONTENT_DIR, { recursive: true });

  const assetManifest = await readJson(ASSET_MANIFEST_PATH);
  const buildingEntries = assetManifest.filter((entry) => entry.family === "building");
  const convertedEntries = [];

  for (const entry of buildingEntries) {
    convertedEntries.push(await convertBuildingAsset(entry));
  }

  const json = `${JSON.stringify(convertedEntries, null, 2)}\n`;
  await fs.writeFile(OUTPUT_MANIFEST_PATH, json, "utf8");
  await fs.writeFile(PUBLIC_MANIFEST_PATH, json, "utf8");
  await fs.writeFile(GENERATED_TS_PATH, generateTypeScriptManifest(convertedEntries), "utf8");

  console.log(`Exported ${convertedEntries.length} building runtime assets.`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
