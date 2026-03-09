#!/usr/bin/env node

import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const BUILDING_EXPORTER_PATH = path.join(ROOT_DIR, "scripts", "art", "export-building-pixel-assets.mjs");

function printHelp() {
  console.log(`Pixel art helper

Usage:
  pnpm art:pixel help
  pnpm art:pixel convert --input path/to/source.png --output path/to/output.png
  pnpm art:pixel buildings
  pnpm art:pixel web-ui

The convert command runs proper-pixel-art via uvx.
The buildings command delegates to the tracked batch exporter and writes:
  - apps/web/public/assets/precinct/buildings/
  - output/art/pixel/buildings/
  - packages/content/src/generated/buildingArt.ts
  - apps/web/public/assets/precinct/building-art-manifest.json

Any extra flags are passed through to ppa.
`);
}

function parseArgs(argv) {
  const [command = "help", ...rest] = argv;
  const options = {};
  const passthrough = [];
  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (!token.startsWith("--")) {
      passthrough.push(token);
      continue;
    }
    const key = token.slice(2);
    const next = rest[index + 1];
    if (!next || next.startsWith("--")) {
      options[key] = "true";
      continue;
    }
    options[key] = next;
    index += 1;
  }
  return { command, options, passthrough };
}

function requireOption(options, key) {
  const value = options[key];
  if (!value) {
    throw new Error(`Missing required option: --${key}`);
  }
  return value;
}

function resolveFromRoot(targetPath) {
  return path.isAbsolute(targetPath) ? targetPath : path.resolve(ROOT_DIR, targetPath);
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

function conversionArgs(inputPath, outputPath, options, passthrough) {
  const args = ["--from", "proper-pixel-art", "ppa", "-i", inputPath, "-o", outputPath];
  const pixelWidth = options["pixel-width"];
  if (pixelWidth) {
    args.push("-w", pixelWidth);
  }
  return [...args, ...passthrough];
}

async function ensureParent(filePath) {
  await mkdir(path.dirname(filePath), { recursive: true });
}

async function convertFile(inputPath, outputPath, options, passthrough) {
  await ensureParent(outputPath);
  await run("uvx", conversionArgs(inputPath, outputPath, options, passthrough));
}

async function convert(options, passthrough) {
  const input = resolveFromRoot(requireOption(options, "input"));
  const output = resolveFromRoot(requireOption(options, "output"));
  await convertFile(input, output, options, passthrough);
}

async function buildings(options, passthrough) {
  if (Object.keys(options).length > 0 || passthrough.length > 0) {
    throw new Error("pnpm art:pixel buildings does not accept extra flags. Use the tracked exporter script if you need to modify the batch.");
  }
  await run("node", [BUILDING_EXPORTER_PATH]);
}

async function webUi() {
  await run("uvx", ["--from", "proper-pixel-art[web]", "ppa-web"]);
}

async function main() {
  const { command, options, passthrough } = parseArgs(process.argv.slice(2));
  switch (command) {
    case "help":
    case "--help":
    case "-h":
      printHelp();
      break;
    case "convert":
      await convert(options, passthrough);
      break;
    case "buildings":
      await buildings(options, passthrough);
      break;
    case "web-ui":
      await webUi();
      break;
    default:
      throw new Error(`Unknown command: ${command}`);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
