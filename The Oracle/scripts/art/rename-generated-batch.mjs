#!/usr/bin/env node

import { readdir, readFile, rename } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[index + 1];
    if (next && !next.startsWith('--')) {
      options[key] = next;
      index += 1;
    }
  }
  return options;
}

function requireOption(options, key) {
  const value = options[key];
  if (!value) throw new Error(`Missing required option: --${key}`);
  return value;
}

function resolvePath(value) {
  return path.isAbsolute(value) ? value : path.join(ROOT_DIR, value);
}

function sanitizeAssetId(value) {
  return String(value).replace(/[^a-z0-9._-]+/gi, '-').toLowerCase();
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const jsonlPath = resolvePath(requireOption(options, 'jsonl'));
  const dirPath = resolvePath(requireOption(options, 'dir'));
  const entries = (await readFile(jsonlPath, 'utf8'))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const files = await readdir(dirPath);

  for (let index = 0; index < entries.length; index += 1) {
    const prefix = `${String(index + 1).padStart(3, '0')}-`;
    const match = files.find((file) => file.startsWith(prefix));
    if (!match) continue;
    const extension = path.extname(match);
    const target = `${String(index + 1).padStart(3, '0')}-${sanitizeAssetId(entries[index].asset_id ?? `asset-${index + 1}`)}${extension}`;
    if (match === target) continue;
    await rename(path.join(dirPath, match), path.join(dirPath, target));
    console.log(`${match} -> ${target}`);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
