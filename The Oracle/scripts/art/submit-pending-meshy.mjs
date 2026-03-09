#!/usr/bin/env node

import { appendFile, mkdir, readFile, readdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const MANIFEST_PATH = path.join(ROOT_DIR, 'art-library', 'generated', 'asset-manifest.json');
const DOWNLOADS_DIR = path.join(ROOT_DIR, 'output', 'art', 'meshy', 'downloads', 'text-to-3d');
const LEDGER_PATH = path.join(ROOT_DIR, 'output', 'art', 'meshy', 'submission-ledger.jsonl');

function parseArgs(argv) {
  const options = {
    limit: Infinity,
    submit: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'submit') {
      options.submit = true;
      continue;
    }
    const next = argv[index + 1];
    if (next && !next.startsWith('--')) {
      options[key] = next;
      index += 1;
    }
  }

  if (options.limit !== Infinity) {
    options.limit = Number(options.limit);
  }

  return options;
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

function meshyArgs(asset) {
  return [
    'scripts/art/meshy.mjs',
    'text-preview',
    '--asset-id',
    asset.assetId,
    '--prompt',
    asset.meshyPrompt,
    '--negative-prompt',
    asset.negativePrompt,
    '--art-style',
    'realistic',
    '--topology',
    'triangle',
  ];
}

async function listDownloadedAssetIds() {
  try {
    const entries = await readdir(DOWNLOADS_DIR, { withFileTypes: true });
    return new Set(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith('.glb'))
        .map((entry) => entry.name.replace(/\.glb$/i, ''))
    );
  } catch {
    return new Set();
  }
}

async function loadManifest() {
  const text = await readFile(MANIFEST_PATH, 'utf8');
  return JSON.parse(text);
}

async function appendLedger(entry) {
  await mkdir(path.dirname(LEDGER_PATH), { recursive: true });
  await appendFile(LEDGER_PATH, `${JSON.stringify(entry)}\n`, 'utf8');
}

async function runNode(args) {
  return new Promise((resolve, reject) => {
    const child = spawn('node', args, {
      cwd: ROOT_DIR,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdout += text;
      process.stdout.write(text);
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(text);
    });

    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`node ${args.join(' ')} exited with ${code}`));
    });
  });
}

function pickPendingAssets(manifest, downloaded, options) {
  let items = manifest.filter((asset) => String(asset.generationLane).includes('meshy'));
  if (options.wave) items = items.filter((asset) => asset.wave === options.wave);
  if (options.family) items = items.filter((asset) => asset.family === options.family);
  if (options.priority) items = items.filter((asset) => asset.priority === options.priority);
  if (options.asset) items = items.filter((asset) => asset.assetId === options.asset);

  items = items.filter((asset) => asset.meshyPrompt);
  items = items.filter((asset) => !downloaded.has(asset.assetId));

  if (Number.isFinite(options.limit)) {
    items = items.slice(0, options.limit);
  }

  return items;
}

function parseTaskId(stdout) {
  const match = stdout.match(/Created Meshy text-to-3d preview task:\s+([a-z0-9-]+)/i);
  return match?.[1] ?? null;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifest = await loadManifest();
  const downloaded = await listDownloadedAssetIds();
  const pending = pickPendingAssets(manifest, downloaded, options);

  if (pending.length === 0) {
    console.log('No pending Meshy assets matched the current filters.');
    return;
  }

  if (!options.submit) {
    console.log(`Dry run: ${pending.length} pending Meshy assets`);
    for (const asset of pending) {
      const command = ['node', ...meshyArgs(asset)].map(shellQuote).join(' ');
      console.log(command);
    }
    return;
  }

  console.log(`Submitting ${pending.length} pending Meshy assets...`);
  for (const asset of pending) {
    console.log(`Submitting ${asset.assetId} (${asset.wave}/${asset.family}/${asset.priority})`);
    const startedAt = new Date().toISOString();
    const { stdout } = await runNode(meshyArgs(asset));
    const taskId = parseTaskId(stdout);
    await appendLedger({
      startedAt,
      assetId: asset.assetId,
      wave: asset.wave,
      family: asset.family,
      priority: asset.priority,
      taskId,
    });
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
