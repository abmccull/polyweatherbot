#!/usr/bin/env node

import { appendFile, mkdir, readFile, readdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const LEDGER_PATH = path.join(ROOT_DIR, 'output', 'art', 'meshy', 'submission-ledger.jsonl');
const DOWNLOADS_DIR = path.join(ROOT_DIR, 'output', 'art', 'meshy', 'downloads', 'text-to-3d');
const HARVEST_LEDGER_PATH = path.join(ROOT_DIR, 'output', 'art', 'meshy', 'harvest-ledger.jsonl');

function parseArgs(argv) {
  const options = { limit: Infinity };
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
  if (options.limit !== Infinity) options.limit = Number(options.limit);
  return options;
}

async function readJsonLines(filePath) {
  const text = await readFile(filePath, 'utf8');
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
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

function parseStatus(stdout) {
  const start = stdout.indexOf('{');
  const end = stdout.lastIndexOf('}');
  if (start === -1 || end === -1) return null;
  return JSON.parse(stdout.slice(start, end + 1));
}

async function appendHarvestLedger(entry) {
  await mkdir(path.dirname(HARVEST_LEDGER_PATH), { recursive: true });
  await appendFile(HARVEST_LEDGER_PATH, `${JSON.stringify(entry)}\n`, 'utf8');
}

function sortEntries(entries, options) {
  let items = entries;
  if (options.priority) items = items.filter((entry) => entry.priority === options.priority);
  if (options.wave) items = items.filter((entry) => entry.wave === options.wave);
  if (options.family) items = items.filter((entry) => entry.family === options.family);
  if (Number.isFinite(options.limit)) items = items.slice(0, options.limit);
  return items;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const ledger = await readJsonLines(LEDGER_PATH);
  const downloaded = await listDownloadedAssetIds();
  const pending = sortEntries(ledger, options).filter((entry) => !downloaded.has(entry.assetId));

  if (pending.length === 0) {
    console.log('No undownloaded assets matched the current filters.');
    return;
  }

  console.log(`Polling ${pending.length} submitted Meshy assets...`);
  for (const entry of pending) {
    console.log(`Checking ${entry.assetId} (${entry.taskId})`);
    const { stdout } = await runNode([
      'scripts/art/meshy.mjs',
      'status',
      '--kind',
      'text-to-3d',
      '--task-id',
      entry.taskId,
    ]);
    const status = parseStatus(stdout);
    if (!status) continue;

    await appendHarvestLedger({
      checkedAt: new Date().toISOString(),
      assetId: entry.assetId,
      taskId: entry.taskId,
      status: status.status,
    });

    if (status.status !== 'SUCCEEDED') continue;

    await runNode([
      'scripts/art/meshy.mjs',
      'download',
      '--kind',
      'text-to-3d',
      '--task-id',
      entry.taskId,
      '--format',
      'glb',
      '--asset-id',
      entry.assetId,
    ]);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
