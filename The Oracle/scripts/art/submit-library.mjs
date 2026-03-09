#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const MANIFEST_PATH = path.join(ROOT_DIR, 'art-library', 'generated', 'asset-manifest.json');

function parseArgs(argv) {
  const options = { provider: 'meshy', limit: 5, submit: false };
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
  options.limit = Number(options.limit ?? 5);
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

async function runNode(args) {
  await new Promise((resolve, reject) => {
    const child = spawn('node', args, { cwd: ROOT_DIR, stdio: 'inherit', env: process.env });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`node ${args.join(' ')} exited with ${code}`));
    });
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, 'utf8'));
  let items = manifest.filter((asset) => String(asset.generationLane).includes(options.provider));
  if (options.wave) items = items.filter((asset) => asset.wave === options.wave);
  if (options.family) items = items.filter((asset) => asset.family === options.family);
  if (options.priority) items = items.filter((asset) => asset.priority === options.priority);
  items = items.filter((asset) => asset.meshyPrompt);
  items = items.slice(0, options.limit);

  if (items.length === 0) {
    throw new Error('No matching assets found. Run `pnpm art:library` first or change filters.');
  }

  if (!options.submit) {
    console.log(`Dry run: ${items.length} ${options.provider} jobs`);
    for (const asset of items) {
      const command = ['node', ...meshyArgs(asset)].map(shellQuote).join(' ');
      console.log(command);
    }
    return;
  }

  if (options.provider !== 'meshy') {
    throw new Error('Only Meshy submission is supported in this script right now.');
  }

  for (const asset of items) {
    console.log(`Submitting ${asset.assetId}...`);
    await runNode(meshyArgs(asset));
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
