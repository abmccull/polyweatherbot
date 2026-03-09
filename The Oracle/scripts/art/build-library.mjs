#!/usr/bin/env node

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { assetCatalog } from '../../art-library/source/catalog.mjs';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const GENERATED_DIR = path.join(ROOT_DIR, 'art-library', 'generated');
const PROMPT_PACKS_DIR = path.join(GENERATED_DIR, 'prompt-packs');
const COMMANDS_DIR = path.join(GENERATED_DIR, 'commands');

function ensureArray(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function csvEscape(value) {
  const text = String(value ?? '');
  if (/[,"\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function imageJobFor(asset) {
  if (!String(asset.generationLane).includes('imagegen')) return null;
  const transparent = ['ui', 'portrait'].includes(asset.family);
  return {
    asset_id: asset.assetId,
    prompt: asset.imagePrompt,
    size: asset.family === 'terrain' ? '1536x1024' : '1024x1024',
    quality: 'high',
    output_format: 'png',
    background: transparent ? 'transparent' : undefined,
  };
}

function meshyJobFor(asset) {
  if (!String(asset.generationLane).includes('meshy') || !asset.meshyPrompt) return null;
  return {
    asset_id: asset.assetId,
    command: 'text-preview',
    prompt: asset.meshyPrompt,
    negative_prompt: asset.negativePrompt,
    art_style: 'realistic',
    topology: 'triangle',
  };
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

function meshyCommandFor(job) {
  return [
    'pnpm art:meshy text-preview',
    `--asset-id ${shellQuote(job.asset_id)}`,
    `--prompt ${shellQuote(job.prompt)}`,
    job.negative_prompt ? `--negative-prompt ${shellQuote(job.negative_prompt)}` : '',
    job.art_style ? `--art-style ${shellQuote(job.art_style)}` : '',
    job.topology ? `--topology ${shellQuote(job.topology)}` : '',
  ].filter(Boolean).join(' ');
}

function imagegenCommandFor(job) {
  const imageGenPath = '$CODEX_HOME/skills/imagegen/scripts/image_gen.py';
  return [
    `python ${shellQuote(imageGenPath)} generate`,
    `--prompt ${shellQuote(job.prompt)}`,
    `--size ${shellQuote(job.size)}`,
    `--quality ${shellQuote(job.quality)}`,
    job.background ? `--background ${shellQuote(job.background)}` : '',
    `--output-format ${shellQuote(job.output_format)}`,
  ].filter(Boolean).join(' ');
}

function groupBy(items, key) {
  const map = new Map();
  for (const item of items) {
    const value = item[key];
    const bucket = map.get(value) ?? [];
    bucket.push(item);
    map.set(value, bucket);
  }
  return map;
}

async function writeJsonl(filePath, rows) {
  const lines = rows.map((row) => JSON.stringify(row)).join('\n');
  await writeFile(filePath, lines ? `${lines}\n` : '', 'utf8');
}

async function main() {
  await mkdir(path.join(PROMPT_PACKS_DIR, 'imagegen'), { recursive: true });
  await mkdir(path.join(PROMPT_PACKS_DIR, 'meshy'), { recursive: true });
  await mkdir(path.join(COMMANDS_DIR, 'imagegen'), { recursive: true });
  await mkdir(path.join(COMMANDS_DIR, 'meshy'), { recursive: true });

  const manifest = assetCatalog.map((asset, index) => ({
    order: index + 1,
    providerHints: ensureArray(asset.generationLane.split('+')),
    ...asset,
  }));

  const summaryByFamily = groupBy(manifest, 'family');
  const summaryByWave = groupBy(manifest, 'wave');
  const summaryByLane = new Map();
  for (const item of manifest) {
    for (const lane of item.providerHints) {
      const bucket = summaryByLane.get(lane) ?? [];
      bucket.push(item);
      summaryByLane.set(lane, bucket);
    }
  }

  const csvHeaders = [
    'order','assetId','displayName','family','wave','priority','renderProfile','generationLane','footprint','animationProfile','eraProfile','variantProfile','status','cleanupTier','tags','visualBrief'
  ];
  const csvLines = [csvHeaders.join(',')];
  for (const item of manifest) {
    csvLines.push(csvHeaders.map((header) => csvEscape(item[header])).join(','));
  }

  await writeFile(path.join(GENERATED_DIR, 'asset-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  await writeFile(path.join(GENERATED_DIR, 'asset-manifest.csv'), `${csvLines.join('\n')}\n`, 'utf8');

  const summaryLines = [
    '# Asset Library Summary',
    '',
    `- Total assets: ${manifest.length}`,
    `- Families: ${[...summaryByFamily.keys()].sort().join(', ')}`,
    `- Waves: ${[...summaryByWave.keys()].sort().join(', ')}`,
    '',
    '## Counts by Family',
    '',
    '| Family | Count |',
    '|---|---:|',
    ...[...summaryByFamily.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([family, items]) => `| ${family} | ${items.length} |`),
    '',
    '## Counts by Wave',
    '',
    '| Wave | Count |',
    '|---|---:|',
    ...[...summaryByWave.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([wave, items]) => `| ${wave} | ${items.length} |`),
    '',
    '## Counts by Generation Lane',
    '',
    '| Lane | Count |',
    '|---|---:|',
    ...[...summaryByLane.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([lane, items]) => `| ${lane} | ${items.length} |`),
    '',
    '## First Waves',
    '',
    '- `core_slice`: replace the live graybox slice and advisor/UI ornament set first.',
    '- `precinct_expansion`: widen the buildable precinct roster next.',
    '- `characters_and_fauna`: build walker sheets, Pythia variants, and omen creatures.',
    '- `world_ui`: world-map, faction, threat, and political-icon packs.',
    '- `late_game`: origins, age emblems, excavation, rivals, and prestige content.',
  ];
  await writeFile(path.join(GENERATED_DIR, 'summary.md'), `${summaryLines.join('\n')}\n`, 'utf8');

  const imageJobs = manifest.map(imageJobFor).filter(Boolean);
  const meshyJobs = manifest.map(meshyJobFor).filter(Boolean);
  const imageJobsByWave = groupBy(imageJobs.map((job) => ({ ...job, wave: manifest.find((m) => m.assetId === job.asset_id)?.wave })), 'wave');
  const meshyJobsByWave = groupBy(meshyJobs.map((job) => ({ ...job, wave: manifest.find((m) => m.assetId === job.asset_id)?.wave })), 'wave');

  for (const [wave, jobs] of imageJobsByWave.entries()) {
    await writeJsonl(path.join(PROMPT_PACKS_DIR, 'imagegen', `${wave}.jsonl`), jobs.map(({ wave: _wave, ...job }) => job));
    const script = [
      '#!/usr/bin/env bash',
      'set -euo pipefail',
      'export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"',
      'export IMAGE_GEN="$CODEX_HOME/skills/imagegen/scripts/image_gen.py"',
      'python "$IMAGE_GEN" generate-batch --input ' + shellQuote(path.join('art-library', 'generated', 'prompt-packs', 'imagegen', `${wave}.jsonl`)) + ' --out-dir ' + shellQuote(path.join('output', 'art', 'imagegen', wave)),
    ].join('\n');
    await writeFile(path.join(COMMANDS_DIR, 'imagegen', `${wave}.sh`), `${script}\n`, 'utf8');
  }

  for (const [wave, jobs] of meshyJobsByWave.entries()) {
    await writeJsonl(path.join(PROMPT_PACKS_DIR, 'meshy', `${wave}.jsonl`), jobs.map(({ wave: _wave, ...job }) => job));
    const script = ['#!/usr/bin/env bash', 'set -euo pipefail', ...jobs.map(({ wave: _wave, ...job }) => meshyCommandFor(job))].join('\n');
    await writeFile(path.join(COMMANDS_DIR, 'meshy', `${wave}.sh`), `${script}\n`, 'utf8');
  }

  console.log(`Wrote manifest for ${manifest.length} assets.`);
  console.log(`Meshy jobs: ${meshyJobs.length}`);
  console.log(`ImageGen jobs: ${imageJobs.length}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
