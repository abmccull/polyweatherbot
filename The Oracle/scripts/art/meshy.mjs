#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const OUTPUT_DIR = path.join(ROOT_DIR, 'output', 'art', 'meshy');
const TASKS_DIR = path.join(OUTPUT_DIR, 'tasks');
const DEFAULT_BASE_URL = 'https://api.meshy.ai';

async function loadEnvFile() {
  const envPath = path.join(ROOT_DIR, '.env');
  try {
    const text = await readFile(envPath, 'utf8');
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const eqIndex = trimmed.indexOf('=');
      if (eqIndex === -1) continue;
      const key = trimmed.slice(0, eqIndex).trim();
      const value = trimmed.slice(eqIndex + 1).trim();
      if (!(key in process.env)) process.env[key] = value;
    }
  } catch {
    return;
  }
}

function printHelp() {
  console.log(`Meshy art pipeline helper

Usage:
  pnpm art:meshy help
  pnpm art:meshy text-preview --prompt "..." [--asset-id name]
  pnpm art:meshy image-to-3d --image path/to/reference.png [--asset-id name]
  pnpm art:meshy retexture --model path/to/model.glb --prompt "..." [--image path/to/style.png]
  pnpm art:meshy status --kind text-to-3d --task-id TASK_ID
  pnpm art:meshy download --kind text-to-3d --task-id TASK_ID --format glb
`);
}

function parseArgs(argv) {
  const [command = 'help', ...rest] = argv;
  const options = {};
  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = rest[index + 1];
    if (!next || next.startsWith('--')) {
      options[key] = 'true';
      continue;
    }
    options[key] = next;
    index += 1;
  }
  return { command, options };
}

function requireOption(options, key) {
  const value = options[key];
  if (!value) {
    throw new Error(`Missing required option: --${key}`);
  }
  return value;
}

function resolvePath(value) {
  return path.isAbsolute(value) ? value : path.join(ROOT_DIR, value);
}

function sanitizeName(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function guessMimeType(filePath) {
  switch (path.extname(filePath).toLowerCase()) {
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.webp':
      return 'image/webp';
    case '.glb':
      return 'model/gltf-binary';
    case '.gltf':
      return 'model/gltf+json';
    case '.obj':
      return 'text/plain';
    default:
      return 'application/octet-stream';
  }
}

async function asDataUri(fileOrUrl) {
  if (/^https?:\/\//i.test(fileOrUrl) || fileOrUrl.startsWith('data:')) {
    return fileOrUrl;
  }
  const resolved = resolvePath(fileOrUrl);
  const buffer = await readFile(resolved);
  const mimeType = guessMimeType(resolved);
  return `data:${mimeType};base64,${buffer.toString('base64')}`;
}

async function callMeshy({ method = 'GET', apiPath, body }) {
  const apiKey = process.env.MESHY_API_KEY;
  if (!apiKey) {
    throw new Error('Missing MESHY_API_KEY in .env or environment');
  }
  const response = await fetch(`${DEFAULT_BASE_URL}${apiPath}`, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Meshy API ${method} ${apiPath} failed: ${response.status} ${text}`);
  }
  return response.json();
}

async function ensureTaskDirs(kind) {
  await mkdir(path.join(TASKS_DIR, kind), { recursive: true });
  await mkdir(path.join(OUTPUT_DIR, 'downloads', kind), { recursive: true });
}

async function saveTaskSnapshot(kind, taskId, task) {
  await ensureTaskDirs(kind);
  const snapshotPath = path.join(TASKS_DIR, kind, `${taskId}.json`);
  await writeFile(snapshotPath, `${JSON.stringify(task, null, 2)}\n`, 'utf8');
  return snapshotPath;
}

function apiPathForKind(kind, taskId) {
  switch (kind) {
    case 'text-to-3d':
      return `/openapi/v2/text-to-3d/${taskId}`;
    case 'image-to-3d':
      return `/openapi/v1/image-to-3d/${taskId}`;
    case 'retexture':
      return `/openapi/v1/text-to-texture/${taskId}`;
    default:
      throw new Error(`Unsupported kind: ${kind}`);
  }
}

function taskIdFromResponse(task) {
  return task.result || task.id || task.task_id;
}

function downloadUrlForFormat(task, format) {
  return (
    task?.model_urls?.[format] ||
    task?.result?.model_urls?.[format] ||
    task?.result?.[format] ||
    null
  );
}

async function submitTextPreview(options) {
  const body = {
    mode: 'preview',
    prompt: requireOption(options, 'prompt'),
    negative_prompt: options['negative-prompt'],
    art_style: options['art-style'] ?? 'realistic',
    topology: options.topology ?? 'triangle',
    ai_model: options['ai-model'],
    target_polycount: options['target-polycount'] ? Number(options['target-polycount']) : undefined,
  };
  const task = await callMeshy({ method: 'POST', apiPath: '/openapi/v2/text-to-3d', body });
  const taskId = taskIdFromResponse(task);
  const snapshotPath = await saveTaskSnapshot('text-to-3d', taskId, task);
  console.log(`Created Meshy text-to-3d preview task: ${taskId}`);
  console.log(`Saved snapshot: ${path.relative(ROOT_DIR, snapshotPath)}`);
}

async function submitImageTo3d(options) {
  const body = {
    image_prompt: await asDataUri(requireOption(options, 'image')),
    topology: options.topology ?? 'triangle',
    ai_model: options['ai-model'],
    target_polycount: options['target-polycount'] ? Number(options['target-polycount']) : undefined,
    enable_pbr: options['enable-pbr'] ? options['enable-pbr'] === 'true' : true,
  };
  const task = await callMeshy({ method: 'POST', apiPath: '/openapi/v1/image-to-3d', body });
  const taskId = taskIdFromResponse(task);
  const snapshotPath = await saveTaskSnapshot('image-to-3d', taskId, task);
  console.log(`Created Meshy image-to-3d task: ${taskId}`);
  console.log(`Saved snapshot: ${path.relative(ROOT_DIR, snapshotPath)}`);
}

async function submitRetexture(options) {
  const body = {
    model_url: await asDataUri(requireOption(options, 'model')),
    prompt: requireOption(options, 'prompt'),
    image_url: options.image ? await asDataUri(options.image) : undefined,
    ai_model: options['ai-model'],
    enable_original_uv: options['enable-original-uv'] ? options['enable-original-uv'] === 'true' : undefined,
  };
  const task = await callMeshy({ method: 'POST', apiPath: '/openapi/v1/text-to-texture', body });
  const taskId = taskIdFromResponse(task);
  const snapshotPath = await saveTaskSnapshot('retexture', taskId, task);
  console.log(`Created Meshy retexture task: ${taskId}`);
  console.log(`Saved snapshot: ${path.relative(ROOT_DIR, snapshotPath)}`);
}

async function fetchStatus(options) {
  const kind = requireOption(options, 'kind');
  const taskId = requireOption(options, 'task-id');
  const task = await callMeshy({ apiPath: apiPathForKind(kind, taskId) });
  const snapshotPath = await saveTaskSnapshot(kind, taskId, task);
  console.log(JSON.stringify({ kind, taskId, status: task.status, snapshot: path.relative(ROOT_DIR, snapshotPath) }, null, 2));
}

async function downloadAsset(options) {
  const kind = requireOption(options, 'kind');
  const taskId = requireOption(options, 'task-id');
  const format = options.format ?? 'glb';
  const task = await callMeshy({ apiPath: apiPathForKind(kind, taskId) });
  const downloadUrl = downloadUrlForFormat(task, format);
  if (!downloadUrl) {
    throw new Error(`No ${format} URL found on task ${taskId}`);
  }
  await ensureTaskDirs(kind);
  const outPath = options.out
    ? resolvePath(options.out)
    : path.join(OUTPUT_DIR, 'downloads', kind, `${sanitizeName(options['asset-id'] ?? taskId)}.${format}`);
  await mkdir(path.dirname(outPath), { recursive: true });
  const response = await fetch(downloadUrl);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  await writeFile(outPath, buffer);
  console.log(`Downloaded ${format} to ${path.relative(ROOT_DIR, outPath)}`);
}

async function main() {
  await loadEnvFile();
  const { command, options } = parseArgs(process.argv.slice(2));
  switch (command) {
    case 'help':
    case '--help':
    case '-h':
      printHelp();
      break;
    case 'text-preview':
      await submitTextPreview(options);
      break;
    case 'image-to-3d':
      await submitImageTo3d(options);
      break;
    case 'retexture':
      await submitRetexture(options);
      break;
    case 'status':
      await fetchStatus(options);
      break;
    case 'download':
      await downloadAsset(options);
      break;
    default:
      throw new Error(`Unknown command: ${command}`);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
