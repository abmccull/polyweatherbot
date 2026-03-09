import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/replay-debug-smoke"
  };

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--url" && next) {
      args.url = next;
      index += 1;
    } else if (arg === "--out-dir" && next) {
      args.outDir = next;
      index += 1;
    }
  }

  return args;
}

function runSmoke(scriptPath, url, outDir) {
  const result = spawnSync(process.execPath, [scriptPath, "--url", url, "--out-dir", outDir], {
    stdio: "inherit"
  });

  if (result.status !== 0) {
    throw new Error(`${path.basename(scriptPath)} exited with status ${result.status ?? "unknown"}`);
  }
}

function main() {
  const { url, outDir } = parseArgs(process.argv);
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const originsOutDir = path.join(outDir, "origins");
  const worldmapOutDir = path.join(outDir, "worldmap");

  fs.mkdirSync(originsOutDir, { recursive: true });
  fs.mkdirSync(worldmapOutDir, { recursive: true });

  runSmoke(path.join(scriptDir, "origins-smoke.mjs"), url, originsOutDir);
  runSmoke(path.join(scriptDir, "worldmap-smoke.mjs"), url, worldmapOutDir);

  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify({
    url,
    generatedAt: new Date().toISOString(),
    runs: [
      {
        name: "origins-smoke",
        outDir: originsOutDir
      },
      {
        name: "worldmap-smoke",
        outDir: worldmapOutDir
      }
    ]
  }, null, 2));
}

main();
