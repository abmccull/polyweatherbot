import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { _electron as electron } from "playwright";

function parseArgs(argv) {
  const args = {
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/desktop-smoke"
  };

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--out-dir" && next) {
      args.outDir = next;
      index += 1;
    }
  }

  return args;
}

function collectFiles(rootDir, predicate, matches = []) {
  for (const entry of fs.readdirSync(rootDir, { withFileTypes: true })) {
    const entryPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      collectFiles(entryPath, predicate, matches);
      continue;
    }
    if (predicate(entryPath)) {
      matches.push(entryPath);
    }
  }
  return matches;
}

function resolvePackagedExecutable(outRoot) {
  const candidates = collectFiles(
    outRoot,
    (entryPath) => {
      const executableName = path.basename(entryPath);
      return entryPath.includes(".app/Contents/MacOS/")
        && !entryPath.includes("/Contents/Frameworks/")
        && !executableName.includes("Helper")
        && Boolean(fs.statSync(entryPath).mode & 0o111);
    }
  ).sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);

  if (!candidates[0]) {
    throw new Error(`No packaged Electron executable found under ${outRoot}`);
  }

  return candidates[0];
}

function stableSignature(state) {
  return {
    clock: state.clock,
    campaign: state.campaign,
    resources: state.resources,
    selected: state.selected,
    buildings: state.buildings,
    walkers: state.walkers
  };
}

async function loadState(page) {
  return page.evaluate(() => JSON.parse(window.render_game_to_text()));
}

const require = createRequire(import.meta.url);

async function launchBuiltApp(desktopDir) {
  const electronBinary = require("electron");
  const mainEntry = path.join(desktopDir, ".vite", "build", "main.cjs");
  const app = await electron.launch({
    executablePath: electronBinary,
    args: [mainEntry],
    cwd: desktopDir
  });
  const window = await app.firstWindow();
  await window.waitForSelector("#precinct-canvas");
  await window.waitForTimeout(400);
  return { app, window };
}

async function main() {
  const { outDir } = parseArgs(process.argv);
  fs.mkdirSync(outDir, { recursive: true });

  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = path.resolve(scriptDir, "../../..");
  const desktopDir = path.join(workspaceRoot, "apps/desktop");
  const packagedSmokeFile = path.join(outDir, "packaged-smoke.json");
  const packageResult = spawnSync("pnpm", ["--filter", "@the-oracle/desktop", "package"], {
    cwd: workspaceRoot,
    stdio: "inherit"
  });

  if (packageResult.status !== 0) {
    throw new Error(`Desktop package failed with status ${packageResult.status ?? "unknown"}`);
  }

  const executablePath = resolvePackagedExecutable(path.join(desktopDir, "out"));
  fs.rmSync(packagedSmokeFile, { force: true });
  const packagedBoot = spawnSync(executablePath, ["--smoke-test", "--smoke-file", packagedSmokeFile], {
    env: {
      ...process.env,
      ORACLE_SMOKE_TEST: "1",
      ORACLE_SMOKE_FILE: packagedSmokeFile
    },
    timeout: 20000,
    stdio: "inherit"
  });
  if (packagedBoot.error) {
    throw packagedBoot.error;
  }
  if (packagedBoot.status !== 0) {
    throw new Error(`Packaged desktop executable failed its smoke boot with status ${packagedBoot.status ?? "unknown"}`);
  }
  if (!fs.existsSync(packagedSmokeFile)) {
    throw new Error(`Packaged desktop smoke file was not created at ${packagedSmokeFile}`);
  }
  const packagedSmoke = JSON.parse(fs.readFileSync(packagedSmokeFile, "utf8"));
  if (!packagedSmoke.loaded) {
    throw new Error(`Packaged desktop smoke did not finish loading: ${JSON.stringify(packagedSmoke)}`);
  }

  const firstRun = await launchBuiltApp(desktopDir);
  const isDesktop = await firstRun.window.evaluate(() => window.oracleDesktop?.runtime.isDesktop ?? false);
  if (!isDesktop) {
    throw new Error("Renderer did not expose the desktop bridge");
  }

  await firstRun.window.evaluate(() => {
    window.__oracleDebug.injectScenario("campaign-lab");
  });
  await firstRun.window.waitForTimeout(200);
  await firstRun.window.evaluate(async () => {
    await window.advanceTime(900);
    await window.__oracleDebug.save("desktop-slot");
  });
  await firstRun.window.waitForTimeout(200);

  const runtimeInfo = await firstRun.window.evaluate(() => window.oracleDesktop.runtime.getInfo());
  const slotList = await firstRun.window.evaluate(() => window.oracleDesktop.persistence.listSlots());
  const savedState = await loadState(firstRun.window);
  await firstRun.window.screenshot({ path: path.join(outDir, "desktop-first-run.png"), fullPage: true });

  if (!fs.existsSync(runtimeInfo.saveDbPath)) {
    throw new Error(`SQLite database was not created at ${runtimeInfo.saveDbPath}`);
  }
  if (!slotList.some((slot) => slot.id === "desktop-slot")) {
    throw new Error(`Desktop save slot was not persisted. Received ${JSON.stringify(slotList)}`);
  }

  fs.writeFileSync(path.join(outDir, "runtime-info.json"), JSON.stringify(runtimeInfo, null, 2));
  fs.writeFileSync(path.join(outDir, "saved-state.json"), JSON.stringify(savedState, null, 2));
  await firstRun.app.close();

  const secondRun = await launchBuiltApp(desktopDir);
  await secondRun.window.evaluate(async () => {
    await window.__oracleDebug.load("desktop-slot");
  });
  await secondRun.window.waitForTimeout(200);

  const loadedState = await loadState(secondRun.window);
  await secondRun.window.screenshot({ path: path.join(outDir, "desktop-loaded-run.png"), fullPage: true });
  fs.writeFileSync(path.join(outDir, "loaded-state.json"), JSON.stringify(loadedState, null, 2));

  const expected = JSON.stringify(stableSignature(savedState));
  const received = JSON.stringify(stableSignature(loadedState));
  if (expected !== received) {
    throw new Error(`Desktop save/load round-trip drifted.\nExpected: ${expected}\nReceived: ${received}`);
  }

  await secondRun.app.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
