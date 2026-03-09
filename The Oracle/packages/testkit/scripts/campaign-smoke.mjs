import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/campaign-smoke"
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

async function readState(page) {
  return JSON.parse(await page.evaluate(() => window.render_game_to_text()));
}

async function readDeepState(page) {
  return page.evaluate(() => window.__oracleDebug.getState());
}

function stableSignature(state) {
  return {
    clock: state.clock,
    selected: state.selected ?? null,
    hovered_tile: state.hovered_tile ?? null,
    resources: state.resources,
    buildings: [...state.buildings].sort((left, right) => left.id.localeCompare(right.id)),
    walkers: [...state.walkers].sort((left, right) => left.id.localeCompare(right.id)),
    resource_jobs: [...state.resource_jobs].sort((left, right) => left.id.localeCompare(right.id)),
    consultation: state.consultation,
    consequences_due: [...state.consequences_due].sort((left, right) => {
      if (left.dueDay !== right.dueDay) {
        return left.dueDay - right.dueDay;
      }
      return left.factionId.localeCompare(right.factionId);
    }),
    chronicle: state.chronicle
  };
}

async function waitForStateMatch(page, expectedState, label) {
  const expected = JSON.stringify(stableSignature(expectedState));

  for (let attempt = 0; attempt < 24; attempt += 1) {
    const current = await readState(page);
    const signature = JSON.stringify(stableSignature(current));
    if (signature === expected) {
      return current;
    }
    await page.waitForTimeout(120);
  }

  const current = await readState(page);
  throw new Error(`${label} did not match saved state after load.\nExpected: ${expected}\nReceived: ${JSON.stringify(stableSignature(current))}`);
}

async function clickPrecinctTile(page, tile) {
  for (let attempt = 0; attempt < 24; attempt += 1) {
    const ready = await page.evaluate(() => Boolean(window.__oracleDebug.viewportForTile));
    if (ready) {
      break;
    }
    await page.waitForTimeout(100);
    if (attempt === 23) {
      throw new Error("viewportForTile hook is unavailable");
    }
  }

  const viewportPoint = await page.evaluate((targetTile) => {
    if (!window.__oracleDebug.viewportForTile) {
      throw new Error("viewportForTile hook is unavailable");
    }
    return window.__oracleDebug.viewportForTile(targetTile);
  }, tile);
  const canvas = page.locator("#precinct-canvas");
  const box = await canvas.boundingBox();
  if (!box) {
    throw new Error("precinct canvas is not visible");
  }
  await page.mouse.click(box.x + viewportPoint.x, box.y + viewportPoint.y);
}

const TOOL_CATEGORY = {
  sacred_way: "processional",
  priest_quarters: "housing",
  storehouse: "storage",
  castalian_spring: "ritual",
  inner_sanctum: "ritual",
  eternal_flame_brazier: "ritual"
};

async function selectTool(page, toolId) {
  if (toolId === "select") {
    await page.click("#tool-select");
    return;
  }

  const selector = `#tool-${toolId}`;
  const button = page.locator(selector);
  if (!(await button.isVisible().catch(() => false))) {
    await page.click(`#tool-category-${TOOL_CATEGORY[toolId]}`);
    await page.waitForSelector(selector, { state: "visible" });
  }
  await page.click(selector);
}

async function advanceAndRead(page, ms) {
  await page.evaluate(async (duration) => {
    await window.advanceTime(duration);
  }, ms);
  await page.waitForTimeout(100);
  return readState(page);
}

async function selectCarrierWalker(page) {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const state = await readState(page);
    const carrier = state.walkers.find((walker) => walker.role === "carrier");
    if (!carrier) {
      break;
    }

    const candidateTiles = [
      { x: carrier.x, y: carrier.y },
      { x: carrier.x - 1, y: carrier.y },
      { x: carrier.x + 1, y: carrier.y },
      { x: carrier.x, y: carrier.y - 1 },
      { x: carrier.x, y: carrier.y + 1 }
    ];
    for (const tile of candidateTiles) {
      await clickPrecinctTile(page, tile);
      await page.waitForTimeout(100);
      const selectedState = await readState(page);
      const selectedWalker = selectedState.walkers.find((walker) => walker.id === selectedState.selected);
      if (selectedWalker?.role === "carrier") {
        return selectedState;
      }
    }

    await advanceAndRead(page, 1200);
  }

  throw new Error("Carrier selection flow did not complete");
}

function scorePlacedTiles(tiles, pythiaAttunement) {
  if (tiles.length === 0) {
    return { clarity: 0, value: 0, risk: 0 };
  }

  const categories = new Set(tiles.map((tile) => tile.category));
  const domainSpread = new Set(tiles.map((tile) => tile.semantics.domain)).size;
  const targetSpread = new Set(tiles.map((tile) => tile.semantics.target)).size;
  const specificCount = tiles.filter((tile) => tile.semantics.ambiguity === "specific").length;
  const balancedCount = tiles.filter((tile) => tile.semantics.ambiguity === "balanced").length;
  const crypticCount = tiles.filter((tile) => tile.semantics.ambiguity === "cryptic").length;
  const structureBonus =
    (categories.has("subject") ? 16 : 0)
    + (categories.has("action") ? 16 : 0)
    + (categories.has("condition") ? 8 : 0)
    + (categories.has("modifier") ? 6 : 0)
    + (categories.has("seal") ? 8 : 0);
  const coherenceBonus = domainSpread === 1 ? 12 : 6 - (domainSpread - 1) * 4;
  const clarity = Math.max(
    20,
    Math.min(
      100,
      18
        + structureBonus
        + balancedCount * 7
        + specificCount * 4
        + crypticCount * 2
        + coherenceBonus
        - Math.max(0, targetSpread - 1) * 10
    )
  );
  const value = Math.max(
    20,
    Math.min(100, clarity + (categories.has("seal") ? 10 : 0) + (domainSpread === 1 ? 12 : 4) + pythiaAttunement * 0.12)
  );
  const risk = Math.max(
    5,
    Math.min(
      100,
      16
        + specificCount * 18
        + Math.max(0, targetSpread - 1) * 12
        + Math.max(0, domainSpread - 1) * 8
        + (categories.has("subject") && categories.has("action") ? 0 : 12)
        - crypticCount * 5
        - pythiaAttunement * 0.08
    )
  );

  return {
    clarity: Math.round(clarity),
    value: Math.round(value),
    risk: Math.round(risk)
  };
}

function combinations(items, size, start = 0, current = [], results = []) {
  if (current.length === size) {
    results.push([...current]);
    return results;
  }

  for (let index = start; index <= items.length - (size - current.length); index += 1) {
    current.push(items[index]);
    combinations(items, size, index + 1, current, results);
    current.pop();
  }

  return results;
}

function chooseBestProphecyTileIds(tilePool, pythiaAttunement) {
  let best = {
    ids: tilePool.slice(0, 4).map((tile) => tile.id),
    rank: Number.NEGATIVE_INFINITY,
    score: { clarity: 0, value: 0, risk: 100 }
  };

  for (const size of [4, 5]) {
    for (const combo of combinations(tilePool, Math.min(size, tilePool.length))) {
      const score = scorePlacedTiles(combo, pythiaAttunement);
      const rank = score.value * 1.4 + score.clarity * 0.35 - score.risk * 0.8;
      if (
        rank > best.rank
        || (rank === best.rank && score.risk < best.score.risk)
        || (rank === best.rank && score.risk === best.score.risk && score.value > best.score.value)
      ) {
        best = {
          ids: combo.map((tile) => tile.id),
          rank,
          score
        };
      }
    }
  }

  return best;
}

async function reloadAndLoad(page, expectedState, screenshotPath) {
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector("#precinct-canvas");
  await page.waitForTimeout(500);
  await page.evaluate(async () => {
    await window.__oracleDebug.load("slot-1");
  });
  await page.waitForTimeout(200);
  const loadedState = await waitForStateMatch(page, expectedState, "Loaded save");
  if (screenshotPath) {
    await page.screenshot({ path: screenshotPath, fullPage: true });
  }
  return loadedState;
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

async function main() {
  const { url, outDir } = parseArgs(process.argv);
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch({
    headless: true,
    args: ["--use-gl=angle", "--use-angle=swiftshader"]
  });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 }
  });
  const errors = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push({ type: "console.error", text: msg.text() });
    }
  });
  page.on("pageerror", (error) => {
    errors.push({ type: "pageerror", text: String(error) });
  });

  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForSelector("#precinct-canvas");
  await page.waitForTimeout(600);
  await page.click("#speed-0");
  await page.waitForTimeout(100);

  await selectTool(page, "sacred_way");
  for (const tile of [
    { x: 28, y: 50 },
    { x: 29, y: 50 },
    { x: 30, y: 50 },
    { x: 31, y: 50 },
    { x: 32, y: 50 },
    { x: 33, y: 50 }
  ]) {
    await clickPrecinctTile(page, tile);
    await page.waitForTimeout(30);
  }

  const placements = [
    ["priest_quarters", { x: 29, y: 49 }],
    ["storehouse", { x: 30, y: 49 }],
    ["castalian_spring", { x: 31, y: 49 }],
    ["inner_sanctum", { x: 32, y: 49 }],
    ["eternal_flame_brazier", { x: 33, y: 49 }]
  ];
  for (const [toolId, tile] of placements) {
    await selectTool(page, toolId);
    await clickPrecinctTile(page, tile);
    await page.waitForTimeout(40);
  }

  await selectTool(page, "select");
  await clickPrecinctTile(page, { x: 31, y: 49 });
  await page.waitForTimeout(160);
  await page.click("#assign-priest-btn");
  await page.waitForTimeout(160);
  await clickPrecinctTile(page, { x: 33, y: 49 });
  await page.waitForTimeout(160);

  const buildState = await readState(page);
  await page.click("#trigger-world");
  await page.waitForSelector("#faction-card-athens");
  const athensCardText = await page.locator("#faction-card-athens").textContent();
  const athensRelationsText = await page.locator("#faction-relations-athens").textContent();
  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  writeJson(path.join(outDir, "state-built.json"), buildState);
  await page.screenshot({ path: path.join(outDir, "built-full.png"), fullPage: true });

  let logisticsState = buildState;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    logisticsState = await advanceAndRead(page, 2000);
    const carrier = logisticsState.walkers.find((walker) => walker.role === "carrier");
    if (logisticsState.resource_jobs.length > 0 || (carrier && carrier.state !== "idle")) {
      break;
    }
  }

  logisticsState = await selectCarrierWalker(page);
  writeJson(path.join(outDir, "state-logistics.json"), logisticsState);
  await page.screenshot({ path: path.join(outDir, "logistics-full.png"), fullPage: true });

  await page.evaluate(async () => {
    await window.__oracleDebug.save("slot-1");
  });
  await page.waitForTimeout(250);
  const reloadedLogisticsState = await reloadAndLoad(
    page,
    logisticsState,
    path.join(outDir, "logistics-reloaded-full.png")
  );
  writeJson(path.join(outDir, "state-logistics-reloaded.json"), reloadedLogisticsState);

  let campaignState = reloadedLogisticsState;
  while (campaignState.clock.absoluteDay < 15 || campaignState.consultation.mode !== "pending") {
    campaignState = await advanceAndRead(page, 60000);
    if (campaignState.clock.absoluteDay > 20) {
      throw new Error("Consultation did not arrive by absolute day 15");
    }
  }

  await page.click("#envoy-btn");
  await page.waitForTimeout(250);

  const consultationDeepState = await readDeepState(page);
  const consultationCurrent = consultationDeepState.consultation.current;
  if (!consultationCurrent) {
    throw new Error("Consultation opened without a current consultation payload");
  }
  const selection = chooseBestProphecyTileIds(consultationCurrent.tilePool, consultationDeepState.pythia.attunement);
  for (const tileId of selection.ids) {
    await page.click(`#prophecy-tile-${tileId}`);
    await page.waitForTimeout(40);
  }

  const consultationOpenState = await readState(page);
  const assembledProphecyText = await page.locator("#consultation-assembled-prophecy").textContent();
  const consultationScoreText = await page.locator("#consultation-score-row").textContent();
  writeJson(path.join(outDir, "state-consultation-open.json"), consultationOpenState);
  writeJson(path.join(outDir, "selected-prophecy-score.json"), selection);
  await page.screenshot({ path: path.join(outDir, "consultation-open-full.png"), fullPage: true });

  await page.waitForFunction(() => {
    const button = document.querySelector("#deliver-prophecy-btn");
    return button instanceof HTMLButtonElement && button.disabled === false;
  });
  await page.click("#deliver-prophecy-btn");
  await page.waitForTimeout(250);
  const deliveredState = await readState(page);
  const deliveredDeepState = await readDeepState(page);
  writeJson(path.join(outDir, "state-delivered.json"), deliveredState);
  await page.screenshot({ path: path.join(outDir, "delivered-full.png"), fullPage: true });

  await page.evaluate(async () => {
    await window.__oracleDebug.save("slot-1");
  });
  await page.waitForTimeout(250);
  const deliveredReloadedState = await reloadAndLoad(
    page,
    deliveredState,
    path.join(outDir, "delivered-reloaded-full.png")
  );
  const deliveredReloadedDeepState = await readDeepState(page);
  writeJson(path.join(outDir, "state-delivered-reloaded.json"), deliveredReloadedState);

  const primaryProphecy = deliveredReloadedDeepState.consultation.history[0];
  if (!primaryProphecy) {
    throw new Error("Delivered prophecy was not present in consultation history after reload");
  }
  const dueDay = primaryProphecy.dueDay;
  const consultedFactionId = primaryProphecy.factionId;
  const preResolutionHistories = Object.fromEntries(
    Object.entries(deliveredReloadedDeepState.factions).map(([factionId, faction]) => [factionId, [...faction.history]])
  );

  let resolvedState = deliveredReloadedState;
  let resolvedDeepState = deliveredReloadedDeepState;
  while (!resolvedDeepState.consultation.history[0]?.resolved) {
    resolvedState = await advanceAndRead(page, 60000);
    resolvedDeepState = await readDeepState(page);
    if (resolvedState.clock.absoluteDay > dueDay + 5) {
      throw new Error(`Consequence for ${consultedFactionId} did not resolve by absolute day ${dueDay}`);
    }
  }

  writeJson(path.join(outDir, "state-resolved.json"), resolvedState);
  await page.screenshot({ path: path.join(outDir, "resolved-full.png"), fullPage: true });
  await page.evaluate(async () => {
    await window.__oracleDebug.save("slot-1");
  });
  await page.waitForTimeout(250);
  const resolvedReloadedState = await reloadAndLoad(
    page,
    resolvedState,
    path.join(outDir, "resolved-reloaded-full.png")
  );
  writeJson(path.join(outDir, "state-resolved-reloaded.json"), resolvedReloadedState);

  const observerShift = Object.values(resolvedDeepState.factions).some((faction) => {
    if (faction.id === consultedFactionId) {
      return false;
    }
    return JSON.stringify(faction.history) !== JSON.stringify(preResolutionHistories[faction.id] ?? []);
  });

  writeJson(path.join(outDir, "errors.json"), errors);

  if (buildState.buildings.length !== 5) {
    throw new Error(`Expected 5 buildings in build state, found ${buildState.buildings.length}`);
  }
  if (!buildState.buildings.some((building) => building.defId === "eternal_flame_brazier")) {
    throw new Error("Brazier placement failed");
  }
  if (!athensCardText?.includes("Rival: Sparta")) {
    throw new Error(`Athens faction card did not show rival relation text: ${athensCardText}`);
  }
  if (!athensRelationsText?.includes("Trade Open") || !athensRelationsText.includes("Favour")) {
    throw new Error(`Athens faction relations line did not show trade/favour text: ${athensRelationsText}`);
  }
  const selectedLogisticsWalker = logisticsState.walkers.find((walker) => walker.id === logisticsState.selected);
  if (selectedLogisticsWalker?.role !== "carrier") {
    throw new Error("Carrier selection flow did not complete");
  }
  if (consultationOpenState.consultation.mode !== "open") {
    throw new Error(`Expected consultation to be open, found ${consultationOpenState.consultation.mode}`);
  }
  if (!assembledProphecyText || assembledProphecyText.includes("Arrange the tiles")) {
    throw new Error(`Consultation UI did not assemble visible prophecy text: ${assembledProphecyText}`);
  }
  if (!consultationScoreText?.includes("Clarity") || !consultationScoreText.includes("Risk")) {
    throw new Error(`Consultation score UI was not visible: ${consultationScoreText}`);
  }
  if ((deliveredState.chronicle ?? []).length === 0) {
    throw new Error("Delivered prophecy did not reach the chronicle");
  }
  if ((selection.score.value ?? 0) < 70) {
    throw new Error(`Consultation picker chose a weak prophecy score: ${JSON.stringify(selection.score)}`);
  }
  if ((selection.score.risk ?? 100) > 70) {
    throw new Error(`Consultation picker chose an overly risky prophecy: ${JSON.stringify(selection.score)}`);
  }
  if (!deliveredReloadedState.consequences_due.some((entry) => entry.factionId === consultedFactionId && entry.dueDay === dueDay)) {
    throw new Error("Saved-and-reloaded prophecy lost its pending consequence");
  }
  if (!resolvedState.chronicle.some((entry) => entry.kind === "consequence")) {
    throw new Error("Resolved consequence did not reach the chronicle");
  }
  if (resolvedState.consequences_due.some((entry) => entry.factionId === consultedFactionId && entry.dueDay === dueDay)) {
    throw new Error("Resolved consequence still appears in the due list");
  }
  if (resolvedReloadedState.consequences_due.some((entry) => entry.factionId === consultedFactionId && entry.dueDay === dueDay)) {
    throw new Error("Resolved consequence reappeared after save/load");
  }
  if (!resolvedReloadedState.chronicle.some((entry) => entry.kind === "consequence")) {
    throw new Error("Resolved consequence was lost after save/load");
  }
  if (!observerShift) {
    throw new Error("No non-consulted faction history changed after consequence resolution");
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
