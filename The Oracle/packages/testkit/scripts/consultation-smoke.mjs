import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/consultation-smoke"
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

function stableConsultationSignature(state) {
  return {
    clock: state.clock,
    resources: state.resources,
    pythia: state.pythia,
    consultation: state.consultation
  };
}

async function waitForConsultationStateMatch(page, expectedState, label) {
  const expected = JSON.stringify(stableConsultationSignature(expectedState));

  for (let attempt = 0; attempt < 24; attempt += 1) {
    const current = await readState(page);
    const signature = JSON.stringify(stableConsultationSignature(current));
    if (signature === expected) {
      return current;
    }
    await page.waitForTimeout(120);
  }

  const current = await readState(page);
  throw new Error(`${label} did not match saved consultation state.\nExpected: ${expected}\nReceived: ${JSON.stringify(stableConsultationSignature(current))}`);
}

async function reloadAndLoad(page, expectedState, screenshotPath) {
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector("#precinct-canvas");
  await page.waitForTimeout(400);
  await page.evaluate(async () => {
    await window.__oracleDebug.load("slot-1");
  });
  await page.waitForTimeout(200);
  const loadedState = await waitForConsultationStateMatch(page, expectedState, "Loaded consultation save");
  if (screenshotPath) {
    await page.screenshot({ path: screenshotPath, fullPage: true });
  }
  return loadedState;
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

function chooseRiskiestProphecyTileIds(tilePool, pythiaAttunement) {
  let best = {
    ids: tilePool.slice(0, 4).map((tile) => tile.id),
    rank: Number.NEGATIVE_INFINITY,
    score: { clarity: 0, value: 0, risk: 0 }
  };

  for (const size of [4, 5]) {
    for (const combo of combinations(tilePool, Math.min(size, tilePool.length))) {
      const score = scorePlacedTiles(combo, pythiaAttunement);
      const rank = score.risk * 1.6 - score.clarity * 0.15 + score.value * 0.1;
      if (
        rank > best.rank
        || (rank === best.rank && score.risk > best.score.risk)
        || (rank === best.rank && score.risk === best.score.risk && score.clarity < best.score.clarity)
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
  await page.waitForTimeout(400);
  await page.evaluate(() => {
    window.__oracleDebug.injectScenario("consultation-ready");
  });
  await page.waitForTimeout(200);
  await page.click("#envoy-btn");
  await page.waitForSelector("#consultation-assembled-prophecy");

  const openDeepState = await readDeepState(page);
  const current = openDeepState.consultation.current;
  if (!current) {
    throw new Error("Consultation-ready scenario did not produce a live consultation");
  }
  const riskySelection = chooseRiskiestProphecyTileIds(current.tilePool, openDeepState.pythia.attunement);
  for (const tileId of riskySelection.ids) {
    await page.click(`#prophecy-tile-${tileId}`);
  }
  await page.waitForTimeout(200);

  const openState = await readState(page);
  const openDeepSelectedState = await readDeepState(page);
  const assembledProphecyText = await page.locator("#consultation-assembled-prophecy").textContent();
  const omenSummaryText = await page.locator("#consultation-omen-summary").textContent();
  const depthReadoutText = await page.locator("#consultation-depth-readout").textContent();
  const scaffoldSpineText = await page.locator("#consultation-scaffold-spine").textContent();
  const firstOmenReliabilityText = await page.locator("#consultation-omen-0-reliability").textContent();
  const guidanceText = await page.locator("#consultation-guidance-text").textContent();
  const riskWarningText = await page.locator("#consultation-risk-warning").textContent();
  const scoreText = await page.locator("#consultation-score-row").textContent();
  fs.writeFileSync(path.join(outDir, "state-open.json"), JSON.stringify(openState, null, 2));
  await page.screenshot({ path: path.join(outDir, "consultation-open-full.png"), fullPage: true });

  await page.evaluate(async () => {
    await window.__oracleDebug.save("slot-1");
  });
  await page.waitForTimeout(200);
  const reloadedOpenState = await reloadAndLoad(
    page,
    openState,
    path.join(outDir, "consultation-reloaded-full.png")
  );
  const reloadedDeepState = await readDeepState(page);
  const reloadedAssembledProphecyText = await page.locator("#consultation-assembled-prophecy").textContent();
  const reloadedOmenSummaryText = await page.locator("#consultation-omen-summary").textContent();
  const reloadedGuidanceText = await page.locator("#consultation-guidance-text").textContent();
  const reloadedRiskWarningText = await page.locator("#consultation-risk-warning").textContent();
  fs.writeFileSync(path.join(outDir, "state-reloaded.json"), JSON.stringify(reloadedOpenState, null, 2));

  await page.waitForFunction(() => {
    const button = document.querySelector("#deliver-prophecy-btn");
    return button instanceof HTMLButtonElement && button.disabled === false;
  });
  await page.click("#deliver-prophecy-btn");
  await page.waitForTimeout(200);
  const deliveredDeepState = await readDeepState(page);
  const deliveredState = await readState(page);
  await page.click("#trigger-record");
  await page.waitForSelector("#sacred-record-panel");
  const recordDetailText = await page.locator("#sacred-record-detail").textContent();
  fs.writeFileSync(path.join(outDir, "state-delivered.json"), JSON.stringify(deliveredState, null, 2));
  await page.screenshot({ path: path.join(outDir, "consultation-delivered-full.png"), fullPage: true });
  fs.writeFileSync(path.join(outDir, "errors.json"), JSON.stringify(errors, null, 2));

  if (openState.consultation.mode !== "open") {
    throw new Error(`Expected open consultation state, found ${openState.consultation.mode}`);
  }
  if (!omenSummaryText?.includes("Consensus") || !omenSummaryText.includes("Average reliability")) {
    throw new Error(`Consultation omen summary was not visible: ${omenSummaryText}`);
  }
  if (!depthReadoutText || !/Prophecy depth/i.test(depthReadoutText)) {
    throw new Error(`Consultation depth readout was not visible: ${depthReadoutText}`);
  }
  if (!scaffoldSpineText || !/Spine/i.test(scaffoldSpineText)) {
    throw new Error(`Consultation scaffold was not visible: ${scaffoldSpineText}`);
  }
  if (!firstOmenReliabilityText || !/(clear|steady|fragile)/.test(firstOmenReliabilityText)) {
    throw new Error(`Omen reliability tier was not visible: ${firstOmenReliabilityText}`);
  }
  if (!guidanceText || guidanceText.length < 24) {
    throw new Error(`Consultation guidance text was not visible: ${guidanceText}`);
  }
  if (!riskWarningText || !/dangerously|muddy|fragile|subject and action|shallow/i.test(riskWarningText)) {
    throw new Error(`Risk warning was not visible after risky tile selection: ${riskWarningText}`);
  }
  if (!scoreText?.includes("Risk")) {
    throw new Error(`Consultation score row was not visible: ${scoreText}`);
  }
  if (reloadedOpenState.consultation.mode !== "open") {
    throw new Error(`Reloaded consultation did not remain open: ${reloadedOpenState.consultation.mode}`);
  }
  if (assembledProphecyText !== reloadedAssembledProphecyText) {
    throw new Error(`Assembled prophecy text did not round-trip through save/load.\nBefore: ${assembledProphecyText}\nAfter: ${reloadedAssembledProphecyText}`);
  }
  if (omenSummaryText !== reloadedOmenSummaryText) {
    throw new Error(`Omen summary did not round-trip through save/load.\nBefore: ${omenSummaryText}\nAfter: ${reloadedOmenSummaryText}`);
  }
  if (guidanceText !== reloadedGuidanceText) {
    throw new Error(`Guidance text did not round-trip through save/load.\nBefore: ${guidanceText}\nAfter: ${reloadedGuidanceText}`);
  }
  if (riskWarningText !== reloadedRiskWarningText) {
    throw new Error(`Risk warning did not round-trip through save/load.\nBefore: ${riskWarningText}\nAfter: ${reloadedRiskWarningText}`);
  }
  if (
    JSON.stringify(openDeepSelectedState.consultation.current?.placedTileIds ?? [])
    !== JSON.stringify(reloadedDeepState.consultation.current?.placedTileIds ?? [])
  ) {
    throw new Error("Placed prophecy tiles did not round-trip through save/load");
  }
  if (deliveredDeepState.consultation.mode !== "idle") {
    throw new Error(`Consultation did not close after delivery: ${deliveredDeepState.consultation.mode}`);
  }
  if (!deliveredDeepState.consultation.history[0]) {
    throw new Error("Delivered prophecy was not recorded in consultation history");
  }
  if (!deliveredState.sacred_record?.[0]?.depthBand || !deliveredState.sacred_record?.[0]?.omenConsensus) {
    throw new Error(`Delivered sacred record summary was incomplete: ${JSON.stringify(deliveredState.sacred_record?.[0] ?? null)}`);
  }
  if (!recordDetailText || !/Selected Record|Political use|Outcome/i.test(recordDetailText)) {
    throw new Error(`Sacred Record detail view was not visible: ${recordDetailText}`);
  }
  if (errors.length > 0) {
    throw new Error(`Encountered browser errors during consultation smoke: ${JSON.stringify(errors)}`);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
