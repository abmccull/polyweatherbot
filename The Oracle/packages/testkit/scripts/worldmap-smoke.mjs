import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/worldmap-smoke"
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

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

function writeText(filePath, value) {
  fs.writeFileSync(filePath, value);
}

async function ensureWorldTab(page) {
  if (await page.locator("#world-view-atlas").count()) {
    return;
  }
  if (await page.locator("#trigger-world").count()) {
    await page.click("#trigger-world");
    await page.waitForTimeout(150);
    return;
  }
  if (await page.locator("#sidebar-tab-world").count()) {
    await page.click("#sidebar-tab-world");
    await page.waitForTimeout(150);
  }
}

function stableCampaignSignature(state) {
  return {
    clock: state.clock,
    campaign: state.campaign
  };
}

async function waitForCampaignStateMatch(page, expectedState, label) {
  const expected = JSON.stringify(stableCampaignSignature(expectedState));

  for (let attempt = 0; attempt < 24; attempt += 1) {
    const current = await readState(page);
    const signature = JSON.stringify(stableCampaignSignature(current));
    if (signature === expected) {
      return current;
    }
    await page.waitForTimeout(120);
  }

  const current = await readState(page);
  throw new Error(`${label} did not match saved campaign state.\nExpected: ${expected}\nReceived: ${JSON.stringify(stableCampaignSignature(current))}`);
}

function normalizeText(text) {
  return text?.replace(/\s+/g, " ").trim() ?? "";
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function expectIncludes(text, snippet, label) {
  if (!text.includes(snippet)) {
    throw new Error(`${label} missing snippet "${snippet}". Received: ${text}`);
  }
}

function expectPillValue(pills, label, expectedValue) {
  const pill = pills.find((entry) => entry.includes(label));
  if (!pill) {
    throw new Error(`Missing replay KPI pill ${label}. Received: ${JSON.stringify(pills)}`);
  }
  if (!pill.includes(String(expectedValue))) {
    throw new Error(`Replay KPI pill ${label} drifted. Expected ${expectedValue}, received ${pill}`);
  }
}

async function assertWorldViewActive(page, view) {
  const atlasActive = await page.locator("#world-view-atlas").evaluate((node) => node.classList.contains("active"));
  const replayActive = await page.locator("#world-view-timeline").evaluate((node) => node.classList.contains("active"));

  if (view === "atlas") {
    if (!atlasActive || replayActive) {
      throw new Error(`Expected atlas view to be active. Atlas=${atlasActive}, Replay=${replayActive}`);
    }
    return;
  }

  if (!replayActive || atlasActive) {
    throw new Error(`Expected replay view to be active. Atlas=${atlasActive}, Replay=${replayActive}`);
  }
}

async function openWorldView(page, view) {
  await ensureWorldTab(page);
  const selector = view === "atlas" ? "#world-view-atlas" : "#world-view-timeline";
  await page.click(selector);
  await page.waitForTimeout(140);
  await assertWorldViewActive(page, view);
}

async function reloadAndLoad(page, expectedState, screenshotPath) {
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector("#precinct-canvas");
  await page.waitForTimeout(400);
  await page.evaluate(async () => {
    await window.__oracleDebug.load("slot-1");
  });
  await page.waitForTimeout(200);
  await ensureWorldTab(page);
  const loadedState = await waitForCampaignStateMatch(page, expectedState, "Loaded world-map save");
  if (screenshotPath) {
    await page.screenshot({ path: screenshotPath, fullPage: true });
  }
  return loadedState;
}

async function readAtlasText(page) {
  await openWorldView(page, "atlas");
  return normalizeText(await page.locator(".world-map-panel").textContent());
}

function visibleRivalFrontCount(deepState) {
  return [...(deepState.rivalOracles?.roster ?? [])]
    .filter((rival) => rival.active !== false && typeof rival.pressure === "number" && rival.pressure >= 38)
    .sort((left, right) => right.pressure - left.pressure || left.id.localeCompare(right.id))
    .slice(0, 4)
    .length;
}

async function assertAtlasMatchesState(page, state, deepState, options = {}) {
  const atlasText = await readAtlasText(page);
  const selectedNode = state.campaign.worldNodes.find((node) => node.id === state.campaign.selectedWorldNode);
  const winConditionSnippet = state.campaign.winCondition.label.slice(0, 24);
  const kpiText = normalizeText(await page.locator(".world-map-panel .campaign-kpis").first().textContent());
  const pressureCount = await page.locator(".world-map-panel .pressure-row").count();
  const crisisCount = await page.locator(".world-map-panel .history-row").count();
  const nodeCount = await page.locator(".world-map-panel .world-map-node").count();
  const pressureMarkerCount = await page.locator(".world-map-panel .world-map-pressure").count();
  const rivalFrontCount = visibleRivalFrontCount(deepState);
  const expectedFrontCount = state.campaign.activePressures.length + rivalFrontCount;

  expectIncludes(atlasText, "Campaign Atlas", "atlas header");
  expectIncludes(atlasText, capitalize(state.campaign.reputation.currentTier), "campaign tier");
  expectIncludes(atlasText, "Active Pressures", "active pressures section");
  expectIncludes(atlasText, "Crisis Chains", "crisis chain section");
  expectIncludes(kpiText, "Dedications", "dedication KPI");
  if (selectedNode) {
    expectIncludes(atlasText, selectedNode.label, "selected world node label");
  }
  expectIncludes(atlasText, winConditionSnippet, "win condition snippet");

  if (nodeCount !== state.campaign.worldNodes.length) {
    throw new Error(`World-map node count drifted. Expected ${state.campaign.worldNodes.length}, received ${nodeCount}`);
  }
  if (pressureMarkerCount !== expectedFrontCount) {
    throw new Error(`World-map pressure markers drifted. Expected ${expectedFrontCount}, received ${pressureMarkerCount}`);
  }
  if (pressureCount !== expectedFrontCount) {
    throw new Error(`Pressure list count drifted. Expected ${expectedFrontCount}, received ${pressureCount}`);
  }
  if (crisisCount < state.campaign.crisisChains.length) {
    throw new Error(`Crisis list was missing entries. Expected at least ${state.campaign.crisisChains.length}, received ${crisisCount}`);
  }

  if (options.requirePressureLabels) {
    for (const pressure of state.campaign.activePressures) {
      expectIncludes(atlasText, String(pressure.value), `pressure value ${pressure.value}`);
    }
  }

  return {
    atlasText,
    kpiText,
    nodeCount,
    pressureMarkerCount,
    pressureCount,
    crisisCount,
    rivalFrontCount
  };
}

async function readReplayText(page) {
  await openWorldView(page, "timeline");
  return normalizeText(await page.locator(".world-timeline-panel").textContent());
}

async function assertReplayMatchesState(page, state, deepState) {
  const liveState = await readState(page);
  const liveDeepState = await readDeepState(page);
  const replayState = liveState?.clock?.absoluteDay >= state.clock.absoluteDay ? liveState : state;
  const replayDeepState = liveDeepState ?? deepState;
  const replayText = await readReplayText(page);
  const activeCrisis = replayState.campaign.crisisChains[0];
  const pillTexts = (await page.locator(".world-timeline-panel .campaign-pill").allTextContents()).map(normalizeText);
  const campaignLists = page.locator(".world-timeline-panel .campaign-list");
  const horizonRows = await campaignLists.nth(0).locator(".world-timeline-row").count();
  const replayRows = await campaignLists.nth(1).locator(".world-timeline-row").count();
  const originRows = await campaignLists.nth(2).locator(".world-timeline-row").count();
  const openProphecies = replayDeepState.consultation.history.filter((entry) => !entry.resolved).length;
  const originHistoryLabel = replayDeepState.worldGeneration.history[0]?.label;
  const chronicleTitle = replayState.chronicle[0]?.title;

  expectIncludes(replayText, "World Replay", "replay header");
  expectIncludes(replayText, `Day ${replayState.clock.absoluteDay}`, "replay day marker");
  expectIncludes(replayText, replayState.run.originTitle, "replay origin title");
  expectIncludes(replayText, "Immediate Horizon", "immediate horizon section");
  expectIncludes(replayText, "Recent Replay", "recent replay section");
  expectIncludes(replayText, "Founding Pressure", "founding pressure section");

  expectPillValue(pillTexts, "Today", `Day ${replayState.clock.absoluteDay}`);
  expectPillValue(pillTexts, "Open Crises", replayState.campaign.crisisChains.length);
  expectPillValue(pillTexts, "Pending Omens", openProphecies);
  expectPillValue(pillTexts, "Field Notes", replayDeepState.eventFeed.length);

  if (activeCrisis) {
    expectIncludes(replayText, activeCrisis.label, "active crisis label");
  }
  if (originHistoryLabel) {
    expectIncludes(replayText, originHistoryLabel, "founding pressure label");
  }
  if (chronicleTitle) {
    expectIncludes(replayText, chronicleTitle, "recent replay chronicle title");
  }

  if (horizonRows === 0) {
    throw new Error("Immediate Horizon list rendered zero replay rows.");
  }
  if (replayRows === 0) {
    throw new Error("Recent Replay list rendered zero replay rows.");
  }
  if (originRows === 0) {
    throw new Error("Founding Pressure list rendered zero replay rows.");
  }

  return {
    replayText,
    pillTexts,
    horizonRows,
    replayRows,
    originRows
  };
}

function assertCampaignState(state, expected) {
  if (state.campaign.scenarioId !== expected.scenarioId) {
    throw new Error(`Expected scenario ${expected.scenarioId}, received ${state.campaign.scenarioId}`);
  }
  if (state.campaign.reputation.currentTier !== expected.currentTier) {
    throw new Error(`Expected tier ${expected.currentTier}, received ${state.campaign.reputation.currentTier}`);
  }
  if (state.campaign.reputation.score !== expected.score) {
    throw new Error(`Expected reputation score ${expected.score}, received ${state.campaign.reputation.score}`);
  }
  if (state.campaign.selectedWorldNode !== expected.selectedWorldNode) {
    throw new Error(`Expected selected node ${expected.selectedWorldNode}, received ${state.campaign.selectedWorldNode}`);
  }
  if (state.campaign.activePressures.length !== expected.pressureCount) {
    throw new Error(`Expected ${expected.pressureCount} active pressures, received ${state.campaign.activePressures.length}`);
  }
  if (state.campaign.crisisChains.length !== expected.crisisCount) {
    throw new Error(`Expected ${expected.crisisCount} crisis chains, received ${state.campaign.crisisChains.length}`);
  }
  if (state.campaign.treasury.completed !== expected.dedications) {
    throw new Error(`Expected ${expected.dedications} completed dedications, received ${state.campaign.treasury.completed}`);
  }
}

async function captureScenario(page, outDir, scenario, expectations) {
  await page.evaluate((scenarioId) => {
    window.__oracleDebug.injectScenario(scenarioId);
  }, scenario);
  await page.waitForTimeout(250);

  const state = await readState(page);
  const deepState = await readDeepState(page);
  assertCampaignState(state, expectations);

  const atlasCapture = await assertAtlasMatchesState(page, state, deepState, {
    requirePressureLabels: true
  });
  writeText(path.join(outDir, `${scenario}-atlas.txt`), `${atlasCapture.atlasText}\n`);
  writeJson(path.join(outDir, `${scenario}-atlas-summary.json`), atlasCapture);
  writeJson(path.join(outDir, `state-${scenario}.json`), state);
  await page.screenshot({ path: path.join(outDir, `${scenario}-full.png`), fullPage: true });

  const replayCapture = await assertReplayMatchesState(page, state, deepState);
  writeText(path.join(outDir, `${scenario}-replay.txt`), `${replayCapture.replayText}\n`);
  writeJson(path.join(outDir, `${scenario}-replay-summary.json`), replayCapture);
  await page.screenshot({ path: path.join(outDir, `${scenario}-replay.png`), fullPage: true });

  await page.evaluate(async () => {
    await window.__oracleDebug.save("slot-1");
  });
  await page.waitForTimeout(200);
  const loadedState = await reloadAndLoad(
    page,
    state,
    path.join(outDir, `${scenario}-reloaded-full.png`)
  );
  const loadedDeepState = await readDeepState(page);
  writeJson(path.join(outDir, `state-${scenario}-reloaded.json`), loadedState);

  assertCampaignState(loadedState, expectations);
  const reloadedAtlasCapture = await assertAtlasMatchesState(page, loadedState, loadedDeepState, {
    requirePressureLabels: true
  });
  writeText(path.join(outDir, `${scenario}-reloaded-atlas.txt`), `${reloadedAtlasCapture.atlasText}\n`);
  writeJson(path.join(outDir, `${scenario}-reloaded-atlas-summary.json`), reloadedAtlasCapture);

  const reloadedReplayCapture = await assertReplayMatchesState(page, loadedState, loadedDeepState);
  writeText(path.join(outDir, `${scenario}-reloaded-replay.txt`), `${reloadedReplayCapture.replayText}\n`);
  writeJson(path.join(outDir, `${scenario}-reloaded-replay-summary.json`), reloadedReplayCapture);
  await page.screenshot({ path: path.join(outDir, `${scenario}-reloaded-replay.png`), fullPage: true });
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
  await ensureWorldTab(page);

  await captureScenario(page, outDir, "campaign-lab", {
    scenarioId: "rising-oracle",
    currentTier: "recognized",
    score: 24,
    selectedWorldNode: "corinth",
    pressureCount: 1,
    crisisCount: 1,
    dedications: 0
  });

  await captureScenario(page, outDir, "world-map-lab", {
    scenarioId: "rising-oracle",
    currentTier: "revered",
    score: 58,
    selectedWorldNode: "athens",
    pressureCount: 2,
    crisisCount: 1,
    dedications: 1
  });

  writeJson(path.join(outDir, "errors.json"), errors);
  if (errors.length > 0) {
    throw new Error(`Encountered browser errors during world-map smoke: ${JSON.stringify(errors)}`);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
