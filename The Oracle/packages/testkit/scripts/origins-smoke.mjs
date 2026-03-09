import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/origins-smoke"
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

async function readPreview(page, seed, originId) {
  return page.evaluate(
    ({ previewSeed, previewOriginId }) => window.__oracleDebug.getRunSetupPreview({
      seed: previewSeed,
      originId: previewOriginId
    }),
    { previewSeed: seed, previewOriginId: originId }
  );
}

async function readRunDebugContext(page, seed, originId) {
  return page.evaluate(
    ({ draftSeed, draftOriginId }) => window.__oracleDebug.getRunDebugContext(
      draftSeed || draftOriginId
        ? { seed: draftSeed || undefined, originId: draftOriginId || undefined }
        : undefined,
      false
    ),
    {
      draftSeed: seed ?? "",
      draftOriginId: originId ?? ""
    }
  );
}

async function openSetup(page, url, seed, originId) {
  await page.goto(`${url}?setup=1&seed=${encodeURIComponent(seed)}&origin=${encodeURIComponent(originId)}`, {
    waitUntil: "networkidle"
  });
  await page.waitForSelector("[data-testid='run-setup-start-btn']");
  await page.waitForTimeout(300);
}

function normalizeText(text) {
  return text?.replace(/\s+/g, " ").trim() ?? "";
}

function expectIncludes(text, snippet, label) {
  if (!text.includes(snippet)) {
    throw new Error(`${label} missing snippet "${snippet}". Received: ${text}`);
  }
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

function setupHrefFromContext(context) {
  const href = new URL(context.href);
  href.searchParams.set("setup", "1");
  return href.toString();
}

async function expectLocation(page, expected) {
  await page.waitForFunction(
    ({ seed, originId, setupOpen }) => {
      const params = new URLSearchParams(window.location.search);
      return params.get("seed") === seed
        && params.get("origin") === originId
        && (setupOpen ? params.get("setup") === "1" : !params.has("setup"));
    },
    expected
  );
}

async function ensureInspectorOpen(page) {
  const inspector = page.locator("[data-testid='seed-replay-inspector']");
  if (await inspector.count()) {
    return inspector;
  }

  await page.click("[data-testid='seed-replay-inspector-toggle']");
  await page.waitForSelector("[data-testid='seed-replay-inspector']");
  await page.waitForTimeout(150);
  return inspector;
}

async function readInspectorText(page) {
  const inspector = await ensureInspectorOpen(page);
  return normalizeText(await inspector.textContent());
}

async function assertSetupPanel(page, seed, originId, preview) {
  const seedInput = page.locator("[data-testid='run-setup-seed-input']");
  const originCard = page.locator(`[data-testid='run-setup-origin-${originId}']`);
  const setupText = normalizeText(await page.locator(".run-setup-panel").textContent());

  if ((await seedInput.inputValue()) !== seed) {
    throw new Error(`Run setup seed drifted. Expected ${seed}, received ${await seedInput.inputValue()}`);
  }

  const activeOrigin = await originCard.evaluate((node) => node.classList.contains("active"));
  if (!activeOrigin) {
    throw new Error(`Expected active run-setup origin ${originId}.`);
  }

  expectIncludes(setupText, preview.summary, `setup summary for ${originId}`);
  expectIncludes(setupText, preview.climate.value, `setup climate for ${originId}`);
  expectIncludes(setupText, preview.divineMood.value, `setup divine mood for ${originId}`);
  expectIncludes(setupText, preview.oracleDensity.value, `setup oracle density for ${originId}`);
}

async function assertInspector(page, { modeLabel, currentContext, currentState, draftContext, draftMatchesLive }) {
  const inspectorText = await readInspectorText(page);
  const seedInputValue = await page.locator("[data-testid='seed-replay-inspector-seed-input']").inputValue();
  const originValue = await page.locator("[data-testid='seed-replay-inspector-origin-select']").inputValue();

  expectIncludes(inspectorText, "Seed / Replay Inspector", "inspector title");
  expectIncludes(inspectorText, modeLabel, "inspector mode label");
  expectIncludes(inspectorText, currentContext.seed, "live seed");
  expectIncludes(inspectorText, currentState.worldGeneration.originTitle, "live origin title");
  expectIncludes(inspectorText, currentState.worldGeneration.summary, "live summary");
  expectIncludes(inspectorText, currentContext.href, "live replay href");
  expectIncludes(inspectorText, setupHrefFromContext(currentContext), "live setup href");
  expectIncludes(inspectorText, "Preview parity matches the live run.", "live preview parity");

  if (seedInputValue !== draftContext.seed) {
    throw new Error(`Inspector draft seed drifted. Expected ${draftContext.seed}, received ${seedInputValue}`);
  }
  if (originValue !== draftContext.originId) {
    throw new Error(`Inspector draft origin drifted. Expected ${draftContext.originId}, received ${originValue}`);
  }

  expectIncludes(inspectorText, draftContext.preview.summary, "draft summary");
  expectIncludes(inspectorText, draftContext.preview.climate.value, "draft climate");
  expectIncludes(inspectorText, draftContext.preview.divineMood.value, "draft divine mood");
  expectIncludes(inspectorText, draftContext.preview.oracleDensity.value, "draft oracle density");
  expectIncludes(inspectorText, draftContext.href, "draft replay href");

  if (draftMatchesLive) {
    expectIncludes(inspectorText, "Draft matches the live run.", "draft parity");
  } else {
    expectIncludes(inspectorText, "Draft changes:", "draft delta");
  }
}

function assertStartedState(state, seed, originId, preview) {
  if (state.run.originId !== originId) {
    throw new Error(`Expected started origin ${originId}, received ${state.run.originId}`);
  }
  if (state.run.seedText !== seed) {
    throw new Error(`Expected started seed ${seed}, received ${state.run.seedText}`);
  }
  if (state.world_generation.summary !== preview.summary) {
    throw new Error(`Started run summary drifted for ${originId}.`);
  }
}

async function setInspectorDraft(page, seed, originId) {
  await ensureInspectorOpen(page);
  await page.locator("[data-testid='seed-replay-inspector-seed-input']").fill(seed);
  await page.locator("[data-testid='seed-replay-inspector-origin-select']").selectOption(originId);
  await page.waitForTimeout(180);
}

async function exerciseSetupHandoffFlow(page, outDir, draftSeed, draftOriginId, artifactPrefix) {
  const liveState = await readDeepState(page);
  const currentContext = await readRunDebugContext(page);

  await setInspectorDraft(page, draftSeed, draftOriginId);
  const draftContext = await readRunDebugContext(page, draftSeed, draftOriginId);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext,
    currentState: liveState,
    draftContext,
    draftMatchesLive: false
  });

  writeJson(path.join(outDir, `${artifactPrefix}-draft-context.json`), {
    currentContext,
    draftContext
  });
  await page.screenshot({ path: path.join(outDir, `${artifactPrefix}-draft.png`), fullPage: true });

  await page.getByRole("button", { name: "Open in setup" }).click();
  await page.waitForSelector(".run-setup-panel");
  await page.waitForTimeout(200);
  await expectLocation(page, {
    seed: currentContext.seed,
    originId: currentContext.originId,
    setupOpen: true
  });
  await assertSetupPanel(page, draftSeed, draftOriginId, draftContext.preview);
  await assertInspector(page, {
    modeLabel: "Run setup is open.",
    currentContext,
    currentState: liveState,
    draftContext,
    draftMatchesLive: false
  });

  await page.screenshot({ path: path.join(outDir, `${artifactPrefix}-setup-handoff.png`), fullPage: true });
  await page.click("[data-testid='run-setup-start-btn']");
  await page.waitForFunction(() => !document.querySelector(".run-setup-panel"));
  await page.waitForTimeout(250);

  const startedState = await readState(page);
  const startedDeepState = await readDeepState(page);
  const startedContext = await readRunDebugContext(page);
  await expectLocation(page, {
    seed: draftSeed,
    originId: draftOriginId,
    setupOpen: false
  });
  assertStartedState(startedState, draftSeed, draftOriginId, draftContext.preview);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext: startedContext,
    currentState: startedDeepState,
    draftContext: startedContext,
    draftMatchesLive: true
  });

  writeJson(path.join(outDir, `${artifactPrefix}-started-state.json`), startedState);
  writeJson(path.join(outDir, `${artifactPrefix}-started-context.json`), startedContext);
  await page.screenshot({ path: path.join(outDir, `${artifactPrefix}-started.png`), fullPage: true });
}

async function exerciseDirectStartFlow(page, outDir, draftSeed, draftOriginId, artifactPrefix) {
  const liveState = await readDeepState(page);
  const currentContext = await readRunDebugContext(page);

  await setInspectorDraft(page, draftSeed, draftOriginId);
  const draftContext = await readRunDebugContext(page, draftSeed, draftOriginId);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext,
    currentState: liveState,
    draftContext,
    draftMatchesLive: false
  });

  await page.getByRole("button", { name: "Use live run" }).click();
  await page.waitForTimeout(180);
  const resetDraftContext = await readRunDebugContext(page);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext,
    currentState: liveState,
    draftContext: resetDraftContext,
    draftMatchesLive: true
  });

  await setInspectorDraft(page, draftSeed, draftOriginId);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext,
    currentState: liveState,
    draftContext,
    draftMatchesLive: false
  });

  writeJson(path.join(outDir, `${artifactPrefix}-draft-context.json`), {
    currentContext,
    draftContext,
    resetDraftContext
  });
  await page.screenshot({ path: path.join(outDir, `${artifactPrefix}-draft.png`), fullPage: true });

  await page.click("[data-testid='seed-replay-inspector-start-btn']");
  await page.waitForTimeout(250);

  const startedState = await readState(page);
  const startedDeepState = await readDeepState(page);
  const startedContext = await readRunDebugContext(page);
  await expectLocation(page, {
    seed: draftSeed,
    originId: draftOriginId,
    setupOpen: false
  });
  assertStartedState(startedState, draftSeed, draftOriginId, draftContext.preview);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext: startedContext,
    currentState: startedDeepState,
    draftContext: startedContext,
    draftMatchesLive: true
  });

  writeJson(path.join(outDir, `${artifactPrefix}-started-state.json`), startedState);
  writeJson(path.join(outDir, `${artifactPrefix}-started-context.json`), startedContext);
  await page.screenshot({ path: path.join(outDir, `${artifactPrefix}-started.png`), fullPage: true });
}

async function captureOrigin(page, outDir, url, seed, originId, options = {}) {
  await openSetup(page, url, seed, originId);
  const previewA = await readPreview(page, seed, originId);
  const previewB = await readPreview(page, seed, originId);

  if (JSON.stringify(previewA) !== JSON.stringify(previewB)) {
    throw new Error(`Preview drift detected for ${originId} / ${seed}.`);
  }

  await assertSetupPanel(page, seed, originId, previewA);
  const setupState = await readDeepState(page);
  const setupContext = await readRunDebugContext(page);
  await assertInspector(page, {
    modeLabel: "Run setup is open.",
    currentContext: setupContext,
    currentState: setupState,
    draftContext: setupContext,
    draftMatchesLive: true
  });

  writeJson(path.join(outDir, `preview-${originId}.json`), previewA);
  writeJson(path.join(outDir, `setup-context-${originId}.json`), setupContext);
  await page.screenshot({ path: path.join(outDir, `setup-${originId}.png`), fullPage: true });

  await page.click("[data-testid='run-setup-start-btn']");
  await page.waitForFunction(() => !document.querySelector(".run-setup-panel"));
  await page.waitForTimeout(250);
  await expectLocation(page, {
    seed,
    originId,
    setupOpen: false
  });

  const state = await readState(page);
  const deepState = await readDeepState(page);
  const liveContext = await readRunDebugContext(page);
  assertStartedState(state, seed, originId, previewA);
  await assertInspector(page, {
    modeLabel: "Live shell mode.",
    currentContext: liveContext,
    currentState: deepState,
    draftContext: liveContext,
    draftMatchesLive: true
  });

  writeJson(path.join(outDir, `state-${originId}.json`), state);
  writeJson(path.join(outDir, `live-context-${originId}.json`), liveContext);
  await page.screenshot({ path: path.join(outDir, `started-${originId}.png`), fullPage: true });

  if (options.debugFlow === "setup-handoff") {
    await exerciseSetupHandoffFlow(page, outDir, options.draftSeed, options.draftOriginId, `debug-${originId}`);
  } else if (options.debugFlow === "direct-start") {
    await exerciseDirectStartFlow(page, outDir, options.draftSeed, options.draftOriginId, `debug-${originId}`);
  }

  return {
    preview: previewA,
    state
  };
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

  const merchant = await captureOrigin(page, outDir, url, "Delphi-001", "merchant-oracle", {
    debugFlow: "setup-handoff",
    draftSeed: "Delphi-009",
    draftOriginId: "war-oracle"
  });
  const war = await captureOrigin(page, outDir, url, "Delphi-001", "war-oracle", {
    debugFlow: "direct-start",
    draftSeed: "Pythia-4242",
    draftOriginId: "merchant-oracle"
  });

  if (merchant.state.world_generation.summary === war.state.world_generation.summary) {
    throw new Error("Different origins on the same seed produced the same opening summary.");
  }
  if (merchant.state.resources.gold <= war.state.resources.gold) {
    throw new Error("Merchant Oracle should open with a stronger treasury than War Oracle.");
  }

  writeJson(path.join(outDir, "errors.json"), errors);
  if (errors.length > 0) {
    throw new Error(`Encountered browser errors during origins smoke: ${JSON.stringify(errors)}`);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
