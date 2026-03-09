import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/logistics-smoke"
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

async function waitForLogisticsReady(page) {
  await page.waitForFunction(() => {
    const state = JSON.parse(window.render_game_to_text());
    return Array.isArray(state.resource_jobs) && state.resource_jobs.length > 0;
  });
}

async function clickPrecinctTile(page, tile) {
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

async function selectCarrierWalker(page) {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const state = await readState(page);
    const carrier = state.walkers.find((walker) => walker.role === "carrier" && walker.assigned_job);
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

    await page.evaluate(async () => {
      await window.advanceTime(300);
    });
    await page.waitForTimeout(100);
  }

  throw new Error("Carrier selection flow did not complete");
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
    window.__oracleDebug.injectScenario("logistics-lab");
  });
  await page.waitForTimeout(200);
  await waitForLogisticsReady(page);
  await page.evaluate(async () => {
    await window.advanceTime(2400);
  });
  await page.waitForTimeout(200);
  await waitForLogisticsReady(page);

  const stateQueued = await readState(page);
  fs.writeFileSync(path.join(outDir, "state-queued.json"), JSON.stringify(stateQueued, null, 2));
  await page.screenshot({ path: path.join(outDir, "logistics-queued-full.png"), fullPage: true });
  await page.waitForSelector("[data-testid='logistics-job-row-0']");
  const queuedTopJobRowText = await page.locator("[data-testid='logistics-job-row-0']").textContent();

  await page.click("#tool-select");
  const stateSelected = await selectCarrierWalker(page);
  fs.writeFileSync(path.join(outDir, "state-selected.json"), JSON.stringify(stateSelected, null, 2));
  await page.screenshot({ path: path.join(outDir, "logistics-selected-full.png"), fullPage: true });
  await page.evaluate(async () => {
    await window.advanceTime(2600);
  });
  await page.waitForTimeout(200);

  const stateActive = await readState(page);
  fs.writeFileSync(path.join(outDir, "state-active.json"), JSON.stringify(stateActive, null, 2));
  await page.screenshot({ path: path.join(outDir, "logistics-active-full.png"), fullPage: true });
  fs.writeFileSync(path.join(outDir, "errors.json"), JSON.stringify(errors, null, 2));

  const carrierCountText = await page.locator("#carrier-count-pill").textContent();
  const carrierStrainText = await page.locator("#carrier-strain-pill").textContent();
  const carrierFatigueText = await page.locator("#selected-carrier-fatigue").textContent();
  const carrierSkillText = await page.locator("#selected-carrier-skill").textContent();
  const carrierRadiusText = await page.locator("#selected-carrier-radius").textContent();
  const kitchen = stateQueued.buildings.find((building) => building.defId === "kitchen");
  const altar = stateQueued.buildings.find((building) => building.defId === "sacrificial_altar");
  const olivePress = stateQueued.buildings.find((building) => building.defId === "olive_press");
  const stateActiveKitchen = stateActive.buildings.find((building) => building.defId === "kitchen");
  const stateActiveAnimalPen = stateActive.buildings.find((building) => building.defId === "animal_pen");
  const stateActiveOlivePress = stateActive.buildings.find((building) => building.defId === "olive_press");

  if (stateQueued.buildings.filter((building) => building.defId === "storehouse").length !== 2) {
    throw new Error("Expected the logistics lab to contain two storehouses");
  }
  if (!kitchen || !altar || !olivePress) {
    throw new Error("Expected the logistics lab to include the new economy buildings");
  }
  if ((stateQueued.carriers ?? 0) < 3) {
    throw new Error(`Expected at least 3 carriers after staffing, received ${stateQueued.carriers}`);
  }
  if (stateQueued.resource_jobs.length < 4) {
    throw new Error(`Expected at least 4 logistics jobs, received ${stateQueued.resource_jobs.length}`);
  }
  if (!stateQueued.resource_jobs.some((job) => job.priority === "critical")) {
    throw new Error("Expected at least one critical logistics job");
  }
  if (!stateQueued.resource_jobs.some((job) => job.priority === "routine")) {
    throw new Error("Expected at least one routine logistics rebalance job");
  }
  if (!stateQueued.resource_jobs.some((job) => job.target === kitchen.id && job.resource === "grain")) {
    throw new Error("Expected a grain supply job for the kitchen");
  }
  if (!stateQueued.resource_jobs.some((job) => job.target === altar.id && job.resource === "sacred_animals")) {
    throw new Error("Expected a sacred-animal supply job for the altar");
  }
  if (!stateQueued.resource_jobs.some((job) => job.target === olivePress.id && job.resource === "olives")) {
    throw new Error("Expected an olive supply job for the olive press");
  }
  if ((stateQueued.carrier_summary?.distinctRadii ?? []).length < 2) {
    throw new Error(`Expected mixed carrier radii in the logistics lab, received ${JSON.stringify(stateQueued.carrier_summary)}`);
  }
  if (!stateQueued.walkers.some((walker) => walker.role === "carrier" && typeof walker.hauling_skill === "number" && typeof walker.supply_radius === "number")) {
    throw new Error("Expected render_game_to_text to expose carrier hauling skill and supply radius");
  }
  if (!queuedTopJobRowText?.includes("critical")) {
    throw new Error(`Queued top UI job row did not surface a critical job: ${queuedTopJobRowText}`);
  }
  if (!carrierCountText?.includes("Carriers") || !carrierCountText.includes("3")) {
    throw new Error(`Carrier count UI did not show the expanded logistics staffing: ${carrierCountText}`);
  }
  if (!carrierStrainText?.includes("Strain")) {
    throw new Error(`Carrier strain UI did not render: ${carrierStrainText}`);
  }
  if (stateSelected.selected === null || !String(stateSelected.selected).includes("walker-carrier")) {
    throw new Error("Expected a carrier to be selected immediately after the canvas click flow");
  }
  if (!carrierFatigueText?.includes("Fatigue")) {
    throw new Error(`Selected carrier fatigue did not render: ${carrierFatigueText}`);
  }
  if (!carrierSkillText?.includes("Hauling skill")) {
    throw new Error(`Selected carrier skill did not render: ${carrierSkillText}`);
  }
  if (!carrierRadiusText?.includes("Supply radius")) {
    throw new Error(`Selected carrier radius did not render: ${carrierRadiusText}`);
  }
  if (!stateActive.walkers.some((walker) => walker.role === "carrier" && (walker.state === "hauling" || walker.state === "delivering"))) {
    throw new Error("Expected at least one carrier to be hauling or delivering after the second advance");
  }
  if ((stateActive.carrier_summary?.highestFatigue ?? 0) < 18) {
    throw new Error(`Expected carrier fatigue to build under load, received ${JSON.stringify(stateActive.carrier_summary)}`);
  }
  if ((stateActiveKitchen?.stored?.bread ?? 0) <= 0) {
    throw new Error(`Expected the kitchen to have produced bread, received ${JSON.stringify(stateActiveKitchen)}`);
  }
  if ((stateActiveAnimalPen?.stored?.sacred_animals ?? 0) <= 0) {
    throw new Error(`Expected the animal pen to have raised sacred animals, received ${JSON.stringify(stateActiveAnimalPen)}`);
  }
  if ((stateActiveOlivePress?.stored?.olive_oil ?? 0) <= (olivePress?.stored?.olive_oil ?? 0)) {
    throw new Error(`Expected the olive press to have produced oil, received ${JSON.stringify(stateActiveOlivePress)}`);
  }
  if (errors.length > 0) {
    throw new Error(`Encountered browser errors during logistics smoke: ${JSON.stringify(errors)}`);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
