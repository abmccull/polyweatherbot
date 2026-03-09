import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = {
    url: "http://localhost:5173",
    outDir: "/Users/tsc-001/station_sniper/The Oracle/output/playwright/precinct-smoke"
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

async function readPrecinctArtDebug(page) {
  return await page.evaluate(() => window.__oracleDebug.getPrecinctArtDebug?.() ?? null);
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

const TOOL_CATEGORY = {
  sacred_way: "processional",
  priest_quarters: "housing",
  storehouse: "storage",
  castalian_spring: "ritual"
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

  await selectTool(page, "sacred_way");
  await clickPrecinctTile(page, { x: 30, y: 50 });
  await clickPrecinctTile(page, { x: 31, y: 50 });
  await clickPrecinctTile(page, { x: 32, y: 50 });

  await selectTool(page, "priest_quarters");
  await clickPrecinctTile(page, { x: 30, y: 49 });

  await selectTool(page, "storehouse");
  await clickPrecinctTile(page, { x: 31, y: 49 });

  await selectTool(page, "castalian_spring");
  await clickPrecinctTile(page, { x: 32, y: 49 });

  await selectTool(page, "select");
  await clickPrecinctTile(page, { x: 32, y: 49 });
  await page.waitForTimeout(200);
  const assignPriestButton = page.locator("#assign-priest-btn");
  if (await assignPriestButton.isVisible().catch(() => false)) {
    await assignPriestButton.click();
  }
  await page.evaluate(async () => {
    await window.advanceTime(4000);
  });
  await page.waitForTimeout(200);
  await page.waitForSelector("#precinct-soul-panel");

  const soulSummary = (await page.textContent("#precinct-soul-summary"))?.trim() ?? "";
  const siteNote = (await page.textContent("#precinct-site-note"))?.trim() ?? "";

  const state = await readState(page);
  const artDebug = await readPrecinctArtDebug(page);
  fs.writeFileSync(path.join(outDir, "state.json"), JSON.stringify(state, null, 2));
  fs.writeFileSync(path.join(outDir, "art-debug.json"), JSON.stringify(artDebug, null, 2));
  await page.screenshot({ path: path.join(outDir, "precinct-smoke-full.png"), fullPage: true });
  fs.writeFileSync(path.join(outDir, "errors.json"), JSON.stringify(errors, null, 2));

  if (state.buildings.length !== 3) {
    throw new Error(`Expected 3 buildings, found ${state.buildings.length}`);
  }
  if (state.buildings.filter((building) => building.defId === "castalian_spring").length !== 1) {
    throw new Error("Castalian Spring was not placed");
  }
  if (state.selected === null || !String(state.selected).includes("building-")) {
    throw new Error("No building remained selected after placement flow");
  }
  if (soulSummary.length === 0) {
    throw new Error("Precinct soul summary did not render");
  }
  if (siteNote.length === 0 || !siteNote.toLowerCase().includes("selected castalian spring")) {
    throw new Error(`Unexpected precinct site note: ${siteNote}`);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
