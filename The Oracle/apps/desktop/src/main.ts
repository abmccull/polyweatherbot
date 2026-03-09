import { app, BrowserWindow, ipcMain } from "electron";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SqliteProfileRepository } from "@the-oracle/persistence/profile-sqlite";
import { SqliteSaveRepository } from "@the-oracle/persistence/sqlite";

import { ORACLE_DESKTOP_CHANNELS, type OracleDesktopRuntimeInfo } from "./ipc";

declare const MAIN_WINDOW_VITE_DEV_SERVER_URL: string | undefined;
declare const MAIN_WINDOW_VITE_NAME: string;

const moduleDir = path.dirname(fileURLToPath(import.meta.url));

let ipcRegistered = false;
const smokeMode = process.argv.includes("--smoke-test") || process.env.ORACLE_SMOKE_TEST === "1";
let smokeOutcomeWritten = false;

function parseSmokeFile(argv: string[]) {
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--smoke-file") {
      return argv[index + 1];
    }
    if (arg.startsWith("--smoke-file=")) {
      return arg.slice("--smoke-file=".length);
    }
  }
  return process.env.ORACLE_SMOKE_FILE;
}

const smokeFilePath = parseSmokeFile(process.argv);

function resolveDesktopFile(...segments: string[]) {
  return path.resolve(moduleDir, ...segments);
}

function assertFileExists(filePath: string, label: string) {
  if (!existsSync(filePath)) {
    throw new Error(`${label} not found at ${filePath}`);
  }
  return filePath;
}

function resolvePreloadPath() {
  return assertFileExists(resolveDesktopFile("preload.cjs"), "Desktop preload");
}

function resolveRendererEntry() {
  return assertFileExists(resolveDesktopFile("..", "renderer", MAIN_WINDOW_VITE_NAME, "index.html"), "Packaged web entry");
}

function resolveAllowedDevUrl() {
  if (!MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    return undefined;
  }

  const parsed = new URL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  if (!["localhost", "127.0.0.1"].includes(parsed.hostname)) {
    throw new Error(`Refusing to load untrusted dev origin: ${parsed.origin}`);
  }

  return parsed;
}

function writeSmokeStatus(payload: Record<string, unknown>) {
  const smokeFile = smokeFilePath;
  if (!smokeFile) {
    return;
  }
  mkdirSync(path.dirname(smokeFile), { recursive: true });
  writeFileSync(smokeFile, JSON.stringify(payload, null, 2));
  smokeOutcomeWritten = true;
}

function normalizeError(error: unknown) {
  if (error instanceof Error) {
    return {
      message: error.message,
      stack: error.stack
    };
  }

  return {
    message: String(error)
  };
}

function reportSmokeFailure(reason: string, error?: unknown, extra: Record<string, unknown> = {}) {
  writeSmokeStatus({
    loaded: false,
    reason,
    ...extra,
    ...normalizeError(error)
  });
}

function getRuntimeInfo(): OracleDesktopRuntimeInfo {
  const userDataPath = app.getPath("userData");
  const databasePath = path.join(userDataPath, "oracle-data.sqlite");

  return {
    isDesktop: true,
    userDataPath,
    saveDbPath: databasePath,
    profileDbPath: databasePath
  };
}

function registerDesktopIpc(runtimeInfo: OracleDesktopRuntimeInfo) {
  if (ipcRegistered) {
    return;
  }

  const saveRepository = new SqliteSaveRepository(runtimeInfo.saveDbPath);
  const profileRepository = new SqliteProfileRepository(runtimeInfo.profileDbPath);

  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.listSlots, () => saveRepository.listSlots());
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.loadSlot, (_event, slotId: string) => saveRepository.loadSlot(slotId));
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.saveSlot, (_event, slotId: string, snapshot) => saveRepository.saveSlot(slotId, snapshot));
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.deleteSlot, (_event, slotId: string) => saveRepository.deleteSlot(slotId));
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.loadProfile, () => profileRepository.loadProfile());
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.saveProfile, (_event, snapshot) => profileRepository.saveProfile(snapshot));
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.resetProfile, () => profileRepository.resetProfile());
  ipcMain.handle(ORACLE_DESKTOP_CHANNELS.runtimeInfo, () => runtimeInfo);

  ipcRegistered = true;
}

async function createMainWindow() {
  const preloadPath = resolvePreloadPath();
  const rendererEntry = resolveRendererEntry();
  const allowedDevUrl = resolveAllowedDevUrl();
  const recentConsoleMessages: string[] = [];
  let smokeResolved = false;
  let smokeTimeout: ReturnType<typeof setTimeout> | undefined;
  const win = new BrowserWindow({
    width: 1600,
    height: 980,
    backgroundColor: "#d3bf90",
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true
    }
  });

  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  if (smokeMode) {
    const markSmokeLoaded = async (phase: string) => {
      if (smokeResolved) {
        return;
      }

      smokeResolved = true;
      if (smokeTimeout) {
        clearTimeout(smokeTimeout);
      }
      let readyState = "unknown";
      try {
        readyState = await win.webContents.executeJavaScript("document.readyState", true);
      } catch {
        readyState = "unavailable";
      }

      writeSmokeStatus({
        loaded: true,
        phase,
        url: win.webContents.getURL(),
        readyState
      });
      setTimeout(() => {
        for (const openWindow of BrowserWindow.getAllWindows()) {
          openWindow.destroy();
        }
        app.exit(0);
      }, 250);
    };
    writeSmokeStatus({
      loaded: false,
      reason: "booting",
      preloadPath,
      rendererEntry
    });
    win.webContents.on("console-message", (_event, _level, message) => {
      recentConsoleMessages.push(message);
      if (recentConsoleMessages.length > 10) {
        recentConsoleMessages.shift();
      }
    });
    win.webContents.once("dom-ready", () => {
      void markSmokeLoaded("dom-ready");
    });
    win.webContents.once("did-stop-loading", () => {
      void markSmokeLoaded("did-stop-loading");
    });
  }
  win.webContents.on("will-navigate", (event, url) => {
    if (allowedDevUrl) {
      if (new URL(url).origin === allowedDevUrl.origin) {
        return;
      }
      event.preventDefault();
      return;
    }
    if (!url.startsWith("file://")) {
      event.preventDefault();
      return;
    }

    if (fileURLToPath(url) !== rendererEntry) {
      event.preventDefault();
    }
  });
  win.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL, isMainFrame) => {
    if (!smokeMode || !isMainFrame) {
      return;
    }

    reportSmokeFailure("did-fail-load", undefined, {
      errorCode,
      errorDescription,
      validatedURL
    });
    setTimeout(() => app.exit(2), 50);
  });
  win.webContents.on("render-process-gone", (_event, details) => {
    if (!smokeMode) {
      return;
    }

    reportSmokeFailure("render-process-gone", undefined, {
      details
    });
    setTimeout(() => app.exit(2), 50);
  });

  if (allowedDevUrl) {
    await win.loadURL(allowedDevUrl.toString());
  } else {
    await win.loadFile(rendererEntry);
  }

  if (smokeMode) {
    smokeTimeout = setTimeout(() => {
      let loadingURL = "";
      try {
        loadingURL = win.webContents.getURL();
      } catch {
        loadingURL = "";
      }

      writeSmokeStatus({
        loaded: false,
        reason: "timeout",
        url: loadingURL,
        isLoading: win.webContents.isLoading(),
        consoleMessages: recentConsoleMessages
      });
      for (const openWindow of BrowserWindow.getAllWindows()) {
        openWindow.destroy();
      }
      app.exit(3);
    }, 5000);
  }
}

app.whenReady().then(() => {
  registerDesktopIpc(getRuntimeInfo());
  void createMainWindow().catch((error) => {
    if (smokeMode) {
      reportSmokeFailure("create-main-window", error);
      app.exit(1);
      return;
    }

    throw error;
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createMainWindow().catch((error) => {
        if (smokeMode) {
          reportSmokeFailure("create-main-window", error);
          app.exit(1);
          return;
        }

        throw error;
      });
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

process.on("uncaughtException", (error) => {
  if (smokeMode && !smokeOutcomeWritten) {
    reportSmokeFailure("uncaught-exception", error);
    app.exit(1);
    return;
  }

  throw error;
});

process.on("unhandledRejection", (error) => {
  if (smokeMode && !smokeOutcomeWritten) {
    reportSmokeFailure("unhandled-rejection", error);
    app.exit(1);
  }
});

if (smokeMode && smokeFilePath) {
  writeSmokeStatus({
    loaded: false,
    reason: "process-started",
    argv: process.argv.slice(1)
  });
}
