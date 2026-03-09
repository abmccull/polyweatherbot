import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const desktopDir = path.resolve(scriptDir, "..");

const child = spawn("pnpm", ["exec", "electron-forge", "start", "--", "--smoke-test"], {
  cwd: desktopDir,
  env: {
    ...process.env
  },
  stdio: "inherit"
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});
