import { builtinModules } from "node:module";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    sourcemap: true,
    target: "node20",
    lib: {
      entry: "src/main.ts",
      fileName: () => "main.cjs",
      formats: ["cjs"]
    },
    rollupOptions: {
      external: [
        "electron",
        "better-sqlite3",
        ...builtinModules,
        ...builtinModules.map((moduleName) => `node:${moduleName}`)
      ]
    }
  }
});
