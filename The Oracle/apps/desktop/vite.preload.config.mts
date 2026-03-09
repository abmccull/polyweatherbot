import { builtinModules } from "node:module";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    sourcemap: true,
    target: "node20",
    rollupOptions: {
      output: {
        format: "cjs",
        inlineDynamicImports: true,
        entryFileNames: "preload.cjs",
        chunkFileNames: "[name].cjs",
        assetFileNames: "[name].[ext]"
      },
      external: [
        "electron",
        ...builtinModules,
        ...builtinModules.map((moduleName) => `node:${moduleName}`)
      ]
    }
  }
});
