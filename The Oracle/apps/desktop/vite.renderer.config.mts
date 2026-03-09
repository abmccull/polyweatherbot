import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const webRoot = path.resolve(import.meta.dirname, "../web");

export default defineConfig({
  root: webRoot,
  base: "./",
  plugins: [react()],
  publicDir: path.resolve(webRoot, "public"),
  resolve: {
    dedupe: ["react", "react-dom"],
    preserveSymlinks: false
  },
  server: {
    host: "127.0.0.1",
    port: 5173
  },
  build: {
    emptyOutDir: false,
    outDir: path.resolve(import.meta.dirname, ".vite/renderer/main_window"),
    chunkSizeWarningLimit: 1300,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/node_modules/react-dom/") || id.includes("/node_modules/react/") || id.includes("/node_modules/scheduler/")) {
            return "react-vendor";
          }
          if (id.includes("/node_modules/phaser/")) {
            return "phaser-core";
          }
          return undefined;
        }
      }
    }
  }
});
