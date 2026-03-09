import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    dedupe: ["react", "react-dom"],
    preserveSymlinks: false
  },
  server: {
    port: 5173
  },
  build: {
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
