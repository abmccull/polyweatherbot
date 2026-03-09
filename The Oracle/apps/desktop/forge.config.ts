import type { ForgeConfig } from "@electron-forge/shared-types";

const config: ForgeConfig = {
  packagerConfig: {
    asar: true,
    name: "The Oracle"
  },
  makers: [
    {
      name: "@electron-forge/maker-zip",
      platforms: ["darwin"]
    }
  ],
  plugins: [
    {
      name: "@electron-forge/plugin-auto-unpack-natives",
      config: {}
    },
    {
      name: "@electron-forge/plugin-vite",
      config: {
        build: [
          {
            entry: "src/main.ts",
            config: "vite.main.config.mts"
          },
          {
            entry: "src/preload.ts",
            config: "vite.preload.config.mts"
          }
        ],
        renderer: [
          {
            name: "main_window",
            config: "vite.renderer.config.mts"
          }
        ]
      }
    }
  ]
};

export default config;
