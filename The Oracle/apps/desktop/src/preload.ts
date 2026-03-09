import { contextBridge, ipcRenderer } from "electron";

import type { GameSnapshot } from "@the-oracle/core";
import type { OracleProfileSnapshot } from "@the-oracle/persistence";

import { ORACLE_DESKTOP_CHANNELS, type OracleDesktopRuntimeInfo } from "./ipc";

const oracleDesktop = {
  persistence: {
    listSlots: () => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.listSlots) as Promise<Awaited<ReturnType<typeof ipcRenderer.invoke>>>,
    loadSlot: (slotId: string) => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.loadSlot, slotId) as Promise<GameSnapshot | null>,
    saveSlot: (slotId: string, snapshot: GameSnapshot) => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.saveSlot, slotId, snapshot) as Promise<void>,
    deleteSlot: (slotId: string) => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.deleteSlot, slotId) as Promise<void>
  },
  profile: {
    load: () => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.loadProfile) as Promise<OracleProfileSnapshot>,
    save: (snapshot: OracleProfileSnapshot) => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.saveProfile, snapshot) as Promise<void>,
    reset: () => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.resetProfile) as Promise<void>
  },
  runtime: {
    isDesktop: true as const,
    getInfo: () => ipcRenderer.invoke(ORACLE_DESKTOP_CHANNELS.runtimeInfo) as Promise<OracleDesktopRuntimeInfo>
  }
};

contextBridge.exposeInMainWorld("oracleDesktop", oracleDesktop);
