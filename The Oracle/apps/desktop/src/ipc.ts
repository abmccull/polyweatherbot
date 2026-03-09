import type { GameSnapshot } from "@the-oracle/core";
import type { OracleProfileSnapshot, SaveSlotMeta } from "@the-oracle/persistence";

export const ORACLE_DESKTOP_CHANNELS = {
  listSlots: "oracle:save:listSlots",
  loadSlot: "oracle:save:loadSlot",
  saveSlot: "oracle:save:saveSlot",
  deleteSlot: "oracle:save:deleteSlot",
  loadProfile: "oracle:profile:load",
  saveProfile: "oracle:profile:save",
  resetProfile: "oracle:profile:reset",
  runtimeInfo: "oracle:runtime:getInfo"
} as const;

export type OracleDesktopRuntimeInfo = {
  isDesktop: true;
  userDataPath: string;
  saveDbPath: string;
  profileDbPath: string;
};

export type OracleDesktopPersistenceBridge = {
  listSlots: () => Promise<SaveSlotMeta[]>;
  loadSlot: (slotId: string) => Promise<GameSnapshot | null>;
  saveSlot: (slotId: string, snapshot: GameSnapshot) => Promise<void>;
  deleteSlot: (slotId: string) => Promise<void>;
};

export type OracleDesktopProfileBridge = {
  load: () => Promise<OracleProfileSnapshot>;
  save: (snapshot: OracleProfileSnapshot) => Promise<void>;
  reset: () => Promise<void>;
};
