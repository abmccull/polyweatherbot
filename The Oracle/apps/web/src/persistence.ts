import type { GameSnapshot } from "@the-oracle/core";
import {
  WebProfileRepository,
  WebSaveRepository,
  type OracleProfileSnapshot,
  type ProfileRepository,
  type SaveRepository,
  type SaveSlotMeta
} from "@the-oracle/persistence";

export type OracleDesktopRuntimeInfo = {
  isDesktop: true;
  userDataPath: string;
  saveDbPath: string;
  profileDbPath: string;
};

export type OracleDesktopBridge = {
  persistence: {
    listSlots: () => Promise<SaveSlotMeta[]>;
    loadSlot: (slotId: string) => Promise<GameSnapshot | null>;
    saveSlot: (slotId: string, snapshot: GameSnapshot) => Promise<void>;
    deleteSlot: (slotId: string) => Promise<void>;
  };
  profile: {
    load: () => Promise<OracleProfileSnapshot>;
    save: (snapshot: OracleProfileSnapshot) => Promise<void>;
    reset: () => Promise<void>;
  };
  runtime: {
    isDesktop: true;
    getInfo: () => Promise<OracleDesktopRuntimeInfo>;
  };
};

class DesktopSaveRepository implements SaveRepository {
  constructor(private readonly bridge: OracleDesktopBridge) {}

  listSlots(): Promise<SaveSlotMeta[]> {
    return this.bridge.persistence.listSlots();
  }

  loadSlot(slotId: string): Promise<GameSnapshot | null> {
    return this.bridge.persistence.loadSlot(slotId);
  }

  saveSlot(slotId: string, snapshot: GameSnapshot): Promise<void> {
    return this.bridge.persistence.saveSlot(slotId, snapshot);
  }

  deleteSlot(slotId: string): Promise<void> {
    return this.bridge.persistence.deleteSlot(slotId);
  }
}

class DesktopProfileRepository implements ProfileRepository {
  constructor(private readonly bridge: OracleDesktopBridge) {}

  loadProfile(): Promise<OracleProfileSnapshot> {
    return this.bridge.profile.load();
  }

  saveProfile(snapshot: OracleProfileSnapshot): Promise<void> {
    return this.bridge.profile.save(snapshot);
  }

  resetProfile(): Promise<void> {
    return this.bridge.profile.reset();
  }
}

export function createPersistenceAdapters(bridge = window.oracleDesktop) {
  if (bridge?.runtime.isDesktop) {
    return {
      isDesktop: true,
      saveRepository: new DesktopSaveRepository(bridge),
      profileRepository: new DesktopProfileRepository(bridge)
    };
  }

  return {
    isDesktop: false,
    saveRepository: new WebSaveRepository(),
    profileRepository: new WebProfileRepository()
  };
}
