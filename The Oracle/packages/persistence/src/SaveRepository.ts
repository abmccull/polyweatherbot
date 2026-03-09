import type { GameSnapshot } from "@the-oracle/core";

export type SaveSlotMeta = {
  id: string;
  savedAt: string;
};

export interface SaveRepository {
  listSlots(): Promise<SaveSlotMeta[]>;
  loadSlot(slotId: string): Promise<GameSnapshot | null>;
  saveSlot(slotId: string, snapshot: GameSnapshot): Promise<void>;
  deleteSlot(slotId: string): Promise<void>;
}
