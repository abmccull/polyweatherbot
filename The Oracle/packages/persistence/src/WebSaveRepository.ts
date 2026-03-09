import { openDB } from "idb";

import type { GameSnapshot } from "@the-oracle/core";

import { deserializeSnapshot, serializeSnapshot } from "./schema";
import type { SaveRepository, SaveSlotMeta } from "./SaveRepository";

const DB_NAME = "the-oracle-saves";
const STORE_NAME = "slots";

type SaveRow = {
  id: string;
  payload: string;
  savedAt: string;
};

async function db() {
  return openDB(DB_NAME, 1, {
    upgrade(database) {
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    }
  });
}

export class WebSaveRepository implements SaveRepository {
  async listSlots(): Promise<SaveSlotMeta[]> {
    const database = await db();
    const rows = await database.getAll(STORE_NAME);
    return (rows as SaveRow[]).map((row) => ({
      id: row.id,
      savedAt: row.savedAt
    }));
  }

  async loadSlot(slotId: string): Promise<GameSnapshot | null> {
    const database = await db();
    const row = (await database.get(STORE_NAME, slotId)) as SaveRow | undefined;
    if (!row) {
      return null;
    }
    return deserializeSnapshot(row.payload);
  }

  async saveSlot(slotId: string, snapshot: GameSnapshot): Promise<void> {
    const database = await db();
    const row: SaveRow = {
      id: slotId,
      payload: serializeSnapshot(snapshot),
      savedAt: new Date().toISOString()
    };
    await database.put(STORE_NAME, row);
  }

  async deleteSlot(slotId: string): Promise<void> {
    const database = await db();
    await database.delete(STORE_NAME, slotId);
  }
}
