import { openDB } from "idb";

import { createEmptyProfileSnapshot, deserializeProfileSnapshot, serializeProfileSnapshot } from "./profileSchema";
import type { OracleProfileSnapshot, ProfileRepository } from "./ProfileRepository";

const DB_NAME = "the-oracle-profile";
const STORE_NAME = "profile";
const PROFILE_KEY = "main";

type ProfileRow = {
  id: string;
  payload: string;
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

export class WebProfileRepository implements ProfileRepository {
  async loadProfile(): Promise<OracleProfileSnapshot> {
    const database = await db();
    const row = (await database.get(STORE_NAME, PROFILE_KEY)) as ProfileRow | undefined;
    return row ? deserializeProfileSnapshot(row.payload) : createEmptyProfileSnapshot();
  }

  async saveProfile(snapshot: OracleProfileSnapshot): Promise<void> {
    const database = await db();
    await database.put(STORE_NAME, {
      id: PROFILE_KEY,
      payload: serializeProfileSnapshot(snapshot)
    } satisfies ProfileRow);
  }

  async resetProfile(): Promise<void> {
    const database = await db();
    await database.delete(STORE_NAME, PROFILE_KEY);
  }
}
