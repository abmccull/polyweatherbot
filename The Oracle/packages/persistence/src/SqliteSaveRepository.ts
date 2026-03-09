import type { GameSnapshot } from "@the-oracle/core";

import { deserializeSnapshot, serializeSnapshot } from "./schema";
import type { SaveRepository, SaveSlotMeta } from "./SaveRepository";

type BetterSqlite3Database = {
  exec(sql: string): void;
  prepare(sql: string): {
    run(...args: unknown[]): unknown;
    get(...args: unknown[]): { id: string; payload: string; saved_at: string } | undefined;
    all(...args: unknown[]): { id: string; saved_at: string }[];
  };
};

export class SqliteSaveRepository implements SaveRepository {
  private database?: BetterSqlite3Database;

  constructor(private readonly dbPath: string) {}

  private async getDb(): Promise<BetterSqlite3Database> {
    if (this.database) {
      return this.database;
    }

    const mod = (await import("better-sqlite3")) as unknown as { default: new (path: string) => BetterSqlite3Database };
    this.database = new mod.default(this.dbPath);
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS save_slots (
        id TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        saved_at TEXT NOT NULL
      );
    `);
    return this.database;
  }

  async listSlots(): Promise<SaveSlotMeta[]> {
    const database = await this.getDb();
    const rows = database.prepare("SELECT id, saved_at FROM save_slots ORDER BY saved_at DESC").all();
    return rows.map((row) => ({
      id: row.id,
      savedAt: row.saved_at
    }));
  }

  async loadSlot(slotId: string): Promise<GameSnapshot | null> {
    const database = await this.getDb();
    const row = database.prepare("SELECT id, payload, saved_at FROM save_slots WHERE id = ?").get(slotId);
    return row ? deserializeSnapshot(row.payload) : null;
  }

  async saveSlot(slotId: string, snapshot: GameSnapshot): Promise<void> {
    const database = await this.getDb();
    const stmt = database.prepare(`
      INSERT INTO save_slots (id, payload, saved_at)
      VALUES (?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, saved_at = excluded.saved_at
    `);
    stmt.run(slotId, serializeSnapshot(snapshot), new Date().toISOString());
  }

  async deleteSlot(slotId: string): Promise<void> {
    const database = await this.getDb();
    database.prepare("DELETE FROM save_slots WHERE id = ?").run(slotId);
  }
}
