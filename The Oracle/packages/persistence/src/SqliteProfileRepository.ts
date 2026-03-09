import { createEmptyProfileSnapshot, deserializeProfileSnapshot, serializeProfileSnapshot } from "./profileSchema";
import type { OracleProfileSnapshot, ProfileRepository } from "./ProfileRepository";

type BetterSqlite3Database = {
  exec(sql: string): void;
  prepare(sql: string): {
    run(...args: unknown[]): unknown;
    get(...args: unknown[]): { id: string; payload: string } | undefined;
  };
};

const PROFILE_KEY = "main";

export class SqliteProfileRepository implements ProfileRepository {
  private database?: BetterSqlite3Database;

  constructor(private readonly dbPath: string) {}

  private async getDb(): Promise<BetterSqlite3Database> {
    if (this.database) {
      return this.database;
    }

    const mod = (await import("better-sqlite3")) as unknown as { default: new (path: string) => BetterSqlite3Database };
    this.database = new mod.default(this.dbPath);
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS profile_snapshots (
        id TEXT PRIMARY KEY,
        payload TEXT NOT NULL
      );
    `);
    return this.database;
  }

  async loadProfile(): Promise<OracleProfileSnapshot> {
    const database = await this.getDb();
    const row = database.prepare("SELECT id, payload FROM profile_snapshots WHERE id = ?").get(PROFILE_KEY);
    return row ? deserializeProfileSnapshot(row.payload) : createEmptyProfileSnapshot();
  }

  async saveProfile(snapshot: OracleProfileSnapshot): Promise<void> {
    const database = await this.getDb();
    database.prepare(`
      INSERT INTO profile_snapshots (id, payload)
      VALUES (?, ?)
      ON CONFLICT(id) DO UPDATE SET payload = excluded.payload
    `).run(PROFILE_KEY, serializeProfileSnapshot(snapshot));
  }

  async resetProfile(): Promise<void> {
    const database = await this.getDb();
    database.prepare("DELETE FROM profile_snapshots WHERE id = ?").run(PROFILE_KEY);
  }
}
