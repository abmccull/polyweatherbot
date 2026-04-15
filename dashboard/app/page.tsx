type Summary = {
  metrics?: Record<string, unknown>;
  counts?: Record<string, number>;
};

type Leader = {
  wallet: string;
  name: string;
  tier: string;
  eligibility?: { status?: string; eligible?: boolean; reason?: string };
  metrics?: { trades_60d?: number; active_days_60d?: number; month_sports_pnl?: number | null };
};

type Lock = {
  match_key: string;
  side: string;
  outcome?: string;
  opened_by_wallet?: string;
  opened_at?: string;
};

type Redemption = {
  condition_id: string;
  status: string;
  size: number;
  created_at?: string;
};

async function api<T>(path: string): Promise<T> {
  const base = process.env.API_BASE_URL ?? "http://localhost:8000";
  const token = process.env.API_BEARER_TOKEN ?? "";
  const resp = await fetch(`${base}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    cache: "no-store"
  });
  if (!resp.ok) throw new Error(`API ${path} failed with ${resp.status}`);
  return (await resp.json()) as T;
}

export default async function Page() {
  let summary: Summary = {};
  let leaders: Leader[] = [];
  let locks: Lock[] = [];
  let redemptions: Redemption[] = [];
  let err: string | null = null;

  try {
    [summary, leaders, locks, redemptions] = await Promise.all([
      api<Summary>("/api/v1/summary"),
      api<Leader[]>("/api/v1/leaders"),
      api<Lock[]>("/api/v1/copy/locks"),
      api<Redemption[]>("/api/v1/redemptions")
    ]);
  } catch (e) {
    err = e instanceof Error ? e.message : "Unknown API error";
  }

  const counts = summary.counts ?? {};
  const mode = String((summary.metrics ?? {})["mode"] ?? "unknown");
  const balances = ((summary.metrics ?? {})["balances"] as Record<string, unknown> | undefined) ?? {};
  const redeemMode = String(balances["redeem_mode"] ?? "unknown");
  const redeemReady = Boolean(balances["redeem_automation_ready"] ?? false);
  const tradableUsdc = Number(balances["tradable_usdc"] ?? 0);
  const redeemableValue = Number(balances["redeemable_value"] ?? 0);

  return (
    <main className="wrap">
      <h1 className="title">Station Sniper Copycat</h1>
      <p className="sub">Vercel monitor for live sports copy-trading, locks, and redemption flow.</p>

      {err ? <p className="empty">API error: {err}</p> : null}

      <section className="grid">
        <article className="panel">
          <div className="k">Runtime Mode</div>
          <div className="v">{mode}</div>
          <span className="pill">Live Monitor</span>
        </article>
        <article className="panel">
          <div className="k">Redemption Mode</div>
          <div className="v">{redeemMode}</div>
          <span className="pill">{redeemReady ? "Auto-Ready" : "Manual"}</span>
        </article>
        <article className="panel">
          <div className="k">Open Match Locks</div>
          <div className="v">{counts["open_locks"] ?? 0}</div>
        </article>
        <article className="panel">
          <div className="k">Tradable USDC</div>
          <div className="v">{tradableUsdc.toFixed(2)}</div>
        </article>
        <article className="panel">
          <div className="k">Redeemable Value</div>
          <div className="v">{redeemableValue.toFixed(2)}</div>
        </article>
        <article className="panel">
          <div className="k">Signals Accepted</div>
          <div className="v">{counts["signals_accepted"] ?? 0}</div>
        </article>
        <article className="panel">
          <div className="k">Signals Skipped</div>
          <div className="v">{counts["signals_skipped"] ?? 0}</div>
        </article>
        <article className="panel">
          <div className="k">Order Intents</div>
          <div className="v">{counts["order_intents"] ?? 0}</div>
        </article>
        <article className="panel">
          <div className="k">Redemption Events</div>
          <div className="v">{counts["redemption_events"] ?? 0}</div>
        </article>
      </section>

      <section className="section panel">
        <h2>Leaders</h2>
        {leaders.length === 0 ? (
          <p className="empty">No leader rows yet.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Tier</th>
                <th>Trades 60d</th>
                <th>Active Days 60d</th>
                <th>Month Sports PnL</th>
                <th>Wallet</th>
              </tr>
            </thead>
            <tbody>
              {leaders.map((l) => (
                <tr key={l.wallet}>
                  <td>{l.name}</td>
                  <td>{l.eligibility?.status ?? "-"}</td>
                  <td>{l.tier ?? "-"}</td>
                  <td>{l.metrics?.trades_60d ?? "-"}</td>
                  <td>{l.metrics?.active_days_60d ?? "-"}</td>
                  <td>{l.metrics?.month_sports_pnl ?? "-"}</td>
                  <td>{l.wallet.slice(0, 10)}...</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="section panel">
        <h2>Open Locks</h2>
        {locks.filter((l) => l).length === 0 ? (
          <p className="empty">No locks available.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Match Key</th>
                <th>Side</th>
                <th>Outcome</th>
                <th>Wallet</th>
                <th>Opened</th>
              </tr>
            </thead>
            <tbody>
              {locks.slice(0, 20).map((l) => (
                <tr key={l.match_key}>
                  <td>{l.match_key.slice(0, 24)}...</td>
                  <td>{l.side}</td>
                  <td>{l.outcome ?? "-"}</td>
                  <td>{(l.opened_by_wallet ?? "-").slice(0, 10)}...</td>
                  <td>{l.opened_at ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="section panel">
        <h2>Recent Redemptions</h2>
        {redemptions.length === 0 ? (
          <p className="empty">No redemption events yet.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Condition</th>
                <th>Status</th>
                <th>Size</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {redemptions.slice(0, 20).map((r, i) => (
                <tr key={`${r.condition_id}-${i}`}>
                  <td>{r.condition_id.slice(0, 18)}...</td>
                  <td>{r.status}</td>
                  <td>{r.size}</td>
                  <td>{r.created_at ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
