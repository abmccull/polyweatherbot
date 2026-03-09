import { describe, expect, it } from "vitest";

import { buildProphecyDepthSummary, buildProphecyInterpretation, buildProphecyScaffold, selectSacredRecordEntries } from "../src/selectors";
import { createInitialState } from "../src/state/initialState";
import type { OmenReport, ProphecyRecord, TileSemantics, WordTile } from "../src/state/gameState";

const semantics: TileSemantics = {
  target: "alliance",
  action: "endure",
  polarity: "double",
  ambiguity: "balanced",
  timeHorizon: "seasonal",
  domain: "spiritual"
};

const tiles: WordTile[] = [
  { id: "subject", text: "the league", category: "subject", semantics },
  { id: "action", text: "shall endure", category: "action", semantics },
  { id: "condition", text: "if the altars remain pure", category: "condition", semantics: { ...semantics, ambiguity: "cryptic" } },
  { id: "seal", text: "thus speaks the laurel smoke", category: "seal", semantics }
];

const omenReports: OmenReport[] = [
  { id: "omen-1", sourceRole: "Laurel Reader", text: "Smoke rose in a single column.", semantics, reliability: 76 },
  { id: "omen-2", sourceRole: "Spring Keeper", text: "The basin remained unbroken.", semantics, reliability: 71 }
];

describe("Prophecy depth foundations", () => {
  it("builds a compact scaffold with spine, hinge, and seal", () => {
    const scaffold = buildProphecyScaffold(tiles);

    expect(scaffold.map((part) => part.kind)).toEqual(["spine", "hinge", "seal"]);
    expect(scaffold[0]?.state).toBe("stable");
    expect(scaffold[1]?.text).toContain("altars remain pure");
    expect(scaffold[2]?.state).toBe("charged");
  });

  it("raises depth when the reading has omen support and full structure", () => {
    const state = createInitialState();
    const deep = buildProphecyDepthSummary({
      tiles,
      omenReports,
      score: { clarity: 82, value: 86, risk: 34 },
      pythia: state.pythia
    });
    const shallow = buildProphecyDepthSummary({
      tiles: tiles.slice(0, 2),
      omenReports: omenReports.map((omen) => ({ ...omen, reliability: 49 })),
      score: { clarity: 48, value: 50, risk: 68 },
      pythia: {
        ...state.pythia,
        tranceDepth: 34,
        mentalClarity: 44,
        needs: {
          ...state.pythia.needs,
          rest: 72,
          purification: 69
        }
      }
    });

    expect(deep.depth).toBeGreaterThan(shallow.depth);
    expect(deep.depthBand).not.toBe("shallow");
    expect(shallow.depthBand).toBe("shallow");
  });

  it("derives Sacred Record entries with graceful rival context fallback", () => {
    const base = createInitialState();
    const record: ProphecyRecord = {
      id: "prophecy-record",
      factionId: "athens",
      dayIssued: 12,
      text: "the league shall endure if the altars remain pure thus speaks the laurel smoke",
      tileIds: tiles.map((tile) => tile.id),
      semantics: tiles.map((tile) => tile.semantics),
      clarity: 82,
      value: 86,
      risk: 34,
      depth: 79,
      depthBand: "deep",
      omenReliability: 73.5,
      omenConsensus: "aligned",
      scaffold: buildProphecyScaffold(tiles),
      interpretation: buildProphecyInterpretation(base, {
        factionId: "athens",
        semantics: tiles.map((tile) => tile.semantics),
        depth: 79,
        depthBand: "deep",
        risk: 34,
        value: 86,
        dayIssued: 12,
        dueDay: 32
      }),
      dueDay: 32,
      resolved: false
    };
    const state = {
      ...base,
      factions: {
        ...base.factions,
        athens: {
          ...base.factions.athens,
          relations: {}
        }
      },
      consultation: {
        ...base.consultation,
        history: [record]
      }
    };

    const entries = selectSacredRecordEntries(state);

    expect(entries[0]?.depthBand).toBe("deep");
    expect(entries[0]?.status).toBe("awaiting");
    expect(entries[0]?.interpretation.rivalContext ?? "fallback").toBeTruthy();
  });
});
