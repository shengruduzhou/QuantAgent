import { describe, expect, it } from "vitest";
import { compoundNav } from "./MarketPlaybooksPage";

/**
 * DEF-022 reproduced client-side: the playbook NAV curve used to compound
 * `finite(r.netReturn) ?? 0`, so a missing return became a flat 0% day. Ten
 * days with five holes drew five flat segments and reported +0% for each,
 * exactly the failure the backend fix already closed.
 */
describe("compoundNav refuses to invent flat days", () => {
  it("compounds a fully measured series", () => {
    const series = compoundNav([
      { date: "2026-01-05", netReturn: 0.1 },
      { date: "2026-01-06", netReturn: 0.1 },
    ]);
    expect(series[0]).toBeCloseTo(1.1, 10);
    expect(series[1]).toBeCloseTo(1.21, 10);
  });

  it("stops the curve at the first missing return instead of drawing 0%", () => {
    const series = compoundNav([
      { date: "2026-01-05", netReturn: 0.1 },
      { date: "2026-01-06" },
      { date: "2026-01-07", netReturn: 0.1 },
    ]);
    expect(series[0]).toBeCloseTo(1.1, 10);
    // The old code produced 1.1 here (a fabricated flat day) and 1.21 after it.
    expect(series[1]).toBeNull();
    expect(series[2]).toBeNull();
  });

  it("treats non-finite returns as missing", () => {
    expect(compoundNav([{ netReturn: Number.NaN }])).toEqual([null]);
    expect(compoundNav([{ netReturn: "0.1" }])).toEqual([null]);
  });

  it("prefers an explicit nav column when the payload carries one", () => {
    expect(compoundNav([{ nav: 1.5 }, { nav: 1.6 }])).toEqual([1.5, 1.6]);
  });

  it("does not silently resume after a gap", () => {
    const series = compoundNav([{ netReturn: 0.1 }, {}, { nav: 2 }]);
    expect(series).toEqual([1.1, null, null]);
  });
});
