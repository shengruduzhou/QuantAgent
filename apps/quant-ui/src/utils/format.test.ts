import { describe, expect, it } from "vitest";
import { UNMEASURED_TITLE, formatNumber, formatPercent, toneClass, valueTone } from "./format";

describe("missing measurements are never rendered as zero", () => {
  it("formats absent numbers as 暂无, not 0", () => {
    expect(formatNumber(null)).toBe("暂无");
    expect(formatNumber(undefined)).toBe("暂无");
    expect(formatNumber(Number.NaN)).toBe("暂无");
    expect(formatPercent(null)).toBe("暂无");
    expect(formatPercent(undefined)).toBe("暂无");
    // A real zero is a measurement and must still print.
    expect(formatPercent(0)).toBe("0.00%");
  });
});

describe("toneClass", () => {
  // The bug this pins: pages inlined `(x ?? 0) >= 0 ? "tone-positive" : ...`,
  // which routed every absent value through `0 >= 0` and painted it with
  // `--market-up`. The cell read "暂无" while its colour claimed a gain.
  it("gives an absent value no direction", () => {
    expect(toneClass(null)).toBe("tone-unmeasured");
    expect(toneClass(undefined)).toBe("tone-unmeasured");
    expect(toneClass(Number.NaN)).toBe("tone-unmeasured");
  });

  it("keeps A-share direction semantics for measured values", () => {
    expect(toneClass(0.01)).toBe("tone-positive");
    expect(toneClass(0)).toBe("tone-positive");
    expect(toneClass(-0.01)).toBe("tone-negative");
  });

  it("never returns the old fail-open pair for a missing value", () => {
    for (const missing of [null, undefined, Number.NaN]) {
      expect(toneClass(missing)).not.toBe("tone-positive");
      expect(toneClass(missing)).not.toBe("tone-negative");
    }
  });

  it("ships an explanation string so colour is not the only channel", () => {
    expect(UNMEASURED_TITLE).toContain("未测量");
  });
});

describe("valueTone", () => {
  it("treats absent as neutral", () => {
    expect(valueTone(null)).toBe("neutral");
    expect(valueTone(undefined)).toBe("neutral");
  });
});
