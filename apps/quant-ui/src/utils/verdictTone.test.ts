import { describe, expect, it } from "vitest";

import { isAffirmative, verdictTone } from "./verdictTone";

describe("verdictTone — negation must beat the affirmative substring", () => {
  // The two real API values that were rendering as success.
  it('treats DO_NOT_ENABLE as a refusal, not an ENABLE', () => {
    expect("DO_NOT_ENABLE".includes("ENABLE")).toBe(true); // the trap itself
    expect(isAffirmative("DO_NOT_ENABLE")).toBe(false);
    expect(verdictTone("DO_NOT_ENABLE")).toBe("danger");
  });

  it("treats U0 NOT_READY states as refusals, not ready", () => {
    expect("u0_bar_not_ready_coverage".includes("ready")).toBe(true); // the trap
    for (const status of [
      "U0_BAR_NOT_READY_COVERAGE",
      "U0_BAR_NOT_READY_IDENTITY",
      "U0_BAR_NOT_READY_PROVIDER",
      "U0_BAR_NOT_READY_QUALITY",
      "FULL_UNIVERSE_DATA_NOT_READY_INTEGRATION",
    ]) {
      expect(isAffirmative(status)).toBe(false);
      expect(verdictTone(status)).toBe("danger");
    }
  });

  it("still recognises genuine affirmatives", () => {
    expect(verdictTone("ENABLE")).toBe("success");
    expect(verdictTone("READY")).toBe("success");
    expect(verdictTone("passed")).toBe("success");
    expect(isAffirmative("ENABLE")).toBe(true);
  });

  it("ranks PAPER_ONLY as a warning, strictly less severe than DO_NOT_ENABLE", () => {
    expect(verdictTone("PAPER_ONLY")).toBe("warning");
    expect(verdictTone("DO_NOT_ENABLE")).toBe("danger");
    // The original bug had these exactly inverted.
    expect(isAffirmative("PAPER_ONLY")).toBe(false);
  });
});

describe("verdictTone — unknown never becomes success", () => {
  it.each([null, undefined, "", "   "])("maps %p to unknown", (value) => {
    expect(verdictTone(value as string | null | undefined)).toBe("unknown");
    expect(isAffirmative(value as string | null | undefined)).toBe(false);
  });

  it.each(["UNKNOWN", "unverified", "NOT_EVALUATED", "pending"])(
    "keeps %s out of success",
    (value) => {
      expect(verdictTone(value)).toBe("unknown");
    },
  );

  it("does not guess at an unrecognised string", () => {
    expect(verdictTone("SOME_NEW_BACKEND_STATE")).toBe("unknown");
  });
});

describe("verdictTone — explicit failure outranks everything", () => {
  it.each(["ERROR", "pipeline_failed", "gate_violation", "FATAL"])(
    "maps %s to danger",
    (value) => {
      expect(verdictTone(value)).toBe("danger");
    },
  );

  it("classifies blocked and disabled as refusals", () => {
    expect(verdictTone("BLOCKED")).toBe("warning");
    expect(verdictTone("LIVE_DISABLED")).toBe("warning");
    expect(isAffirmative("LIVE_DISABLED")).toBe(false);
  });
});
