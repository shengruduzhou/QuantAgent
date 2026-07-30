import { expect, test } from "vitest";
import { compactApiErrorPayload } from "./client";

test("compacts FastAPI validation issues without echoing request input", () => {
  const result = compactApiErrorPayload(JSON.stringify({
    detail: [{
      loc: ["body", "primaryHorizon"],
      msg: "Value error, primaryHorizon must be included in horizons",
      input: {
        marketPanelPath: "runtime/private/large-input.parquet",
        labelsPath: "must-never-render",
      },
    }],
  }), 422);

  expect(result.message).toBe(
    "primaryHorizon: primaryHorizon must be included in horizons",
  );
  expect(result.message).not.toContain("marketPanelPath");
  expect(result.message).not.toContain("must-never-render");
});

test("compacts governed launch errors into actionable lines", () => {
  const result = compactApiErrorPayload(JSON.stringify({
    detail: {
      message: "strategy validation and Human Gate are required",
      errors: ["labelsPath: missing forward_return_10d"],
      warnings: ["research only"],
    },
  }), 422);

  expect(result.issues).toEqual([
    "strategy validation and Human Gate are required",
    "labelsPath: missing forward_return_10d",
  ]);
});
