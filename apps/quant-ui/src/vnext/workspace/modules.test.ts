import { describe, expect, test } from "vitest";
import { moduleForVNextPath } from "./modules";

describe("VNext module routing", () => {
  test("prefers query-specific Runtime and Settings modules", () => {
    expect(moduleForVNextPath("/runtime?view=data").id).toBe("data");
    expect(moduleForVNextPath("/runtime?view=lineage&runId=run-42").id).toBe("pipeline");
    expect(moduleForVNextPath("/runtime?view=cleanup").id).toBe("runtime");
    expect(moduleForVNextPath("/runtime").id).toBe("runtime");
    expect(moduleForVNextPath("/settings?view=jobs").id).toBe("tasks");
    expect(moduleForVNextPath("/settings?job=train").id).toBe("settings");
  });

  test("keeps entity context parameters from changing the owning module", () => {
    expect(moduleForVNextPath("/models?modelId=model-v12-4").id).toBe("model");
    expect(moduleForVNextPath("/backtests?run=backtest-vnext").id).toBe("backtest");
  });
});
