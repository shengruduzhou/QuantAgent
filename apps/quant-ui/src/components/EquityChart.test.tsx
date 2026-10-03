import { cleanup, render } from "@testing-library/react";
import type { EChartsOption } from "echarts";
import { afterEach, expect, test, vi } from "vitest";
import { VNextThemeContext } from "../vnext/theme";
import { EquityChart, formatDrawdownAxis } from "./EquityChart";

let latest: EChartsOption = {};
vi.mock("./EChart", () => ({ EChart: ({ option }: { option: EChartsOption }) => { latest = option; return <div />; } }));
afterEach(() => cleanup());

const points = Array.from({ length: 40 }, (_, index) => ({ datetime: `2026-01-${String(index + 1).padStart(2, "0")}`, nav: 1_000_000 + index, drawdown: -index / 600 }));

test("drawdown axis: percent labels, two ticks, enough height not to overlap", () => {
  render(<EquityChart points={points} />);
  const yAxes = latest.yAxis as Array<Record<string, any>>;
  const drawdown = yAxes[1];
  expect(drawdown.splitNumber).toBe(2);
  expect(drawdown.max).toBe(0);
  expect(drawdown.axisLabel.formatter(-0.0609)).toBe("-6.1%");
  expect(drawdown.axisLabel.formatter(-0.2)).toBe("-20%");
  expect(drawdown.axisLabel.hideOverlap).toBe(true);
  const grids = latest.grid as Array<Record<string, string>>;
  expect(parseFloat(grids[1].height)).toBeGreaterThanOrEqual(20);
  expect(formatDrawdownAxis(0)).toBe("0%");
});

test("series colours come from the theme palette in every theme (no dark-only literals)", () => {
  for (const theme of ["night", "dawn", "day"] as const) {
    render(<VNextThemeContext.Provider value={theme}><EquityChart points={points} /></VNextThemeContext.Provider>);
    const text = JSON.stringify(latest);
    expect(text).not.toMatch(/#2f83ff|#f05a5a|#6f8798|#4c8dff|#75a9ff|rgba\((47|76|240),/i);
    cleanup();
  }
});
