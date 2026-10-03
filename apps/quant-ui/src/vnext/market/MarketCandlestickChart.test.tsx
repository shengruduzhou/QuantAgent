import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { EChartsOption } from "echarts";
import type { EChartsType } from "echarts/core";
import type { KeyboardEventHandler } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import type { KlineBar, Trade } from "../../api/types";
import { VNextThemeContext } from "../theme";
import { MarketCandlestickChart } from "./MarketCandlestickChart";

/**
 * The K-line human-operation contract (AGENTS.md), tested on the component
 * the /stock-replay route actually renders. The previous contract tests
 * exercised components/CandlestickChart.tsx, which no route imported.
 */

let latestOption: EChartsOption = {};
let latestDataZoom: ((params: unknown, chart: EChartsType) => void) | undefined;

vi.mock("../../components/EChart", () => ({
  EChart: ({ option, ariaLabel, onKeyDown, onDataZoom }: {
    option: EChartsOption;
    ariaLabel?: string;
    onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
    onDataZoom?: (params: unknown, chart: EChartsType) => void;
  }) => {
    latestOption = option;
    latestDataZoom = onDataZoom;
    return <div role="application" aria-label={ariaLabel} tabIndex={0} onKeyDown={onKeyDown} />;
  },
}));

afterEach(() => cleanup());

function createBars(count: number, symbol = "000001.SZ"): KlineBar[] {
  const start = new Date("2025-01-01T00:00:00Z");
  return Array.from({ length: count }, (_, index) => {
    const date = new Date(start);
    date.setUTCDate(start.getUTCDate() + index);
    const close = 10 + index / 100;
    return { datetime: date.toISOString(), symbol, open: close - 0.05, high: close + 0.1, low: close - 0.1, close, volume: 1_000_000 + index };
  });
}

const day = (bars: KlineBar[], index: number): string => bars[index].datetime.slice(0, 10);

function zoomOf(): { startValue?: string; endValue?: string; zoomOnMouseWheel?: boolean; moveOnMouseMove?: boolean; moveOnMouseWheel?: boolean } {
  const zoom = Array.isArray(latestOption.dataZoom) ? latestOption.dataZoom[0] : latestOption.dataZoom;
  return (zoom ?? {}) as ReturnType<typeof zoomOf>;
}

function userZoomsTo(bars: KlineBar[], start: number, end: number): void {
  act(() => latestDataZoom?.({}, {
    getOption: () => ({ dataZoom: [{ startValue: day(bars, start), endValue: day(bars, end) }] }),
  } as unknown as EChartsType));
}

const trade = (bars: KlineBar[], index: number): Trade => ({ id: `t-${index}`, datetime: bars[index].datetime, symbol: "000001.SZ", action: "BUY", price: bars[index].close, quantity: 100 });

describe("MarketCandlestickChart human-operation contract", () => {
  test("wheel zooms and drag pans, without wheel-pan conflict", () => {
    render(<MarketCandlestickChart bars={createBars(180)} />);
    expect(zoomOf()).toMatchObject({ zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false });
    expect(screen.getByText(/滚轮只缩放 · 左键拖拽只平移/)).toBeInTheDocument();
  });

  test("keeps the user's zoom window across rerenders: trade selection, theme change, refetch", async () => {
    const bars = createBars(300);
    const trades = [trade(bars, 40), trade(bars, 60)];
    // The shell always provides the theme; only its value changes.
    const { rerender } = render(<VNextThemeContext.Provider value="night"><MarketCandlestickChart bars={bars} trades={trades} selectedTradeId={null} /></VNextThemeContext.Provider>);
    userZoomsTo(bars, 30, 90);
    await waitFor(() => expect(zoomOf()).toMatchObject({ startValue: day(bars, 30), endValue: day(bars, 90) }));
    // The preset range button is no longer shown as active once the window is manual.
    expect(screen.getByRole("button", { name: "1Y" })).toHaveAttribute("aria-pressed", "false");

    rerender(<VNextThemeContext.Provider value="night"><MarketCandlestickChart bars={bars} trades={trades} selectedTradeId="t-40" /></VNextThemeContext.Provider>);
    expect(zoomOf()).toMatchObject({ startValue: day(bars, 30), endValue: day(bars, 90) });

    rerender(<VNextThemeContext.Provider value="day"><MarketCandlestickChart bars={bars} trades={trades} selectedTradeId="t-40" /></VNextThemeContext.Provider>);
    expect(zoomOf()).toMatchObject({ startValue: day(bars, 30), endValue: day(bars, 90) });

    const refetched = bars.map((bar) => ({ ...bar }));
    rerender(<VNextThemeContext.Provider value="day"><MarketCandlestickChart bars={refetched} trades={trades} selectedTradeId="t-40" /></VNextThemeContext.Provider>);
    expect(zoomOf()).toMatchObject({ startValue: day(bars, 30), endValue: day(bars, 90) });
  });

  test("a different instrument resets the manual window to the preset range", async () => {
    const bars = createBars(300);
    const { rerender } = render(<MarketCandlestickChart bars={bars} symbol="000001.SZ" />);
    userZoomsTo(bars, 30, 90);
    await waitFor(() => expect(zoomOf().startValue).toBe(day(bars, 30)));
    const other = createBars(300, "600519.SH");
    rerender(<MarketCandlestickChart bars={other} symbol="600519.SH" />);
    await waitFor(() => expect(zoomOf().endValue).toBe(day(other, 299)));
    expect(zoomOf().startValue).toBe(day(other, 50));
  });

  test("keyboard: human-scale pan, zoom, latest and all", async () => {
    const bars = createBars(300);
    render(<MarketCandlestickChart bars={bars} />);
    const chart = screen.getByRole("application", { name: /000001.SZ K 线/ });
    // Preset 1Y = last 250 bars.
    expect(zoomOf()).toMatchObject({ startValue: day(bars, 50), endValue: day(bars, 299) });

    fireEvent.keyDown(chart, { key: "ArrowLeft" });
    await waitFor(() => expect(zoomOf()).toMatchObject({ startValue: day(bars, 45), endValue: day(bars, 294) }));
    fireEvent.keyDown(chart, { key: "PageUp" });
    await waitFor(() => expect(zoomOf()).toMatchObject({ startValue: day(bars, 25), endValue: day(bars, 274) }));

    fireEvent.keyDown(chart, { key: "+" });
    await waitFor(() => expect(zoomOf().startValue).not.toBe(day(bars, 25)));
    const zoomedIn = zoomOf();
    const width = bars.findIndex((bar) => bar.datetime.startsWith(zoomedIn.endValue as string)) - bars.findIndex((bar) => bar.datetime.startsWith(zoomedIn.startValue as string)) + 1;
    expect(width).toBe(195);

    fireEvent.keyDown(chart, { key: "End" });
    await waitFor(() => expect(zoomOf()).toMatchObject({ startValue: day(bars, 105), endValue: day(bars, 299) }));

    fireEvent.keyDown(chart, { key: "Home" });
    await waitFor(() => expect(zoomOf()).toMatchObject({ startValue: day(bars, 0), endValue: day(bars, 299) }));
    expect(screen.getByRole("button", { name: "ALL" })).toHaveAttribute("aria-pressed", "true");
  });

  test("a range button replaces the manual window", async () => {
    const bars = createBars(300);
    render(<MarketCandlestickChart bars={bars} />);
    userZoomsTo(bars, 10, 40);
    await waitFor(() => expect(zoomOf().startValue).toBe(day(bars, 10)));
    fireEvent.click(screen.getByRole("button", { name: "60D" }));
    await waitFor(() => expect(zoomOf()).toMatchObject({ startValue: day(bars, 240), endValue: day(bars, 299) }));
  });
});
