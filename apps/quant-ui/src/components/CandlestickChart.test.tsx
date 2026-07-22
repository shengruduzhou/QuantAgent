import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { EChartsOption } from "echarts";
import type { EChartsType } from "echarts/core";
import type { KeyboardEventHandler } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import type { KlineBar, Trade } from "../api/types";
import { CandlestickChart } from "./CandlestickChart";

let latestOption: EChartsOption = {};
let latestDataZoom: ((params: unknown, chart: EChartsType) => void) | undefined;

vi.mock("./EChart", () => ({
  EChart: ({
    option,
    ariaLabel,
    onKeyDown,
    onDataZoom,
  }: {
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

function createBars(count: number): KlineBar[] {
  const start = new Date("2025-01-01T00:00:00Z");
  return Array.from({ length: count }, (_, index) => {
    const date = new Date(start);
    date.setUTCDate(start.getUTCDate() + index);
    const close = 10 + index / 100;
    return {
      datetime: date.toISOString(),
      symbol: "000001.SZ",
      open: close - 0.05,
      high: close + 0.1,
      low: close - 0.1,
      close,
      volume: 1_000_000 + index,
    };
  });
}

function getZoom(): {
  startValue?: string;
  endValue?: string;
  zoomOnMouseWheel?: boolean;
  moveOnMouseMove?: boolean;
  moveOnMouseWheel?: boolean;
} {
  const zoom = Array.isArray(latestOption.dataZoom) ? latestOption.dataZoom[0] : latestOption.dataZoom;
  return (zoom ?? {}) as {
    startValue?: string;
    endValue?: string;
    zoomOnMouseWheel?: boolean;
    moveOnMouseMove?: boolean;
    moveOnMouseWheel?: boolean;
  };
}

afterEach(() => cleanup());

describe("CandlestickChart workstation interaction", () => {
  test("moves to latest and expands to all history", async () => {
    const bars = createBars(300);
    const trade: Trade = {
      id: "trade-1",
      datetime: bars[100].datetime,
      symbol: "000001.SZ",
      action: "BUY",
      price: bars[100].close,
      quantity: 100,
    };

    render(<CandlestickChart bars={bars} trades={[trade]} selectedTradeId={trade.id} />);

    const chart = screen.getByRole("application", { name: /000001.SZ K 线/ });
    const lastDate = bars.at(-1)?.datetime.slice(0, 10);
    const firstDate = bars[0].datetime.slice(0, 10);

    fireEvent.keyDown(chart, { key: "End" });
    await waitFor(() => expect(getZoom().endValue).toBe(lastDate));

    fireEvent.keyDown(chart, { key: "Home" });
    await waitFor(() => {
      expect(getZoom().startValue).toBe(firstDate);
      expect(getZoom().endValue).toBe(lastDate);
    });
  });

  test("uses wheel for zoom and pointer drag for pan without wheel-pan conflict", () => {
    const bars = createBars(180);
    render(<CandlestickChart bars={bars} trades={[]} />);

    expect(getZoom()).toMatchObject({
      zoomOnMouseWheel: true,
      moveOnMouseMove: true,
      moveOnMouseWheel: false,
    });
    expect(screen.getByText(/滚轮只缩放/)).toBeInTheDocument();
    expect(screen.getByText(/左键拖拽只平移/)).toBeInTheDocument();
  });

  test("persists a pointer-dragged dataZoom window across React re-renders", async () => {
    const bars = createBars(180);
    render(<CandlestickChart bars={bars} trades={[]} />);

    act(() => latestDataZoom?.({}, {
      getOption: () => ({ dataZoom: [{ startValue: bars[20].datetime.slice(0, 10), endValue: bars[75].datetime.slice(0, 10) }] }),
    } as unknown as EChartsType));

    await waitFor(() => {
      expect(getZoom().startValue).toBe(bars[20].datetime.slice(0, 10));
      expect(getZoom().endValue).toBe(bars[75].datetime.slice(0, 10));
    });
    fireEvent.click(screen.getByRole("button", { name: "向右平移" }));
    await waitFor(() => expect(getZoom().startValue).toBe(bars[25].datetime.slice(0, 10)));
  });

  test("moves by human-scale keyboard steps instead of one bar", async () => {
    const bars = createBars(300);
    render(<CandlestickChart bars={bars} trades={[]} />);
    const chart = screen.getByRole("application", { name: /000001.SZ K 线/ });
    const before = getZoom().endValue;

    fireEvent.keyDown(chart, { key: "ArrowLeft" });
    await waitFor(() => expect(getZoom().endValue).not.toBe(before));

    const expected = bars[294].datetime.slice(0, 10);
    expect(getZoom().endValue).toBe(expected);
  });
});
