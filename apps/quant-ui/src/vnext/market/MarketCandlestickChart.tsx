import { useCallback, useEffect, useMemo, useState, type KeyboardEvent } from "react";
import type { EChartsOption } from "echarts";
import type { EChartsType } from "echarts/core";
import type { KlineBar, Trade } from "../../api/types";
import { resolveKlineWindow, type KlineWindow } from "../../charting/kline";
import { EChart } from "../../components/EChart";
import { useVNextChartPalette } from "../theme";

interface MarketCandlestickChartProps {
  bars: KlineBar[];
  trades?: Trade[];
  symbol?: string;
  selectedTradeId?: string | null;
  onTradeSelect?: (tradeId: string) => void;
}

type RangeKey = "60D" | "120D" | "1Y" | "ALL";
const RANGE_SESSIONS: Record<RangeKey, number | null> = { "60D": 60, "120D": 120, "1Y": 250, ALL: null };
/** Keyboard pan steps, in bars: human-scale, not one bar per press. */
const PAN_STEP = 5;
const PAGE_STEP = 20;
const MIN_WINDOW = 20;

/**
 * The window the user dragged, wheeled or keyed to, as bar indices. It is
 * React state (not just ECharts' internal state) because `EChart` re-applies
 * the option with `notMerge` on every option change — a theme switch, a trade
 * marker click or a refetch — and a window that lived only inside ECharts was
 * reset to the preset range each time (AGENTS.md K 线人类操作契约).
 */
export function windowFromZoom(
  zoom: { start?: number; end?: number; startValue?: string | number; endValue?: string | number } | undefined,
  dates: string[],
): KlineWindow | null {
  if (!zoom || !dates.length) return null;
  const toIndex = (value: string | number | undefined, percent: number | undefined, fallback: number): number => {
    if (typeof value === "string") {
      const found = dates.indexOf(value);
      if (found >= 0) return found;
    }
    if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value < dates.length) return value;
    if (typeof percent === "number") return Math.round((percent / 100) * Math.max(0, dates.length - 1));
    return fallback;
  };
  const startIndex = toIndex(zoom.startValue, zoom.start, 0);
  const endIndex = toIndex(zoom.endValue, zoom.end, dates.length - 1);
  return { startIndex: Math.min(startIndex, endIndex), endIndex: Math.max(startIndex, endIndex) };
}

function movingAverage(bars: KlineBar[], period: number): Array<number | null> {
  const result: Array<number | null> = Array(bars.length).fill(null);
  let sum = 0;
  for (let index = 0; index < bars.length; index += 1) {
    sum += bars[index].close;
    if (index >= period) sum -= bars[index - period].close;
    if (index >= period - 1) result[index] = sum / period;
  }
  return result;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function MarketCandlestickChart({
  bars,
  trades = [],
  symbol,
  selectedTradeId,
  onTradeSelect,
}: MarketCandlestickChartProps): JSX.Element {
  const palette = useVNextChartPalette();
  const [range, setRange] = useState<RangeKey>("1Y");
  const [manualWindow, setManualWindow] = useState<KlineWindow | null>(null);
  const dates = useMemo(() => bars.map((bar) => bar.datetime.slice(0, 10)), [bars]);
  const seriesIdentity = `${symbol ?? bars[0]?.symbol ?? ""}|${dates[0] ?? ""}`;
  // A different instrument (or history) invalidates the manual window; a
  // refetch of the same series, a theme change or a marker click does not.
  useEffect(() => {
    setManualWindow(null);
  }, [seriesIdentity]);
  const viewWindow = useMemo<KlineWindow>(() => {
    const preset = resolveKlineWindow(dates.length, range);
    if (!manualWindow || !dates.length) return preset;
    const endIndex = Math.min(dates.length - 1, manualWindow.endIndex);
    const startIndex = Math.min(endIndex, Math.max(0, manualWindow.startIndex));
    return { startIndex, endIndex };
  }, [dates.length, manualWindow, range]);
  const startValue = dates[viewWindow.startIndex];
  const endValue = dates[viewWindow.endIndex];

  const option = useMemo<EChartsOption>(() => {
    const ma20 = movingAverage(bars, 20);
    const ma60 = movingAverage(bars, 60);
    const ma120 = movingAverage(bars, 120);
    const markData = trades
      .filter((trade) => dates.includes(trade.datetime.slice(0, 10)) && Number.isFinite(trade.price))
      .map((trade) => {
        const isBuy = trade.action.includes("BUY");
        const isSelected = trade.id === selectedTradeId;
        return {
          id: trade.id,
          name: trade.action,
          value: trade.action,
          coord: [trade.datetime.slice(0, 10), trade.price],
          symbol: isBuy ? "arrow" : "pin",
          symbolRotate: isBuy ? 0 : 180,
          symbolSize: isSelected ? 20 : 14,
          itemStyle: { color: isBuy ? palette.positive : palette.negative, borderColor: palette.tooltipText, borderWidth: isSelected ? 2 : 0 },
          label: { show: isSelected, formatter: trade.action, color: palette.tooltipText, fontSize: 9 },
        };
      });

    return {
      animation: false,
      legend: { top: 2, left: 8, data: ["MA20", "MA60", "MA120"] },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        formatter: (params: unknown) => {
          const list = Array.isArray(params) ? params as Array<Record<string, unknown>> : [];
          const first = list[0];
          const dataIndex = typeof first?.dataIndex === "number" ? first.dataIndex : -1;
          const bar = bars[dataIndex];
          if (!bar) return "";
          const previous = dataIndex > 0 ? bars[dataIndex - 1].close : null;
          const change = previous && previous !== 0 ? bar.close / previous - 1 : null;
          return [
            `<strong>${bar.datetime.slice(0, 10)} · ${symbol ?? bar.symbol}</strong>`,
            `开 ${formatNumber(bar.open)}　高 ${formatNumber(bar.high)}`,
            `低 ${formatNumber(bar.low)}　收 ${formatNumber(bar.close)}`,
            `涨跌 ${change === null ? "—" : `${change >= 0 ? "+" : ""}${(change * 100).toFixed(2)}%`}`,
            `成交量 ${typeof bar.volume === "number" ? bar.volume.toLocaleString("zh-CN") : "—"}`,
            `成交额 ${typeof bar.amount === "number" ? bar.amount.toLocaleString("zh-CN", { notation: "compact", maximumFractionDigits: 2 }) : "—"}`,
          ].join("<br/>");
        },
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: [
        { left: 58, right: 22, top: 34, height: "61%" },
        { left: 58, right: 22, top: "72%", height: "15%" },
      ],
      xAxis: [
        { type: "category", data: dates, boundaryGap: false, axisLine: { onZero: false }, splitLine: { show: false }, min: "dataMin", max: "dataMax" },
        { type: "category", gridIndex: 1, data: dates, boundaryGap: false, axisLine: { onZero: false }, axisLabel: { show: false }, splitLine: { show: false }, min: "dataMin", max: "dataMax" },
      ],
      yAxis: [
        { scale: true, splitArea: { show: false }, position: "right" },
        { scale: true, gridIndex: 1, splitNumber: 2, position: "right", axisLabel: { formatter: (value: number) => new Intl.NumberFormat("zh-CN", { notation: "compact", maximumFractionDigits: 1 }).format(value) } },
      ],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1], startValue, endValue, zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false },
        { type: "slider", xAxisIndex: [0, 1], bottom: 2, height: 18, startValue, endValue, showDetail: false },
      ],
      series: [
        {
          name: "K线",
          type: "candlestick",
          data: bars.map((bar) => [bar.open, bar.close, bar.low, bar.high]),
          itemStyle: { color: palette.positive, color0: palette.negative, borderColor: palette.positive, borderColor0: palette.negative },
          markPoint: markData.length ? { data: markData, silent: false } : undefined,
        },
        { name: "MA20", type: "line", data: ma20, smooth: false, showSymbol: false, lineStyle: { width: 1.2, color: palette.series[0] }, connectNulls: false },
        { name: "MA60", type: "line", data: ma60, smooth: false, showSymbol: false, lineStyle: { width: 1.2, color: palette.series[2] }, connectNulls: false },
        { name: "MA120", type: "line", data: ma120, smooth: false, showSymbol: false, lineStyle: { width: 1.4, color: palette.series[3] }, connectNulls: false },
        {
          name: "成交量",
          type: "bar",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: bars.map((bar) => ({ value: bar.volume ?? 0, itemStyle: { color: bar.close >= bar.open ? palette.positive : palette.negative, opacity: 0.72 } })),
        },
      ],
    };
  }, [bars, dates, endValue, palette, selectedTradeId, startValue, symbol, trades]);

  const handleChartClick = (params: unknown): void => {
    const payload = params as { componentType?: string; componentSubType?: string; data?: { id?: string } };
    const tradeId = payload.data?.id;
    if (payload.componentType === "markPoint" && tradeId) onTradeSelect?.(tradeId);
  };

  const handleDataZoom = useCallback((_params: unknown, chart: EChartsType): void => {
    const current = chart.getOption() as { dataZoom?: Array<{ start?: number; end?: number; startValue?: string | number; endValue?: string | number }> };
    const next = windowFromZoom(current.dataZoom?.[0], dates);
    if (!next) return;
    setManualWindow((previous) => (previous?.startIndex === next.startIndex && previous.endIndex === next.endIndex ? previous : next));
  }, [dates]);

  const pan = (delta: number): void => {
    if (!dates.length) return;
    const width = viewWindow.endIndex - viewWindow.startIndex;
    const startIndex = Math.min(Math.max(0, dates.length - 1 - width), Math.max(0, viewWindow.startIndex + delta));
    setManualWindow({ startIndex, endIndex: Math.min(dates.length - 1, startIndex + width) });
  };

  const zoom = (direction: -1 | 1): void => {
    if (!dates.length) return;
    const size = viewWindow.endIndex - viewWindow.startIndex + 1;
    const nextSize = Math.min(dates.length, Math.max(Math.min(MIN_WINDOW, dates.length), Math.round(size * (direction < 0 ? 0.78 : 1.28))));
    const center = (viewWindow.startIndex + viewWindow.endIndex) / 2;
    let startIndex = Math.max(0, Math.round(center - nextSize / 2));
    const endIndex = Math.min(dates.length - 1, startIndex + nextSize - 1);
    startIndex = Math.max(0, endIndex - nextSize + 1);
    setManualWindow({ startIndex, endIndex });
  };

  const selectRange = (key: RangeKey): void => {
    setRange(key);
    setManualWindow(null);
  };

  const handleKeyboard = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (!dates.length) return;
    const width = viewWindow.endIndex - viewWindow.startIndex;
    const actions: Record<string, () => void> = {
      ArrowLeft: () => pan(-PAN_STEP),
      ArrowRight: () => pan(PAN_STEP),
      PageUp: () => pan(-PAGE_STEP),
      PageDown: () => pan(PAGE_STEP),
      "+": () => zoom(-1),
      "=": () => zoom(-1),
      ArrowUp: () => zoom(-1),
      "-": () => zoom(1),
      _: () => zoom(1),
      ArrowDown: () => zoom(1),
      // Latest keeps the current width and moves it to the newest bar.
      End: () => setManualWindow({ startIndex: Math.max(0, dates.length - 1 - width), endIndex: dates.length - 1 }),
      Home: () => selectRange("ALL"),
    };
    const action = actions[event.key];
    if (!action) return;
    event.preventDefault();
    action();
  };

  return (
    <div className="market-kline-workstation">
      <div className="market-chart-range" aria-label="K线区间">
        {(Object.keys(RANGE_SESSIONS) as RangeKey[]).map((key) => (
          <button key={key} type="button" className={!manualWindow && range === key ? "active" : ""} aria-pressed={!manualWindow && range === key} onClick={() => selectRange(key)}>{key}</button>
        ))}
        {manualWindow ? <em className="market-chart-window" aria-live="polite">手动窗口 {startValue} → {endValue} · {viewWindow.endIndex - viewWindow.startIndex + 1} 根</em> : null}
        <span>滚轮只缩放 · 左键拖拽只平移 · ←/→ 5 根 · PgUp/PgDn 20 根 · +/− 缩放 · End 最新 · Home 全部</span>
      </div>
      <EChart
        option={option}
        className="market-kline-chart"
        ariaLabel={`${symbol ?? bars[0]?.symbol ?? "股票"} K 线、成交量与 MA20/60/120`}
        interactive
        onClick={handleChartClick}
        onDataZoom={handleDataZoom}
        onKeyDown={handleKeyboard}
      />
    </div>
  );
}
