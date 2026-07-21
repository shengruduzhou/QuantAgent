import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import type { KlineBar, Trade } from "../api/types";
import {
  layoutTradeMarkers,
  movingAverage,
  resolveKlineWindow,
  type KlineViewRange,
  type TradeMarkerLayout,
} from "../charting/kline";
import { EChart } from "./EChart";

interface CandlestickChartProps {
  bars: KlineBar[];
  trades: Trade[];
  symbol?: string;
  selectedTradeId?: string | null;
  onTradeSelect?: (tradeId: string) => void;
}

type LayerKey = "ma5" | "ma10" | "ma20" | "ma60" | "trades" | "tTrades" | "risk";

interface TooltipParam {
  axisValue?: string | number;
  dataIndex?: number;
}

const RANGE_OPTIONS: Array<{ value: KlineViewRange; label: string }> = [
  { value: "60D", label: "60 日" },
  { value: "120D", label: "120 日" },
  { value: "1Y", label: "1 年" },
  { value: "ALL", label: "全部" },
];

const MA_OPTIONS: Array<{ key: LayerKey; label: string; period: number; color: string }> = [
  { key: "ma5", label: "MA5", period: 5, color: "#d7b95b" },
  { key: "ma10", label: "MA10", period: 10, color: "#72a7ff" },
  { key: "ma20", label: "MA20", period: 20, color: "#b28cff" },
  { key: "ma60", label: "MA60", period: 60, color: "#7f96a8" },
];

const DEFAULT_LAYERS: Record<LayerKey, boolean> = {
  ma5: true,
  ma10: true,
  ma20: true,
  ma60: false,
  trades: true,
  tTrades: true,
  risk: true,
};

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatValue(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function markerVisible(marker: TradeMarkerLayout, layers: Record<LayerKey, boolean>): boolean {
  if (marker.category === "risk") return layers.risk;
  if (marker.category === "t_trade") return layers.tTrades;
  return layers.trades;
}

function markerColor(marker: TradeMarkerLayout): string {
  if (marker.category === "risk") return "#e7a53a";
  if (marker.category === "t_trade") return marker.side === "buy" ? "#3f8cff" : "#b779ff";
  return marker.side === "buy" ? "#2ac89f" : "#ef5c63";
}

function markerLabel(marker: TradeMarkerLayout): string {
  if (marker.category === "risk") return "RISK";
  if (marker.category === "t_trade") return marker.side === "buy" ? "T·B" : "T·S";
  return marker.side === "buy" ? "B" : "S";
}

export function CandlestickChart({
  bars,
  trades,
  symbol,
  selectedTradeId,
  onTradeSelect,
}: CandlestickChartProps): JSX.Element {
  const [range, setRange] = useState<KlineViewRange>("120D");
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>(DEFAULT_LAYERS);

  const selectedTrade = trades.find((trade) => trade.id === selectedTradeId) ?? trades[0];
  const selectedDate = selectedTrade?.datetime.slice(0, 10);

  const derived = useMemo(() => {
    const dates = bars.map((bar) => bar.datetime.slice(0, 10));
    const dateSet = new Set(dates);
    const selectedIndex = selectedDate ? dates.indexOf(selectedDate) : -1;
    const window = resolveKlineWindow(dates.length, range, selectedIndex);
    const markerLayouts = layoutTradeMarkers(trades, dateSet).filter((marker) => markerVisible(marker, layers));
    const markersByDate = new Map<string, TradeMarkerLayout[]>();

    markerLayouts.forEach((marker) => {
      const bucket = markersByDate.get(marker.date) ?? [];
      bucket.push(marker);
      markersByDate.set(marker.date, bucket);
    });

    const tradePoints = markerLayouts.map((marker) => {
      const isSelected = marker.trade.id === selectedTradeId;
      const offsetDirection = marker.side === "buy" ? 1 : -1;
      const laneDistance = 14 + marker.lane * 13;
      return {
        name: marker.trade.action,
        value: marker.trade.price,
        coord: [marker.date, marker.trade.price],
        tradeId: marker.trade.id,
        symbol: marker.category === "risk" ? "diamond" : marker.side === "buy" ? "triangle" : "pin",
        symbolRotate: marker.side === "sell" && marker.category !== "risk" ? 180 : 0,
        symbolSize: isSelected ? 20 : marker.category === "risk" ? 14 : 13,
        symbolOffset: [0, offsetDirection * laneDistance],
        itemStyle: {
          color: markerColor(marker),
          borderColor: isSelected ? "#ffffff" : "#07131e",
          borderWidth: isSelected ? 2 : 1,
        },
        label: {
          show: isSelected || marker.lane < 2,
          formatter: markerLabel(marker),
          color: markerColor(marker),
          fontSize: isSelected ? 10 : 9,
          fontWeight: isSelected ? 700 : 500,
          position: marker.side === "buy" ? "bottom" : "top",
          distance: 5,
        },
      };
    });

    const maSeries = MA_OPTIONS
      .filter((item) => layers[item.key])
      .map((item) => ({
        name: item.label,
        type: "line" as const,
        data: movingAverage(bars, item.period),
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        smooth: false,
        connectNulls: false,
        silent: true,
        lineStyle: { width: 1, color: item.color, opacity: 0.9 },
        emphasis: { disabled: true },
      }));

    const tooltipFormatter = (params: unknown): string => {
      const items = (Array.isArray(params) ? params : [params]) as TooltipParam[];
      const date = String(items[0]?.axisValue ?? "");
      const index = dates.indexOf(date);
      const bar = index >= 0 ? bars[index] : undefined;
      if (!bar) return escapeHtml(date);

      const dayTrades = markersByDate.get(date) ?? [];
      const change = bar.open ? ((bar.close / bar.open) - 1) * 100 : 0;
      const tradeRows = dayTrades.length
        ? dayTrades.map((marker) => (
          `<div class="qa-tooltip-trade"><span style="color:${markerColor(marker)}">${escapeHtml(markerLabel(marker))}</span>`
          + `<b>${escapeHtml(marker.trade.action)}</b><em>${formatValue(marker.trade.price)}</em></div>`
        )).join("")
        : '<div class="qa-tooltip-muted">当日无成交、做 T 或风控事件</div>';

      return [
        `<div class="qa-tooltip-title">${escapeHtml(symbol ?? bar.symbol)} · ${escapeHtml(date)}</div>`,
        '<div class="qa-tooltip-grid">',
        `<span>开</span><b>${formatValue(bar.open)}</b><span>高</span><b>${formatValue(bar.high)}</b>`,
        `<span>低</span><b>${formatValue(bar.low)}</b><span>收</span><b>${formatValue(bar.close)}</b>`,
        `<span>涨跌</span><b class="${change >= 0 ? "positive" : "negative"}">${change >= 0 ? "+" : ""}${formatValue(change)}%</b>`,
        `<span>成交量</span><b>${formatValue(bar.volume, 0)}</b>`,
        "</div>",
        `<div class="qa-tooltip-events">${tradeRows}</div>`,
      ].join("");
    };

    const option: EChartsOption = {
      animation: false,
      backgroundColor: "transparent",
      legend: { show: false },
      tooltip: {
        trigger: "axis",
        triggerOn: "mousemove|click",
        axisPointer: { type: "cross", snap: true },
        confine: true,
        enterable: true,
        appendToBody: false,
        backgroundColor: "rgba(7, 19, 30, .97)",
        borderColor: "#29475f",
        borderWidth: 1,
        padding: 10,
        textStyle: { color: "#d8e5ef", fontSize: 11 },
        formatter: tooltipFormatter,
      },
      axisPointer: {
        link: [{ xAxisIndex: "all" }],
        label: { backgroundColor: "#24435d", color: "#e8f2f8", fontSize: 10 },
      },
      grid: [
        { left: 58, right: 58, top: 24, height: "62%" },
        { left: 58, right: 58, top: "75%", height: "16%" },
      ],
      xAxis: [
        {
          type: "category",
          data: dates,
          boundaryGap: true,
          axisLabel: { show: false },
          axisTick: { show: false },
          axisLine: { lineStyle: { color: "#20364a" } },
          splitLine: { show: false },
        },
        {
          type: "category",
          gridIndex: 1,
          data: dates,
          boundaryGap: true,
          axisLabel: { color: "#71879a", fontSize: 10, hideOverlap: true },
          axisTick: { show: false },
          axisLine: { lineStyle: { color: "#20364a" } },
          splitLine: { show: false },
        },
      ],
      yAxis: [
        {
          scale: true,
          position: "right",
          axisLabel: { color: "#71879a", fontSize: 10, formatter: (value: number) => value.toFixed(2) },
          axisLine: { show: true, lineStyle: { color: "#20364a" } },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: "#13283a", type: "dashed" } },
        },
        {
          scale: true,
          gridIndex: 1,
          position: "right",
          axisLabel: { color: "#71879a", fontSize: 9 },
          axisLine: { show: true, lineStyle: { color: "#20364a" } },
          axisTick: { show: false },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1],
          startValue: dates[window.startIndex],
          endValue: dates[window.endIndex],
          zoomOnMouseWheel: true,
          moveOnMouseMove: false,
          moveOnMouseWheel: true,
          preventDefaultMouseMove: true,
        },
        {
          type: "slider",
          xAxisIndex: [0, 1],
          startValue: dates[window.startIndex],
          endValue: dates[window.endIndex],
          bottom: 0,
          height: 16,
          borderColor: "#20364a",
          backgroundColor: "#0b1722",
          fillerColor: "rgba(63,140,255,.16)",
          handleStyle: { color: "#3f8cff", borderColor: "#85b5ff" },
          moveHandleStyle: { color: "#315f8e" },
          textStyle: { color: "#71879a", fontSize: 9 },
          brushSelect: false,
        },
      ],
      series: [
        {
          name: "Kline",
          type: "candlestick",
          data: bars.map((bar) => [bar.open, bar.close, bar.low, bar.high]),
          itemStyle: {
            color: "#2ac89f",
            color0: "#ef5c63",
            borderColor: "#2ac89f",
            borderColor0: "#ef5c63",
          },
          markPoint: {
            silent: false,
            symbolKeepAspect: true,
            data: tradePoints,
            label: { hideOverlap: true },
          },
        },
        ...maSeries,
        {
          name: "Volume",
          type: "bar",
          xAxisIndex: 1,
          yAxisIndex: 1,
          barMaxWidth: 8,
          data: bars.map((bar) => ({
            value: bar.volume ?? 0,
            itemStyle: { color: bar.close >= bar.open ? "rgba(42,200,159,.66)" : "rgba(239,92,99,.66)" },
          })),
        },
      ],
    };

    return { option, markerCount: markerLayouts.length };
  }, [bars, layers, range, selectedDate, selectedTradeId, symbol, trades]);

  const toggleLayer = (key: LayerKey): void => {
    setLayers((current) => ({ ...current, [key]: !current[key] }));
  };

  return (
    <div className="kline-workstation">
      <div className="kline-controlbar" aria-label="K 线图层与时间窗口">
        <div className="kline-control-group kline-range-group">
          <span>窗口</span>
          {RANGE_OPTIONS.map((item) => (
            <button
              key={item.value}
              type="button"
              className={range === item.value ? "active" : ""}
              aria-pressed={range === item.value}
              onClick={() => setRange(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="kline-control-group kline-layer-group">
          <span>指标</span>
          {MA_OPTIONS.map((item) => (
            <button
              key={item.key}
              type="button"
              className={layers[item.key] ? "active" : ""}
              aria-pressed={layers[item.key]}
              onClick={() => toggleLayer(item.key)}
            >
              <i style={{ backgroundColor: item.color }} />{item.label}
            </button>
          ))}
        </div>
        <div className="kline-control-group kline-layer-group">
          <span>事件</span>
          {([
            ["trades", "成交"],
            ["tTrades", "做 T"],
            ["risk", "风控"],
          ] as Array<[LayerKey, string]>).map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={layers[key] ? "active" : ""}
              aria-pressed={layers[key]}
              onClick={() => toggleLayer(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="kline-current-state" role="status" aria-live="polite">
          <span>当前</span>
          {selectedTrade ? (
            <>
              <strong>{selectedTrade.action}</strong>
              <time>{selectedTrade.datetime.slice(0, 16)}</time>
              <b>{formatValue(selectedTrade.price)}</b>
            </>
          ) : <strong>未选择交易</strong>}
          <em>{derived.markerCount} 个可见事件</em>
        </div>
      </div>
      <div className="kline-gesture-hint">滚轮缩放 · 拖动平移 · 十字光标查看 OHLCV · 点击信号联动交易记录</div>
      <EChart
        option={derived.option}
        className="chart chart-kline chart-kline-workstation"
        ariaLabel={`${symbol ?? bars[0]?.symbol ?? "股票"} K 线、成交量、均线与交易事件图`}
        onClick={(params) => {
          const value = params as { data?: { tradeId?: string } };
          if (value.data?.tradeId) onTradeSelect?.(value.data.tradeId);
        }}
      />
    </div>
  );
}
