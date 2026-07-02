import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { KlineBar, Trade } from "../api/types";
import { EChart } from "./EChart";

interface CandlestickChartProps {
  bars: KlineBar[];
  trades: Trade[];
  selectedTradeId?: string | null;
  onTradeSelect?: (tradeId: string) => void;
}

export function CandlestickChart({
  bars,
  trades,
  selectedTradeId,
  onTradeSelect,
}: CandlestickChartProps): JSX.Element {
  const option = useMemo(() => {
    const dates = bars.map((bar) => bar.datetime.slice(0, 10));
    const dateIndex = new Set(dates);
    const selectedTrade = trades.find((trade) => trade.id === selectedTradeId) ?? trades[0];
    const selectedDate = selectedTrade?.datetime.slice(0, 10);
    const selectedIndex = selectedDate ? dates.indexOf(selectedDate) : -1;
    const windowStart = Math.max(0, (selectedIndex >= 0 ? selectedIndex : dates.length - 1) - 90);
    const windowEnd = Math.min(dates.length - 1, (selectedIndex >= 0 ? selectedIndex : dates.length - 1) + 90);
    const tradePoints = trades
      .filter((trade) => dateIndex.has(trade.datetime.slice(0, 10)))
      .map((trade) => ({
        name: trade.action,
        value: trade.price,
        coord: [trade.datetime.slice(0, 10), trade.price],
        tradeId: trade.id,
        symbol: trade.action.includes("BUY") ? "triangle" : "pin",
        symbolRotate: trade.action.includes("BUY") ? 0 : 180,
        symbolSize: trade.id === selectedTradeId ? 18 : 13,
        itemStyle: {
          color: trade.action.includes("BUY") ? "#26c79a" : "#f05a5a",
          borderColor: trade.id === selectedTradeId ? "#ffffff" : "transparent",
          borderWidth: trade.id === selectedTradeId ? 1.5 : 0,
        },
        label: {
          show: true,
          formatter: trade.action.replace("T_", "T·"),
          color: trade.action.includes("BUY") ? "#26c79a" : "#ff7373",
          fontSize: 9,
          position: trade.action.includes("BUY") ? "bottom" : "top",
        },
      }));

    return {
      animation: false,
      backgroundColor: "transparent",
      legend: { show: false },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        backgroundColor: "#0b1824",
        borderColor: "#27425a",
        textStyle: { color: "#d7e4ef" },
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: [
        { left: 58, right: 22, top: 24, height: "64%" },
        { left: 58, right: 22, top: "76%", height: "15%" },
      ],
      xAxis: [
        {
          type: "category",
          data: dates,
          boundaryGap: true,
          axisLabel: { show: false },
          axisLine: { lineStyle: { color: "#20364a" } },
        },
        {
          type: "category",
          gridIndex: 1,
          data: dates,
          boundaryGap: true,
          axisLabel: { color: "#71879a", fontSize: 10 },
          axisLine: { lineStyle: { color: "#20364a" } },
        },
      ],
      yAxis: [
        {
          scale: true,
          position: "right",
          axisLabel: { color: "#71879a", fontSize: 10 },
          splitLine: { lineStyle: { color: "#13283a" } },
        },
        {
          scale: true,
          gridIndex: 1,
          position: "right",
          axisLabel: { color: "#71879a", fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1], startValue: dates[windowStart], endValue: dates[windowEnd] },
        { type: "slider", xAxisIndex: [0, 1], startValue: dates[windowStart], endValue: dates[windowEnd], bottom: 0, height: 14, borderColor: "#20364a", backgroundColor: "#0b1722", fillerColor: "rgba(47,131,255,.18)", handleStyle: { color: "#3f8cff" }, textStyle: { color: "#71879a" } },
      ],
      series: [
        {
          name: "Kline",
          type: "candlestick",
          data: bars.map((bar) => [bar.open, bar.close, bar.low, bar.high]),
          itemStyle: {
            color: "#25bc8e",
            color0: "#ef5c63",
            borderColor: "#25bc8e",
            borderColor0: "#ef5c63",
          },
          markPoint: {
            data: tradePoints,
          },
        },
        {
          name: "Volume",
          type: "bar",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: bars.map((bar) => ({
            value: bar.volume ?? 0,
            itemStyle: { color: bar.close >= bar.open ? "rgba(37,188,142,.72)" : "rgba(239,92,99,.72)" },
          })),
        },
      ],
    } as EChartsOption;
  }, [bars, selectedTradeId, trades]);

  return (
    <EChart
      option={option}
      className="chart chart-kline"
      onClick={(params) => {
        const value = params as { data?: { tradeId?: string } };
        if (value.data?.tradeId) onTradeSelect?.(value.data.tradeId);
      }}
    />
  );
}
