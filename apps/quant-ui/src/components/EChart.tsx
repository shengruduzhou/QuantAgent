import { useEffect, useRef, type CSSProperties, type KeyboardEvent } from "react";
import type { EChartsOption } from "echarts";
import { BarChart, CandlestickChart, LineChart, RadarChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkPointComponent,
  RadarComponent,
  TooltipComponent,
} from "echarts/components";
import { init, use, type EChartsCoreOption, type EChartsType } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";

use([
  BarChart,
  CandlestickChart,
  LineChart,
  RadarChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkPointComponent,
  RadarComponent,
  TooltipComponent,
  CanvasRenderer,
]);

interface EChartProps {
  option: EChartsOption;
  className?: string;
  ariaLabel?: string;
  interactive?: boolean;
  onClick?: (params: unknown) => void;
  onDataZoom?: (params: unknown, chart: EChartsType) => void;
  onReady?: (chart: EChartsType) => void;
  onKeyDown?: (event: KeyboardEvent<HTMLDivElement>) => void;
  style?: CSSProperties;
}

export function EChart({
  option,
  className,
  ariaLabel,
  interactive = false,
  onClick,
  onDataZoom,
  onReady,
  onKeyDown,
  style,
}: EChartProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<EChartsType | null>(null);
  const onReadyRef = useRef(onReady);

  useEffect(() => {
    onReadyRef.current = onReady;
  }, [onReady]);

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const chart = init(containerRef.current, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    onReadyRef.current?.(chart);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(containerRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption({
      textStyle: {
        fontFamily: '"Inter Variable", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif',
      },
      ...option,
    } as EChartsCoreOption, { notMerge: true });
  }, [option]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onClick) return undefined;
    chart.on("click", onClick);
    return () => {
      chart.off("click", onClick);
    };
  }, [onClick]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onDataZoom) return undefined;
    const handler = (params: unknown): void => onDataZoom(params, chart);
    chart.on("datazoom", handler);
    return () => {
      chart.off("datazoom", handler);
    };
  }, [onDataZoom]);

  return (
    <div
      ref={containerRef}
      className={className ?? "chart"}
      role={interactive ? "application" : "img"}
      tabIndex={interactive ? 0 : undefined}
      aria-label={ariaLabel ?? "QuantAgent 数据图表"}
      onKeyDown={onKeyDown}
      style={style}
    />
  );
}
