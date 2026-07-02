import { useEffect, useRef } from "react";
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
  onClick?: (params: unknown) => void;
}

export function EChart({ option, className, onClick }: EChartProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<EChartsType | null>(null);

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const chart = init(containerRef.current, undefined, { renderer: "canvas" });
    chartRef.current = chart;
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
        fontFamily: '"Noto Sans SC", "Inter Variable", sans-serif',
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

  return <div ref={containerRef} className={className ?? "chart"} role="img" />;
}
