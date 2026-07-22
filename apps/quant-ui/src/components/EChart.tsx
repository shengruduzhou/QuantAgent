import { useEffect, useMemo, useRef, type CSSProperties, type KeyboardEvent } from "react";
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
import { useVNextChartPalette, type VNextChartPalette } from "../vnext/theme";

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
  const palette = useVNextChartPalette();
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

  const themedOption = useMemo(() => applyWorkstationTheme(option, palette), [option, palette]);

  useEffect(() => {
    chartRef.current?.setOption({
      textStyle: {
        fontFamily: '"Inter Variable", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif',
        color: palette.text,
      },
      ...themedOption,
    } as EChartsCoreOption, { notMerge: true });
  }, [palette.text, themedOption]);

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

function applyWorkstationTheme(option: EChartsOption, palette: VNextChartPalette): EChartsOption {
  const source = option as Record<string, unknown>;
  const themeAxis = (axis: unknown): unknown => {
    if (Array.isArray(axis)) return axis.map(themeAxis);
    if (!axis || typeof axis !== "object") return axis;
    const value = axis as Record<string, unknown>;
    const axisLabel = (value.axisLabel && typeof value.axisLabel === "object" ? value.axisLabel : {}) as Record<string, unknown>;
    const nameTextStyle = (value.nameTextStyle && typeof value.nameTextStyle === "object" ? value.nameTextStyle : {}) as Record<string, unknown>;
    const axisLine = (value.axisLine && typeof value.axisLine === "object" ? value.axisLine : {}) as Record<string, unknown>;
    const axisLineStyle = (axisLine.lineStyle && typeof axisLine.lineStyle === "object" ? axisLine.lineStyle : {}) as Record<string, unknown>;
    const splitLine = (value.splitLine && typeof value.splitLine === "object" ? value.splitLine : {}) as Record<string, unknown>;
    const splitLineStyle = (splitLine.lineStyle && typeof splitLine.lineStyle === "object" ? splitLine.lineStyle : {}) as Record<string, unknown>;
    return {
      ...value,
      axisLabel: { ...axisLabel, color: palette.text },
      nameTextStyle: { ...nameTextStyle, color: palette.muted },
      axisLine: { ...axisLine, lineStyle: { ...axisLineStyle, color: palette.axis } },
      splitLine: { ...splitLine, lineStyle: { ...splitLineStyle, color: palette.grid } },
    };
  };
  const tooltip = (source.tooltip && typeof source.tooltip === "object" ? source.tooltip : {}) as Record<string, unknown>;
  const tooltipText = (tooltip.textStyle && typeof tooltip.textStyle === "object" ? tooltip.textStyle : {}) as Record<string, unknown>;
  const legend = (source.legend && typeof source.legend === "object" && !Array.isArray(source.legend) ? source.legend : {}) as Record<string, unknown>;
  const legendText = (legend.textStyle && typeof legend.textStyle === "object" ? legend.textStyle : {}) as Record<string, unknown>;
  const dataZoom = Array.isArray(source.dataZoom) ? source.dataZoom.map((entry) => {
    if (!entry || typeof entry !== "object") return entry;
    const value = entry as Record<string, unknown>;
    const textStyle = (value.textStyle && typeof value.textStyle === "object" ? value.textStyle : {}) as Record<string, unknown>;
    return { ...value, borderColor: palette.axis, backgroundColor: palette.slider, dataBackground: { lineStyle: { color: palette.sliderData }, areaStyle: { color: palette.sliderData } }, selectedDataBackground: { lineStyle: { color: palette.sliderSelected }, areaStyle: { color: palette.sliderSelected } }, textStyle: { ...textStyle, color: palette.text } };
  }) : source.dataZoom;
  return {
    ...source,
    backgroundColor: "transparent",
    tooltip: { ...tooltip, backgroundColor: palette.tooltip, borderColor: palette.tooltipBorder, textStyle: { ...tooltipText, color: palette.tooltipText } },
    legend: { ...legend, textStyle: { ...legendText, color: palette.text } },
    xAxis: themeAxis(source.xAxis),
    yAxis: themeAxis(source.yAxis),
    dataZoom,
  } as EChartsOption;
}
