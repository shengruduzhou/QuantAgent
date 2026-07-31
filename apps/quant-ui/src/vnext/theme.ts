import { createContext, useContext } from "react";
import type { WorkspaceTheme } from "./workspace/types";

export const VNextThemeContext = createContext<WorkspaceTheme>("night");

export interface VNextChartPalette {
  text: string;
  muted: string;
  grid: string;
  axis: string;
  tooltip: string;
  tooltipBorder: string;
  tooltipText: string;
  slider: string;
  sliderData: string;
  sliderSelected: string;
  /** Market direction: A-share convention, red is up and green is down. */
  positive: string;
  negative: string;
  /** Interaction and system accents, mirroring the CSS token families. */
  primary: string;
  agent: string;
  realtime: string;
  warning: string;
  danger: string;
  success: string;
  /** Categorical series ramp, mirroring `--viz-1` … `--viz-8` in tokens.css. */
  series: string[];
}

const chartPalettes: Record<WorkspaceTheme, VNextChartPalette> = {
  night: {
    text: "#9bb0bd",
    muted: "#6f8899",
    grid: "rgba(148, 178, 208, .10)",
    axis: "#35485c",
    tooltip: "#0c1219",
    tooltipBorder: "#35485c",
    tooltipText: "#e6eef5",
    slider: "#0c1219",
    sliderData: "#172230",
    sliderSelected: "#16304e",
    positive: "#f0525f",
    negative: "#22b889",
    primary: "#5b9dff",
    agent: "#a98bff",
    realtime: "#3ec9e0",
    warning: "#e0a838",
    danger: "#ec5f6a",
    success: "#2fb98d",
    series: ["#5b9dff", "#2fb98d", "#e0a838", "#a98bff", "#3ec9e0", "#ef7d63", "#7e93a8", "#d08fc4"],
  },
  dawn: {
    text: "#a9bbc5",
    muted: "#8197a3",
    grid: "rgba(163, 195, 215, .12)",
    axis: "#486370",
    tooltip: "#16232d",
    tooltipBorder: "#486370",
    tooltipText: "#ecf4f7",
    slider: "#16232d",
    sliderData: "#253844",
    sliderSelected: "#1e4463",
    positive: "#f0525f",
    negative: "#22b889",
    primary: "#63b0ff",
    agent: "#b49aff",
    realtime: "#4bcfc7",
    warning: "#e6b24a",
    danger: "#ec5f6a",
    success: "#35c295",
    series: ["#63b0ff", "#35c295", "#e6b24a", "#b49aff", "#4bcfc7", "#f28a70", "#8ea3b6", "#d79ecd"],
  },
  day: {
    text: "#304853",
    muted: "#5c7280",
    grid: "rgba(45, 78, 100, .12)",
    axis: "#9db2bf",
    tooltip: "#ffffff",
    tooltipBorder: "#9db2bf",
    tooltipText: "#16262f",
    slider: "#eaeff3",
    sliderData: "#cbdbe2",
    sliderSelected: "#d9e8f8",
    positive: "#c8323f",
    negative: "#0f7a5e",
    primary: "#1667c4",
    agent: "#6d45c4",
    realtime: "#077e88",
    warning: "#8f6210",
    danger: "#c03846",
    success: "#10795e",
    series: ["#1667c4", "#10795e", "#8f6210", "#6d45c4", "#077e88", "#c05a37", "#566b78", "#9c4a8c"],
  },
};

export function useVNextTheme(): WorkspaceTheme {
  return useContext(VNextThemeContext);
}

export function useVNextChartPalette(): VNextChartPalette {
  return chartPalettes[useVNextTheme()];
}
