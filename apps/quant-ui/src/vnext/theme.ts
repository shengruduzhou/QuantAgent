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
}

const chartPalettes: Record<WorkspaceTheme, VNextChartPalette> = {
  night: {
    text: "#9bb0bd",
    muted: "#6f8899",
    grid: "#182b37",
    axis: "#263b49",
    tooltip: "#08131b",
    tooltipBorder: "#304657",
    tooltipText: "#e6edf2",
    slider: "#0c171f",
    sliderData: "#172c3a",
    sliderSelected: "#183c5a",
  },
  dawn: {
    text: "#a9bbc5",
    muted: "#8197a3",
    grid: "#273b47",
    axis: "#3a515e",
    tooltip: "#111f28",
    tooltipBorder: "#456170",
    tooltipText: "#edf5f7",
    slider: "#15242d",
    sliderData: "#203946",
    sliderSelected: "#25516f",
  },
  day: {
    text: "#304853",
    muted: "#627983",
    grid: "#d8e3e8",
    axis: "#b7c9d1",
    tooltip: "#ffffff",
    tooltipBorder: "#9fb8c4",
    tooltipText: "#15252e",
    slider: "#e4edf1",
    sliderData: "#cbdbe2",
    sliderSelected: "#9ec7df",
  },
};

export function useVNextTheme(): WorkspaceTheme {
  return useContext(VNextThemeContext);
}

export function useVNextChartPalette(): VNextChartPalette {
  return chartPalettes[useVNextTheme()];
}
