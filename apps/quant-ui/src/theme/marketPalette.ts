export const marketPalette = {
  canvas: "#07131e",
  panel: "#08141e",
  border: "#29475f",
  grid: "#13283a",
  axis: "#71879a",
  text: "#d8e5ef",
  selected: "#ffffff",
  up: "#3f8cff",
  down: "#ef5c63",
  buy: "#2ac89f",
  sell: "#ef5c63",
  tBuy: "#6f9dff",
  tSell: "#b779ff",
  risk: "#e7a53a",
  ma5: "#d7b95b",
  ma10: "#72a7ff",
  ma20: "#b28cff",
  ma60: "#7f96a8",
} as const;

export type MarketPalette = typeof marketPalette;
