export const marketPalette = {
  canvas: "#07131e",
  panel: "#08141e",
  border: "#29475f",
  grid: "#13283a",
  axis: "#71879a",
  text: "#d8e5ef",
  selected: "#ffffff",
  up: "#f0525f",
  down: "#22b889",
  buy: "#f0525f",
  sell: "#22b889",
  tBuy: "#6f9dff",
  tSell: "#b779ff",
  risk: "#e7a53a",
  ma5: "#d7b95b",
  ma10: "#72a7ff",
  ma20: "#b28cff",
  ma60: "#7f96a8",
} as const;

export type MarketPalette = typeof marketPalette;
