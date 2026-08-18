import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TradeTable } from "./TradeTable";
import type { Trade } from "../api/types";

function trade(id: string, pnl: number | null | undefined): Trade {
  return {
    id,
    datetime: "2026-08-18T09:35:00",
    symbol: "600519.SH",
    action: "BUY",
    price: 1500,
    quantity: 100,
    pnl,
  } as Trade;
}

describe("TradeTable P&L cell", () => {
  it("keeps A-share direction colours for measured P&L", () => {
    render(<TradeTable trades={[trade("gain", 120.5), trade("loss", -80.25)]} />);
    const gain = screen.getByText("120.50");
    const loss = screen.getByText("-80.25");
    expect(gain.className).toContain("tone-positive");
    expect(loss.className).toContain("tone-negative");
  });

  it("does not paint an absent P&L as a gain", () => {
    // Before the fix the cell rendered "暂无" in `--market-up` red, because
    // `(trade.pnl ?? 0) >= 0` sent every missing value down the positive
    // branch. The number said unmeasured; the colour said the trade made money.
    const { container } = render(
      <TradeTable trades={[trade("missing", null), trade("undef", undefined)]} />,
    );
    const pnlCells = [...container.querySelectorAll("tbody tr")].map(
      (row) => row.querySelectorAll("td")[8],
    );
    expect(pnlCells).toHaveLength(2);
    for (const cell of pnlCells) {
      expect(cell.textContent).toBe("暂无");
      expect(cell.className).toContain("tone-unmeasured");
      expect(cell.className).not.toContain("tone-positive");
      expect(cell.className).not.toContain("tone-negative");
      // Colour is not the only channel: the reason is readable on hover.
      expect(cell.getAttribute("title")).toContain("未测量");
    }
  });
});
