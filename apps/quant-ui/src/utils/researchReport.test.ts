import { describe, expect, it } from "vitest";
import { buildOfflineResearchHtml } from "./researchReport";

describe("offline research report", () => {
  it("renders a self-contained evidence document without secrets", () => {
    const html = buildOfflineResearchHtml({
      title: "研究报告",
      subtitle: "测试",
      generatedAt: "2026-08-08T00:00:00Z",
      sections: [{ title: "绩效", rows: [["Sharpe", "1.20"]] }],
      provenance: [["行情", "/api/market/stocks/600519.SH/overview"]],
      limitations: ["非投资建议"],
    });
    expect(html).toContain("<!doctype html>");
    expect(html).toContain("Sharpe");
    expect(html).toContain("/api/market/stocks/600519.SH/overview");
    expect(html).toContain("非投资建议");
    expect(html).not.toContain("HITHINK_FINANCE_API_KEY=");
    expect(html).not.toContain("<script src=");
    expect(html).not.toContain("<link rel=");
  });
});
