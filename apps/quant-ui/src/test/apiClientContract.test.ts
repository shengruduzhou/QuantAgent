import { describe, expect, it } from "vitest";

const sources = import.meta.glob("../**/*.{ts,tsx}", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

/**
 * Every request must go through `src/api/client.ts`.
 *
 * That module is the single place holding the `{status, data, issues}`
 * envelope, the AbortSignal wiring and `compactApiErrorPayload` (which turns a
 * FastAPI 422 body into readable operator text). A page that calls `fetch`
 * directly gets none of it: `WalkForwardRiskPage` used to poll every 2s with no
 * cancellation and surfaced a bare "HTTP 500" to the operator.
 */
describe("all requests go through the shared API client", () => {
  // `refetch()` from react-query is not a network primitive; exclude it.
  const rawFetch = /(?<![A-Za-z.])fetch\s*\(/g;

  const productionFiles = Object.entries(sources).filter(
    ([path]) => !path.includes(".test.") && !path.endsWith("api/client.ts") && !path.includes("/test/"),
  );

  it("actually scans the product source", () => {
    expect(productionFiles.length).toBeGreaterThan(60);
    expect(productionFiles.some(([path]) => path.endsWith("WalkForwardRiskPage.tsx"))).toBe(true);
  });

  it("detects a raw fetch when one exists", () => {
    expect("const r = await fetch(`${API}/results`);".match(rawFetch)).not.toBeNull();
    expect("void query.refetch();".match(rawFetch)).toBeNull();
    expect("window.fetch(url)".match(rawFetch)).toBeNull();
  });

  it("has no direct fetch() outside the client", () => {
    const offenders: string[] = [];
    for (const [path, text] of productionFiles) {
      if (rawFetch.test(text)) offenders.push(path);
      rawFetch.lastIndex = 0;
    }
    expect(offenders).toEqual([]);
  });
});
