import { describe, expect, it } from "vitest";

// Read the product source through Vite rather than `node:fs`, so the check
// needs no @types/node and runs under the same resolver as the app build.
const sources = import.meta.glob("../**/*.tsx", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

/**
 * AGENTS.md: an empty state must say which artifact is missing, what safe
 * state is being held, and what to do next. A bare `<StateView state="empty" />`
 * falls through to the generic default copy, which says none of those things —
 * the reader cannot tell "this run produced no artifact" from "this page is
 * broken", which is the same ambiguity the honesty rules exist to remove.
 */
describe("empty states explain themselves", () => {
  const bare = /<StateView\s+state=(?:"empty"|\{[^}]*"empty"[^}]*\})\s*\/>/g;

  it("actually scans the product source", () => {
    // Without this the suite would pass vacuously if the glob ever broke.
    const files = Object.keys(sources).filter((path) => !path.includes(".test."));
    expect(files.length).toBeGreaterThan(40);
    expect(files.some((path) => path.endsWith("RiskCenterPage.tsx"))).toBe(true);
  });

  it("detects a bare empty state when one exists", () => {
    // Pins the matcher itself, so the check below cannot go quietly blind.
    expect('<StateView state="empty" />'.match(bare)).not.toBeNull();
    expect('<StateView state={loading ? "loading" : "empty"} />'.match(bare)).not.toBeNull();
    expect('<StateView state="empty" detail="x" />'.match(bare)).toBeNull();
  });

  it("has no StateView empty without title or detail", () => {
    const offenders: string[] = [];
    for (const [path, text] of Object.entries(sources)) {
      if (path.includes(".test.")) continue;
      for (const match of text.matchAll(bare)) {
        offenders.push(`${path}: ${match[0]}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
