import { afterEach, expect, test } from "vitest";
import { resolveUiVersion, setUiVersion } from "./featureFlags";

afterEach(() => {
  window.localStorage.clear();
  window.history.replaceState({}, "", "/");
});

test("defaults to VNext and persists an explicit query override", () => {
  expect(resolveUiVersion()).toBe("vnext");
  window.history.replaceState({}, "", "/?ui=legacy");
  expect(resolveUiVersion()).toBe("legacy");
  window.history.replaceState({}, "", "/");
  expect(resolveUiVersion()).toBe("legacy");
});

test("supports an explicit rollback setter", () => {
  setUiVersion("legacy");
  expect(resolveUiVersion()).toBe("legacy");
  setUiVersion("vnext");
  expect(resolveUiVersion()).toBe("vnext");
});
