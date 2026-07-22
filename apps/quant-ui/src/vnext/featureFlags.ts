export type UiVersion = "vnext" | "legacy";

const STORAGE_KEY = "quantagent.ui.version";

function normalize(value: string | null | undefined): UiVersion | null {
  if (value === "vnext" || value === "legacy") return value;
  return null;
}

export function resolveUiVersion(): UiVersion {
  const params = new URLSearchParams(window.location.search);
  const queryValue = normalize(params.get("ui"));
  if (queryValue) {
    try {
      window.localStorage.setItem(STORAGE_KEY, queryValue);
    } catch {
      // Query selection still works when storage is unavailable.
    }
    return queryValue;
  }

  try {
    const stored = normalize(window.localStorage.getItem(STORAGE_KEY));
    if (stored) return stored;
  } catch {
    // Continue to the build-time/default contract.
  }

  return normalize(import.meta.env.VITE_WORKSTATION_VNEXT) ?? "vnext";
}

export function setUiVersion(version: UiVersion): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, version);
  } catch {
    // A reload can still use the query override.
  }
}
