import type { ApiResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number | boolean | null | undefined>,
  signal?: AbortSignal,
): Promise<ApiResponse<T>> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(text || `Request failed: ${response.status}`, response.status);
  }
  return response.json() as Promise<ApiResponse<T>>;
}

export async function apiPost<T>(
  path: string,
  value: unknown,
  signal?: AbortSignal,
): Promise<ApiResponse<T>> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
    signal,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(text || `Request failed: ${response.status}`, response.status);
  }
  return response.json() as Promise<ApiResponse<T>>;
}

export function downloadJson(filename: string, value: unknown): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
