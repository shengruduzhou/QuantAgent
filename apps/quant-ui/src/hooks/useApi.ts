import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { apiGet } from "../api/client";
import type { ApiResponse } from "../api/types";

interface ApiQueryOptions {
  refetchInterval?: number | false;
  staleTime?: number;
}

export function useApi<T>(
  key: readonly unknown[],
  path: string | null,
  params?: Record<string, string | number | boolean | null | undefined>,
  options?: ApiQueryOptions,
): UseQueryResult<ApiResponse<T>, Error> {
  return useQuery({
    queryKey: [...key, params],
    queryFn: ({ signal }) => apiGet<T>(path as string, params, signal),
    enabled: Boolean(path),
    refetchInterval: options?.refetchInterval,
    staleTime: options?.staleTime,
  });
}
