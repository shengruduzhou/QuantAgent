import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { fetchPaperAccount, type PaperAccount } from "../api/paperAccount";
import type { ApiResponse } from "../api/types";

/**
 * One cache entry for the paper account, shared by the command bar, the
 * decision overview, the T+1 workspace and the Risk page, so the workstation
 * polls `/api/paper/account` once rather than once per surface.
 */
export const PAPER_ACCOUNT_QUERY_KEY = ["paper-account"] as const;

export function usePaperAccount(): UseQueryResult<ApiResponse<PaperAccount>, Error> {
  return useQuery({
    queryKey: PAPER_ACCOUNT_QUERY_KEY,
    queryFn: ({ signal }) => fetchPaperAccount(signal),
    refetchInterval: 15_000,
    staleTime: 5_000,
  });
}
