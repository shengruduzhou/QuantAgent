import { StockReplayPage } from "../../pages/StockReplayPage";

/**
 * Financial-API-inspired market workstation.
 *
 * The former implementation wrapped StockReplayPage in another institutional
 * page shell, which duplicated page padding/header chrome and left the legacy
 * replay anatomy untouched. StockReplayPage now owns the complete market-first
 * layout, so this route intentionally renders it directly.
 */
export function MarketWorkbenchPage(): JSX.Element {
  return <StockReplayPage />;
}
