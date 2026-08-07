import { StockReplayPage } from "../../pages/StockReplayPage";
import {
  TruthNotice,
  WorkbenchHeader,
} from "../workbench/InstitutionalWorkbench";

/**
 * Canonical VNext market-reading frame.
 *
 * The information hierarchy follows the public HiThink/Fuyao stock-overview
 * reference without copying vendor branding: identity/source/as-of first,
 * dominant K-line + moving averages + linked volume second, then interval/risk
 * metrics and QuantAgent-only decision evidence. The existing StockReplayPage
 * remains the functional body so trade markers, T+1 analysis and the decision
 * inspector are not forked into a second implementation.
 */
export function MarketWorkbenchPage(): JSX.Element {
  return (
    <div className="page institutional-workbench market-workbench-page">
      <WorkbenchHeader
        eyebrow="Market Workbench · 行情工作台"
        title="单股行情与研究复盘"
        description="先读行情快照与趋势，再读成交量、区间风险和 QuantAgent 决策证据；所有字段沿用 persisted runtime，不用 mock 补空。"
        asOf="PIT / source-backed"
        context="行情源 · 数据时间 · 复权口径必须显式"
      />
      <TruthNotice tone="info">
        行情阅读基线：最新价 / 涨跌 → K 线 + 可验证均线层 → 联动成交量 → 区间收益 / 最大回撤 / 平均成交额 → 交易与决策证据。若 artifact 缺字段，保持 unavailable，不推断或伪造。
      </TruthNotice>
      <StockReplayPage />
    </div>
  );
}
