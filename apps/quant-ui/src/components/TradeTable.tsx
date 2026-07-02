import type { Trade } from "../api/types";
import { formatCompact, formatNumber } from "../utils/format";

interface TradeTableProps {
  trades: Trade[];
  selectedId?: string | null;
  onSelect?: (trade: Trade) => void;
  compact?: boolean;
}

export function TradeTable({
  trades,
  selectedId,
  onSelect,
  compact = false,
}: TradeTableProps): JSX.Element {
  return (
    <div className="table-scroll">
      <table className={`data-table ${compact ? "data-table-compact" : ""}`}>
        <thead>
          <tr>
            <th>时间</th>
            <th>代码 / 名称</th>
            <th>操作</th>
            <th className="numeric">价格</th>
            <th className="numeric">数量</th>
            <th className="numeric">金额</th>
            <th className="numeric">成交后仓位</th>
            <th className="numeric">模型分</th>
            <th className="numeric">单笔 PnL</th>
            <th>原因</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => (
            <tr
              key={trade.id}
              className={trade.id === selectedId ? "row-selected" : ""}
              onClick={() => onSelect?.(trade)}
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect?.(trade);
              }}
            >
              <td className="mono">{trade.datetime.slice(0, 16)}</td>
              <td>
                <strong>{trade.symbol}</strong>
                <span>{trade.name ?? "名称暂无"}</span>
              </td>
              <td><span className={`trade-action action-${trade.action.toLowerCase()}`}>{trade.action}</span></td>
              <td className="numeric mono">{formatNumber(trade.price)}</td>
              <td className="numeric mono">{formatCompact(trade.quantity)}</td>
              <td className="numeric mono">{formatCompact(trade.amount)}</td>
              <td className="numeric mono">{formatCompact(trade.positionAfter)}</td>
              <td className="numeric mono">{formatNumber(trade.modelScore)}</td>
              <td className={`numeric mono ${(trade.pnl ?? 0) >= 0 ? "tone-positive" : "tone-negative"}`}>
                {formatNumber(trade.pnl)}
              </td>
              <td className="truncate-cell">{trade.riskReason ?? trade.failureReason ?? trade.signalSource ?? "暂无记录"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
