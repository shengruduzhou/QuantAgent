import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { ArrowsClockwise, DownloadSimple, MagicWand } from "@phosphor-icons/react";
import { StateView } from "./StateView";

export type MonitorAlign = "left" | "center" | "right";
export type MonitorSortDirection = "asc" | "desc";

export interface MonitorColumn<T> {
  id: string;
  header: string;
  value: (row: T) => unknown;
  render?: (row: T) => ReactNode;
  csvValue?: (row: T) => unknown;
  sortable?: boolean;
  align?: MonitorAlign;
  width?: number;
  minWidth?: number;
  maxWidth?: number;
}

interface MonitorTableProps<T> {
  monitorId: string;
  ariaLabel: string;
  rows: readonly T[];
  columns: readonly MonitorColumn<T>[];
  rowKey: (row: T) => string;
  selectedKey?: string | null;
  onSelect?: (row: T) => void;
  emptyDetail?: string;
  exportFilename?: string;
  maxRows?: number;
  className?: string;
}

interface SortState {
  columnId: string;
  direction: MonitorSortDirection;
}

type WidthMap = Record<string, number>;

const STORAGE_PREFIX = "quantagent.monitor.columns.v1";
const DEFAULT_MIN_WIDTH = 72;
const DEFAULT_MAX_WIDTH = 360;

function toComparable(value: unknown): string | number {
  if (typeof value === "number") return Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (value === null || value === undefined) return "";
  return String(value).toLocaleLowerCase();
}

function compareValues(left: unknown, right: unknown): number {
  const a = toComparable(left);
  const b = toComparable(right);
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), "zh-CN", { numeric: true, sensitivity: "base" });
}

function csvEscape(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  if (!/[",\r\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function initialWidths<T>(columns: readonly MonitorColumn<T>[]): WidthMap {
  return Object.fromEntries(columns.map((column) => [column.id, column.width ?? 120]));
}

function loadWidths<T>(monitorId: string, columns: readonly MonitorColumn<T>[]): WidthMap {
  const fallback = initialWidths(columns);
  try {
    const raw = window.localStorage.getItem(`${STORAGE_PREFIX}.${monitorId}`);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return Object.fromEntries(columns.map((column) => {
      const stored = parsed[column.id];
      return [column.id, typeof stored === "number" && Number.isFinite(stored) ? stored : fallback[column.id]];
    }));
  } catch {
    return fallback;
  }
}

function clampWidth<T>(column: MonitorColumn<T>, value: number): number {
  const min = column.minWidth ?? DEFAULT_MIN_WIDTH;
  const max = column.maxWidth ?? DEFAULT_MAX_WIDTH;
  return Math.min(max, Math.max(min, Math.round(value)));
}

export function MonitorTable<T>({
  monitorId,
  ariaLabel,
  rows,
  columns,
  rowKey,
  selectedKey,
  onSelect,
  emptyDetail = "当前没有可显示的监控数据。",
  exportFilename = `${monitorId}.csv`,
  maxRows,
  className = "",
}: MonitorTableProps<T>): JSX.Element {
  const [sort, setSort] = useState<SortState | null>(null);
  const [widths, setWidths] = useState<WidthMap>(() => loadWidths(monitorId, columns));

  useEffect(() => {
    setWidths(loadWidths(monitorId, columns));
  }, [columns, monitorId]);

  useEffect(() => {
    try {
      window.localStorage.setItem(`${STORAGE_PREFIX}.${monitorId}`, JSON.stringify(widths));
    } catch {
      // Persistence is optional; the monitor remains functional without storage access.
    }
  }, [monitorId, widths]);

  const visibleRows = useMemo(() => {
    const indexed = rows.map((row, index) => ({ row, index }));
    if (sort) {
      const column = columns.find((item) => item.id === sort.columnId);
      if (column) {
        indexed.sort((left, right) => {
          const compared = compareValues(column.value(left.row), column.value(right.row));
          if (compared !== 0) return sort.direction === "asc" ? compared : -compared;
          return left.index - right.index;
        });
      }
    }
    const ordered = indexed.map((item) => item.row);
    return typeof maxRows === "number" ? ordered.slice(0, maxRows) : ordered;
  }, [columns, maxRows, rows, sort]);

  const toggleSort = (column: MonitorColumn<T>): void => {
    if (column.sortable === false) return;
    setSort((current) => {
      if (!current || current.columnId !== column.id) return { columnId: column.id, direction: "asc" };
      if (current.direction === "asc") return { columnId: column.id, direction: "desc" };
      return null;
    });
  };

  const resetWidths = (): void => {
    setWidths(initialWidths(columns));
  };

  const autoFit = (): void => {
    const sample = rows.slice(0, 500);
    setWidths(Object.fromEntries(columns.map((column) => {
      const longest = sample.reduce((length, row) => {
        const value = column.csvValue ? column.csvValue(row) : column.value(row);
        return Math.max(length, String(value ?? "").length);
      }, column.header.length);
      return [column.id, clampWidth(column, 30 + longest * 8)];
    })));
  };

  const exportCsv = (): void => {
    const header = columns.map((column) => csvEscape(column.header)).join(",");
    const body = visibleRows.map((row) => columns.map((column) => {
      const value = column.csvValue ? column.csvValue(row) : column.value(row);
      return csvEscape(value);
    }).join(","));
    const blob = new Blob(["\uFEFF", [header, ...body].join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = exportFilename;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const resizeColumn = useCallback((column: MonitorColumn<T>, startX: number, startWidth: number): void => {
    const handleMove = (event: PointerEvent): void => {
      const next = clampWidth(column, startWidth + event.clientX - startX);
      setWidths((current) => ({ ...current, [column.id]: next }));
    };
    const handleUp = (): void => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp, { once: true });
  }, []);

  const selectAdjacent = (event: KeyboardEvent<HTMLTableRowElement>, index: number): void => {
    if (!onSelect || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return;
    event.preventDefault();
    const direction = event.key === "ArrowDown" ? 1 : -1;
    const next = visibleRows[Math.min(visibleRows.length - 1, Math.max(0, index + direction))];
    if (next) onSelect(next);
  };

  return (
    <section className={`monitor-table ${className}`.trim()} aria-label={ariaLabel}>
      <header className="monitor-toolbar">
        <span><strong>{visibleRows.length.toLocaleString()}</strong> / {rows.length.toLocaleString()} rows</span>
        <span className="monitor-sort-state">
          {sort ? `sort ${sort.columnId} ${sort.direction}` : "source order"}
        </span>
        <button type="button" onClick={autoFit} title="根据当前数据自适应列宽"><MagicWand size={13} /> 自适应</button>
        <button type="button" onClick={resetWidths} title="恢复默认列宽"><ArrowsClockwise size={13} /> 复位列宽</button>
        <button type="button" onClick={exportCsv} title={`导出 ${exportFilename}`}><DownloadSimple size={13} /> CSV</button>
      </header>

      {visibleRows.length ? (
        <div className="monitor-table-scroll">
          <table className="data-table" aria-label={ariaLabel}>
            <colgroup>
              {columns.map((column) => <col key={column.id} style={{ width: widths[column.id] ?? column.width ?? 120 }} />)}
            </colgroup>
            <thead>
              <tr>
                {columns.map((column) => {
                  const active = sort?.columnId === column.id;
                  return (
                    <th key={column.id} className={column.align === "right" ? "numeric" : ""}>
                      <button
                        type="button"
                        className="monitor-header-button"
                        onClick={() => toggleSort(column)}
                        aria-sort={active ? (sort?.direction === "asc" ? "ascending" : "descending") : "none"}
                        disabled={column.sortable === false}
                      >
                        <span>{column.header}</span>
                        <small>{active ? (sort?.direction === "asc" ? "▲" : "▼") : ""}</small>
                      </button>
                      <i
                        className="monitor-resize-handle"
                        role="separator"
                        aria-orientation="vertical"
                        aria-label={`调整 ${column.header} 列宽`}
                        onPointerDown={(event) => {
                          event.preventDefault();
                          resizeColumn(column, event.clientX, widths[column.id] ?? column.width ?? 120);
                        }}
                      />
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => {
                const key = rowKey(row);
                const selected = selectedKey === key;
                return (
                  <tr
                    key={key}
                    tabIndex={onSelect ? 0 : undefined}
                    aria-selected={selected}
                    className={selected ? "selected" : ""}
                    onClick={() => onSelect?.(row)}
                    onKeyDown={(event) => selectAdjacent(event, index)}
                  >
                    {columns.map((column) => (
                      <td
                        key={column.id}
                        className={column.align === "right" ? "numeric" : column.align === "center" ? "center" : ""}
                      >
                        {column.render ? column.render(row) : String(column.value(row) ?? "—")}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : <StateView state="empty" detail={emptyDetail} />}
    </section>
  );
}
