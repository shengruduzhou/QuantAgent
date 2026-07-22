import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, MagnifyingGlass, X } from "@phosphor-icons/react";
import type { GlobalSearchResult, SearchEntity } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { vnextModules } from "../workspace/modules";

interface CommandResult {
  id: string;
  group: string;
  label: string;
  detail: string;
  path: string;
  status?: string;
}

interface EntityCommandPaletteProps {
  open: boolean;
  onClose: () => void;
  openPath: (path: string, newInstance?: boolean) => void;
}

function directStockPath(value: string): string | null {
  const stock = value.trim().toUpperCase().match(/^(\d{6})(?:\.(SZ|SH|BJ))?$/);
  if (!stock) return null;
  const suffix = stock[2] ?? (stock[1].startsWith("6") ? "SH" : "SZ");
  return `/stock-replay?symbol=${stock[1]}.${suffix}`;
}

export function EntityCommandPalette({ open, onClose, openPath }: EntityCommandPaletteProps): JSX.Element | null {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const deferredQuery = useDeferredValue(query.trim());
  const inputRef = useRef<HTMLInputElement>(null);
  const remote = useApi<GlobalSearchResult>(
    ["vnext-global-search", deferredQuery],
    open && deferredQuery.length >= 2 ? "/search" : null,
    { q: deferredQuery, limit: 8 },
    { staleTime: 10_000 },
  );

  useEffect(() => {
    if (!open) return undefined;
    const timer = window.setTimeout(() => inputRef.current?.focus(), 20);
    const close = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("keydown", close);
    };
  }, [onClose, open]);

  const groups = useMemo(() => {
    const needle = deferredQuery.toLowerCase();
    const moduleItems: CommandResult[] = vnextModules
      .filter((module) => !needle || `${module.label} ${module.caption} ${module.keywords}`.toLowerCase().includes(needle))
      .slice(0, needle ? 8 : 10)
      .map((module) => ({
        id: `module-${module.id}`,
        group: "Modules & commands",
        label: module.label,
        detail: module.caption,
        path: module.path,
      }));
    const stockPath = directStockPath(deferredQuery);
    if (stockPath) {
      const symbol = new URL(stockPath, window.location.origin).searchParams.get("symbol") ?? deferredQuery.toUpperCase();
      moduleItems.unshift({
        id: `stock-direct-${symbol}`,
        group: "Stocks",
        label: symbol,
        detail: "Open exact symbol in Chart Workstation",
        path: stockPath,
      });
    }
    const remoteGroups = (remote.data?.data.groups ?? []).map((group) => ({
      label: group.label,
      items: group.items.map((item: SearchEntity): CommandResult => ({
        id: `${item.kind}-${item.id}`,
        group: group.label,
        label: item.label,
        detail: item.detail,
        path: item.path,
        status: item.status,
      })),
    }));
    const groupedModules = moduleItems.reduce<Array<{ label: string; items: CommandResult[] }>>((result, item) => {
      const existing = result.find((group) => group.label === item.group);
      if (existing) existing.items.push(item);
      else result.push({ label: item.group, items: [item] });
      return result;
    }, []);
    return [...groupedModules, ...remoteGroups].reduce<Array<{ label: string; items: CommandResult[] }>>((result, group) => {
      const existing = result.find((item) => item.label === group.label);
      if (!existing) {
        result.push({ ...group, items: [...group.items] });
        return result;
      }
      for (const item of group.items) {
        if (!existing.items.some((candidate) => candidate.path === item.path)) existing.items.push(item);
      }
      return result;
    }, []);
  }, [deferredQuery, remote.data?.data.groups]);

  const flatResults = useMemo(() => groups.flatMap((group) => group.items), [groups]);

  useEffect(() => setSelectedIndex(0), [deferredQuery, open]);

  if (!open) return null;

  const select = (result: CommandResult): void => {
    openPath(result.path);
    onClose();
  };

  return (
    <div className="vnext-command-overlay" role="presentation" onMouseDown={onClose}>
      <section className="vnext-command-palette" role="dialog" aria-modal="true" aria-label="全局实体与命令搜索" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <MagnifyingGlass size={20} />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="股票 / 因子 / 模型 / Experiment / Run / Artifact / 命令"
            aria-label="搜索 QuantAgent 实体和命令"
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setSelectedIndex((current) => flatResults.length ? Math.min(current + 1, flatResults.length - 1) : 0);
              } else if (event.key === "ArrowUp") {
                event.preventDefault();
                setSelectedIndex((current) => Math.max(current - 1, 0));
              } else if (event.key === "Enter") {
                const stockPath = directStockPath(query);
                const target = stockPath
                  ? { id: stockPath, group: "Stocks", label: query, detail: "", path: stockPath }
                  : flatResults[selectedIndex];
                if (!target) return;
                event.preventDefault();
                select(target);
              }
            }}
          />
          <kbd>⌘K</kbd>
          <button type="button" onClick={onClose} aria-label="关闭全局搜索"><X size={17} /></button>
        </header>
        <div className="vnext-command-results">
          {groups.map((group) => (
            <section key={group.label}>
              <h3>{group.label}</h3>
              {group.items.map((item) => {
                const index = flatResults.indexOf(item);
                return (
                  <button
                    type="button"
                    key={item.id}
                    className={index === selectedIndex ? "active" : ""}
                    onMouseEnter={() => setSelectedIndex(index)}
                    onClick={() => select(item)}
                  >
                    <span><strong>{item.label}</strong><small>{item.detail}</small></span>
                    {item.status ? <em>{item.status}</em> : null}
                    <ArrowRight size={15} />
                  </button>
                );
              })}
            </section>
          ))}
          {!flatResults.length && remote.isLoading ? <div className="vnext-command-empty">正在搜索可信实体…</div> : null}
          {!flatResults.length && !remote.isLoading ? <div className="vnext-command-empty">没有匹配实体；结果不会使用模拟或模糊回退。</div> : null}
        </div>
        <footer><span>↑↓ 选择</span><span>Enter 打开</span><span>Esc 关闭</span><span>搜索结果来自 Runtime adapters</span></footer>
      </section>
    </div>
  );
}
