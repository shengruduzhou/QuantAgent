import { useEffect, useMemo, useRef, useState } from "react";
import { MagnifyingGlass, X } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import { workstationModules as commands } from "../workstation/modules";

interface CommandPaletteProps {
  open: boolean;
  initialQuery?: string;
  onClose: () => void;
}

export function CommandPalette({ open, initialQuery = "", onClose }: CommandPaletteProps): JSX.Element | null {
  const [query, setQuery] = useState(initialQuery);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!open) return;
    setQuery(initialQuery);
    const timer = window.setTimeout(() => inputRef.current?.focus(), 20);
    return () => window.clearTimeout(timer);
  }, [initialQuery, open]);

  useEffect(() => {
    if (!open) return undefined;
    const close = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose, open]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return commands;
    if (/^\d{6}(?:\.(?:sz|sh|bj))?$/.test(needle)) return [commands[1]];
    return commands.filter((item) =>
      `${item.label} ${item.caption} ${item.keywords}`.toLowerCase().includes(needle),
    );
  }, [query]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query, open]);

  if (!open) return null;

  const select = (path: string): void => {
    const stock = query.trim().toUpperCase().match(/^(\d{6})(?:\.(SZ|SH|BJ))?$/);
    if (stock && path === "/stock-replay") {
      const suffix = stock[2] ?? (stock[1].startsWith("6") ? "SH" : "SZ");
      navigate(`/stock-replay?symbol=${stock[1]}.${suffix}`);
    } else {
      navigate(path);
    }
    onClose();
  };

  return (
    <div className="command-overlay" role="presentation" onMouseDown={onClose}>
      <section className="command-palette" role="dialog" aria-modal="true" aria-label="全局命令面板" onMouseDown={(event) => event.stopPropagation()}>
        <div className="command-input">
          <MagnifyingGlass size={19} />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="输入页面、股票代码、因子、模型或功能"
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setSelectedIndex((current) => Math.min(current + 1, visible.length - 1));
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setSelectedIndex((current) => Math.max(current - 1, 0));
              }
              if (event.key === "Enter" && visible[selectedIndex]) {
                event.preventDefault();
                select(visible[selectedIndex].path);
              }
            }}
          />
          <button onClick={onClose} aria-label="关闭命令面板"><X size={17} /></button>
        </div>
        <div className="command-results">
          {visible.map((item, index) => {
            const Icon = item.icon;
            return (
              <button
                key={item.path}
                className={index === selectedIndex ? "active" : ""}
                onClick={() => select(item.path)}
                onMouseEnter={() => setSelectedIndex(index)}
              >
                <span className="command-icon"><Icon size={19} weight="duotone" /></span>
                <span>
                  <strong>{item.path === "/stock-replay" && /^\d{6}/.test(query.trim()) ? `复盘 ${query.trim().toUpperCase()}` : item.label}</strong>
                  <small>{item.caption}</small>
                </span>
                <kbd>↵</kbd>
              </button>
            );
          })}
          {!visible.length ? <div className="command-empty">没有匹配命令；股票代码可直接跳转到复盘页。</div> : null}
        </div>
        <footer><span>↑↓ 浏览</span><span>Enter 打开</span><span>Esc 关闭</span></footer>
      </section>
    </div>
  );
}
