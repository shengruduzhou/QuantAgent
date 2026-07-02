import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  ArrowsClockwise,
  Atom,
  Brain,
  ChartLine,
  ChartLineUp,
  Database,
  FileText,
  Flask,
  Funnel,
  Gear,
  ListMagnifyingGlass,
  MagnifyingGlass,
  Pulse,
  ShieldCheck,
  SidebarSimple,
  TrendUp,
  Warning,
} from "@phosphor-icons/react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import type { SystemOverview } from "../api/types";
import { StatusBadge } from "./StatusBadge";
import { CommandPalette } from "./CommandPalette";

const navItems = [
  { to: "/", label: "工作台", caption: "Dashboard", icon: ChartLineUp },
  { to: "/stock-replay", label: "股票复盘", caption: "Stock Replay", icon: ChartLine },
  { to: "/backtests", label: "回测实验", caption: "Backtest Lab", icon: Flask },
  { to: "/t-plus-one", label: "T+1 做 T", caption: "T+1 Analysis", icon: ArrowsClockwise },
  { to: "/factors", label: "因子中心", caption: "Factor Center", icon: Atom },
  { to: "/selection", label: "选股逻辑", caption: "Selection Logic", icon: Funnel },
  { to: "/models", label: "模型中心", caption: "Model Lab", icon: Brain },
  { to: "/risk", label: "风险中心", caption: "Risk Center", icon: ShieldCheck },
  { to: "/runtime", label: "运行监控", caption: "Runtime Explorer", icon: Database },
  { to: "/reports", label: "研究报告", caption: "Reports", icon: FileText },
] as const;

const pageLabels: Record<string, { title: string; caption: string }> = {
  "/": { title: "量化工作台", caption: "Portfolio Command Center" },
  "/stock-replay": { title: "股票复盘", caption: "Stock Replay Workbench" },
  "/backtests": { title: "回测实验", caption: "Backtest Laboratory" },
  "/t-plus-one": { title: "T+1 合规做 T", caption: "Intraday Overlay Analysis" },
  "/factors": { title: "因子中心", caption: "Factor Research Library" },
  "/selection": { title: "透明选股漏斗", caption: "Selection Decision Chain" },
  "/models": { title: "模型中心", caption: "Training & Inference" },
  "/risk": { title: "风险中心", caption: "Exposure & Control" },
  "/runtime": { title: "运行监控", caption: "Runtime Artifact Explorer" },
  "/reports": { title: "研究报告", caption: "Research Briefs" },
  "/settings": { title: "系统设置", caption: "Terminal Preferences" },
};

export function AppShell(): JSX.Element {
  const [collapsed, setCollapsed] = useState(false);
  const [search, setSearch] = useState("");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const page = pageLabels[location.pathname] ?? pageLabels["/"];
  const overview = useApi<SystemOverview>(["system-overview-shell"], "/system/overview");
  const data = overview.data?.data;
  const closePalette = useCallback(() => setPaletteOpen(false), []);

  useEffect(() => {
    const openPalette = (event: KeyboardEvent): void => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", openPalette);
    return () => window.removeEventListener("keydown", openPalette);
  }, []);

  return (
    <div className={`app-frame ${collapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><TrendUp size={24} weight="bold" /></div>
          <div>
            <strong>QuantAgent</strong>
            <span>AI 量化研究系统</span>
          </div>
          <button className="icon-button sidebar-toggle" onClick={() => setCollapsed((value) => !value)} aria-label="切换导航栏">
            <SidebarSimple size={18} />
          </button>
        </div>
        <nav className="main-nav" aria-label="主导航">
          {navItems.map(({ to, label, caption, icon: NavIcon }) => (
            <NavLink key={to} to={to} end={to === "/"}>
              <NavIcon size={21} weight="duotone" />
              <span>
                <strong>{label}</strong>
                <small>{caption}</small>
              </span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <NavLink to="/settings">
            <Gear size={20} />
            <span>
              <strong>系统设置</strong>
              <small>Settings</small>
            </span>
          </NavLink>
          <div className="user-chip">
            <div>QA</div>
            <span>
              <strong>Quant Research</strong>
              <small>Read-only workspace</small>
            </span>
          </div>
        </div>
      </aside>

      <header className="topbar">
        <div className="page-title">
          <h1>{page.title}</h1>
          <span>{page.caption}</span>
        </div>
        <form
          className="global-search"
          onSubmit={(event: FormEvent<HTMLFormElement>) => {
            event.preventDefault();
            const value = search.trim();
            if (!value) {
              setPaletteOpen(true);
              return;
            }
            const stock = value.toUpperCase().match(/^(\d{6})(?:\.(SZ|SH|BJ))?$/);
            if (stock) {
              const suffix = stock[2] ?? (stock[1].startsWith("6") ? "SH" : "SZ");
              navigate(`/stock-replay?symbol=${stock[1]}.${suffix}`);
            } else {
              navigate(`/runtime?query=${encodeURIComponent(value)}`);
            }
          }}
        >
          <MagnifyingGlass size={17} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            onFocus={() => setPaletteOpen(true)}
            placeholder="搜索股票 / 因子 / 模型 / 回测 ID"
            aria-label="全局搜索"
          />
          <kbd>⌘K</kbd>
        </form>
        <div className="topbar-status">
          <StatusBadge status={data?.modelStatus ?? "loading"} label={`模型 ${data?.modelStatus ?? "连接中"}`} />
          <span className="topbar-stat"><Database size={15} /> {data?.runtime.artifactCount?.toLocaleString() ?? "—"} artifacts</span>
          <span className="topbar-stat"><Pulse size={15} /> PIT Research</span>
          {(data?.riskStatus ?? "") === "warning" ? <Warning size={18} className="tone-warning" /> : <ShieldCheck size={18} className="tone-positive" />}
        </div>
      </header>

      <main className="workspace">
        {overview.isError ? (
          <div className="api-banner">
            <Warning size={18} />
            <span>Quant API 未连接；页面会保持空数据态，不使用模拟结果。</span>
            <code>python3 -m services.quant_api</code>
          </div>
        ) : null}
        <Outlet />
      </main>

      <footer className="statusbar">
        <span><i className="health-dot" /> 数据状态：{overview.isError ? "离线" : "已连接"}</span>
        <span>数据延迟：runtime snapshot</span>
        <span>安全模式：No live orders</span>
        <span className="statusbar-right"><ListMagnifyingGlass size={14} /> 所有路径 repository-relative</span>
      </footer>
      <CommandPalette open={paletteOpen} initialQuery={search} onClose={closePalette} />
    </div>
  );
}
