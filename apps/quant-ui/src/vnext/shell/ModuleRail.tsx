import { useMemo, useState } from "react";
import { CaretLeft, CaretRight, Star } from "@phosphor-icons/react";
import type { JobSummary } from "../../api/types";
import { vnextModuleGroups, vnextModules, type VNextModule } from "../workspace/modules";

const FAVORITES_KEY = "quantagent.workstation.vnext.favorites";

function loadFavorites(): string[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(FAVORITES_KEY) ?? "[]") as unknown;
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

interface ModuleRailProps {
  expanded: boolean;
  activeModuleId: string;
  jobs: JobSummary[];
  riskEventCount: number;
  onToggle: () => void;
  openPath: (path: string) => void;
}

export function ModuleRail({ expanded, activeModuleId, jobs, riskEventCount, onToggle, openPath }: ModuleRailProps): JSX.Element {
  const [favorites, setFavorites] = useState(loadFavorites);
  const activeJobs = jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.status));
  const favoriteModules = useMemo(() => favorites.map((id) => vnextModules.find((module) => module.id === id)).filter((module): module is VNextModule => Boolean(module)), [favorites]);

  const toggleFavorite = (moduleId: string): void => {
    const next = favorites.includes(moduleId) ? favorites.filter((id) => id !== moduleId) : [...favorites, moduleId];
    setFavorites(next);
    try {
      window.localStorage.setItem(FAVORITES_KEY, JSON.stringify(next));
    } catch {
      // Favorites remain available for this session.
    }
  };

  const badgeFor = (module: VNextModule): number => {
    if (module.id === "training") return activeJobs.filter((job) => job.type === "train").length;
    if (module.id === "backtest") return activeJobs.filter((job) => job.type === "backtest").length;
    if (module.id === "tasks") return activeJobs.length;
    if (module.id === "risk") return riskEventCount;
    return 0;
  };

  const renderModule = (module: VNextModule, favoriteContext = false): JSX.Element => {
    const Icon = module.icon;
    const badge = badgeFor(module);
    return (
      <div className={`vnext-module-row ${activeModuleId === module.id ? "active" : ""}`} key={`${favoriteContext ? "favorite" : "module"}-${module.id}`}>
        <button type="button" className="vnext-module-link" onClick={() => openPath(module.path)} title={`${module.label} · ${module.caption}`}>
          <span className="vnext-module-icon"><Icon size={19} weight="duotone" />{badge ? <em>{badge > 99 ? "99+" : badge}</em> : null}</span>
          {expanded ? <span><strong>{module.label}</strong><small>{module.caption}</small></span> : null}
        </button>
        {expanded && !favoriteContext ? (
          <button type="button" className={favorites.includes(module.id) ? "vnext-favorite active" : "vnext-favorite"} onClick={() => toggleFavorite(module.id)} aria-label={`${favorites.includes(module.id) ? "取消收藏" : "收藏"} ${module.label}`}>
            <Star size={13} weight={favorites.includes(module.id) ? "fill" : "regular"} />
          </button>
        ) : null}
      </div>
    );
  };

  return (
    <aside className="vnext-module-rail" aria-label="QuantAgent 模块栏">
      <div className="vnext-rail-brand"><span>QA</span>{expanded ? <strong>Institutional<br />Workstation</strong> : null}</div>
      <nav>
        {favoriteModules.length ? <section><h2>{expanded ? "FAVORITES" : "★"}</h2>{favoriteModules.map((module) => renderModule(module, true))}</section> : null}
        {vnextModuleGroups.map((group) => (
          <section key={group.id}>
            <h2>{expanded ? group.label : group.label.slice(0, 1)}</h2>
            {vnextModules.filter((module) => module.group === group.id).map((module) => renderModule(module))}
          </section>
        ))}
      </nav>
      <footer>
        <button type="button" onClick={onToggle} aria-label={expanded ? "收起模块栏" : "展开模块栏"}>
          {expanded ? <CaretLeft size={16} /> : <CaretRight size={16} />}
          {expanded ? <span>Collapse rail</span> : null}
        </button>
        {expanded ? <small>RESEARCH / PAPER ONLY</small> : null}
      </footer>
    </aside>
  );
}
