import { Check, CloudSun, MoonStars, SunHorizon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import type { WorkspaceTheme } from "../workspace/types";

const themeOptions: Array<{
  id: WorkspaceTheme;
  label: string;
  detail: string;
  icon: typeof MoonStars;
}> = [
  { id: "night", label: "深空", detail: "低照度交易与训练监控", icon: MoonStars },
  { id: "dawn", label: "曙光", detail: "中等亮度的混合工作台", icon: CloudSun },
  { id: "day", label: "日间", detail: "高环境光下的清晰阅读", icon: SunHorizon },
];

export function ThemeSwitcher({ theme, onChange }: { theme: WorkspaceTheme; onChange: (theme: WorkspaceTheme) => void }): JSX.Element {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const active = themeOptions.find((option) => option.id === theme) ?? themeOptions[0];
  const ActiveIcon = active.icon;

  useEffect(() => {
    if (!open) return undefined;
    const close = (event: PointerEvent): void => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent): void => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", escape);
    };
  }, [open]);

  return (
    <div className="vnext-theme-switcher" ref={rootRef}>
      <button
        type="button"
        className="vnext-icon-button"
        aria-label="切换界面主题"
        aria-haspopup="menu"
        aria-expanded={open}
        title={`Theme: ${active.label}`}
        onClick={() => setOpen((value) => !value)}
      >
        <ActiveIcon size={18} weight="duotone" />
      </button>
      {open ? (
        <div className="vnext-theme-menu" role="menu" aria-label="界面主题">
          <header><span>VISUAL MODE</span><strong>工作台主题</strong></header>
          {themeOptions.map((option) => {
            const Icon = option.icon;
            return (
              <button
                type="button"
                key={option.id}
                role="menuitemradio"
                aria-checked={theme === option.id}
                className={theme === option.id ? "active" : ""}
                onClick={() => {
                  onChange(option.id);
                  setOpen(false);
                }}
              >
                <Icon size={19} weight="duotone" />
                <span><strong>{option.label}</strong><small>{option.detail}</small></span>
                {theme === option.id ? <Check size={15} weight="bold" /> : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
