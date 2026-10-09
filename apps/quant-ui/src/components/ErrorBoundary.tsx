import { ArrowCounterClockwise, Bug, WarningOctagon } from "@phosphor-icons/react";
import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * React 19 unmounts the whole root on an uncaught render error, so one
 * malformed artifact used to blank the entire workstation — including the
 * LIVE / KILL / RISK safety chrome. Two boundaries now contain that:
 *
 *  - `scope="workspace"` wraps each workspace pane's routes inside the shell,
 *    so a page crash leaves the command bar, module rail and dock intact and
 *    shows a diagnosis where the page was. It resets when the route changes.
 *  - `scope="root"` wraps the shell itself as the last line.
 *
 * The panel states what failed and what did not; it never claims the data is
 * fine or guesses at a cause.
 */

interface ErrorBoundaryProps {
  scope: "workspace" | "root";
  /** Changing this value (e.g. the route path) clears a caught error. */
  resetKey?: string;
  /** Where the failure happened, for the diagnosis (e.g. "/risk"). */
  location?: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
  componentStack: string | null;
}

const THEME_KEY = "quantagent.workstation.vnext.v2";

function storedTheme(): string {
  try {
    const raw = window.localStorage.getItem(THEME_KEY);
    const theme = raw ? (JSON.parse(raw) as { theme?: unknown }).theme : null;
    return theme === "dawn" || theme === "day" ? theme : "night";
  } catch {
    return "night";
  }
}

function firstFrames(stack: string | null | undefined, count = 6): string {
  if (!stack) return "component stack unavailable";
  return stack.split("\n").map((line) => line.trim()).filter(Boolean).slice(0, count).join("\n");
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null, componentStack: null };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ componentStack: info.componentStack ?? null });
    // Keep the original error visible to developers; the panel is for operators.
    console.error(`[QuantAgent ${this.props.scope} boundary]`, error, info.componentStack);
  }

  componentDidUpdate(previous: ErrorBoundaryProps): void {
    if (this.state.error && previous.resetKey !== this.props.resetKey) {
      this.setState({ error: null, componentStack: null });
    }
  }

  private retry = (): void => {
    this.setState({ error: null, componentStack: null });
  };

  render(): ReactNode {
    const { error, componentStack } = this.state;
    if (!error) return this.props.children;
    const workspace = this.props.scope === "workspace";
    const panel = (
      <section className="atlas-surface error-boundary-panel" data-rail="danger" role="alert" aria-label={workspace ? "工作区渲染失败" : "工作站渲染失败"}>
        <header>
          <WarningOctagon size={26} weight="duotone" aria-hidden="true" />
          <div>
            <span className="atlas-eyebrow">{workspace ? "WORKSPACE RENDER FAILURE · CONTAINED" : "WORKSTATION RENDER FAILURE"}</span>
            <h2>{workspace ? "此工作区无法渲染" : "工作站外壳无法渲染"}</h2>
            <p>
              {workspace
                ? "只有这个页面失败了：命令栏、LIVE / KILL / RISK 安全状态、模块栏与任务坞仍然可用。"
                : "浏览器端渲染失败。服务端的 paper/live 策略、风控与账本不依赖浏览器，不受此错误影响。"}
              {" "}这是界面错误，不代表数据为空或健康；在修复前不要依据此页做判断。
            </p>
          </div>
        </header>
        <dl className="error-boundary-facts">
          <div><dt>Error</dt><dd className="mono">{error.name}: {error.message || "(no message)"}</dd></div>
          {this.props.location ? <div><dt>Route</dt><dd className="mono">{this.props.location}</dd></div> : null}
          <div><dt>Likely source</dt><dd>页面读取的某个 artifact 字段形状与界面预期不符（例如缺字段或类型变化）；请对照该页面调用的 /api 响应。</dd></div>
        </dl>
        <details className="error-boundary-stack">
          <summary><Bug size={13} /> Component stack</summary>
          <pre>{firstFrames(componentStack)}</pre>
        </details>
        <div className="atlas-row">
          <button type="button" className="atlas-action" data-variant="primary" onClick={workspace ? this.retry : () => window.location.reload()}>
            <ArrowCounterClockwise size={13} />{workspace ? "重试渲染" : "重新加载工作站"}
          </button>
          {workspace ? <span className="error-boundary-hint">或从左侧模块栏切换到其他工作区；切换路由会自动清除此错误。</span> : null}
        </div>
      </section>
    );
    if (workspace) return <div className="error-boundary-workspace">{panel}</div>;
    const theme = storedTheme();
    return <div className={`vnext-shell theme-${theme} error-boundary-root`} data-theme={theme}>{panel}</div>;
  }
}
