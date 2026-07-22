import {
  ArrowRight,
  BookOpenText,
  Broom,
  ChartLine,
  Database,
  Flask,
  Keyboard,
  ListChecks,
  Play,
  ShieldCheck,
} from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { Panel } from "../components/Panel";

const shortcuts = [
  ["⌘/Ctrl + K", "打开全局模块与命令搜索"],
  ["K 线 ← / →", "按 5 根 K 线移动可见窗口"],
  ["K 线 ↑ / ↓", "缩小 / 扩大时间窗口"],
  ["K 线 Home / End", "显示全部 / 跳到最新"],
  ["表格 ↑ / ↓", "移动当前行；Enter 打开或选择"],
];

const workflows = [
  {
    title: "研究实验",
    detail: "一次选择一个实验作为当前上下文，查看真实 NAV、指标和 artifact 能力。",
    path: "/backtests",
    icon: Flask,
  },
  {
    title: "全宇宙训练",
    detail: "进入受控任务启动器；不设置 symbols 或 symbols_file 时使用数据集中的全部股票。",
    path: "/settings?job=train&universe=all",
    icon: Play,
  },
  {
    title: "数据与删除",
    detail: "在 Runtime Cleanup 中只删除后端批准的候选，并保留审计记录与受保护路径。",
    path: "/runtime?view=cleanup",
    icon: Broom,
  },
  {
    title: "能力补全",
    detail: "查看未审计、缺失、规划中和部分实现的 vn.py 能力，以及当前差距和下一步。",
    path: "/parity",
    icon: ListChecks,
  },
];

export function HelpCenterPage(): JSX.Element {
  return (
    <div className="page help-center-page">
      <section className="help-hero">
        <div>
          <span className="page-kicker">QUANTAGENT WORKSTATION GUIDE</span>
          <h1>帮助中心</h1>
          <p>这里是 QuantAgent 的操作说明、快捷键、任务入口和安全边界。官方 VeighNa 文档仅作为设计来源，不再替代产品内帮助。</p>
        </div>
        <div className="help-safety">
          <ShieldCheck size={22} weight="duotone" />
          <span><strong>RESEARCH / PAPER ONLY</strong><small>网页任务不能绕过 allowlist、RiskGate 或 Runtime 路径策略。</small></span>
        </div>
      </section>

      <section className="help-workflow-grid">
        {workflows.map(({ title, detail, path, icon: Icon }) => (
          <Link key={path} to={path} className="help-workflow-card">
            <Icon size={21} weight="duotone" />
            <span><strong>{title}</strong><small>{detail}</small></span>
            <ArrowRight size={15} />
          </Link>
        ))}
      </section>

      <section className="help-main-grid">
        <Panel title="快捷操作" eyebrow="Keyboard and pointer contract" className="help-shortcut-panel">
          <div className="help-shortcuts">
            {shortcuts.map(([key, description]) => (
              <div key={key}><kbd>{key}</kbd><span>{description}</span></div>
            ))}
          </div>
        </Panel>

        <Panel title="K 线操作" eyebrow="Human-scale chart interaction" className="help-chart-panel">
          <div className="help-guide-list">
            <div><ChartLine size={17} /><span><strong>滚轮</strong><small>只负责以鼠标位置为中心缩放，不同时平移。</small></span></div>
            <div><Keyboard size={17} /><span><strong>左键拖拽</strong><small>平移时间窗口；不会改变缩放比例。</small></span></div>
            <div><BookOpenText size={17} /><span><strong>信号与详情</strong><small>点击买卖、做 T 或风控标记，联动当前交易记录。</small></span></div>
          </div>
        </Panel>

        <Panel title="数据操作边界" eyebrow="DataManager-style workflow" className="help-data-panel">
          <div className="help-guide-list">
            <div><Database size={17} /><span><strong>Catalog</strong><small>查看真实 artifact、manifest、时间范围、质量和血缘。</small></span></div>
            <div><Broom size={17} /><span><strong>Cleanup</strong><small>删除前重新校验候选；canonical 数据、manifest 和主模型注册表受保护。</small></span></div>
            <div><ShieldCheck size={17} /><span><strong>训练输出</strong><small>所有网页训练输出必须位于 runtime，不能写入任意系统路径。</small></span></div>
          </div>
        </Panel>

        <Panel title="设计来源" eyebrow="Internal documentation only" className="help-reference-panel">
          <div className="help-guide-list">
            <div><BookOpenText size={17} /><span><strong>VeighNa / vn.py 对齐</strong><small>DataManager、DataRecorder、事件引擎和交易对象只作为实现依据；帮助入口始终停留在 QuantAgent 内。</small></span></div>
            <div><ListChecks size={17} /><span><strong>能力状态</strong><small>已实现、部分实现、未审计和缺失均以产品内 Capability Registry 的机器可读状态为准。</small></span></div>
          </div>
          <p>本页没有外部跳转。需要核对实现来源时，由开发审计流程记录具体版本与代码位置。</p>
        </Panel>
      </section>
    </div>
  );
}
