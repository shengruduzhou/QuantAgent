import { ArrowRight, Database, FileHtml, ShieldCheck, WarningCircle } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { StateView } from "../../components/StateView";
import { StatusBadge } from "../../components/StatusBadge";
import { useApi } from "../../hooks/useApi";

interface DataGroup {
  id: string;
  title: string;
  capabilities: string[];
  quantagent: string[];
}

interface BestPractice {
  id: string;
  slug: string;
  title: string;
  category: string;
  quantagent_path: string;
  endpoints: string[];
  outputs: string[];
  contract: string[];
  boundaries: string[];
}

interface BestPracticePayload {
  source: string;
  count: number;
  dataGroups: DataGroup[];
  items: BestPractice[];
  outputContract: {
    offlineHtml: boolean;
    showDataTime: boolean;
    showMode: boolean;
    showSourceEndpoint: boolean;
    showCalculationBasis: boolean;
    showNonInvestmentAdvice: boolean;
    browserApiKey: boolean;
    unavailableDataPolicy: string;
  };
}

const CATEGORY_LABELS: Record<string, string> = {
  market: "行情 / 横截面",
  financial: "财务 / PIT",
  special: "特色盘面",
  backtest: "严格回测",
};

export function FuyaoBestPracticesPage(): JSX.Element {
  const query = useApi<BestPracticePayload>(["fuyao-best-practices"], "/market/best-practices", undefined, { staleTime: 60 * 60_000 });
  const data = query.data?.data;

  if (query.isLoading) return <StateView state="loading" detail="加载 Fuyao 产品契约。" />;
  if (!data) return <StateView state="unavailable" title="Fuyao 产品契约不可用" detail="无法读取 /api/market/best-practices。" />;

  return (
    <div className="page institutional-workbench market-intelligence-page">
      <header className="mi-hero">
        <div>
          <div className="mi-eyebrow"><Database size={16} /> Fuyao / Financial-API · Best Practices Parity</div>
          <h1>Fuyao 全场景研究工场</h1>
          <p>不是示例链接清单。这里把 16 个官方最佳实践、六类数据产品、页面产物和不可跨越的研究边界注册成 QuantAgent 的可审计产品契约，并直接指向实际工作站。</p>
        </div>
        <div className="mi-hero-meta">
          <StatusBadge status="ready" label={`${data.count}/16 registered`} />
          <span>真实数据走服务端 Fuyao</span>
          <span>unavailable 不造 mock</span>
        </div>
      </header>

      <section className="mi-metric-grid">
        {data.dataGroups.map((group) => (
          <div className="mi-metric" key={group.id}>
            <span>{group.title}</span>
            <strong>{group.capabilities.length}</strong>
            <small>{group.capabilities.join(" · ")}</small>
          </div>
        ))}
      </section>

      <section className="mi-panel mi-span-2">
        <header><div><strong>页面 / 报告统一产物契约</strong><span>所有场景必须遵守，而不是每个页面各写一套口径</span></div><FileHtml size={18} /></header>
        <div className="capability-grid">
          <Contract label="离线单文件 HTML" ready={data.outputContract.offlineHtml} />
          <Contract label="显著数据时间" ready={data.outputContract.showDataTime} />
          <Contract label="真实 / 模拟模式" ready={data.outputContract.showMode} />
          <Contract label="来源 endpoint" ready={data.outputContract.showSourceEndpoint} />
          <Contract label="计算口径" ready={data.outputContract.showCalculationBasis} />
          <Contract label="非投资建议" ready={data.outputContract.showNonInvestmentAdvice} />
          <Contract label="浏览器不含 API Key" ready={!data.outputContract.browserApiKey} />
          <div><span>缺失数据</span><strong>{data.outputContract.unavailableDataPolicy}</strong></div>
        </div>
      </section>

      <section className="mi-coverage-grid">
        {data.items.map((item) => (
          <article className="mi-panel" key={item.id}>
            <header>
              <div><strong>{item.id} · {item.title}</strong><span>{CATEGORY_LABELS[item.category] ?? item.category}</span></div>
              <StatusBadge status="ready" label="contract registered" />
            </header>
            <div className="fuyao-contract-block">
              <b>必须输出</b><p>{item.outputs.join(" · ")}</p>
              <b>计算 / 执行契约</b><ul>{item.contract.map((line) => <li key={line}>{line}</li>)}</ul>
              <b>边界</b><ul>{item.boundaries.map((line) => <li key={line}><WarningCircle size={13} /> {line}</li>)}</ul>
              <details><summary>数据 endpoints ({item.endpoints.length})</summary><div className="mi-source-strip">{item.endpoints.map((endpoint) => <code key={endpoint}>{endpoint}</code>)}</div></details>
            </div>
            <Link className="primary-button" to={item.quantagent_path}>进入 QuantAgent 实际工作站 <ArrowRight size={14} /></Link>
          </article>
        ))}
      </section>

      <section className="mi-source-strip">
        <ShieldCheck size={16} /><strong>QuantAgent 额外治理：</strong>
        <span>PIT / available_at</span><span>purged walk-forward + embargo</span><span>explicit benchmark</span><span>PBO / DSR / SPA</span><span>final holdout</span><span>T+1 close-signal execution</span>
      </section>
    </div>
  );
}

function Contract({ label, ready }: { label: string; ready: boolean }): JSX.Element {
  return <div><span>{label}</span><StatusBadge status={ready ? "ready" : "blocked"} label={ready ? "required" : "blocked"} /></div>;
}
