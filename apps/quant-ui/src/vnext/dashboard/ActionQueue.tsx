import { ArrowRight, CheckCircle, Info, WarningCircle, XCircle } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import type { ActionQueueItem } from "./types";

const icons = {
  critical: XCircle,
  warning: WarningCircle,
  info: Info,
  success: CheckCircle,
};

export function ActionQueue({ items }: { items: ActionQueueItem[] }): JSX.Element {
  return (
    <section className="vnext-action-queue" aria-label="需要处理的事项">
      <header><div><span>NEEDS ATTENTION</span><h2>可执行队列</h2></div><strong>{items.filter((item) => item.severity !== "success").length}</strong></header>
      <div className="vnext-action-rows">
        {items.map((item) => {
          const Icon = icons[item.severity];
          return (
            <article key={item.id} className={`severity-${item.severity}`}>
              <Icon size={17} weight="duotone" />
              <div><strong>{item.entity}</strong><p>{item.reason}</p><small>{item.timestamp} · {item.source}</small></div>
              <Link to={item.path}>{item.action} <ArrowRight size={13} /></Link>
            </article>
          );
        })}
      </div>
    </section>
  );
}
