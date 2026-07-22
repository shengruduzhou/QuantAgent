import { useMemo, useState } from "react";
import { DownloadSimple, MagnifyingGlass, Pause, Play } from "@phosphor-icons/react";

export function TrainingConsole({ lines, jobId }: { lines: string[]; jobId?: string }): JSX.Element {
  const [query, setQuery] = useState("");
  const [level, setLevel] = useState("all");
  const [paused, setPaused] = useState(false);
  const visible = useMemo(() => lines.filter((line) => {
    const text = line.toLowerCase();
    const queryMatches = !query || text.includes(query.toLowerCase());
    const levelMatches = level === "all" || text.includes(level);
    return queryMatches && levelMatches;
  }), [level, lines, query]);
  const download = (): void => {
    const blob = new Blob([visible.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${jobId ?? "training"}.log`;
    anchor.click();
    URL.revokeObjectURL(url);
  };
  return (
    <section className="vnext-training-console">
      <header><div><span>TRAINING CONSOLE</span><strong>{jobId ?? "No run selected"}</strong></div><label><MagnifyingGlass size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search log / traceback / step" /></label><select value={level} onChange={(event) => setLevel(event.target.value)}><option value="all">All levels</option><option value="warning">Warning</option><option value="error">Error</option><option value="exception">Exception</option></select><button type="button" onClick={() => setPaused((value) => !value)}>{paused ? <Play size={14} /> : <Pause size={14} />}{paused ? "Resume scroll" : "Pause scroll"}</button><button type="button" onClick={download} disabled={!visible.length}><DownloadSimple size={14} /> Download</button></header>
      <pre aria-live={paused ? "off" : "polite"}>{visible.length ? visible.join("\n") : "No persisted log lines match the current filter."}</pre>
    </section>
  );
}
