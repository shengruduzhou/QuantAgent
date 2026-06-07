#!/usr/bin/env python3
"""批量生成产业链池 (date × news_mode) — 供混合实验/OOS消融/并集回测复用.

并发受限地调用 scripts/industry_chain_research.py。已存在的产出默认跳过 (--force 重生成)。
LLM 慢且非确定 → 默认并发 2，避免端点限流。
"""
from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "AI_quant_venv/bin/python3"
MON = Path("runtime/reports/monthly")
SFX = {"real": "", "nonews": "_nonews", "scrambled": "_scrambled"}


def run_one(d, mode, enrich, n_chains, sps, force):
    sfx = SFX[mode]
    out = MON / f"chain_pool_{d}{sfx}.parquet"
    if out.exists() and not force:
        return d, mode, "skip(exists)"
    pf = f"runtime/tmp/real_preds_{d.replace('-', '')}.parquet"
    log = MON / f"chain_gen_{d}{sfx}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(
            [PY, "scripts/industry_chain_research.py", "--as-of", d, "--predictions", pf,
             "--news-mode", mode, "--n-chains", str(n_chains), "--stocks-per-segment", str(sps),
             "--max-stocks-enrich", str(enrich)],
            stdout=fh, stderr=subprocess.STDOUT).returncode
    ok = "ok" if out.exists() else f"FAIL(rc={rc})"
    return d, mode, f"{ok} {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", default=["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"])
    ap.add_argument("--modes", nargs="+", default=["real"])
    ap.add_argument("--enrich", type=int, default=24)
    ap.add_argument("--n-chains", type=int, default=6)
    ap.add_argument("--stocks-per-segment", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(d, m) for d in args.dates for m in args.modes]
    print(f"generating {len(jobs)} pools (concurrency={args.concurrency}): "
          + ", ".join(f"{d}/{m}" for d, m in jobs))
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, d, m, args.enrich, args.n_chains, args.stocks_per_segment, args.force)
                for d, m in jobs]
        for f in futs:
            d, m, status = f.result()
            print(f"  [{d}/{m}] {status}", flush=True)


if __name__ == "__main__":
    main()
