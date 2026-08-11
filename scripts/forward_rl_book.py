#!/usr/bin/env python3
"""Fail-closed guard for the retired RL forward-target exporter.

The historical implementation is intentionally disabled. It built a delayed
hold-band book and then passed that book through an environment whose reward
clock was itself misaligned with strict T->T+1 execution. It could therefore
emit a stale/right-censored research date as if it were a current forward
signal and could reuse a policy trained under the superseded reward contract.

Use ``scripts/rl_pit_train_eval.py`` for research under the audited
T(signal)->T+1(execution)->T+2(reward) clock. A separate current-signal inference
state machine, policy-contract check, and fresh forward-shadow acceptance are
required before RL target export can be reintroduced.
"""

from __future__ import annotations

import argparse
import sys


_BLOCK_REASON = (
    "RL_FORWARD_BLOCKED: the legacy forward exporter used a superseded reward/"
    "execution clock. No RL target is emitted. Retrain and validate with "
    "scripts/rl_pit_train_eval.py; current-signal inference remains a separate "
    "audited implementation task."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-start", default=None)
    parser.add_argument("--warmup-start", default=None)
    parser.add_argument("--policy", default=None)
    parser.parse_args()
    print(_BLOCK_REASON, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
