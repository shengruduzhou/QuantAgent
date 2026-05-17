#!/usr/bin/env python3
"""Fetch qlib CN data into /home/shanhefu/QuantAgent/data/raw/qlib/cn_data/.

Calls qlib.tests.data.GetData.qlib_data which pulls zipped archives from
the SunsetWolf/qlib_dataset GitHub releases mirror.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--interval", default="1d", choices=["1d", "1min"])
    parser.add_argument("--region", default="cn")
    parser.add_argument("--version", default=None)
    parser.add_argument("--delete-zip", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    from qlib.tests.data import GetData

    print(f"[fetch] target={target} interval={args.interval} region={args.region}", flush=True)
    GetData(delete_zip_file=args.delete_zip).qlib_data(
        target_dir=str(target),
        region=args.region,
        interval=args.interval,
        version=args.version,
        exists_skip=True,
    )
    print(f"[fetch] DONE target={target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
