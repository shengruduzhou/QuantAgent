from __future__ import annotations

import argparse

from quantagent.data.features import add_benchmark_features, add_technical_features
from quantagent.data.io import read_frame, write_frame
from quantagent.data.labels import add_forward_return_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily alpha features and labels.")
    parser.add_argument("--prices", required=True, help="CSV or Parquet with OHLCV rows.")
    parser.add_argument("--benchmark", required=True, help="CSV or Parquet benchmark OHLCV.")
    parser.add_argument("--benchmark-symbol", default="000300.SH")
    parser.add_argument("--output", default="data/processed/daily_features.parquet")
    args = parser.parse_args()

    prices = read_frame(args.prices)
    benchmark = read_frame(args.benchmark)
    features = add_technical_features(prices)
    features = add_benchmark_features(features, benchmark, args.benchmark_symbol)
    features = add_forward_return_labels(features)
    write_frame(features, args.output)
    print(f"wrote {len(features):,} rows to {args.output}")


if __name__ == "__main__":
    main()
