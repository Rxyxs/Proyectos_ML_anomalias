"""Python reference used to validate the Go realized-volatility streamer.

This mirrors, in one non-incremental numpy expression, the same quantity that
go/streamer.go accumulates online: sqrt(sum(r_i^2)) over log-returns
r_i = ln(P_i / P_{i-1}). The two implementations compute the sum in a
different order (numpy's pairwise summation vs. Go's sequential running sum),
so streamer_test.go compares them with a small tolerance (1e-5) rather than
requiring bit-for-bit equality.

Run this script to regenerate golden.json:
    python go/testdata/reference_volatility.py
"""

import json
from pathlib import Path

import numpy as np


def realized_volatility(prices):
    prices = np.asarray(prices, dtype=np.float64)
    log_returns = np.diff(np.log(prices))
    return float(np.sqrt(np.sum(log_returns ** 2)))


def generate_prices(n=500, seed=7, start=100.0, sigma=0.01):
    rng = np.random.default_rng(seed)
    step_returns = rng.normal(loc=0.0, scale=sigma, size=n - 1)
    log_prices = np.concatenate([[0.0], np.cumsum(step_returns)])
    return (start * np.exp(log_prices)).tolist()


def main():
    prices = generate_prices()
    expected = realized_volatility(prices)
    payload = {"prices": prices, "expected_realized_volatility": expected}

    out_path = Path(__file__).parent / "golden.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path} (expected realized volatility = {expected!r})")


if __name__ == "__main__":
    main()
