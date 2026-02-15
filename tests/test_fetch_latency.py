#!/usr/bin/env python3
"""
IC-R8600 fetch_iq() Latency vs Block Size

Measures average I/Q fetch latency at 480 kHz, 24-bit across block sizes
from 4096 to 65536 in 4096 increments. Each block size runs for 30 seconds.

Usage:
    ./test_fetch_latency.py
    ./test_fetch_latency.py --freq 98.1
    ./test_fetch_latency.py --duration 10       # Shorter runs for quick iteration
    ./test_fetch_latency.py --16bit             # Compare with 16-bit mode
"""

import argparse
import os
import sys
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from icom_r8600 import IcomR8600


def run_block_size_test(radio, block_size, duration_sec):
    """Run fetch_iq() at a given block size for duration_sec and return timing stats."""
    radio.flush_iq()
    time.sleep(0.2)

    # Warmup: 20 fetches to stabilize
    for _ in range(20):
        radio.fetch_iq(block_size)

    radio.flush_iq()
    time.sleep(0.1)

    fetch_times_ms = []
    start = time.time()

    while time.time() - start < duration_sec:
        t0 = time.perf_counter()
        radio.fetch_iq(block_size)
        dt_ms = (time.perf_counter() - t0) * 1000
        fetch_times_ms.append(dt_ms)

    return fetch_times_ms


def main():
    parser = argparse.ArgumentParser(description="IC-R8600 fetch_iq() Latency vs Block Size")
    parser.add_argument("--freq", type=float, default=89.9,
                        help="Frequency in MHz (default: 89.9)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration per block size in seconds (default: 30)")
    parser.add_argument("--16bit", dest="use_16bit", action="store_true",
                        help="Use 16-bit mode (default: 24-bit)")
    args = parser.parse_args()

    use_24bit = not args.use_16bit
    bit_label = "24-bit" if use_24bit else "16-bit"
    freq_hz = int(args.freq * 1e6)
    sample_rate = 480000

    block_sizes = list(range(4096, 65536 + 1, 4096))

    print(f"\nIC-R8600 fetch_iq() Latency Test")
    print(f"  Mode:        {bit_label}")
    print(f"  Sample rate: {sample_rate / 1e3:.0f} kHz")
    print(f"  Frequency:   {args.freq:.1f} MHz")
    print(f"  Duration:    {args.duration:.0f}s per block size")
    print(f"  Block sizes: {block_sizes[0]} - {block_sizes[-1]} (step 4096)")
    print(f"  Total time:  ~{len(block_sizes) * args.duration / 60:.0f} minutes")

    radio = IcomR8600(use_24bit=use_24bit)
    radio.open()
    radio.configure_iq_streaming(freq=freq_hz, sample_rate=sample_rate)
    time.sleep(1.0)

    results = []

    try:
        for block_size in block_sizes:
            expected_ms = block_size / sample_rate * 1000

            print(f"\n{'='*65}")
            print(f"  Block size: {block_size:>6}  "
                  f"(expected {expected_ms:.1f} ms at {sample_rate/1e3:.0f} kHz)")
            print(f"{'='*65}")

            fetch_times = run_block_size_test(radio, block_size, args.duration)

            avg = statistics.mean(fetch_times)
            med = statistics.median(fetch_times)
            sd = statistics.stdev(fetch_times) if len(fetch_times) > 1 else 0
            mn = min(fetch_times)
            mx = max(fetch_times)
            p95 = sorted(fetch_times)[int(len(fetch_times) * 0.95)]
            p99 = sorted(fetch_times)[int(len(fetch_times) * 0.99)]
            overhead = avg - expected_ms

            diag = radio.get_diagnostics()
            sync_misses = diag.get('sync_misses', 0)
            sample_loss = diag.get('total_sample_loss', 0)

            print(f"  Fetches:   {len(fetch_times)}")
            print(f"  Mean:      {avg:>8.2f} ms")
            print(f"  Median:    {med:>8.2f} ms")
            print(f"  Stdev:     {sd:>8.2f} ms")
            print(f"  Min:       {mn:>8.2f} ms")
            print(f"  Max:       {mx:>8.2f} ms")
            print(f"  P95:       {p95:>8.2f} ms")
            print(f"  P99:       {p99:>8.2f} ms")
            print(f"  Overhead:  {overhead:>+8.2f} ms (vs {expected_ms:.1f} ms ideal)")
            if sync_misses or sample_loss:
                print(f"  Sync miss: {sync_misses}  Sample loss: {sample_loss}")

            results.append({
                'block_size': block_size,
                'expected_ms': expected_ms,
                'count': len(fetch_times),
                'mean': avg,
                'median': med,
                'stdev': sd,
                'min': mn,
                'max': mx,
                'p95': p95,
                'p99': p99,
                'overhead': overhead,
            })

    finally:
        radio.close()

    # Summary table
    print(f"\n{'='*85}")
    print(f"  SUMMARY  ({bit_label}, {sample_rate/1e3:.0f} kHz, {args.duration:.0f}s/test)")
    print(f"{'='*85}")
    print(f"  {'Block':>7}  {'Expected':>8}  {'Mean':>8}  {'Median':>8}  "
          f"{'Stdev':>7}  {'P95':>8}  {'P99':>8}  {'Over':>8}  {'N':>6}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

    for r in results:
        print(f"  {r['block_size']:>7}  {r['expected_ms']:>7.1f}ms  "
              f"{r['mean']:>7.2f}ms  {r['median']:>7.2f}ms  "
              f"{r['stdev']:>6.2f}ms  {r['p95']:>7.2f}ms  "
              f"{r['p99']:>7.2f}ms  {r['overhead']:>+7.2f}ms  {r['count']:>6}")

    # Find sweet spot
    best = min(results, key=lambda r: r['overhead'])
    print(f"\n  Lowest overhead: block_size={best['block_size']} "
          f"({best['overhead']:+.2f} ms, mean={best['mean']:.2f} ms)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
