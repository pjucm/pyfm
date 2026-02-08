#!/usr/bin/env python3
"""
Rate controller simulation for panadapter audio clock drift compensation.

Models the bursty audio delivery pattern in the panadapter's Qt event loop:
  - IQ source produces audio blocks at IQ clock rate (~8.5ms per block)
  - Qt event loop batches them into bursts every ~100ms
  - Audio callback drains buffer steadily at 48kHz

Tests two approaches:
  1. PI controller with heavy EMA filtering
  2. Slope estimator using linear regression on buffer level

Usage:
    python3 pi_tuner_panadapter.py
"""

import numpy as np
import itertools
import time as time_mod
from collections import deque

# ─── System constants (from panadapter.py) ─────────────────────────
AUDIO_RATE = 48000           # Audio DAC sample rate (Hz)
BUF_SIZE = 57600             # Ring buffer: 4 * latency * rate
PREFILL = 14400              # Prefill: latency * rate (300ms)
CALLBACK_SIZE = 1024         # sounddevice blocksize
AUDIO_BLOCK = 410            # Samples per demod output block

# IQ fetch interval: 8192 IQ samples at 960kHz = 8.53ms
IQ_FETCH_S = 8192 / 960000   # 0.00853s

# ─── Burst model (from log analysis) ──────────────────────────────
BURST_PERIOD_S = 0.100       # Mean time between burst processing
BURST_JITTER_S = 0.015       # ±jitter in burst timing


def simulate_pi(Kp, Ki, alpha, target_ms, drift_ppm,
                duration_s=300, seed=42, integrator_max=0.002,
                log_path=None):
    """
    Simulate PI controller on buffer level.

    PI fires once per burst (after first block write), matching the real
    panadapter's on_audio_ready() behavior.
    """
    rng = np.random.RandomState(seed)
    dt = 0.0001  # 0.1ms time steps
    steps = int(duration_s / dt)

    buf = PREFILL
    iq_interval = IQ_FETCH_S / (1 + drift_ppm * 1e-6)
    next_iq = iq_interval
    qt_queue = []
    next_burst = BURST_PERIOD_S + rng.uniform(-BURST_JITTER_S, BURST_JITTER_S)

    callback_interval = CALLBACK_SIZE / AUDIO_RATE
    next_callback = callback_interval

    rate_adj = 1.0
    integrator = 0.0
    filtered_err = 0.0

    total_drops = 0
    total_underruns = 0

    log_interval = 0.1
    next_log = log_interval
    log_data = []

    for step in range(steps):
        t = step * dt

        # 1. IQ production → queue audio block
        if t >= next_iq:
            adjusted = int(round(AUDIO_BLOCK * rate_adj))
            qt_queue.append(adjusted)
            next_iq += iq_interval

        # 2. Qt burst processing — PI fires once, after first block write
        if t >= next_burst and qt_queue:
            pi_done = False
            for block_samples in qt_queue:
                space = BUF_SIZE - buf - 1
                if block_samples > space:
                    total_drops += block_samples - space
                    block_samples = space
                if block_samples > 0:
                    buf += block_samples

                if not pi_done:
                    pi_done = True
                    buf_ms = buf / AUDIO_RATE * 1000
                    err = buf_ms - target_ms
                    filtered_err = alpha * err + (1 - alpha) * filtered_err

                    p = filtered_err * Kp
                    integrator += filtered_err * Ki
                    integrator = max(-integrator_max, min(integrator_max, integrator))
                    rate_adj = 1.0 - (p + integrator)
                    rate_adj = max(0.98, min(1.02, rate_adj))

            qt_queue.clear()
            jitter = rng.uniform(-BURST_JITTER_S, BURST_JITTER_S)
            next_burst = t + BURST_PERIOD_S + jitter

        # 3. Audio callback
        if t >= next_callback:
            if buf >= CALLBACK_SIZE:
                buf -= CALLBACK_SIZE
            else:
                total_underruns += 1
                buf = 0
            next_callback += callback_interval

        # 4. Logging
        if t >= next_log:
            buf_ms = buf / AUDIO_RATE * 1000
            adj_ppm = (rate_adj - 1.0) * 1e6
            int_ppm = integrator * 1e6
            log_data.append((t, buf_ms, adj_ppm, int_ppm))
            next_log += log_interval

    return _analyze(log_data, drift_ppm, total_drops, total_underruns,
                    integrator_max, log_path,
                    extra={'Kp': Kp, 'Ki': Ki, 'alpha': alpha})


def simulate_slope(window_s, update_s, ema_alpha, target_ms, drift_ppm,
                   duration_s=300, seed=42, log_path=None):
    """
    Simulate slope-estimator rate controller.

    Measures buffer level at each burst, fits a line over the last window_s
    seconds, and uses the slope to estimate drift. Updates rate_adj every
    update_s seconds with EMA smoothing.
    """
    rng = np.random.RandomState(seed)
    dt = 0.0001
    steps = int(duration_s / dt)

    buf = PREFILL
    iq_interval = IQ_FETCH_S / (1 + drift_ppm * 1e-6)
    next_iq = iq_interval
    qt_queue = []
    next_burst = BURST_PERIOD_S + rng.uniform(-BURST_JITTER_S, BURST_JITTER_S)

    callback_interval = CALLBACK_SIZE / AUDIO_RATE
    next_callback = callback_interval

    rate_adj = 1.0

    # Slope estimator state
    buf_history = deque()  # (time, buf_ms) measurements
    next_update = window_s + 5.0  # Wait for window to fill + warm-up

    total_drops = 0
    total_underruns = 0

    log_interval = 0.1
    next_log = log_interval
    log_data = []

    for step in range(steps):
        t = step * dt

        # 1. IQ production
        if t >= next_iq:
            adjusted = int(round(AUDIO_BLOCK * rate_adj))
            qt_queue.append(adjusted)
            next_iq += iq_interval

        # 2. Qt burst processing — record buffer level at each burst
        if t >= next_burst and qt_queue:
            for block_samples in qt_queue:
                space = BUF_SIZE - buf - 1
                if block_samples > space:
                    total_drops += block_samples - space
                    block_samples = space
                if block_samples > 0:
                    buf += block_samples
            qt_queue.clear()

            # Record measurement (after burst write, before next drain)
            buf_ms = buf / AUDIO_RATE * 1000
            buf_history.append((t, buf_ms))

            # Trim old measurements
            cutoff = t - window_s - 1.0
            while buf_history and buf_history[0][0] < cutoff:
                buf_history.popleft()

            jitter = rng.uniform(-BURST_JITTER_S, BURST_JITTER_S)
            next_burst = t + BURST_PERIOD_S + jitter

        # 3. Slope update
        if t >= next_update and len(buf_history) >= 20:
            next_update = t + update_s

            # Fit line to buffer level over window
            pts = np.array(buf_history)
            # Only use points within the window
            mask = pts[:, 0] >= (t - window_s)
            pts = pts[mask]

            if len(pts) >= 20:
                times = pts[:, 0] - pts[0, 0]  # Relative time
                levels = pts[:, 1]

                # Least-squares slope
                n = len(times)
                sum_t = np.sum(times)
                sum_l = np.sum(levels)
                sum_tl = np.sum(times * levels)
                sum_t2 = np.sum(times ** 2)
                denom = n * sum_t2 - sum_t ** 2
                if denom > 0:
                    slope_ms_per_s = (n * sum_tl - sum_t * sum_l) / denom

                    # slope in ms/s → drift in fractional rate
                    # If buffer grows at S ms/s, production exceeds consumption
                    # by S/1000 * AUDIO_RATE samples/sec
                    drift_frac = slope_ms_per_s / 1000.0

                    # Compute new rate_adj: reduce production by drift fraction
                    new_adj = rate_adj - drift_frac
                    new_adj = max(0.98, min(1.02, new_adj))

                    # EMA smooth the update
                    rate_adj = ema_alpha * new_adj + (1 - ema_alpha) * rate_adj

        # 4. Audio callback
        if t >= next_callback:
            if buf >= CALLBACK_SIZE:
                buf -= CALLBACK_SIZE
            else:
                total_underruns += 1
                buf = 0
            next_callback += callback_interval

        # 5. Logging
        if t >= next_log:
            buf_ms = buf / AUDIO_RATE * 1000
            adj_ppm = (rate_adj - 1.0) * 1e6
            log_data.append((t, buf_ms, adj_ppm, 0.0))
            next_log += log_interval

    return _analyze(log_data, drift_ppm, total_drops, total_underruns,
                    0.02, log_path,
                    extra={'window_s': window_s, 'update_s': update_s,
                           'ema_alpha': ema_alpha})


def _analyze(log_data, drift_ppm, total_drops, total_underruns,
             integrator_max, log_path, extra=None):
    """Analyze simulation results."""
    log_arr = np.array(log_data)

    # Use last 40% for steady-state (more conservative than 30%)
    ss_start = int(len(log_arr) * 0.6)
    ss = log_arr[ss_start:]

    adj_ss = ss[:, 2]  # ppm
    buf_ss = ss[:, 1]  # ms

    adj_range = np.ptp(adj_ss)
    int_max_ppm = integrator_max * 1e6
    hit_limits = np.any(np.abs(ss[:, 3]) >= int_max_ppm * 0.99)

    result = {
        'drift_ppm': drift_ppm,
        'adj_mean': np.mean(adj_ss),
        'adj_std': np.std(adj_ss),
        'adj_range': adj_range,
        'buf_mean': np.mean(buf_ss),
        'buf_std': np.std(buf_ss),
        'hit_limits': hit_limits,
        'drops': total_drops,
        'underruns': total_underruns,
        'converged': adj_range < 200 and not hit_limits and total_drops == 0 and total_underruns == 0,
    }
    if extra:
        result.update(extra)

    if log_path:
        with open(log_path, 'w') as f:
            f.write('time_s,buf_ms,adj_ppm,int_ppm\n')
            for row in log_arr:
                f.write(f'{row[0]:.2f},{row[1]:.1f},{row[2]:.1f},{row[3]:.1f}\n')

    return result


# ─── PI Parameter Sweep ─────────────────────────────────────────────

def sweep_pi():
    """Sweep PI parameters with improved ranges."""
    print("=" * 78)
    print("APPROACH 1: PI Controller with Heavy EMA")
    print("=" * 78)

    # Key insights:
    # - PI fires once per burst (~10 Hz), not 60 Hz
    # - Burst jitter causes ±15ms measurement noise
    # - EMA needs to be very heavy (alpha < 0.005) to filter burst noise
    # - Ki must be small enough to avoid integrator overshoot
    # - Need to handle drift from 50 to 1000+ ppm

    Kp_vals = [0.000005, 0.00001, 0.00002, 0.00005]
    Ki_vals = [0.000000005, 0.00000001, 0.00000002, 0.00000005]
    alpha_vals = [0.001, 0.002, 0.005]
    target_ms = 300  # prefill level
    drift_test = 200  # Test at moderate drift

    combos = list(itertools.product(Kp_vals, Ki_vals, alpha_vals))
    total = len(combos)

    print(f"\nSweeping {total} combinations at {drift_test} ppm drift (300s)")
    print(f"  Kp: {Kp_vals}")
    print(f"  Ki: {Ki_vals}")
    print(f"  alpha: {alpha_vals}")
    print()

    results = []
    t0 = time_mod.monotonic()

    for i, (Kp, Ki, alpha) in enumerate(combos):
        if (i + 1) % 10 == 0:
            elapsed = time_mod.monotonic() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

        r = simulate_pi(Kp, Ki, alpha, target_ms, drift_test, duration_s=300)
        results.append(r)

    elapsed = time_mod.monotonic() - t0
    print(f"  Sweep complete in {elapsed:.0f}s")

    converged = [r for r in results if r['converged']]
    print(f"\n─── Results: {len(converged)}/{len(results)} converged ───")

    if not converged:
        results.sort(key=lambda r: r['adj_range'])
        print("No configurations converged. Best attempts:")
        print(f"{'Kp':>12} {'Ki':>14} {'alpha':>7} | "
              f"{'adj_mean':>9} {'adj_std':>8} {'adj_rng':>8} {'buf_ms':>7} {'drops':>6} {'lim':>4}")
        print("-" * 85)
        for r in results[:10]:
            lim = 'YES' if r['hit_limits'] else 'no'
            print(f"{r['Kp']:>12.9f} {r['Ki']:>14.11f} {r['alpha']:>7.3f} | "
                  f"{r['adj_mean']:>+9.1f} {r['adj_std']:>8.1f} {r['adj_range']:>8.1f} "
                  f"{r['buf_mean']:>7.1f} {r['drops']:>6} {lim:>4}")
        return None

    converged.sort(key=lambda r: r['adj_std'])
    print(f"{'Kp':>12} {'Ki':>14} {'alpha':>7} | "
          f"{'adj_mean':>9} {'adj_std':>8} {'adj_rng':>8} {'buf_ms':>7}")
    print("-" * 80)
    for r in converged[:10]:
        print(f"{r['Kp']:>12.9f} {r['Ki']:>14.11f} {r['alpha']:>7.3f} | "
              f"{r['adj_mean']:>+9.1f} {r['adj_std']:>8.1f} {r['adj_range']:>8.1f} "
              f"{r['buf_mean']:>7.1f}")

    return converged


def sweep_slope():
    """Sweep slope estimator parameters."""
    print()
    print("=" * 78)
    print("APPROACH 2: Slope Estimator (Linear Regression)")
    print("=" * 78)

    window_vals = [10, 20, 30, 60]
    update_vals = [2, 5, 10]
    ema_vals = [0.2, 0.5, 0.8, 1.0]
    target_ms = 300
    drift_test = 200

    combos = list(itertools.product(window_vals, update_vals, ema_vals))
    total = len(combos)

    print(f"\nSweeping {total} combinations at {drift_test} ppm drift (300s)")
    print(f"  window_s: {window_vals}")
    print(f"  update_s: {update_vals}")
    print(f"  ema_alpha: {ema_vals}")
    print()

    results = []
    t0 = time_mod.monotonic()

    for i, (win, upd, ema) in enumerate(combos):
        if (i + 1) % 10 == 0:
            elapsed = time_mod.monotonic() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

        r = simulate_slope(win, upd, ema, target_ms, drift_test, duration_s=300)
        results.append(r)

    elapsed = time_mod.monotonic() - t0
    print(f"  Sweep complete in {elapsed:.0f}s")

    converged = [r for r in results if r['converged']]
    print(f"\n─── Results: {len(converged)}/{len(results)} converged ───")

    if not converged:
        results.sort(key=lambda r: r['adj_range'])
        print("No configurations converged. Best attempts:")
        print(f"{'win':>6} {'upd':>6} {'ema':>6} | "
              f"{'adj_mean':>9} {'adj_std':>8} {'adj_rng':>8} {'buf_ms':>7} {'drops':>6}")
        print("-" * 70)
        for r in results[:10]:
            print(f"{r['window_s']:>6} {r['update_s']:>6} {r['ema_alpha']:>6.2f} | "
                  f"{r['adj_mean']:>+9.1f} {r['adj_std']:>8.1f} {r['adj_range']:>8.1f} "
                  f"{r['buf_mean']:>7.1f} {r['drops']:>6}")
        return None

    converged.sort(key=lambda r: r['adj_std'])
    print(f"{'win':>6} {'upd':>6} {'ema':>6} | "
          f"{'adj_mean':>9} {'adj_std':>8} {'adj_rng':>8} {'buf_ms':>7}")
    print("-" * 65)
    for r in converged[:10]:
        print(f"{r['window_s']:>6} {r['update_s']:>6} {r['ema_alpha']:>6.2f} | "
              f"{r['adj_mean']:>+9.1f} {r['adj_std']:>8.1f} {r['adj_range']:>8.1f} "
              f"{r['buf_mean']:>7.1f}")

    return converged


def validate(configs, approach, n=3):
    """Validate top N configurations across drift values."""
    print()
    print("=" * 78)
    print(f"Validating top {n} {approach} configurations (600s, multiple drift values)")
    print("=" * 78)

    drift_vals = [-200, -100, -50, 0, 50, 100, 200, 500, 1000]

    for rank, cfg in enumerate(configs[:n]):
        if approach == 'PI':
            Kp, Ki, alpha = cfg['Kp'], cfg['Ki'], cfg['alpha']
            label = f"Kp={Kp:.9f}, Ki={Ki:.11f}, alpha={alpha:.3f}"
        else:
            win, upd, ema = cfg['window_s'], cfg['update_s'], cfg['ema_alpha']
            label = f"window={win}s, update={upd}s, ema={ema:.2f}"

        print(f"\n── #{rank+1}: {label} ──")
        print(f"  {'drift':>6} | {'adj_mean':>9} {'adj_std':>8} {'adj_rng':>8} "
              f"{'buf_mean':>8} {'drops':>6} {'under':>6} {'status':>8}")
        print("  " + "-" * 82)

        all_ok = True
        for drift in drift_vals:
            if approach == 'PI':
                r = simulate_pi(Kp, Ki, alpha, 300, drift, duration_s=600,
                               log_path=f'/tmp/pi_v2_{rank}_{drift:+d}.csv'
                               if drift in [0, 200, 1000] else None)
            else:
                r = simulate_slope(win, upd, ema, 300, drift, duration_s=600,
                                  log_path=f'/tmp/slope_{rank}_{drift:+d}.csv'
                                  if drift in [0, 200, 1000] else None)

            status = "OK" if r['converged'] else "FAIL"
            if not r['converged']:
                all_ok = False
                if r['hit_limits']:
                    status = "LIMIT"
                elif r['drops'] > 0:
                    status = "DROPS"
                elif r['underruns'] > 0:
                    status = "UNDRN"

            print(f"  {drift:>+5d} | {r['adj_mean']:>+9.1f} {r['adj_std']:>8.1f} "
                  f"{r['adj_range']:>8.1f} {r['buf_mean']:>8.1f} {r['drops']:>6} "
                  f"{r['underruns']:>6} {status:>8}")

        if all_ok:
            print(f"\n  *** ALL DRIFT VALUES PASSED ***")
            if approach == 'PI':
                print(f"\n  Recommended PI parameters for panadapter.py:")
                print(f"    Kp = {Kp}")
                print(f"    Ki = {Ki}")
                print(f"    alpha = {alpha}")
                print(f"    integrator_max = 0.002")
            else:
                print(f"\n  Recommended slope estimator parameters:")
                print(f"    window_s = {win}")
                print(f"    update_s = {upd}")
                print(f"    ema_alpha = {ema}")
            return cfg

    print(f"\nNo {approach} configuration passed all drift values.")
    return None


def main():
    print("Panadapter Rate Controller Simulation")
    print(f"Buffer: {BUF_SIZE} samples ({BUF_SIZE/AUDIO_RATE*1000:.0f}ms)")
    print(f"Prefill: {PREFILL} samples ({PREFILL/AUDIO_RATE*1000:.0f}ms)")
    print(f"Audio block: {AUDIO_BLOCK} samples, burst ~{BURST_PERIOD_S*1000:.0f}ms")
    print(f"Callback: {CALLBACK_SIZE} samples every {CALLBACK_SIZE/AUDIO_RATE*1000:.1f}ms")
    print()

    # Test both approaches
    pi_converged = sweep_pi()
    slope_converged = sweep_slope()

    # Validate winners
    best_pi = None
    best_slope = None

    if pi_converged:
        best_pi = validate(pi_converged, 'PI')
    if slope_converged:
        best_slope = validate(slope_converged, 'Slope')

    # Summary
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if best_pi:
        print(f"  PI controller: PASSED (Kp={best_pi['Kp']}, Ki={best_pi['Ki']}, "
              f"alpha={best_pi['alpha']})")
    else:
        print("  PI controller: FAILED")
    if best_slope:
        print(f"  Slope estimator: PASSED (window={best_slope['window_s']}s, "
              f"update={best_slope['update_s']}s, ema={best_slope['ema_alpha']})")
    else:
        print("  Slope estimator: FAILED")

    if not best_pi and not best_slope:
        print("\n  Neither approach converged. Consider hybrid approach.")


if __name__ == '__main__':
    main()
