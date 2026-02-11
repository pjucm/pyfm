#!/usr/bin/env python3
"""
PI Loop Tuner for pjfm Resampler

This script provides automated optimization of the PI controller that
manages audio buffer level by adjusting the resample rate.

The PI controller compensates for clock drift between the IQ source
(BB60D or IC-R8600) and the audio output (sound card).

Usage:
    python3 tests/test_pi_tuner.py [--run | --analyze | --optimize]

Modes:
    --run       Run pjfm for 90 seconds with detailed logging
    --analyze   Analyze the last log file and report metrics
    --optimize  Run multiple iterations to find optimal PI gains
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import numpy as np
from pathlib import Path

# Log file locations
LOG_FILE = "/tmp/pjfm_pi_detailed.log"
STARTUP_LOG_FILE = "/tmp/pjfm_startup_prefill.log"
METRICS_FILE = "/tmp/pjfm_pi_metrics.txt"


def parse_log_file(log_path=LOG_FILE):
    """Parse the PI loop log file and return structured data."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return None

    data = {
        'time_s': [],
        'buf_ms': [],
        'target_ms': [],
        'error_ms': [],  # This is now filtered_error
        'p_ppm': [],
        'i_ppm': [],
        'adj_ppm': [],
        'integrator': [],
        'raw_error_ms': [],
        'filtered_error_ms': [],
    }

    with open(log_path, 'r') as f:
        header = f.readline().strip()
        cols = header.split(',')

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    data['time_s'].append(float(parts[0]))
                    data['buf_ms'].append(float(parts[1]))
                    data['target_ms'].append(float(parts[2]))
                    data['error_ms'].append(float(parts[3]))
                    data['p_ppm'].append(float(parts[4]))
                    data['i_ppm'].append(float(parts[5]))
                    data['adj_ppm'].append(float(parts[6]))
                    if len(parts) >= 8:
                        data['integrator'].append(float(parts[7]))
                    else:
                        data['integrator'].append(0.0)
                    if len(parts) >= 10:
                        data['raw_error_ms'].append(float(parts[8]))
                        data['filtered_error_ms'].append(float(parts[9]))
                    else:
                        data['raw_error_ms'].append(data['error_ms'][-1])
                        data['filtered_error_ms'].append(data['error_ms'][-1])
                except ValueError:
                    continue

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def parse_startup_log(log_path=STARTUP_LOG_FILE):
    """Parse startup prefill CSV log."""
    if not os.path.exists(log_path):
        return None

    data = {
        'elapsed_s': [],
        'buffer_ms': [],
        'target_ms': [],
        'fill_pct': [],
        'rate_ppm': [],
        'drop_ms_total': [],
        'drop_ms_delta': [],
        'iq_queue_len': [],
        'iq_queue_drops': [],
        'iq_loss_events': [],
        'loss_now': [],
    }

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 15 or parts[0] == 'elapsed_s':
                continue
            try:
                data['elapsed_s'].append(float(parts[0]))
                # parts[1] is stage string
                data['buffer_ms'].append(float(parts[2]))
                data['target_ms'].append(float(parts[3]))
                data['fill_pct'].append(float(parts[4]))
                data['rate_ppm'].append(float(parts[5]))
                data['drop_ms_total'].append(float(parts[6]))
                data['drop_ms_delta'].append(float(parts[7]))
                data['iq_queue_len'].append(float(parts[8]))
                data['iq_queue_drops'].append(float(parts[9]))
                data['iq_loss_events'].append(float(parts[10]))
                data['loss_now'].append(float(parts[14]))
            except ValueError:
                continue

    for key in data:
        data[key] = np.array(data[key])
    return data if len(data['elapsed_s']) > 0 else None


def compute_metrics(data):
    """Compute performance metrics from log data."""
    if data is None or len(data['time_s']) < 10:
        return None

    metrics = {}

    error = data['error_ms']
    time_s = data['time_s']

    # Find settling time (time to reach and stay within ±5ms of target)
    settling_threshold = 5.0  # ms
    settled_idx = None
    window_size = 10  # Must stay settled for this many samples

    for i in range(len(error) - window_size):
        window = error[i:i + window_size]
        if np.all(np.abs(window) < settling_threshold):
            settled_idx = i
            break

    if settled_idx is not None:
        metrics['settling_time_s'] = time_s[settled_idx]
    else:
        metrics['settling_time_s'] = float('inf')

    # Startup metrics (first 10 seconds)
    startup_mask = time_s < 10.0
    if np.any(startup_mask):
        startup_error = error[startup_mask]
        metrics['startup_max_overshoot_ms'] = np.max(np.abs(startup_error))
        metrics['startup_rms_error_ms'] = np.sqrt(np.mean(startup_error ** 2))
    else:
        metrics['startup_max_overshoot_ms'] = 0.0
        metrics['startup_rms_error_ms'] = 0.0

    # Steady-state metrics (last 60 seconds, or after settling)
    if metrics['settling_time_s'] < float('inf'):
        steady_start = max(metrics['settling_time_s'] + 5.0, time_s[-1] - 60.0)
    else:
        steady_start = time_s[-1] - 30.0  # Last 30 seconds

    steady_mask = time_s >= steady_start
    if np.any(steady_mask):
        steady_error = error[steady_mask]
        steady_adj = data['adj_ppm'][steady_mask]

        metrics['steady_mean_error_ms'] = np.mean(steady_error)
        metrics['steady_rms_error_ms'] = np.sqrt(np.mean(steady_error ** 2))
        metrics['steady_max_error_ms'] = np.max(np.abs(steady_error))
        metrics['steady_mean_adj_ppm'] = np.mean(steady_adj)
        metrics['steady_std_adj_ppm'] = np.std(steady_adj)

        # Estimate clock drift from integrator
        if len(data['integrator']) > 0:
            steady_integrator = data['integrator'][steady_mask]
            metrics['estimated_drift_ppm'] = np.mean(steady_integrator) * 1e6
    else:
        metrics['steady_mean_error_ms'] = 0.0
        metrics['steady_rms_error_ms'] = 0.0
        metrics['steady_max_error_ms'] = 0.0
        metrics['steady_mean_adj_ppm'] = 0.0
        metrics['steady_std_adj_ppm'] = 0.0

    # Overall quality score (lower is better)
    # Weights: settling time (important), startup overshoot, steady-state error
    settling_penalty = metrics['settling_time_s'] if np.isfinite(metrics['settling_time_s']) else 180.0
    metrics['quality_score'] = (
        settling_penalty * 10.0 +
        metrics['startup_max_overshoot_ms'] * 2.0 +
        metrics['startup_rms_error_ms'] * 5.0 +
        metrics['steady_rms_error_ms'] * 10.0 +
        metrics['steady_std_adj_ppm'] * 0.1
    )

    return metrics


def compute_startup_metrics(startup_data):
    """Compute startup-prefill-specific metrics from startup log."""
    if startup_data is None or len(startup_data['elapsed_s']) < 2:
        return None

    metrics = {}
    buf = startup_data['buffer_ms']
    target = startup_data['target_ms']
    err = buf - target

    metrics['startup_log_duration_s'] = float(startup_data['elapsed_s'][-1])
    metrics['prefill_max_drop_ms_total'] = float(np.max(startup_data['drop_ms_total']))
    metrics['prefill_drop_ms_sum'] = float(np.sum(startup_data['drop_ms_delta']))
    metrics['prefill_max_abs_rate_ppm'] = float(np.max(np.abs(startup_data['rate_ppm'])))
    metrics['prefill_max_abs_error_ms'] = float(np.max(np.abs(err)))
    metrics['prefill_rms_error_ms'] = float(np.sqrt(np.mean(err ** 2)))
    metrics['prefill_iq_loss_events'] = float(np.max(startup_data['iq_loss_events']))
    metrics['prefill_iq_queue_drops'] = float(np.max(startup_data['iq_queue_drops']))

    # Time to remain within ±20 ms of target for 0.5 s worth of samples.
    within = np.abs(err) <= 20.0
    settle_time = float('inf')
    for i in range(len(within)):
        if not within[i]:
            continue
        t0 = startup_data['elapsed_s'][i]
        window_mask = (startup_data['elapsed_s'] >= t0) & (startup_data['elapsed_s'] <= t0 + 0.5)
        if np.any(window_mask) and np.all(within[window_mask]):
            settle_time = t0
            break
    metrics['prefill_settle_time_s'] = settle_time

    settling_penalty = settle_time if np.isfinite(settle_time) else 10.0
    metrics['prefill_quality_score'] = (
        settling_penalty * 8.0 +
        metrics['prefill_max_drop_ms_total'] * 12.0 +
        metrics['prefill_max_abs_error_ms'] * 1.5 +
        metrics['prefill_rms_error_ms'] * 3.0 +
        metrics['prefill_max_abs_rate_ppm'] * 0.004 +
        metrics['prefill_iq_loss_events'] * 100.0 +
        metrics['prefill_iq_queue_drops'] * 40.0
    )
    return metrics


def combine_metrics(pi_metrics, startup_metrics):
    """Combine steady-state PI metrics with startup-prefill metrics."""
    if pi_metrics is None:
        return None
    combined = dict(pi_metrics)
    if startup_metrics:
        combined.update(startup_metrics)
        combined['quality_score'] = (
            pi_metrics['quality_score'] * 0.7 +
            startup_metrics['prefill_quality_score'] * 0.3
        )
    return combined


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    if metrics is None:
        print("No metrics available (insufficient data)")
        return

    print("\n" + "=" * 60)
    print("PI LOOP PERFORMANCE METRICS")
    print("=" * 60)

    settle_text = (
        f"{metrics['settling_time_s']:.2f} s"
        if np.isfinite(metrics['settling_time_s'])
        else "not settled"
    )
    print("\nStartup Performance:")
    print(f"  Settling time:       {settle_text}")
    print(f"  Max overshoot:       {metrics['startup_max_overshoot_ms']:.1f} ms")
    print(f"  RMS error (0-10s):   {metrics['startup_rms_error_ms']:.2f} ms")

    print("\nSteady-State Performance:")
    print(f"  Mean error:          {metrics['steady_mean_error_ms']:.2f} ms")
    print(f"  RMS error:           {metrics['steady_rms_error_ms']:.2f} ms")
    print(f"  Max error:           {metrics['steady_max_error_ms']:.1f} ms")
    print(f"  Mean rate adj:       {metrics['steady_mean_adj_ppm']:.1f} ppm")
    print(f"  Rate adj std dev:    {metrics['steady_std_adj_ppm']:.2f} ppm")

    if 'estimated_drift_ppm' in metrics:
        print(f"  Est. clock drift:    {metrics['estimated_drift_ppm']:.1f} ppm")

    if 'prefill_quality_score' in metrics:
        prefill_settle = metrics.get('prefill_settle_time_s', float('inf'))
        prefill_settle_text = (
            f"{prefill_settle:.2f} s"
            if np.isfinite(prefill_settle)
            else "not settled"
        )
        print("\nStartup Prefill Metrics:")
        print(f"  Prefill settle:      {prefill_settle_text} (±20 ms)")
        print(f"  Max drop total:      {metrics['prefill_max_drop_ms_total']:.2f} ms")
        print(f"  Max abs error:       {metrics['prefill_max_abs_error_ms']:.1f} ms")
        print(f"  RMS error:           {metrics['prefill_rms_error_ms']:.2f} ms")
        print(f"  Max |rate|:          {metrics['prefill_max_abs_rate_ppm']:.1f} ppm")
        print(f"  IQ loss events:      {metrics['prefill_iq_loss_events']:.0f}")
        print(f"  IQ queue drops:      {metrics['prefill_iq_queue_drops']:.0f}")

    print(f"\nQuality Score:         {metrics['quality_score']:.1f} (lower is better)")
    print("=" * 60)


def generate_plot(data, output_path="/tmp/pjfm_pi_plot.png"):
    """Generate a plot of the PI loop behavior."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return None

    if data is None or len(data['time_s']) < 10:
        print("Insufficient data for plotting")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Buffer level vs target
    ax1 = axes[0]
    ax1.plot(data['time_s'], data['buf_ms'], 'b-', alpha=0.7, label='Buffer Level')
    ax1.axhline(y=data['target_ms'][0], color='r', linestyle='--', label='Target')
    ax1.fill_between(data['time_s'],
                     data['target_ms'][0] - 5, data['target_ms'][0] + 5,
                     alpha=0.2, color='green', label='±5ms zone')
    ax1.set_ylabel('Buffer (ms)')
    ax1.legend(loc='upper right')
    ax1.set_title('PI Loop Buffer Level Control')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    ax2 = axes[1]
    ax2.plot(data['time_s'], data['error_ms'], 'r-', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(y=5, color='g', linestyle=':', alpha=0.5)
    ax2.axhline(y=-5, color='g', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Error (ms)')
    ax2.set_title('Buffer Error')
    ax2.grid(True, alpha=0.3)

    # Plot 3: P and I terms
    ax3 = axes[2]
    ax3.plot(data['time_s'], data['p_ppm'], 'b-', alpha=0.7, label='P term')
    ax3.plot(data['time_s'], data['i_ppm'], 'g-', alpha=0.7, label='I term')
    ax3.plot(data['time_s'], data['adj_ppm'], 'r-', alpha=0.5, label='Total adj')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Rate adj (ppm)')
    ax3.legend(loc='upper right')
    ax3.set_title('PI Controller Output')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Integrator state
    ax4 = axes[3]
    ax4.plot(data['time_s'], np.array(data['integrator']) * 1e6, 'g-', alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Integrator (ppm)')
    ax4.set_title('Integrator State (accumulated clock drift estimate)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Plot saved to: {output_path}")
    return output_path


def generate_ascii_plot(data, width=70, height=15):
    """Generate an ASCII plot of the PI loop behavior for terminal display."""
    if data is None or len(data['time_s']) < 10:
        return "Insufficient data for plotting"

    time_s = data['time_s']
    buf_ms = data['buf_ms']
    target = data['target_ms'][0] if len(data['target_ms']) > 0 else 100

    # Downsample data to fit width
    step = max(1, len(time_s) // width)
    t_plot = time_s[::step][:width]
    b_plot = buf_ms[::step][:width]

    # Determine y-axis range
    y_min = max(0, min(b_plot) - 10)
    y_max = max(b_plot) + 10

    lines = []
    lines.append("Buffer Level (ms) vs Time (s)")
    lines.append("-" * (width + 10))

    for row in range(height - 1, -1, -1):
        y_val = y_min + (y_max - y_min) * row / (height - 1)
        line = f"{y_val:6.0f} |"

        for i, (t, b) in enumerate(zip(t_plot, b_plot)):
            b_row = int((b - y_min) / (y_max - y_min) * (height - 1))
            target_row = int((target - y_min) / (y_max - y_min) * (height - 1))

            if b_row == row:
                line += "*"
            elif target_row == row:
                line += "-"
            else:
                line += " "

        lines.append(line)

    # X-axis
    lines.append("       +" + "-" * width)
    x_label = f"       0{' ' * (width // 2 - 2)}{t_plot[-1]:.0f}s"
    lines.append(x_label)
    lines.append(f"       Target: {target:.0f}ms (---)")

    return "\n".join(lines)


def run_pjfm_headless(duration_s=90, frequency=89.9, kp=None, ki=None, alpha=None,
                      prefill_ms=None, use_icom=False, use_bb60d=False):
    """
    Run pjfm in headless mode for the specified duration.

    Args:
        duration_s: How long to run in seconds
        frequency: FM frequency in MHz
        kp: Optional Kp override
        ki: Optional Ki override
        alpha: Optional PI error EMA alpha override
        prefill_ms: Optional startup audio prefill level override
        use_icom: Use IC-R8600 device path
        use_bb60d: Use BB60D device path

    Returns:
        True if successful, False otherwise
    """
    # Build environment with PI gain overrides if specified
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if kp is not None:
        env['PYFM_PI_KP'] = str(kp)
    if ki is not None:
        env['PYFM_PI_KI'] = str(ki)
    if alpha is not None:
        env['PYFM_PI_ALPHA'] = str(alpha)
    if prefill_ms is not None:
        env['PYFM_AUDIO_PREFILL_MS'] = str(prefill_ms)

    # Enable detailed logging
    env['PYFM_PI_LOG'] = LOG_FILE
    env['PYFM_STARTUP_PREFILL_LOG'] = STARTUP_LOG_FILE
    env['PYFM_STARTUP_PREFILL_SECONDS'] = '8'
    env['PYFM_HEADLESS'] = '1'
    env['PYFM_DURATION'] = str(duration_s)

    print(f"Starting pjfm on {frequency} MHz for {duration_s} seconds...")
    if kp is not None:
        print(f"  Kp override: {kp}")
    if ki is not None:
        print(f"  Ki override: {ki}")
    if alpha is not None:
        print(f"  Alpha override: {alpha}")
    if prefill_ms is not None:
        print(f"  Prefill override: {prefill_ms} ms")

    # Run pjfm
    pjfm_path = Path(__file__).resolve().parents[1] / "pjfm.py"
    if not pjfm_path.exists():
        print(f"pjfm.py not found at expected path: {pjfm_path}")
        return False

    # Remove stale logs so each run evaluates clean output.
    for path in (LOG_FILE, STARTUP_LOG_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    cmd = [sys.executable, str(pjfm_path), str(frequency)]
    if use_icom:
        cmd.append("--icom")
    if use_bb60d:
        cmd.append("--bb60d")

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration_s + 5:
            if proc.poll() is not None:
                break
            time.sleep(1)
            elapsed = time.time() - start_time
            print(f"\r  Running... {elapsed:.0f}s / {duration_s}s", end='', flush=True)

        print()

        # Send SIGINT to gracefully stop
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Collect output for robust success/failure detection.
        try:
            out_bytes, err_bytes = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            out_bytes, err_bytes = b"", b""
        stdout = out_bytes.decode('utf-8', errors='ignore') if isinstance(out_bytes, (bytes, bytearray)) else (out_bytes or "")
        stderr = err_bytes.decode('utf-8', errors='ignore') if isinstance(err_bytes, (bytes, bytearray)) else (err_bytes or "")

        # pjfm currently hard-exits with code 0 even on startup failure.
        # Require at least one log file and reject known startup error markers.
        startup_failed = (
            "Error starting radio:" in stdout
            or "Error starting radio:" in stderr
            or "Error: " in stdout and "Goodbye!" in stdout and not os.path.exists(LOG_FILE)
        )
        logs_present = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
        if proc.returncode in (0, -2, None) and logs_present and not startup_failed:  # -2 is SIGINT
            return True

        # Show concise failure context to make tuning runs debuggable.
        if stdout.strip():
            print("  pjfm stdout (tail):")
            for line in stdout.strip().splitlines()[-6:]:
                print(f"    {line}")
        if stderr.strip():
            print("  pjfm stderr (tail):")
            for line in stderr.strip().splitlines()[-6:]:
                print(f"    {line}")
        print(f"  pjfm exited with code {proc.returncode}")
        return False

    except Exception as e:
        print(f"Error running pjfm: {e}")
        return False


def optimize_pi_gains(frequency=89.9, duration_s=60, use_icom=False, use_bb60d=False):
    """
    Run multiple iterations with different PI gains to find optimal values.

    Uses a simple grid search followed by gradient descent.
    """
    print("\n" + "=" * 60)
    print("PI GAIN OPTIMIZATION")
    print("=" * 60)

    # Search around current defaults in pjfm.py:
    # Kp=0.000015, Ki=0.0000006, alpha=0.25, prefill=35 ms
    kp_values = [0.000010, 0.000015, 0.000020]
    ki_values = [0.0000003, 0.0000006, 0.0000009]
    alpha_values = [0.20, 0.25, 0.30]
    prefill_values = [35, 40, 50]

    results = []

    # Keep optimization run count manageable by doing staged sweeps.
    staged_candidates = []
    for prefill in prefill_values:
        staged_candidates.append((0.000015, 0.0000006, 0.25, prefill))
    for kp in kp_values:
        for ki in ki_values:
            staged_candidates.append((kp, ki, 0.25, 50))
    for alpha in alpha_values:
        staged_candidates.append((0.000015, 0.0000006, alpha, 50))

    # Remove duplicates while preserving order.
    seen = set()
    candidates = []
    for item in staged_candidates:
        if item in seen:
            continue
        seen.add(item)
        candidates.append(item)

    for kp, ki, alpha, prefill in candidates:
        print(
            f"\n--- Testing Kp={kp:.6f}, Ki={ki:.8f}, "
            f"alpha={alpha:.2f}, prefill={prefill:.0f}ms ---"
        )

        ok = run_pjfm_headless(
            duration_s=duration_s,
            frequency=frequency,
            kp=kp,
            ki=ki,
            alpha=alpha,
            prefill_ms=prefill,
            use_icom=use_icom,
            use_bb60d=use_bb60d,
        )
        if not ok:
            print("  pjfm run failed")
            continue

        pi_data = parse_log_file()
        startup_data = parse_startup_log()
        pi_metrics = compute_metrics(pi_data)
        startup_metrics = compute_startup_metrics(startup_data)
        metrics = combine_metrics(pi_metrics, startup_metrics)
        if metrics is None:
            print("  Failed to compute metrics")
            continue

        results.append({
            'kp': kp,
            'ki': ki,
            'alpha': alpha,
            'prefill_ms': prefill,
            'metrics': metrics,
            'score': metrics['quality_score']
        })
        print(f"  Score: {metrics['quality_score']:.1f}")

    if not results:
        print("\nNo successful runs!")
        return None

    # Find best result
    results.sort(key=lambda x: x['score'])
    best = results[0]

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest configuration:")
    print(f"  Kp = {best['kp']:.8f}  ({best['kp'] * 1e6:.1f} ppm per ms error)")
    print(f"  Ki = {best['ki']:.10f}  ({best['ki'] * 1e6:.3f} ppm per ms*s accumulated)")
    print(f"  alpha = {best['alpha']:.2f}")
    print(f"  prefill = {best['prefill_ms']:.0f} ms")
    print_metrics(best['metrics'])

    print("\nAll results (sorted by score):")
    for r in results[:10]:
        print(
            f"  Kp={r['kp']:.6f}, Ki={r['ki']:.8f}, "
            f"alpha={r['alpha']:.2f}, prefill={r['prefill_ms']:.0f}ms: "
            f"score={r['score']:.1f}"
        )

    return best


def main():
    parser = argparse.ArgumentParser(description="PI Loop Tuner for pjfm")
    parser.add_argument('--run', action='store_true',
                        help='Run pjfm for 90 seconds with detailed logging')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze the last log file')
    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization to find best PI gains')
    parser.add_argument('--frequency', '-f', type=float, default=89.9,
                        help='FM frequency in MHz (default: 89.9)')
    parser.add_argument('--duration', '-d', type=int, default=90,
                        help='Duration in seconds (default: 90)')
    parser.add_argument('--kp', type=float, default=None,
                        help='Override Kp value')
    parser.add_argument('--ki', type=float, default=None,
                        help='Override Ki value')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Override PI error EMA alpha (PYFM_PI_ALPHA)')
    parser.add_argument('--prefill-ms', type=float, default=None,
                        help='Override startup prefill in ms (PYFM_AUDIO_PREFILL_MS)')
    parser.add_argument('--icom', action='store_true',
                        help='Force IC-R8600 device path for tuning runs')
    parser.add_argument('--bb60d', action='store_true',
                        help='Force BB60D device path for tuning runs')

    args = parser.parse_args()

    if args.analyze:
        data = parse_log_file()
        startup_data = parse_startup_log()
        if data is not None:
            print(f"Loaded {len(data['time_s'])} samples from PI log")
            print(f"PI time range: {data['time_s'][0]:.1f}s - {data['time_s'][-1]:.1f}s")
            if startup_data is not None:
                print(
                    f"Loaded {len(startup_data['elapsed_s'])} samples from startup log "
                    f"(0-{startup_data['elapsed_s'][-1]:.1f}s)"
                )
            else:
                print("Startup log not found; scoring steady-state PI only.")
            pi_metrics = compute_metrics(data)
            startup_metrics = compute_startup_metrics(startup_data)
            metrics = combine_metrics(pi_metrics, startup_metrics)
            print_metrics(metrics)

            # Generate plots
            print("\n" + generate_ascii_plot(data))
            generate_plot(data)
        return

    if args.optimize:
        optimize_pi_gains(
            frequency=args.frequency,
            duration_s=max(30, args.duration),
            use_icom=args.icom,
            use_bb60d=args.bb60d,
        )
        return

    if args.run:
        success = run_pjfm_headless(
            duration_s=args.duration,
            frequency=args.frequency,
            kp=args.kp,
            ki=args.ki,
            alpha=args.alpha,
            prefill_ms=args.prefill_ms,
            use_icom=args.icom,
            use_bb60d=args.bb60d,
        )
        if success:
            print("\nRun completed. Analyzing results...")
            data = parse_log_file()
            startup_data = parse_startup_log()
            if data is not None:
                print(f"Loaded {len(data['time_s'])} samples from PI log")
                if startup_data is not None:
                    print(
                        f"Loaded {len(startup_data['elapsed_s'])} samples from startup log "
                        f"(0-{startup_data['elapsed_s'][-1]:.1f}s)"
                    )
                pi_metrics = compute_metrics(data)
                startup_metrics = compute_startup_metrics(startup_data)
                metrics = combine_metrics(pi_metrics, startup_metrics)
                print_metrics(metrics)
                print("\n" + generate_ascii_plot(data))
                generate_plot(data)
        return

    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
