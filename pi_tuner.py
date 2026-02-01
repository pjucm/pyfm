#!/usr/bin/env python3
"""
PI Loop Tuner for pyfm Resampler

This script provides automated optimization of the PI controller that
manages audio buffer level by adjusting the resample rate.

The PI controller compensates for clock drift between the IQ source
(BB60D or IC-R8600) and the audio output (sound card).

Usage:
    ./pi_tuner.py [--run | --analyze | --optimize]

Modes:
    --run       Run pyfm for 90 seconds with detailed logging
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

# Log file location
LOG_FILE = "/tmp/pyfm_pi_detailed.log"
METRICS_FILE = "/tmp/pyfm_pi_metrics.txt"


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
    metrics['quality_score'] = (
        metrics['settling_time_s'] * 10.0 +
        metrics['startup_max_overshoot_ms'] * 2.0 +
        metrics['startup_rms_error_ms'] * 5.0 +
        metrics['steady_rms_error_ms'] * 10.0 +
        metrics['steady_std_adj_ppm'] * 0.1
    )

    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    if metrics is None:
        print("No metrics available (insufficient data)")
        return

    print("\n" + "=" * 60)
    print("PI LOOP PERFORMANCE METRICS")
    print("=" * 60)

    print("\nStartup Performance:")
    print(f"  Settling time:       {metrics['settling_time_s']:.2f} s")
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

    print(f"\nQuality Score:         {metrics['quality_score']:.1f} (lower is better)")
    print("=" * 60)


def generate_plot(data, output_path="/tmp/pyfm_pi_plot.png"):
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


def run_pyfm_headless(duration_s=90, frequency=89.9, kp=None, ki=None):
    """
    Run pyfm in headless mode for the specified duration.

    Args:
        duration_s: How long to run in seconds
        frequency: FM frequency in MHz
        kp: Optional Kp override
        ki: Optional Ki override

    Returns:
        True if successful, False otherwise
    """
    # Build environment with PI gain overrides if specified
    env = os.environ.copy()
    if kp is not None:
        env['PYFM_PI_KP'] = str(kp)
    if ki is not None:
        env['PYFM_PI_KI'] = str(ki)

    # Enable detailed logging
    env['PYFM_PI_LOG'] = LOG_FILE
    env['PYFM_HEADLESS'] = '1'
    env['PYFM_DURATION'] = str(duration_s)

    print(f"Starting pyfm on {frequency} MHz for {duration_s} seconds...")
    if kp is not None:
        print(f"  Kp override: {kp}")
    if ki is not None:
        print(f"  Ki override: {ki}")

    # Run pyfm
    pyfm_path = Path(__file__).parent / "pyfm.py"
    cmd = [sys.executable, str(pyfm_path), str(frequency)]

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

        return proc.returncode in (0, -2, None)  # -2 is SIGINT

    except Exception as e:
        print(f"Error running pyfm: {e}")
        return False


def optimize_pi_gains(iterations=5, frequency=89.9):
    """
    Run multiple iterations with different PI gains to find optimal values.

    Uses a simple grid search followed by gradient descent.
    """
    print("\n" + "=" * 60)
    print("PI GAIN OPTIMIZATION")
    print("=" * 60)

    # Initial search grid (based on current values)
    # Current: Kp = 0.00005 (50 ppm/ms), Ki = 0.0000004 (~24 ppm/ms/s)
    kp_values = [0.00003, 0.00005, 0.00008, 0.0001]
    ki_values = [0.0000002, 0.0000004, 0.0000008, 0.000001]

    results = []

    for kp in kp_values:
        for ki in ki_values:
            print(f"\n--- Testing Kp={kp:.6f}, Ki={ki:.8f} ---")

            if run_pyfm_headless(duration_s=60, frequency=frequency, kp=kp, ki=ki):
                data = parse_log_file()
                metrics = compute_metrics(data)

                if metrics:
                    results.append({
                        'kp': kp,
                        'ki': ki,
                        'metrics': metrics,
                        'score': metrics['quality_score']
                    })
                    print(f"  Score: {metrics['quality_score']:.1f}")
                else:
                    print("  Failed to compute metrics")
            else:
                print("  pyfm run failed")

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
    print_metrics(best['metrics'])

    print("\nAll results (sorted by score):")
    for r in results[:10]:
        print(f"  Kp={r['kp']:.6f}, Ki={r['ki']:.8f}: score={r['score']:.1f}")

    return best


def main():
    parser = argparse.ArgumentParser(description="PI Loop Tuner for pyfm")
    parser.add_argument('--run', action='store_true',
                        help='Run pyfm for 90 seconds with detailed logging')
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

    args = parser.parse_args()

    if args.analyze:
        data = parse_log_file()
        if data is not None:
            print(f"Loaded {len(data['time_s'])} samples from log")
            print(f"Time range: {data['time_s'][0]:.1f}s - {data['time_s'][-1]:.1f}s")
            metrics = compute_metrics(data)
            print_metrics(metrics)

            # Generate plots
            print("\n" + generate_ascii_plot(data))
            generate_plot(data)
        return

    if args.optimize:
        optimize_pi_gains(frequency=args.frequency)
        return

    if args.run:
        success = run_pyfm_headless(
            duration_s=args.duration,
            frequency=args.frequency,
            kp=args.kp,
            ki=args.ki
        )
        if success:
            print("\nRun completed. Analyzing results...")
            data = parse_log_file()
            if data is not None:
                print(f"Loaded {len(data['time_s'])} samples from log")
                metrics = compute_metrics(data)
                print_metrics(metrics)
                print("\n" + generate_ascii_plot(data))
                generate_plot(data)
        return

    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
