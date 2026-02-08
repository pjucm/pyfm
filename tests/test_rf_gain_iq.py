#!/usr/bin/env python3
"""
Test RF Gain vs I/Q Output Level Relationship

This script tests whether RF gain settings affect the I/Q output level.
The hypothesis is that the I/Q output may be tapped from a fixed point
in the signal chain, independent of RF gain settings.

Stop pjfm before running this.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import icom_r8600 as r8600

def analyze_samples(samples, label):
    """Analyze I/Q samples and print statistics."""
    # Get raw values (undo the gain normalization to see true ADC values)
    # The driver normalizes by /32768 and multiplies by iq_gain
    power = np.mean(np.abs(samples)**2)
    peak = np.abs(samples).max()
    std_i = np.real(samples).std()
    std_q = np.imag(samples).std()

    print(f"  {label}:")
    print(f"    Power: {power:.6f} ({10*np.log10(power + 1e-10):.1f} dB)")
    print(f"    Peak:  {peak:.4f}")
    print(f"    Std I: {std_i:.4f}, Std Q: {std_q:.4f}")
    return power, peak

def test_rf_gain_levels(radio, gains_to_test):
    """Test I/Q output at different RF gain levels."""
    results = []

    for gain in gains_to_test:
        print(f"\n--- Testing RF Gain = {gain} ---")

        # Disable I/Q output to change RF gain
        radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x00]))
        time.sleep(0.1)

        # Set RF gain
        try:
            radio.set_rf_gain(gain)
            print(f"  RF Gain set to {gain}")
        except RuntimeError as e:
            print(f"  Failed to set RF gain: {e}")
            continue

        # Verify it was set
        settings = radio.query_rf_settings()
        actual_gain = settings.get('rf_gain', -1)
        print(f"  Verified RF Gain: {actual_gain}")

        # Re-enable I/Q output
        resp = radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x01, 0x00, 0x05]))  # 16-bit, 480 KSPS
        time.sleep(0.3)

        # Flush old samples
        radio.flush_iq()
        time.sleep(0.2)

        # Collect fresh samples
        all_samples = []
        for _ in range(5):
            samples = radio.fetch_iq(8192)
            all_samples.append(samples)
            time.sleep(0.1)

        combined = np.concatenate(all_samples)
        power, peak = analyze_samples(combined, f"RF Gain {gain}")
        results.append((gain, actual_gain, power, peak))

    return results

def main():
    print("=" * 60)
    print("RF Gain vs I/Q Output Level Test")
    print("=" * 60)

    radio = r8600.IcomR8600()
    radio._iq_gain = 1.0  # Use unity gain to see true normalized levels

    try:
        radio.open()
        print(f"\nDevice: PID={radio.device.idProduct:04x}")

        # Enable I/Q mode first
        print("\n--- Initial Setup ---")
        radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x00, 0x01]))  # REMOTE ON
        time.sleep(0.1)

        # Set frequency to known FM station
        freq = 89.9e6
        bcd_freq = r8600._freq_to_bcd(freq)
        radio._send_command(r8600._build_civ_command([0x05] + list(bcd_freq)))
        print(f"Frequency: {freq/1e6:.1f} MHz")

        # Get initial settings
        settings = radio.query_rf_settings()
        print(f"Initial RF Settings: {settings}")

        # Start I/Q streaming
        radio.streaming_mode = "iq"
        radio.iq_sample_rate = 480000
        radio._bit_depth = 16
        radio._bytes_per_sample = 4

        # Enable I/Q output
        resp = radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x01, 0x00, 0x05]))  # 16-bit, 480 KSPS
        time.sleep(0.2)

        # Start reader thread
        radio._running = True
        radio._iq_buffer = []
        import threading
        radio._iq_thread = threading.Thread(target=radio._iq_reader_loop, daemon=True)
        radio._iq_thread.start()
        time.sleep(0.5)

        # Test different RF gain levels
        gains_to_test = [0, 50, 100, 150, 200, 255]
        print("\n" + "=" * 60)
        print("Testing RF Gain Levels")
        print("=" * 60)

        results = test_rf_gain_levels(radio, gains_to_test)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Set Gain':>10} {'Actual':>10} {'Power':>12} {'Peak':>10}")
        print("-" * 45)
        for set_gain, actual_gain, power, peak in results:
            print(f"{set_gain:>10} {actual_gain:>10} {power:>12.6f} {peak:>10.4f}")

        # Analysis
        if len(results) >= 2:
            powers = [r[2] for r in results]
            power_range_db = 10 * np.log10(max(powers) / (min(powers) + 1e-10))
            print(f"\nPower range: {power_range_db:.1f} dB across RF gain settings")

            if power_range_db < 3:
                print("\nCONCLUSION: I/Q output level is INDEPENDENT of RF gain!")
                print("The I/Q output is likely tapped before the RF gain stage.")
            else:
                print(f"\nCONCLUSION: I/Q output varies with RF gain ({power_range_db:.1f} dB range)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            radio._running = False
            if radio._iq_thread:
                radio._iq_thread.join(timeout=1.0)
            radio.close()
        except:
            pass

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
