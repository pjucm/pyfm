#!/usr/bin/env python3
"""
Test Preamp and Attenuator vs I/Q Output Level

Tests whether preamp and attenuator settings affect I/Q output level.
These are earlier in the signal chain than RF gain.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import icom_r8600 as r8600

def analyze_samples(samples):
    """Analyze I/Q samples and return power."""
    power = np.mean(np.abs(samples)**2)
    peak = np.abs(samples).max()
    std_i = np.real(samples).std()
    std_q = np.imag(samples).std()
    return power, peak, std_i, std_q

def collect_samples(radio, label, count=5, samples_per=8192):
    """Collect samples and print analysis."""
    time.sleep(0.3)
    radio.flush_iq()
    time.sleep(0.2)

    all_samples = []
    for _ in range(count):
        samples = radio.fetch_iq(samples_per)
        all_samples.append(samples)
        time.sleep(0.05)

    combined = np.concatenate(all_samples)
    power, peak, std_i, std_q = analyze_samples(combined)
    power_db = 10 * np.log10(power + 1e-10)

    print(f"  {label}: Power={power_db:.1f} dB, Peak={peak:.4f}, Std={std_i:.4f}")
    return power

def main():
    print("=" * 60)
    print("Preamp/Attenuator vs I/Q Output Level Test")
    print("=" * 60)

    radio = r8600.IcomR8600()
    radio._iq_gain = 1.0  # Unity gain

    try:
        radio.open()

        # Setup
        radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x00, 0x01]))  # REMOTE ON
        time.sleep(0.1)

        # Set frequency
        freq = 89.9e6
        bcd_freq = r8600._freq_to_bcd(freq)
        radio._send_command(r8600._build_civ_command([0x05] + list(bcd_freq)))
        print(f"\nFrequency: {freq/1e6:.1f} MHz")

        # Start I/Q streaming
        radio.streaming_mode = "iq"
        radio.iq_sample_rate = 480000
        radio._bit_depth = 16
        radio._bytes_per_sample = 4
        radio._running = True
        radio._iq_buffer = []

        import threading
        radio._iq_thread = threading.Thread(target=radio._iq_reader_loop, daemon=True)
        radio._iq_thread.start()

        # Enable I/Q output
        radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x01, 0x00, 0x05]))
        time.sleep(0.5)

        results = []

        def set_rf_config(preamp, att):
            """Disable I/Q, change settings, re-enable I/Q."""
            # Disable I/Q output
            radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x00]))
            time.sleep(0.1)

            # Change settings
            radio.set_preamp(preamp)
            radio.set_attenuator(att)

            # Re-enable I/Q output
            radio._send_command(r8600._build_civ_command([0x1A, 0x13, 0x01, 0x01, 0x00, 0x05]))
            time.sleep(0.2)

        # Test preamp ON/OFF
        print("\n--- Preamp Tests ---")

        set_rf_config(preamp=True, att=0)
        power = collect_samples(radio, "Preamp ON, ATT 0dB")
        results.append(("Preamp ON, ATT 0dB", power))

        set_rf_config(preamp=False, att=0)
        power = collect_samples(radio, "Preamp OFF, ATT 0dB")
        results.append(("Preamp OFF, ATT 0dB", power))

        # Test attenuator
        print("\n--- Attenuator Tests ---")

        for att in [0, 10, 20, 30]:
            set_rf_config(preamp=True, att=att)
            power = collect_samples(radio, f"Preamp ON, ATT {att}dB")
            results.append((f"ATT {att}dB", power))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Setting':<25} {'Power (dB)':>12}")
        print("-" * 40)
        for label, power in results:
            power_db = 10 * np.log10(power + 1e-10)
            print(f"{label:<25} {power_db:>12.1f}")

        # Analysis
        powers = [p for _, p in results]
        power_range_db = 10 * np.log10(max(powers) / (min(powers) + 1e-10))

        print(f"\nPower variation: {power_range_db:.1f} dB")

        if power_range_db < 3:
            print("\nCONCLUSION: I/Q output is FIXED and not affected by front-end controls!")
            print("The I/Q tap point is after all RF processing stages.")
        elif power_range_db >= 10:
            print("\nCONCLUSION: Preamp/attenuator DO affect I/Q output!")
            print(f"This gives us {power_range_db:.0f} dB of control range.")
        else:
            print(f"\nCONCLUSION: Some effect on I/Q output ({power_range_db:.1f} dB range)")

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
