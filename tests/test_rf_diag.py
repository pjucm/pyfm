#!/usr/bin/env python3
"""
IC-R8600 RF Diagnostics

Run this to check and adjust RF settings for optimal I/Q streaming.
Stop pjfm first before running this.
"""

import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import icom_r8600 as r8600

def main():
    print("=" * 60)
    print("IC-R8600 RF Diagnostics")
    print("=" * 60)

    radio = r8600.IcomR8600()

    try:
        radio.open()
        print(f"\nDevice: PID={radio.device.idProduct:04x}")

        # Query current settings
        print("\n--- Current RF Settings ---")
        settings = radio.query_rf_settings()
        print(f"  RF Gain:    {settings.get('rf_gain', 'Unknown')}")
        print(f"  Attenuator: {settings.get('attenuator', 'Unknown')} dB")
        print(f"  Preamp:     {'ON' if settings.get('preamp') else 'OFF'}")
        print(f"  IP+:        {'ON' if settings.get('ip_plus') else 'OFF'}")

        # Enable I/Q mode to read S-meter in I/Q context
        print("\n--- Enabling I/Q Mode ---")
        radio.configure_iq_streaming(freq=89.9e6, sample_rate=480000)

        # Read some samples to check levels
        print("\n--- I/Q Sample Analysis ---")
        for i in range(5):
            time.sleep(0.2)
            samples = radio.fetch_iq(8192)

            # Calculate statistics
            power = (samples.real**2 + samples.imag**2).mean()
            peak = abs(samples).max()

            import math
            power_db = 10 * math.log10(power + 1e-10)
            peak_db = 20 * math.log10(peak + 1e-10)

            print(f"  Sample {i+1}: Power={power:.2f} ({power_db:.1f} dB), Peak={peak:.2f} ({peak_db:.1f} dB)")

            # Warn if clipping
            if peak > 0.9:
                print("    WARNING: Peak near clipping! Reduce RF gain.")

        # Recommendations
        print("\n--- Recommendations ---")
        rf_gain = settings.get('rf_gain', 128)
        att = settings.get('attenuator', 0)

        if rf_gain > 200:
            print(f"  * RF Gain ({rf_gain}) is very high - try reducing to ~128")
            print(f"    Command: radio.set_rf_gain(128)")

        if att == 0 and rf_gain > 150:
            print(f"  * Consider enabling 10-20dB attenuator for strong signals")
            print(f"    Command: radio.set_attenuator(10)")

        if settings.get('preamp'):
            print(f"  * Preamp is ON - for FM broadcast, try turning it OFF")
            print(f"    Command: radio.set_preamp(False)")

        print("\n--- Interactive Commands ---")
        print("  radio.set_rf_gain(128)    # Set RF gain 0-255")
        print("  radio.set_attenuator(10)  # Set attenuator 0/10/20/30 dB")
        print("  radio.set_preamp(False)   # Disable preamp")
        print("  radio.iq_gain = 50        # Set software I/Q gain")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            radio.close()
        except:
            pass

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
