#!/usr/bin/env python3
"""Test sample loss on channel change (simulates pjfm tune_up/tune_down)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from icom_r8600 import IcomR8600
import time

def test_channel_change(use_24bit=False, num_changes=50):
    """Test channel change and check for sample loss."""
    bit_label = "24-bit" if use_24bit else "16-bit"
    sample_rate = 480000  # Same as pjfm default

    print(f"\n{'='*60}")
    print(f"Channel Change Stress Test ({bit_label})")
    print('='*60)

    radio = IcomR8600(use_24bit=use_24bit)
    radio.open()
    radio.configure_iq_streaming(freq=89.9e6, sample_rate=sample_rate)

    # Let things stabilize
    time.sleep(0.5)

    # Warmup fetches
    for _ in range(10):
        radio.fetch_iq(8192)

    print(f"\nInitial state after warmup:")
    d = radio.get_diagnostics()
    print(f"  sample_loss={d['total_sample_loss']}, sync_misses={d['sync_misses']}")

    # Simulate many channel changes
    loss_events = []
    slow_fetches = []
    for i in range(num_changes):
        # Simulate what pjfm does on tune_up:
        # 1. tune_up() calls set_frequency() and flush_iq()
        radio.tune_up()  # This calls flush_iq()

        # 2. NO delay - immediate fetch (worst case)
        # This simulates the audio loop immediately resuming

        # 3. First fetch after channel change
        t0 = time.perf_counter()
        iq = radio.fetch_iq(8192)
        fetch_ms = (time.perf_counter() - t0) * 1000

        # Check for sample loss (total_sample_loss was reset by flush_iq)
        d = radio.get_diagnostics()
        loss = d['total_sample_loss']
        if loss > 0:
            loss_events.append((i, loss, fetch_ms))
            print(f"  Ch {i}: LOSS={loss}, fetch={fetch_ms:.1f}ms, "
                  f"aligns={d['initial_aligns']}, sync_misses={d['sync_misses']}")
        if fetch_ms > 100:
            slow_fetches.append((i, fetch_ms))

        # A few more fetches to simulate normal operation
        for _ in range(3):
            radio.fetch_iq(8192)

    radio.close()

    print(f"\nResults:")
    print(f"  Channel changes: {num_changes}")
    print(f"  Changes with sample loss: {len(loss_events)}")
    print(f"  Slow first fetches (>100ms): {len(slow_fetches)}")

    return len(loss_events) == 0

def test_startup(use_24bit=False, iterations=10):
    """Test startup sequence for sample loss."""
    bit_label = "24-bit" if use_24bit else "16-bit"
    sample_rate = 480000

    print(f"\n{'='*60}")
    print(f"Startup Test ({bit_label})")
    print('='*60)

    loss_at_startup = 0
    for i in range(iterations):
        radio = IcomR8600(use_24bit=use_24bit)
        radio.open()
        radio.configure_iq_streaming(freq=89.9e6, sample_rate=sample_rate)

        # Small delay after configure (simulates pjfm startup)
        time.sleep(0.1)

        # First fetch - this is where startup loss would occur
        t0 = time.perf_counter()
        iq = radio.fetch_iq(8192)
        fetch_ms = (time.perf_counter() - t0) * 1000

        d = radio.get_diagnostics()
        if d['total_sample_loss'] > 0:
            loss_at_startup += 1
            print(f"  Startup {i}: LOSS, fetch={fetch_ms:.1f}ms, "
                  f"aligns={d['initial_aligns']}, usb_buf={d['usb_buffer_kb']:.1f}KB")

        radio.close()
        time.sleep(0.3)  # Let USB settle between runs

    print(f"\nResults:")
    print(f"  Startup iterations: {iterations}")
    print(f"  Iterations with sample loss: {loss_at_startup}")

    return loss_at_startup == 0

if __name__ == '__main__':
    # Test startup
    ok_startup_16 = test_startup(use_24bit=False, iterations=5)
    ok_startup_24 = test_startup(use_24bit=True, iterations=5)

    # Test channel changes
    ok_16 = test_channel_change(use_24bit=False, num_changes=50)
    ok_24 = test_channel_change(use_24bit=True, num_changes=50)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"  Startup 16-bit: {'PASS' if ok_startup_16 else 'FAIL'}")
    print(f"  Startup 24-bit: {'PASS' if ok_startup_24 else 'FAIL'}")
    print(f"  Channel change 16-bit: {'PASS' if ok_16 else 'FAIL'}")
    print(f"  Channel change 24-bit: {'PASS' if ok_24 else 'FAIL'}")
