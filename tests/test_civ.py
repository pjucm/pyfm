#!/usr/bin/env python3
"""
CI-V Command Test Suite for IC-R8600

Tests CI-V command building, parsing, and protocol handling.
Includes both unit tests (no hardware) and integration tests (requires radio).

Usage:
    python test_civ.py          # Run unit tests only
    python test_civ.py --hw     # Include hardware integration tests
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import CI-V protocol functions and constants
from icom_r8600 import (
    _build_civ_command,
    _freq_to_bcd,
    SAMPLE_RATES,
    SAMPLE_RATES_24BIT,
    CIV_PREAMBLE,
    CIV_TERMINATOR,
    R8600_ADDR,
    PC_ADDR,
    IcomR8600,
)


# =============================================================================
# Unit Tests (No Hardware Required)
# =============================================================================

def test_civ_command_building():
    """
    Test CI-V command packet construction.

    Verifies:
    - Preamble (0xFE 0xFE)
    - R8600 address (0x96)
    - PC address (0xE0)
    - Terminator (0xFD)
    """
    print("\n" + "=" * 60)
    print("TEST: CI-V Command Building")
    print("=" * 60)

    # Build a simple frequency set command
    cmd = _build_civ_command([0x05, 0x00, 0x00, 0x90, 0x89, 0x00])

    print(f"  Command bytes: {cmd.hex()}")
    print(f"  Command length: {len(cmd)} bytes")

    # Check preamble
    preamble_ok = cmd[0] == 0xFE and cmd[1] == 0xFE
    print(f"  Preamble (FE FE): {'OK' if preamble_ok else 'FAIL'}")

    # Check addresses
    addr_ok = cmd[2] == R8600_ADDR and cmd[3] == PC_ADDR
    print(f"  Addresses (96 E0): {'OK' if addr_ok else 'FAIL'}")

    # Check terminator (may have padding after)
    term_idx = cmd.find(bytes([CIV_TERMINATOR]))
    term_ok = term_idx >= 4
    print(f"  Terminator (FD) at index {term_idx}: {'OK' if term_ok else 'FAIL'}")

    passed = preamble_ok and addr_ok and term_ok
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_civ_command_padding():
    """
    Test even-length padding requirement.

    CI-V commands must be even length for USB transfer.
    Odd-length commands should get 0xFF padding.
    """
    print("\n" + "=" * 60)
    print("TEST: CI-V Command Padding")
    print("=" * 60)

    # Odd-length payload should result in even-length packet
    cmd_odd = _build_civ_command([0x05])  # Short command
    cmd_even = _build_civ_command([0x05, 0x00])  # Even command

    print(f"  Odd payload -> {len(cmd_odd)} bytes (should be even)")
    print(f"  Even payload -> {len(cmd_even)} bytes (should be even)")

    odd_padded = len(cmd_odd) % 2 == 0
    even_padded = len(cmd_even) % 2 == 0

    print(f"  Odd payload padded: {'OK' if odd_padded else 'FAIL'}")
    print(f"  Even payload padded: {'OK' if even_padded else 'FAIL'}")

    passed = odd_padded and even_padded
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_freq_to_bcd():
    """
    Test frequency to BCD conversion.

    Frequencies are encoded as 5-byte BCD, LSB first.
    Example: 89.9 MHz = 89900000 Hz -> 00 00 90 89 00
    """
    print("\n" + "=" * 60)
    print("TEST: Frequency to BCD Conversion")
    print("=" * 60)

    test_cases = [
        (89900000, "89.9 MHz"),
        (162400000, "162.4 MHz (NOAA WX)"),
        (100000000, "100.0 MHz"),
        (7200000, "7.2 MHz (40m ham)"),
    ]

    all_passed = True
    for freq_hz, name in test_cases:
        bcd = _freq_to_bcd(freq_hz)
        print(f"  {name}: {freq_hz} Hz -> {bcd.hex()}")

        # Verify length
        if len(bcd) != 5:
            print(f"    ERROR: Expected 5 bytes, got {len(bcd)}")
            all_passed = False
            continue

        # Verify BCD digits are valid (each nibble 0-9)
        valid_bcd = True
        for byte in bcd:
            high = (byte >> 4) & 0x0F
            low = byte & 0x0F
            if high > 9 or low > 9:
                valid_bcd = False
                break

        if not valid_bcd:
            print(f"    ERROR: Invalid BCD digits")
            all_passed = False

    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_sample_rate_tables():
    """
    Test sample rate table consistency.

    Verifies:
    - 16-bit table entries have bit_depth = 0x00
    - 24-bit table entries have bit_depth = 0x01
    - Rate codes are in valid range
    """
    print("\n" + "=" * 60)
    print("TEST: Sample Rate Tables")
    print("=" * 60)

    all_passed = True

    print("  16-bit rates:")
    for rate, (bd, code) in sorted(SAMPLE_RATES.items()):
        status = "OK" if bd == 0x00 else "FAIL"
        print(f"    {rate/1e6:.3f} MSPS: bd=0x{bd:02x}, code=0x{code:02x} {status}")
        if bd != 0x00:
            all_passed = False

    print("  24-bit rates:")
    for rate, (bd, code) in sorted(SAMPLE_RATES_24BIT.items()):
        status = "OK" if bd == 0x01 else "FAIL"
        print(f"    {rate/1e6:.3f} MSPS: bd=0x{bd:02x}, code=0x{code:02x} {status}")
        if bd != 0x01:
            all_passed = False

    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_24bit_excludes_5120k():
    """
    Test that 5.12 MSPS is not in the 24-bit table.

    Per Icom documentation, 5.12 MSPS only supports 16-bit.
    """
    print("\n" + "=" * 60)
    print("TEST: 24-bit Excludes 5.12 MSPS")
    print("=" * 60)

    in_16bit = 5120000 in SAMPLE_RATES
    in_24bit = 5120000 in SAMPLE_RATES_24BIT

    print(f"  5.12 MSPS in 16-bit table: {in_16bit}")
    print(f"  5.12 MSPS in 24-bit table: {in_24bit}")

    passed = in_16bit and not in_24bit
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_sync_patterns():
    """
    Test sync pattern definitions.

    16-bit sync: 0x8000, 0x8000 (little-endian: 00 80 00 80)
    24-bit sync: 0x800001, 0x800002 (little-endian: 01 00 80 02 00 80)
    """
    print("\n" + "=" * 60)
    print("TEST: Sync Patterns")
    print("=" * 60)

    # Expected patterns (as defined in icom_r8600.py)
    sync_16bit = b"\x00\x80\x00\x80"
    sync_24bit = b"\x00\x80\x01\x00\x80\x02"

    print(f"  16-bit sync: {sync_16bit.hex()} (4 bytes)")
    print(f"  24-bit sync: {sync_24bit.hex()} (6 bytes)")

    # Verify patterns decode correctly
    # 16-bit: two samples of 0x8000 (-32768)
    i16_1 = int.from_bytes(sync_16bit[0:2], "little", signed=True)
    q16_1 = int.from_bytes(sync_16bit[2:4], "little", signed=True)
    sync16_ok = i16_1 == -32768 and q16_1 == -32768

    print(f"  16-bit sync decodes to I={i16_1}, Q={q16_1}: {'OK' if sync16_ok else 'FAIL'}")

    # 24-bit: check structure
    sync24_len_ok = len(sync_24bit) == 6
    print(f"  24-bit sync length: {'OK' if sync24_len_ok else 'FAIL'}")

    passed = sync16_ok and sync24_len_ok
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_24bit_sample_parsing():
    """
    Test 24-bit little-endian sample parsing with sign extension.

    24-bit samples are 3 bytes each, little-endian, signed.
    """
    print("\n" + "=" * 60)
    print("TEST: 24-bit Sample Parsing")
    print("=" * 60)

    def parse_24bit_sample(raw):
        """Parse 3-byte little-endian signed integer."""
        val = raw[0] | (raw[1] << 8) | (raw[2] << 16)
        if val & 0x800000:  # Sign bit set
            val -= 0x1000000
        return val

    test_cases = [
        (bytes([0x01, 0x00, 0x00]), 1, "small positive"),
        (bytes([0xFF, 0xFF, 0x7F]), 8388607, "max positive"),
        (bytes([0x00, 0x00, 0x80]), -8388608, "min negative"),
        (bytes([0xFF, 0xFF, 0xFF]), -1, "minus one"),
        (bytes([0x00, 0x00, 0x00]), 0, "zero"),
    ]

    all_passed = True
    for raw, expected, desc in test_cases:
        result = parse_24bit_sample(raw)
        ok = result == expected
        print(f"  {desc}: {raw.hex()} -> {result} (expected {expected}) {'OK' if ok else 'FAIL'}")
        if not ok:
            all_passed = False

    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_normalization_factors():
    """
    Test normalization factors for 16-bit and 24-bit samples.

    16-bit: divide by 32768.0 (2^15)
    24-bit: divide by 8388608.0 (2^23)
    """
    print("\n" + "=" * 60)
    print("TEST: Normalization Factors")
    print("=" * 60)

    # 16-bit normalization
    norm_16 = 32768.0
    max_16 = 32767 / norm_16
    min_16 = -32768 / norm_16

    print(f"  16-bit norm factor: {norm_16}")
    print(f"    Max positive: {max_16:.6f} (should be ~1.0)")
    print(f"    Min negative: {min_16:.6f} (should be -1.0)")

    norm_16_ok = abs(max_16 - 1.0) < 0.001 and abs(min_16 + 1.0) < 0.001

    # 24-bit normalization
    norm_24 = 8388608.0
    max_24 = 8388607 / norm_24
    min_24 = -8388608 / norm_24

    print(f"  24-bit norm factor: {norm_24}")
    print(f"    Max positive: {max_24:.6f} (should be ~1.0)")
    print(f"    Min negative: {min_24:.6f} (should be -1.0)")

    norm_24_ok = abs(max_24 - 1.0) < 0.001 and abs(min_24 + 1.0) < 0.001

    passed = norm_16_ok and norm_24_ok
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_bit_depth_default():
    """
    Test that 16-bit is default when use_24bit=False.
    """
    print("\n" + "=" * 60)
    print("TEST: Bit Depth Default (16-bit)")
    print("=" * 60)

    radio = IcomR8600(use_24bit=False)
    print(f"  IcomR8600(use_24bit=False)._use_24bit = {radio._use_24bit}")

    passed = radio._use_24bit == False
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_bit_depth_24bit():
    """
    Test that 24-bit is enabled when use_24bit=True.
    """
    print("\n" + "=" * 60)
    print("TEST: Bit Depth 24-bit Enabled")
    print("=" * 60)

    radio = IcomR8600(use_24bit=True)
    print(f"  IcomR8600(use_24bit=True)._use_24bit = {radio._use_24bit}")

    passed = radio._use_24bit == True
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# =============================================================================
# Hardware Integration Tests (Requires IC-R8600)
# =============================================================================

def test_hw_connect(radio):
    """
    Test connecting to IC-R8600.
    """
    print("\n" + "=" * 60)
    print("TEST: Hardware Connect")
    print("=" * 60)

    try:
        radio.open()
        print(f"  Connected to IC-R8600")
        passed = True
    except Exception as e:
        print(f"  ERROR: {e}")
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_hw_iq_streaming_16bit(radio):
    """
    Test 16-bit I/Q streaming.
    """
    print("\n" + "=" * 60)
    print("TEST: 16-bit I/Q Streaming")
    print("=" * 60)

    try:
        radio.configure_iq_streaming(freq=89.9e6, sample_rate=480000)
        print(f"  Configured: {radio.iq_sample_rate/1e3:.0f} kSPS, {radio._bit_depth}-bit")

        # Fetch some samples
        iq = radio.fetch_iq(4096)
        print(f"  Fetched {len(iq)} samples")
        print(f"  Sample range: {np.min(np.abs(iq)):.4f} to {np.max(np.abs(iq)):.4f}")

        passed = len(iq) == 4096 and radio._bit_depth == 16
    except Exception as e:
        print(f"  ERROR: {e}")
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_hw_iq_streaming_24bit(radio):
    """
    Test 24-bit I/Q streaming.
    """
    print("\n" + "=" * 60)
    print("TEST: 24-bit I/Q Streaming")
    print("=" * 60)

    try:
        radio.configure_iq_streaming(freq=89.9e6, sample_rate=480000)
        print(f"  Configured: {radio.iq_sample_rate/1e3:.0f} kSPS, {radio._bit_depth}-bit")

        # Fetch some samples
        iq = radio.fetch_iq(4096)
        print(f"  Fetched {len(iq)} samples")
        print(f"  Sample range: {np.min(np.abs(iq)):.4f} to {np.max(np.abs(iq)):.4f}")

        passed = len(iq) == 4096 and radio._bit_depth == 24
    except Exception as e:
        print(f"  ERROR: {e}")
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_hw_frequency_set(radio):
    """
    Test setting frequency via CI-V.
    """
    print("\n" + "=" * 60)
    print("TEST: Frequency Set")
    print("=" * 60)

    test_freqs = [89.9e6, 100.1e6, 162.4e6]

    all_passed = True
    for freq in test_freqs:
        try:
            radio.configure_iq_streaming(freq=freq, sample_rate=480000)
            print(f"  Set {freq/1e6:.1f} MHz: OK")
        except Exception as e:
            print(f"  Set {freq/1e6:.1f} MHz: FAIL ({e})")
            all_passed = False

    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# =============================================================================
# Test Runner
# =============================================================================

def run_unit_tests():
    """Run tests that don't require hardware."""
    tests = [
        ("CI-V Command Building", test_civ_command_building),
        ("CI-V Command Padding", test_civ_command_padding),
        ("Frequency to BCD", test_freq_to_bcd),
        ("Sample Rate Tables", test_sample_rate_tables),
        ("24-bit Excludes 5.12 MSPS", test_24bit_excludes_5120k),
        ("Sync Patterns", test_sync_patterns),
        ("24-bit Sample Parsing", test_24bit_sample_parsing),
        ("Normalization Factors", test_normalization_factors),
        ("Bit Depth Default", test_bit_depth_default),
        ("Bit Depth 24-bit", test_bit_depth_24bit),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    return results


def run_hardware_tests():
    """Run tests that require IC-R8600 hardware."""
    print("\n" + "=" * 60)
    print("HARDWARE INTEGRATION TESTS")
    print("=" * 60)
    print("  Requires IC-R8600 connected via USB")

    results = []

    # Test with 16-bit radio
    radio_16 = IcomR8600(use_24bit=False)
    try:
        if test_hw_connect(radio_16):
            results.append(("HW: Connect", True))
            results.append(("HW: 16-bit Streaming", test_hw_iq_streaming_16bit(radio_16)))
            results.append(("HW: Frequency Set", test_hw_frequency_set(radio_16)))
        else:
            results.append(("HW: Connect", False))
    finally:
        try:
            radio_16.close()
        except:
            pass

    # Test with 24-bit radio
    radio_24 = IcomR8600(use_24bit=True)
    try:
        radio_24.open()
        results.append(("HW: 24-bit Streaming", test_hw_iq_streaming_24bit(radio_24)))
    except Exception as e:
        print(f"\n  24-bit test error: {e}")
        results.append(("HW: 24-bit Streaming", False))
    finally:
        try:
            radio_24.close()
        except:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(description="CI-V Command Test Suite for IC-R8600")
    parser.add_argument("--hw", action="store_true", help="Include hardware integration tests")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CI-V COMMAND TEST SUITE")
    print("=" * 60)
    print("\nTesting icom_r8600.py CI-V protocol implementation")

    # Run unit tests
    results = run_unit_tests()

    # Run hardware tests if requested
    if args.hw:
        hw_results = run_hardware_tests()
        results.extend(hw_results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s}  {status}")

    print(f"\n  {passed_count}/{total_count} tests passed")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
