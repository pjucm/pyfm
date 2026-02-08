#!/usr/bin/env python3
"""
RDS FM Isolation Test

Tests FM modulation effects on RDS decoding to isolate the cause of degradation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy import signal
from datetime import date

from rds_decoder import (
    RDSDecoder,
    RDS_CARRIER_FREQ,
    RDS_SYMBOL_RATE,
    OFFSET_WORDS,
    compute_syndrome,
)

PILOT_FREQ = 19000.0


def _crc_checkword(data16, offset_word):
    remainder = compute_syndrome((data16 & 0xFFFF) << 10)
    return (remainder ^ offset_word) & 0x3FF


def _build_block(data16, offset_word):
    check = _crc_checkword(data16, offset_word)
    return ((data16 & 0xFFFF) << 10) | check


def _block_to_bits(block26):
    return [(block26 >> i) & 1 for i in range(25, -1, -1)]


def _encode_group_0a(pi_code, pty, segment, ps_name):
    segment &= 0x03
    ps_name = (ps_name[:8]).ljust(8)
    char1 = ord(ps_name[segment * 2])
    char2 = ord(ps_name[segment * 2 + 1])
    block_a = pi_code
    block_b = (0 << 12) | (0 << 11) | (pty << 5) | segment
    block_c = 0x0000
    block_d = (char1 << 8) | char2
    bits = []
    for block, offset_name in [(block_a, 'A'), (block_b, 'B'), (block_c, 'C'), (block_d, 'D')]:
        full_block = _build_block(block, OFFSET_WORDS[offset_name])
        bits.extend(_block_to_bits(full_block))
    return bits


def _build_groups(num_groups):
    pi_code = 0x54A8
    pty = 1
    ps_name = "TESTPJFM"
    all_bits = []
    for i in range(num_groups):
        segment = i % 4
        all_bits.extend(_encode_group_0a(pi_code, pty, segment, ps_name))
    return np.array(all_bits, dtype=np.uint8)


def _differential_encode(bits):
    symbols = np.zeros(len(bits), dtype=np.float64)
    prev = 1.0
    for i, bit in enumerate(bits):
        if bit == 1:
            prev = -prev
        symbols[i] = prev
    return symbols


def _upsample_symbols(symbols, samples_per_symbol):
    n_samples = int(len(symbols) * samples_per_symbol)
    t = np.arange(n_samples) / samples_per_symbol
    symbol_idx = np.floor(t).astype(int)
    symbol_idx = np.clip(symbol_idx, 0, len(symbols) - 1)
    return symbols[symbol_idx]


def run_fm_test(sample_rate, num_groups, with_audio, audio_amp, pilot_amp, rds_amp,
                with_fm, description):
    """Run a single FM isolation test."""
    sps = sample_rate / RDS_SYMBOL_RATE

    # Generate RDS signal
    bits = _build_groups(num_groups)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    # Shape symbols
    nyq = sample_rate / 2.0
    cutoff = 2400 / nyq
    taps = signal.firwin(101, cutoff, window=('kaiser', 7.0))
    shaped = signal.lfilter(taps, 1.0, upsampled)

    n_samples = len(shaped)
    t = np.arange(n_samples) / sample_rate

    # Generate baseband components
    pilot = pilot_amp * np.cos(2 * np.pi * PILOT_FREQ * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = rds_amp * shaped * carrier

    if with_audio:
        audio = audio_amp * np.sin(2 * np.pi * 1000 * t)
        baseband = pilot + audio + rds
    else:
        baseband = pilot + rds

    if with_fm:
        # FM modulate
        deviation = 75000
        dt = 1.0 / sample_rate
        phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
        iq = np.cos(phase) + 1j * np.sin(phase)

        # FM demodulate
        product = iq[1:] * np.conj(iq[:-1])
        phase_diff = np.angle(product)
        recovered = phase_diff * sample_rate / (2 * np.pi * deviation)
        recovered = np.concatenate([recovered, recovered[-1:]])
    else:
        recovered = baseband

    # Run decoder
    decoder = RDSDecoder(sample_rate=sample_rate)
    block_size = 8192
    result = None
    for i in range(0, len(recovered), block_size):
        result = decoder.process(recovered[i:i + block_size], use_coherent=True)

    print(f"  {description}")
    print(f"    Block rate: {result['block_rate']*100:.1f}%")
    print(f"    Blocks: {result['blocks_received']}/{result['blocks_expected']}")
    print(f"    Groups: {result['groups_received']}")
    print(f"    Block errors: {result['block_errors']}")
    return result


def test_fm_isolation():
    """Test different FM configurations to isolate the issue."""
    print("\n" + "=" * 60)
    print("FM MODULATION ISOLATION TESTS")
    print("=" * 60)

    sample_rate = 250000
    num_groups = 20

    # Test 1: No FM, no audio (should pass)
    run_fm_test(sample_rate, num_groups,
                with_audio=False, audio_amp=0, pilot_amp=0.1, rds_amp=0.06,
                with_fm=False, description="No FM, no audio")

    # Test 2: No FM, with audio (should pass)
    run_fm_test(sample_rate, num_groups,
                with_audio=True, audio_amp=0.5, pilot_amp=0.1, rds_amp=0.06,
                with_fm=False, description="No FM, with audio (0.5)")

    # Test 3: FM, no audio
    run_fm_test(sample_rate, num_groups,
                with_audio=False, audio_amp=0, pilot_amp=0.1, rds_amp=0.06,
                with_fm=True, description="FM, no audio")

    # Test 4: FM, small audio
    run_fm_test(sample_rate, num_groups,
                with_audio=True, audio_amp=0.1, pilot_amp=0.1, rds_amp=0.06,
                with_fm=True, description="FM, small audio (0.1)")

    # Test 5: FM, medium audio
    run_fm_test(sample_rate, num_groups,
                with_audio=True, audio_amp=0.3, pilot_amp=0.1, rds_amp=0.06,
                with_fm=True, description="FM, medium audio (0.3)")

    # Test 6: FM, large audio
    run_fm_test(sample_rate, num_groups,
                with_audio=True, audio_amp=0.5, pilot_amp=0.1, rds_amp=0.06,
                with_fm=True, description="FM, large audio (0.5)")

    # Test 7: FM, very large audio
    run_fm_test(sample_rate, num_groups,
                with_audio=True, audio_amp=0.7, pilot_amp=0.1, rds_amp=0.06,
                with_fm=True, description="FM, very large audio (0.7)")

    print()


def test_sample_rate():
    """Test different sample rates."""
    print("\n" + "=" * 60)
    print("SAMPLE RATE TESTS")
    print("=" * 60)

    num_groups = 20

    for sample_rate in [237500, 250000, 312500, 500000]:
        sps = sample_rate / RDS_SYMBOL_RATE
        print(f"\n  Sample rate: {sample_rate} Hz (SPS: {sps:.3f})")

        run_fm_test(sample_rate, num_groups,
                    with_audio=True, audio_amp=0.5, pilot_amp=0.1, rds_amp=0.06,
                    with_fm=True, description=f"  FM with audio")


def test_deviation():
    """Test different FM deviation values."""
    print("\n" + "=" * 60)
    print("FM DEVIATION TESTS")
    print("=" * 60)

    sample_rate = 250000
    num_groups = 20
    sps = sample_rate / RDS_SYMBOL_RATE

    bits = _build_groups(num_groups)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    nyq = sample_rate / 2.0
    cutoff = 2400 / nyq
    taps = signal.firwin(101, cutoff, window=('kaiser', 7.0))
    shaped = signal.lfilter(taps, 1.0, upsampled)

    n_samples = len(shaped)
    t = np.arange(n_samples) / sample_rate

    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * shaped * carrier
    baseband = pilot + audio + rds

    for deviation in [50000, 75000, 100000, 150000]:
        dt = 1.0 / sample_rate
        phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
        iq = np.cos(phase) + 1j * np.sin(phase)

        product = iq[1:] * np.conj(iq[:-1])
        phase_diff = np.angle(product)
        recovered = phase_diff * sample_rate / (2 * np.pi * deviation)
        recovered = np.concatenate([recovered, recovered[-1:]])

        decoder = RDSDecoder(sample_rate=sample_rate)
        block_size = 8192
        result = None
        for i in range(0, len(recovered), block_size):
            result = decoder.process(recovered[i:i + block_size], use_coherent=True)

        print(f"  Deviation {deviation/1000:.0f} kHz: {result['block_rate']*100:.1f}% "
              f"({result['blocks_received']}/{result['blocks_expected']})")


def test_phase_continuity():
    """Test if phase discontinuity in FM is causing issues."""
    print("\n" + "=" * 60)
    print("PHASE CONTINUITY TEST")
    print("=" * 60)

    sample_rate = 250000
    num_groups = 20
    sps = sample_rate / RDS_SYMBOL_RATE

    bits = _build_groups(num_groups)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    nyq = sample_rate / 2.0
    cutoff = 2400 / nyq
    taps = signal.firwin(101, cutoff, window=('kaiser', 7.0))
    shaped = signal.lfilter(taps, 1.0, upsampled)

    n_samples = len(shaped)
    t = np.arange(n_samples) / sample_rate

    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * shaped * carrier
    baseband = pilot + audio + rds

    # FM mod/demod
    deviation = 75000
    dt = 1.0 / sample_rate
    phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
    iq = np.cos(phase) + 1j * np.sin(phase)

    product = iq[1:] * np.conj(iq[:-1])
    phase_diff = np.angle(product)
    recovered = phase_diff * sample_rate / (2 * np.pi * deviation)
    recovered = np.concatenate([recovered, recovered[-1:]])

    # Check phase discontinuities
    # Extract pilot from recovered signal
    pilot_bpf = signal.firwin(201, [18500/nyq, 19500/nyq],
                              pass_zero=False, window=('kaiser', 7.0))
    pilot_recovered = signal.lfilter(pilot_bpf, 1.0, recovered)

    # Analyze pilot phase
    analytic_pilot = signal.hilbert(pilot_recovered)
    pilot_phase = np.angle(analytic_pilot)

    # Unwrap phase
    pilot_phase_unwrapped = np.unwrap(pilot_phase)

    # Check phase derivative (should be constant = 2*pi*19000)
    phase_deriv = np.diff(pilot_phase_unwrapped) * sample_rate
    expected_deriv = 2 * np.pi * PILOT_FREQ

    # Skip transient
    skip = 5000
    phase_deriv = phase_deriv[skip:]

    mean_deriv = np.mean(phase_deriv)
    std_deriv = np.std(phase_deriv)

    print(f"  Pilot phase derivative:")
    print(f"    Expected: {expected_deriv:.1f} rad/s")
    print(f"    Mean: {mean_deriv:.1f} rad/s")
    print(f"    Std: {std_deriv:.1f} rad/s")
    print(f"    Frequency error: {(mean_deriv - expected_deriv)/(2*np.pi):.2f} Hz")

    # Check for phase jumps
    phase_jumps = np.abs(np.diff(pilot_phase_unwrapped)) > 0.5
    num_jumps = np.sum(phase_jumps)
    print(f"  Phase jumps (>0.5 rad): {num_jumps}")


def main():
    test_fm_isolation()
    test_deviation()
    test_phase_continuity()
    test_sample_rate()


if __name__ == "__main__":
    main()
