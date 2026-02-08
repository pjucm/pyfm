#!/usr/bin/env python3
"""Test if pulse shaping filter is causing the issue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy import signal

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


def run_test(sample_rate, shaping, filter_taps, description):
    """Run single test."""
    sps = sample_rate / RDS_SYMBOL_RATE
    num_groups = 20

    bits = _build_groups(num_groups)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    if shaping:
        nyq = sample_rate / 2.0
        cutoff = 2400 / nyq
        taps = signal.firwin(filter_taps, cutoff, window=('kaiser', 7.0))
        shaped = signal.lfilter(taps, 1.0, upsampled)
    else:
        shaped = upsampled

    n_samples = len(shaped)
    t = np.arange(n_samples) / sample_rate

    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * shaped * carrier
    baseband = pilot + rds

    decoder = RDSDecoder(sample_rate=sample_rate)
    block_size = 8192
    result = None
    for i in range(0, len(baseband), block_size):
        result = decoder.process(baseband[i:i + block_size], use_coherent=True)

    print(f"  {description}: {result['block_rate']*100:.1f}% "
          f"({result['blocks_received']}/{result['blocks_expected']}) "
          f"errors: {result['block_errors']}")
    return result


def main():
    print("\n" + "=" * 60)
    print("PULSE SHAPING FILTER TESTS")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE
    print(f"\nSample rate: {sample_rate} Hz, SPS: {sps:.3f}")

    print("\nWithout shaping:")
    run_test(sample_rate, shaping=False, filter_taps=0, description="No shaping")

    print("\nWith shaping (different filter lengths):")
    for taps in [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 151, 201]:
        delay = (taps - 1) // 2
        run_test(sample_rate, shaping=True, filter_taps=taps,
                 description=f"{taps:3d} taps (delay {delay:3d})")

    print("\n" + "=" * 60)
    print("GROUP DELAY ANALYSIS")
    print("=" * 60)

    # The receiver has BPF with 201 taps (100 sample delay)
    # and pilot BPF with 201 taps (100 sample delay)
    # So there's no mismatch there.

    # But the transmit shaping filter adds extra delay that isn't
    # compensated in the receiver.

    print("\nReceiver filters:")
    print("  RDS BPF: 201 taps -> 100 sample delay")
    print("  Pilot BPF: 201 taps -> 100 sample delay")
    print("  (Matched, so no phase error)")

    print("\nTransmit filter (when used):")
    for taps in [101]:
        delay = (taps - 1) // 2
        delay_ms = delay / sample_rate * 1000
        delay_symbols = delay / sps
        print(f"  {taps} taps: {delay} samples = {delay_ms:.3f} ms = {delay_symbols:.3f} symbols")

    # The issue is that the transmit filter delay interacts with the
    # differential decoding delay. The receiver expects to compare
    # sample N with sample N-211, but the filter delay means the
    # actual symbol boundaries are shifted.

    print("\nHypothesis: Transmit filter delay shifts symbol boundaries")
    print("relative to where the receiver expects them.")


if __name__ == "__main__":
    main()
