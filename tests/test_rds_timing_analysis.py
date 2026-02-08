#!/usr/bin/env python3
"""
RDS Timing Analysis

Deep dive into what's happening with symbol timing during FM processing.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy import signal

from rds_decoder import RDS_CARRIER_FREQ, RDS_SYMBOL_RATE, OFFSET_WORDS, compute_syndrome

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


def analyze_fm_distortion():
    """Analyze how FM modulation affects the RDS signal."""
    print("\n" + "=" * 60)
    print("FM DISTORTION ANALYSIS")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE
    duration = 0.2  # 200ms

    # Generate simple alternating symbol pattern
    n_symbols = int(duration * RDS_SYMBOL_RATE)
    symbols = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_symbols)])

    # Upsample
    n_samples = int(n_symbols * sps)
    t = np.arange(n_samples) / sample_rate
    symbol_idx = np.floor(t * RDS_SYMBOL_RATE).astype(int)
    symbol_idx = np.clip(symbol_idx, 0, len(symbols) - 1)
    upsampled = symbols[symbol_idx]

    # Shape symbols
    nyq = sample_rate / 2.0
    cutoff = 2400 / nyq
    shape_taps = signal.firwin(101, cutoff, window=('kaiser', 7.0))
    shaped = signal.lfilter(shape_taps, 1.0, upsampled)

    # Modulate to 57 kHz
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * shaped * carrier

    # Add pilot
    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    baseband = pilot + audio + rds

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

    # Extract pilot
    pilot_bpf = signal.firwin(201, [18500/nyq, 19500/nyq],
                              pass_zero=False, window=('kaiser', 7.0))
    pilot_recovered = signal.lfilter(pilot_bpf, 1.0, recovered)

    # Extract RDS
    rds_bpf = signal.firwin(201, [54600/nyq, 59400/nyq],
                            pass_zero=False, window=('kaiser', 7.0))
    rds_recovered = signal.lfilter(rds_bpf, 1.0, recovered)

    # Coherent demod: triple pilot to get 57 kHz carrier
    pilot_rms = np.sqrt(np.mean(pilot_recovered[5000:]**2))
    if pilot_rms > 1e-10:
        pilot_norm = pilot_recovered / (pilot_rms * np.sqrt(2))
        carrier_57k = 4 * pilot_norm**3 - 3 * pilot_norm
    else:
        print("  ERROR: No pilot detected")
        return

    # Mix RDS with carrier
    coherent_product = rds_recovered * carrier_57k

    # Differential decode: multiply by delayed version
    delay_samples = int(round(sps))
    n = len(coherent_product)
    delayed = np.zeros(n)
    delayed[delay_samples:] = coherent_product[:n - delay_samples]
    diff_decoded = coherent_product * delayed

    # LPF to extract baseband
    lpf_b, lpf_a = signal.butter(2, 800 / nyq, btype='low')
    demod_bb = signal.lfilter(lpf_b, lpf_a, diff_decoded)

    # Analyze demodulated baseband
    # Skip transients (5000 samples)
    skip = 5000
    analysis_bb = demod_bb[skip:]
    analysis_shaped = shaped[:len(analysis_bb)]

    # Compare with original shaped signal
    # Need to account for group delays
    # BPF: 100 samples, pilot BPF: 100 samples, LPF: small
    total_delay = 200  # Approximate

    # Normalize both signals
    bb_norm = analysis_bb / (np.std(analysis_bb) + 1e-10)
    shaped_norm = analysis_shaped / (np.std(analysis_shaped) + 1e-10)

    # Find correlation at different lags
    max_lag = 500
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(bb_norm[lag:lag+10000], shaped_norm[:10000])[0, 1]
        else:
            corr = np.corrcoef(bb_norm[:10000], shaped_norm[-lag:-lag+10000])[0, 1]
        correlations.append((lag, corr))

    best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))
    print(f"  Best correlation: {best_corr:.4f} at lag {best_lag} samples")
    print(f"  Lag in symbols: {best_lag / sps:.2f}")

    # Analyze symbol-rate content
    print("\n  Symbol Rate Analysis:")
    # FFT of demodulated baseband
    fft_bb = np.fft.rfft(analysis_bb[:50000])
    freqs = np.fft.rfftfreq(50000, 1/sample_rate)

    # Find power at symbol rate
    sym_mask = (freqs > RDS_SYMBOL_RATE - 50) & (freqs < RDS_SYMBOL_RATE + 50)
    sym_power = np.max(np.abs(fft_bb[sym_mask]))

    # Find power at 2x symbol rate (eye pattern)
    sym2_mask = (freqs > 2*RDS_SYMBOL_RATE - 50) & (freqs < 2*RDS_SYMBOL_RATE + 50)
    sym2_power = np.max(np.abs(fft_bb[sym2_mask]))

    dc_power = np.abs(fft_bb[0])

    print(f"  DC power: {dc_power:.2f}")
    print(f"  Symbol rate ({RDS_SYMBOL_RATE} Hz) power: {sym_power:.2f}")
    print(f"  2x symbol rate power: {sym2_power:.2f}")

    # Check for timing jitter
    # Find zero crossings
    signs = np.sign(analysis_bb)
    crossings = np.where(np.diff(signs) != 0)[0]

    if len(crossings) > 100:
        crossing_intervals = np.diff(crossings)
        expected_interval = sps / 2  # Half symbol period for alternating
        print(f"\n  Zero Crossing Analysis:")
        print(f"  Expected interval: {expected_interval:.1f} samples")
        print(f"  Mean interval: {np.mean(crossing_intervals):.1f} samples")
        print(f"  Std interval: {np.std(crossing_intervals):.1f} samples")
        print(f"  Min interval: {np.min(crossing_intervals):.1f} samples")
        print(f"  Max interval: {np.max(crossing_intervals):.1f} samples")

        # Jitter as percentage
        jitter_pct = 100 * np.std(crossing_intervals) / expected_interval
        print(f"  Timing jitter: {jitter_pct:.1f}%")

    print(f"\n  Analysis complete")


def analyze_group_delay_mismatch():
    """Check for group delay mismatch between BPFs."""
    print("\n" + "=" * 60)
    print("GROUP DELAY ANALYSIS")
    print("=" * 60)

    sample_rate = 250000
    nyq = sample_rate / 2.0

    # RDS BPF (57 kHz)
    rds_bw = 2000
    low = (RDS_CARRIER_FREQ - rds_bw) / nyq
    high = (RDS_CARRIER_FREQ + rds_bw) / nyq
    rds_bpf = signal.firwin(201, [low, high], pass_zero=False, window=('kaiser', 7.0))

    # Pilot BPF (19 kHz)
    pilot_low = 18500 / nyq
    pilot_high = 19500 / nyq
    pilot_bpf = signal.firwin(201, [pilot_low, pilot_high], pass_zero=False, window=('kaiser', 7.0))

    # Group delay of FIR filter = (N-1)/2 samples
    rds_delay = (len(rds_bpf) - 1) / 2
    pilot_delay = (len(pilot_bpf) - 1) / 2

    print(f"  RDS BPF taps: {len(rds_bpf)}, group delay: {rds_delay} samples")
    print(f"  Pilot BPF taps: {len(pilot_bpf)}, group delay: {pilot_delay} samples")
    print(f"  Delay mismatch: {rds_delay - pilot_delay} samples")

    # In time
    delay_ms = (rds_delay - pilot_delay) / sample_rate * 1000
    print(f"  Delay mismatch: {delay_ms:.4f} ms")

    # As fraction of symbol period
    symbol_period = 1.0 / RDS_SYMBOL_RATE
    delay_symbols = (rds_delay - pilot_delay) / sample_rate / symbol_period
    print(f"  Delay mismatch: {delay_symbols:.4f} symbol periods")


def analyze_lpf_effect():
    """Analyze how LPF cutoff affects symbol shape."""
    print("\n" + "=" * 60)
    print("LPF CUTOFF EFFECT ON SYMBOL SHAPE")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE
    nyq = sample_rate / 2.0

    # Create single symbol transition
    n_samples = int(10 * sps)  # 10 symbol periods
    t = np.arange(n_samples) / sample_rate

    # Symbol pattern: 5 highs, 5 lows
    symbols = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.float64)
    symbol_idx = np.floor(t * RDS_SYMBOL_RATE).astype(int)
    symbol_idx = np.clip(symbol_idx, 0, len(symbols) - 1)
    upsampled = symbols[symbol_idx]

    # Test different LPF cutoffs
    cutoffs = [600, 800, 1000, 1200, 1500]

    print(f"  Symbol rate: {RDS_SYMBOL_RATE} Hz")
    print(f"  Nyquist for symbols: {RDS_SYMBOL_RATE/2:.1f} Hz")
    print()

    for cutoff in cutoffs:
        lpf_b, lpf_a = signal.butter(2, cutoff / nyq, btype='low')
        filtered = signal.lfilter(lpf_b, lpf_a, upsampled)

        # Measure rise time (10% to 90%)
        # Find transition region
        transition_start = int(5 * sps) - int(sps)
        transition_end = int(5 * sps) + int(sps)
        segment = filtered[transition_start:transition_end]

        # Normalize
        seg_min = np.min(segment)
        seg_max = np.max(segment)
        seg_norm = (segment - seg_min) / (seg_max - seg_min + 1e-10)

        # Find 10% and 90% points
        try:
            idx_10 = np.where(seg_norm < 0.1)[0][-1]
            idx_90 = np.where(seg_norm < 0.9)[0][-1]
            rise_samples = idx_90 - idx_10
            rise_time_ms = rise_samples / sample_rate * 1000
            rise_time_symbols = rise_samples / sps
        except:
            rise_time_ms = 0
            rise_time_symbols = 0

        print(f"  Cutoff {cutoff:4d} Hz: rise time = {rise_time_ms:.3f} ms ({rise_time_symbols:.2f} symbols)")


def analyze_symbol_bandwidth():
    """Calculate theoretical symbol bandwidth requirements."""
    print("\n" + "=" * 60)
    print("SYMBOL BANDWIDTH REQUIREMENTS")
    print("=" * 60)

    # RDS symbol rate
    symbol_rate = RDS_SYMBOL_RATE
    print(f"  Symbol rate: {symbol_rate} Hz")

    # For NRZ signaling, bandwidth ~ symbol_rate (first null)
    # For raised-cosine with alpha=0.35, BW = (1+alpha) * symbol_rate / 2
    for alpha in [0.0, 0.25, 0.35, 0.5, 1.0]:
        bw = (1 + alpha) * symbol_rate / 2
        print(f"  Raised-cosine alpha={alpha}: BW = {bw:.1f} Hz")

    print()
    print("  RDS spec: ±2.4 kHz around 57 kHz = 4.8 kHz total bandwidth")
    print(f"  This supports alpha up to: {(4800 / symbol_rate - 1):.2f}")

    # For our LPF after demodulation
    print()
    print("  Post-demod LPF considerations:")
    print("  - Too narrow: attenuates symbol transitions, causes ISI")
    print("  - Too wide: passes more noise")
    print(f"  - Symbol rate / 2 = {symbol_rate/2:.1f} Hz (Nyquist)")
    print(f"  - Current LPF at 800 Hz is {800/(symbol_rate/2)*100:.0f}% of Nyquist")
    print(f"  - This attenuates high-frequency symbol components")


def analyze_differential_decoding():
    """Check differential decoding is working correctly."""
    print("\n" + "=" * 60)
    print("DIFFERENTIAL DECODING VERIFICATION")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE
    delay_samples = int(round(sps))

    print(f"  Samples per symbol: {sps:.2f}")
    print(f"  Delay samples (rounded): {delay_samples}")
    print(f"  Delay error: {(delay_samples - sps) / sps * 100:.2f}%")

    # This delay error accumulates over symbols
    # After 104 bits (one group), the error is:
    bits_per_group = 104
    accumulated_error = bits_per_group * (delay_samples - sps) / sps
    print(f"  After 1 group ({bits_per_group} bits): {accumulated_error:.2f} samples drift")
    print(f"  This is {accumulated_error/sps*100:.2f}% of a symbol period")

    # After multiple groups
    for n_groups in [1, 5, 10, 20]:
        drift = n_groups * bits_per_group * (delay_samples - sps)
        drift_symbols = drift / sps
        print(f"  After {n_groups:2d} groups: {drift:.1f} samples = {drift_symbols:.3f} symbols")


def main():
    """Run all timing analyses."""
    analyze_symbol_bandwidth()
    analyze_lpf_effect()
    analyze_group_delay_mismatch()
    analyze_differential_decoding()
    analyze_fm_distortion()


if __name__ == "__main__":
    main()
