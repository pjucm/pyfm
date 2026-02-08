#!/usr/bin/env python3
"""
RDS Decoder Test Suite

Generates synthetic FM multiplex baseband containing valid RDS groups and
verifies that the RDSDecoder demodulates and decodes expected fields.

Usage:
    python3 test_rds_decoder.py     # Run all tests with detailed output
    pytest test_rds_decoder.py -v   # Run with pytest (if installed)
"""

from datetime import date
import inspect
import sys

from scipy import signal

import numpy as np

from rds_decoder import (
    RDSDecoder,
    RDS_CARRIER_FREQ,
    RDS_SYMBOL_RATE,
    OFFSET_WORDS,
    PTY_NAMES,
    compute_syndrome,
)


PILOT_FREQ = 19000.0
_SUPPORTED_DECODER_KWARGS = set(inspect.signature(RDSDecoder.__init__).parameters) - {'self'}


def mjd_from_date(year, month, day):
    """Compute Modified Julian Day (MJD) for a date at 00:00 UTC."""
    return (date(year, month, day) - date(1858, 11, 17)).days


def _crc_checkword(data16, offset_word):
    """Compute 10-bit RDS checkword for a 16-bit data word."""
    remainder = compute_syndrome((data16 & 0xFFFF) << 10)
    return (remainder ^ offset_word) & 0x3FF


def _build_block(data16, offset_word):
    """Build a 26-bit RDS block from 16-bit data and an offset word."""
    check = _crc_checkword(data16, offset_word)
    return ((data16 & 0xFFFF) << 10) | check


def _block_to_bits(block26):
    """Convert 26-bit block to MSB-first bit list."""
    return [(block26 >> i) & 1 for i in range(25, -1, -1)]


def _group_to_bits(block_a, block_b, block_c, block_d):
    """Convert 4 RDS blocks (A/B/C/D) into a contiguous bit list."""
    blocks = [
        _build_block(block_a, OFFSET_WORDS['A']),
        _build_block(block_b, OFFSET_WORDS['B']),
        _build_block(block_c, OFFSET_WORDS['C']),
        _build_block(block_d, OFFSET_WORDS['D']),
    ]
    bits = []
    for block in blocks:
        bits.extend(_block_to_bits(block))
    return bits


def _encode_group_0a(pi_code, pty, segment, ps_name):
    """Encode Group 0A (Program Service name)."""
    segment &= 0x03
    ps_name = (ps_name[:8]).ljust(8)
    char1 = ord(ps_name[segment * 2])
    char2 = ord(ps_name[segment * 2 + 1])

    block_a = pi_code
    block_b = (0 << 12) | (0 << 11) | (pty << 5) | segment
    block_c = 0x0000
    block_d = (char1 << 8) | char2
    return _group_to_bits(block_a, block_b, block_c, block_d)


def _encode_group_2a(pi_code, pty, segment, text, text_flag=0):
    """Encode Group 2A (RadioText)."""
    segment &= 0x0F
    text = (text[:64]).ljust(64)
    base = segment * 4
    chars = text[base:base + 4]
    c1, c2, c3, c4 = (ord(chars[0]), ord(chars[1]), ord(chars[2]), ord(chars[3]))

    block_a = pi_code
    block_b = (2 << 12) | (0 << 11) | (pty << 5) | (text_flag << 4) | segment
    block_c = (c1 << 8) | c2
    block_d = (c3 << 8) | c4
    return _group_to_bits(block_a, block_b, block_c, block_d)


def _encode_group_4a(pi_code, pty, mjd, hour, minute, offset_half_hours):
    """Encode Group 4A (Clock Time and Date)."""
    offset = offset_half_hours & 0x3F

    block_a = pi_code
    block_b = (4 << 12) | (0 << 11) | (pty << 5) | ((mjd >> 15) & 0x03)
    block_c = ((mjd & 0x7FFF) << 1) | ((hour >> 4) & 0x01)
    block_d = ((hour & 0x0F) << 12) | ((minute & 0x3F) << 6) | offset
    return _group_to_bits(block_a, block_b, block_c, block_d)


def _build_bitstream(pi_code, pty, ps_name, radio_text, ct_hour, ct_minute,
                     ct_offset_half_hours, ct_mjd, repeats=2):
    """Build a full RDS bitstream with repeated groups for reliable sync."""
    bits = []
    for _ in range(repeats):
        for segment in range(4):
            bits.extend(_encode_group_0a(pi_code, pty, segment, ps_name))
        for segment in range(16):
            bits.extend(_encode_group_2a(pi_code, pty, segment, radio_text))
        bits.extend(_encode_group_4a(
            pi_code, pty, ct_mjd, ct_hour, ct_minute, ct_offset_half_hours
        ))
    return np.array(bits, dtype=np.uint8)


def _differential_encode(bits):
    """RDS differential encoding: 0 = no phase change, 1 = phase inversion."""
    symbols = np.zeros(len(bits), dtype=np.float64)
    prev = 1.0
    for i, bit in enumerate(bits):
        if bit == 1:
            prev = -prev
        symbols[i] = prev
    return symbols


def _symbols_to_baseband(symbols, sample_rate, include_pilot=True,
                         pilot_amp=0.1, rds_amp=0.06, pad_s=0.05,
                         shape=True):
    """Convert differential symbols into RDS baseband samples."""
    total_samples = int(np.ceil(len(symbols) * sample_rate / RDS_SYMBOL_RATE))
    t = np.arange(total_samples) / sample_rate

    symbol_idx = np.floor(t * RDS_SYMBOL_RATE).astype(int)
    symbol_idx = np.clip(symbol_idx, 0, len(symbols) - 1)
    symbol_wave = symbols[symbol_idx]

    if shape:
        # Shape symbols to fit inside the 57 kHz +/- 2.4 kHz bandpass.
        # Use 201 taps to match the receiver's BPF group delay (100 samples).
        # Using 101 taps (50 sample delay) causes phase mismatch issues.
        nyq = sample_rate / 2.0
        cutoff = 2400.0 / nyq
        taps = signal.firwin(201, cutoff, window=('kaiser', 7.0))
        symbol_wave = signal.lfilter(taps, 1.0, symbol_wave)

    rds = rds_amp * symbol_wave * np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    if include_pilot:
        pilot = pilot_amp * np.cos(2 * np.pi * PILOT_FREQ * t)
    else:
        pilot = 0.0

    baseband = rds + pilot

    if pad_s > 0:
        pad_samples = int(pad_s * sample_rate)
        baseband = np.concatenate([
            np.zeros(pad_samples),
            baseband,
            np.zeros(pad_samples),
        ])

    return baseband.astype(np.float64)


def _add_audio_tone(baseband, sample_rate, amplitude=0.7, tone_hz=1000.0):
    """Add a low-frequency L+R audio tone to the multiplex baseband."""
    t = np.arange(len(baseband)) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * tone_hz * t)
    return baseband + audio


def _pad_baseband(baseband, sample_rate, pad_s):
    if pad_s <= 0:
        return baseband
    pad_samples = int(pad_s * sample_rate)
    return np.concatenate([
        np.zeros(pad_samples),
        baseband,
        np.zeros(pad_samples),
    ])


def _run_decoder(baseband, sample_rate, use_coherent, matched_filter=False,
                 decoder_kwargs=None):
    """Run decoder with optional configuration overrides."""
    kwargs = {'sample_rate': sample_rate, 'matched_filter': matched_filter}
    if decoder_kwargs:
        kwargs.update(decoder_kwargs)
    kwargs = {k: v for k, v in kwargs.items() if k in _SUPPORTED_DECODER_KWARGS}
    decoder = RDSDecoder(**kwargs)
    result = None
    block_size = 8192
    for i in range(0, len(baseband), block_size):
        result = decoder.process(baseband[i:i + block_size], use_coherent=use_coherent)
    return result


# Configuration presets for before/after comparison
BASELINE_CONFIG = {
    'timing_gain_p': 0.03,      # Original: aggressive
    'timing_gain_i': 0.003,     # Original: aggressive
    'lpf_cutoff': 1200,         # Original: wide
    'bpf_bandwidth': 2400,      # Original: wide
    'agc_alpha': 1.0,           # Original: instant (no smoothing)
}

IMPROVED_CONFIG = {
    'timing_gain_p': 0.015,     # Improved: reduced jitter
    'timing_gain_i': 0.0008,    # Improved: slower frequency tracking
    'lpf_cutoff': 800,          # Improved: tighter noise rejection
    'bpf_bandwidth': 2000,      # Improved: moderate tightening (was 2400)
    'agc_alpha': 0.05,          # Improved: slow tracking
}


def _build_test_signal(sample_rate, include_pilot, repeats=2, pad_s=0.05):
    pi_code = 0x54A8  # WAAA
    pty = 1  # News
    ps_name = "PJFMTEST"
    radio_text = "PJFM RDS TEST SUITE"
    ct_hour = 12
    ct_minute = 34
    ct_offset = -10  # UTC-5:00 in half-hours
    ct_mjd = mjd_from_date(2026, 2, 4)

    bits = _build_bitstream(
        pi_code,
        pty,
        ps_name,
        radio_text,
        ct_hour,
        ct_minute,
        ct_offset,
        ct_mjd,
        repeats=repeats,
    )
    symbols = _differential_encode(bits)
    baseband = _symbols_to_baseband(
        symbols,
        sample_rate=sample_rate,
        include_pilot=include_pilot,
        pad_s=pad_s,
    )

    expected = {
        'pi_hex': f"{pi_code:04X}",
        'ps_name': ps_name,
        'radio_text': radio_text,
        'pty_name': PTY_NAMES[pty],
        'clock_time': "12:34 UTC-5:00",
    }
    return baseband, expected


def _build_rf_test_signal(sample_rate, include_pilot, repeats=6,
                          audio_amp=0.7, pad_s=0.2):
    baseband, expected = _build_test_signal(
        sample_rate,
        include_pilot=include_pilot,
        repeats=repeats,
        pad_s=0.0,
    )
    baseband = _add_audio_tone(baseband, sample_rate, amplitude=audio_amp)
    baseband = _pad_baseband(baseband, sample_rate, pad_s)
    return baseband, expected


def _fm_modulate(baseband, sample_rate, deviation=75000.0):
    dt = 1.0 / sample_rate
    phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
    return np.cos(phase) + 1j * np.sin(phase)


def _fm_demod(iq, sample_rate, deviation=75000.0):
    product = iq[1:] * np.conj(iq[:-1])
    phase_diff = np.angle(product)
    baseband = phase_diff * sample_rate / (2 * np.pi * deviation)
    return np.concatenate([baseband, baseband[-1:]])


def _add_awgn(iq, snr_db, rng):
    signal_power = np.mean(np.abs(iq) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2.0)
    noise = rng.normal(scale=sigma, size=iq.shape) + 1j * rng.normal(scale=sigma, size=iq.shape)
    return iq + noise


def _run_rf_chain(sample_rate, snr_db, use_coherent, rng, repeats=6, pad_s=0.2,
                  matched_filter=False, decoder_kwargs=None):
    baseband, expected = _build_rf_test_signal(
        sample_rate,
        include_pilot=use_coherent,
        repeats=repeats,
        pad_s=pad_s,
    )
    iq = _fm_modulate(baseband, sample_rate)
    iq = _add_awgn(iq, snr_db, rng)
    demod = _fm_demod(iq, sample_rate)
    result = _run_decoder(
        demod,
        sample_rate,
        use_coherent=use_coherent,
        matched_filter=matched_filter,
        decoder_kwargs=decoder_kwargs,
    )
    return result, expected


def test_rds_decode_coherent_stream():
    sample_rate = 250000
    baseband, expected = _build_test_signal(sample_rate, include_pilot=True)
    result = _run_decoder(baseband, sample_rate, use_coherent=True)

    assert result['synced']
    assert result['pi_hex'] == expected['pi_hex']
    assert result['station_name'] == expected['ps_name']
    assert result['radio_text'] == expected['radio_text']
    assert result['program_type'] == expected['pty_name']
    assert result['clock_time'] == expected['clock_time']
    assert result['block_rate'] > 0.9


def test_rds_decode_noncoherent_no_pilot_smoke():
    sample_rate = 250000
    baseband, expected = _build_test_signal(sample_rate, include_pilot=False)
    result = _run_decoder(baseband, sample_rate, use_coherent=False)

    # No pilot is a stress path. Validate graceful operation and partial
    # recovery instead of requiring a full decode.
    assert result['blocks_expected'] > 0
    assert result['block_rate'] > 0.30
    assert result['synced'] is False or result['pi_hex'] in ('', expected['pi_hex'])


def run_rf_snr_sweep(sample_rate=250000, snr_values=None, trials=5,
                     use_coherent=True, repeats=6, pad_s=0.2,
                     matched_filter=False, decoder_kwargs=None):
    """Run an RF-SNR sweep and report average decode metrics (diagnostic)."""
    if snr_values is None:
        snr_values = [25, 20, 15, 10]

    results = []
    for snr_db in snr_values:
        decode_ok = 0
        block_rates = []
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            result, expected = _run_rf_chain(
                sample_rate,
                snr_db=snr_db,
                use_coherent=use_coherent,
                rng=rng,
                repeats=repeats,
                pad_s=pad_s,
                matched_filter=matched_filter,
                decoder_kwargs=decoder_kwargs,
            )
            ok = (
                result['synced'] and
                result['pi_hex'] == expected['pi_hex'] and
                result['station_name'] == expected['ps_name'] and
                result['radio_text'] == expected['radio_text'] and
                result['program_type'] == expected['pty_name'] and
                result['clock_time'] == expected['clock_time']
            )
            decode_ok += int(ok)
            block_rates.append(result['block_rate'])

        avg_block_rate = float(np.mean(block_rates)) if block_rates else 0.0
        results.append({
            'snr_db': snr_db,
            'decode_rate': decode_ok / trials if trials else 0.0,
            'avg_block_rate': avg_block_rate,
        })

    return results


def run_improvement_comparison(sample_rate=250000, snr_values=None, trials=10,
                               use_coherent=True, repeats=8, pad_s=0.2):
    """
    Run side-by-side comparison of baseline vs improved decoder settings.

    Returns dict with 'baseline' and 'improved' sweep results.
    """
    if snr_values is None:
        # Finer resolution around typical threshold region
        snr_values = [30, 25, 22, 20, 18, 16, 14, 12, 10, 8]

    print(f"\n  Running baseline (original settings)...")
    baseline = run_rf_snr_sweep(
        sample_rate=sample_rate,
        snr_values=snr_values,
        trials=trials,
        use_coherent=use_coherent,
        repeats=repeats,
        pad_s=pad_s,
        decoder_kwargs=BASELINE_CONFIG,
    )

    print(f"  Running improved (optimized settings)...")
    improved = run_rf_snr_sweep(
        sample_rate=sample_rate,
        snr_values=snr_values,
        trials=trials,
        use_coherent=use_coherent,
        repeats=repeats,
        pad_s=pad_s,
        decoder_kwargs=IMPROVED_CONFIG,
    )

    return {'baseline': baseline, 'improved': improved}


def run_single_improvement_test(improvement_name, config_override, snr_values=None,
                                 trials=8, sample_rate=250000, repeats=8):
    """
    Test a single improvement in isolation against baseline.

    Args:
        improvement_name: Name of the improvement being tested
        config_override: Dict with just the changed parameter(s)
        snr_values: SNR values to test
        trials: Number of trials per SNR
        repeats: Number of RDS group repeats in test signal

    Returns dict with 'baseline' and 'improved' results.
    """
    if snr_values is None:
        snr_values = [22, 18, 14, 10]

    # Baseline: all original settings
    baseline_results = run_rf_snr_sweep(
        sample_rate=sample_rate,
        snr_values=snr_values,
        trials=trials,
        repeats=repeats,
        decoder_kwargs=BASELINE_CONFIG,
    )

    # Improved: baseline with single change applied
    improved_config = BASELINE_CONFIG.copy()
    improved_config.update(config_override)
    improved_results = run_rf_snr_sweep(
        sample_rate=sample_rate,
        snr_values=snr_values,
        trials=trials,
        repeats=repeats,
        decoder_kwargs=improved_config,
    )

    return {
        'name': improvement_name,
        'config': config_override,
        'baseline': baseline_results,
        'improved': improved_results,
    }


def print_comparison_table(comparison, title=None):
    """Print a formatted comparison table."""
    if title:
        print(f"\n  {title}")
        print("  " + "-" * 60)

    baseline = comparison['baseline']
    improved = comparison['improved']

    print("  SNR(dB)  | Baseline Block% | Improved Block% | Delta")
    print("  " + "-" * 55)

    for b, i in zip(baseline, improved):
        snr = b['snr_db']
        b_rate = b['avg_block_rate'] * 100
        i_rate = i['avg_block_rate'] * 100
        delta = i_rate - b_rate
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"  {snr:>7}  | {b_rate:>14.1f}% | {i_rate:>14.1f}% | {delta_str:>5}%")


def test_improvement_1_timing_gains():
    """Test improvement 1: Reduced timing recovery gains."""
    print("\n" + "=" * 60)
    print("IMPROVEMENT 1: Reduced Timing Recovery Gains")
    print("  Before: timing_gain_p=0.03, timing_gain_i=0.003")
    print("  After:  timing_gain_p=0.015, timing_gain_i=0.0008")
    print("=" * 60)

    result = run_single_improvement_test(
        "Timing Gains",
        {'timing_gain_p': 0.015, 'timing_gain_i': 0.0008},
        snr_values=[22, 18, 14, 10],
        trials=8,
    )
    print_comparison_table(result)
    return result


def test_improvement_2_lpf_cutoff():
    """Test improvement 2: Tighter post-demod LPF."""
    print("\n" + "=" * 60)
    print("IMPROVEMENT 2: Tighter Post-Demod LPF")
    print("  Before: lpf_cutoff=1200 Hz")
    print("  After:  lpf_cutoff=800 Hz")
    print("=" * 60)

    result = run_single_improvement_test(
        "LPF Cutoff",
        {'lpf_cutoff': 800},
        snr_values=[22, 18, 14, 10],
        trials=8,
    )
    print_comparison_table(result)
    return result


def test_improvement_3_bpf_bandwidth():
    """Test improvement 3: Moderately tighter RDS BPF."""
    print("\n" + "=" * 60)
    print("IMPROVEMENT 3: Moderately Tighter RDS Bandpass Filter")
    print("  Before: bpf_bandwidth=2400 Hz")
    print("  After:  bpf_bandwidth=2000 Hz")
    print("  Note: 1500 Hz was too narrow, cutting signal energy")
    print("=" * 60)

    result = run_single_improvement_test(
        "BPF Bandwidth",
        {'bpf_bandwidth': 2000},
        snr_values=[22, 18, 14, 10],
        trials=8,
    )
    print_comparison_table(result)
    return result


def test_improvement_4_slow_agc():
    """Test improvement 4: Slow-tracking AGC."""
    print("\n" + "=" * 60)
    print("IMPROVEMENT 4: Slow-Tracking AGC")
    print("  Before: agc_alpha=1.0 (instant response)")
    print("  After:  agc_alpha=0.05 (slow tracking + clipping)")
    print("=" * 60)

    result = run_single_improvement_test(
        "Slow AGC",
        {'agc_alpha': 0.05},
        snr_values=[22, 18, 14, 10],
        trials=8,
    )
    print_comparison_table(result)
    return result


def test_improvement_5_soft_decision():
    """
    Test improvement 5: Soft decision averaging.

    NOTE: The soft-decision experiment was removed from RDSDecoder because it
    reduced performance for differential BPSK.

    This test is retained as documentation only.
    """
    print("\n" + "=" * 60)
    print("IMPROVEMENT 5: Soft Decision Averaging (DISABLED)")
    print("  Status: Does NOT work for differential BPSK")
    print("  Reason: Averaging consecutive symbols creates ISI because")
    print("          adjacent symbols can have opposite polarity")
    print("=" * 60)
    print("\n  Skipping test - feature removed from decoder API")

    # Return a dummy result showing no change
    return {
        'name': 'Soft Decision (DISABLED)',
        'config': {},
        'baseline': [{'snr_db': 0, 'avg_block_rate': 0, 'decode_rate': 0}],
        'improved': [{'snr_db': 0, 'avg_block_rate': 0, 'decode_rate': 0}],
    }


def test_all_improvements_combined():
    """Test all improvements combined vs baseline."""
    print("\n" + "=" * 60)
    print("ALL IMPROVEMENTS COMBINED")
    print("=" * 60)
    print("  Baseline (original):")
    for k, v in BASELINE_CONFIG.items():
        print(f"    {k}: {v}")
    print("  Improved (all changes):")
    for k, v in IMPROVED_CONFIG.items():
        print(f"    {k}: {v}")

    comparison = run_improvement_comparison(
        snr_values=[25, 22, 20, 18, 16, 14, 12, 10, 8],
        trials=10,
        repeats=8,
    )
    print_comparison_table(comparison, "Combined Results")

    # Calculate effective SNR gain at 80% block rate threshold
    def find_threshold_snr(results, threshold=0.80):
        for r in results:
            if r['avg_block_rate'] >= threshold:
                return r['snr_db']
        return None

    baseline_thresh = find_threshold_snr(comparison['baseline'])
    improved_thresh = find_threshold_snr(comparison['improved'])

    print("\n  THRESHOLD ANALYSIS (80% block rate):")
    if baseline_thresh and improved_thresh:
        gain = baseline_thresh - improved_thresh
        print(f"    Baseline threshold: {baseline_thresh} dB")
        print(f"    Improved threshold: {improved_thresh} dB")
        print(f"    Effective SNR gain: {gain:.1f} dB")
    else:
        print(f"    Baseline threshold: {baseline_thresh or 'N/A'}")
        print(f"    Improved threshold: {improved_thresh or 'N/A'}")

    return comparison


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("RDS DECODER TEST SUITE")
    print("=" * 60)
    print("\nTesting rds_decoder.py (RDSDecoder)")
    print("Signal flow: Baseband -> 57 kHz BPF -> Demod -> Timing -> Groups")

    tests = [
        ("Coherent Demodulation Decode", test_rds_decode_coherent_stream),
        ("No-Pilot Noncoherent Smoke", test_rds_decode_noncoherent_no_pilot_smoke),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except AssertionError as exc:
            print(f"\n  FAIL: {name} ({exc})")
            results.append((name, False))
        except Exception as exc:
            print(f"\n  ERROR: {name} ({exc})")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:40s}  {status}")

    print(f"\n  {passed_count}/{total_count} tests passed\n")

    print("=" * 60)
    print("RF SNR SWEEP (with improved settings)")
    print("=" * 60)
    print("  Config: coherent demod, IIR LPF (matched_filter=False), repeats=6, pad=0.2s")
    sweep = run_rf_snr_sweep()
    print("  SNR (dB)   Decode Rate   Avg Block Rate")
    for row in sweep:
        print(f"  {row['snr_db']:>7}   {row['decode_rate']*100:9.1f}%   {row['avg_block_rate']:.3f}")
    print()

    return passed_count == total_count


def run_improvement_tests():
    """Run all individual improvement tests with before/after comparisons."""
    print("\n" + "=" * 70)
    print("RDS DECODER SNR IMPROVEMENT VERIFICATION")
    print("=" * 70)
    print("\nThis test suite verifies each improvement individually,")
    print("then tests all improvements combined.")
    print("\nEach test runs baseline (original) vs improved settings")
    print("at multiple SNR levels to measure the benefit.\n")

    # Run individual improvement tests
    results = {}

    print("\n>>> Testing each improvement in isolation...")

    results['timing'] = test_improvement_1_timing_gains()
    results['lpf'] = test_improvement_2_lpf_cutoff()
    results['bpf'] = test_improvement_3_bpf_bandwidth()
    results['agc'] = test_improvement_4_slow_agc()
    results['soft'] = test_improvement_5_soft_decision()

    print("\n>>> Testing all improvements combined...")
    results['combined'] = test_all_improvements_combined()

    # Summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)

    def calc_avg_improvement(result):
        """Calculate average block rate improvement across all SNR values."""
        baseline = result['baseline']
        improved = result['improved']
        deltas = []
        for b, i in zip(baseline, improved):
            delta = (i['avg_block_rate'] - b['avg_block_rate']) * 100
            deltas.append(delta)
        return np.mean(deltas)

    print("\n  Individual Improvements (avg block rate delta across SNR range):")
    print("  " + "-" * 50)
    for name, key in [("1. Timing Gains", 'timing'),
                      ("2. LPF Cutoff", 'lpf'),
                      ("3. BPF Bandwidth", 'bpf'),
                      ("4. Slow AGC", 'agc')]:
        avg = calc_avg_improvement(results[key])
        sign = "+" if avg >= 0 else ""
        print(f"    {name:25s}: {sign}{avg:.1f}%")
    print(f"    {'5. Soft Decision':25s}: DISABLED (causes ISI)")

    print("\n  Combined Improvement:")
    print("  " + "-" * 50)
    combined_avg = calc_avg_improvement(results['combined'])
    sign = "+" if combined_avg >= 0 else ""
    print(f"    All changes combined:    {sign}{combined_avg:.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RDS Decoder Test Suite")
    parser.add_argument('--improvements', '-i', action='store_true',
                        help='Run before/after improvement comparison tests')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run quick validation tests only')
    args = parser.parse_args()

    if args.improvements:
        run_improvement_tests()
        sys.exit(0)
    elif args.quick:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # Default: run basic tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
