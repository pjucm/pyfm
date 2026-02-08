#!/usr/bin/env python3
"""
RDS Decoder Diagnostic Test

This test bypasses FM modulation to isolate RDS decoder issues.
Tests the decoder at baseband to identify where performance degrades.
"""

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
    """Build continuous stream of valid RDS groups."""
    pi_code = 0x54A8
    pty = 1
    ps_name = "TESTPJFM"
    all_bits = []
    for i in range(num_groups):
        segment = i % 4
        all_bits.extend(_encode_group_0a(pi_code, pty, segment, ps_name))
    return np.array(all_bits, dtype=np.uint8)


def _differential_encode(bits):
    """RDS differential encoding."""
    symbols = np.zeros(len(bits), dtype=np.float64)
    prev = 1.0
    for i, bit in enumerate(bits):
        if bit == 1:
            prev = -prev
        symbols[i] = prev
    return symbols


def _upsample_symbols(symbols, samples_per_symbol):
    """Upsample symbols to baseband sample rate."""
    n_samples = int(len(symbols) * samples_per_symbol)
    t = np.arange(n_samples) / samples_per_symbol
    symbol_idx = np.floor(t).astype(int)
    symbol_idx = np.clip(symbol_idx, 0, len(symbols) - 1)
    return symbols[symbol_idx]


def test_1_symbol_upsample():
    """Test 1: Verify symbol upsampling creates correct waveform."""
    print("\n" + "=" * 60)
    print("TEST 1: Symbol Upsampling")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE  # ~210.5 samples/symbol
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Symbol rate: {RDS_SYMBOL_RATE} Hz")
    print(f"  Samples per symbol: {sps:.2f}")

    # Create 10 symbols: alternating pattern
    symbols = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    upsampled = _upsample_symbols(symbols, sps)

    print(f"  Input symbols: {len(symbols)}")
    print(f"  Output samples: {len(upsampled)}")
    print(f"  Expected samples: ~{len(symbols) * sps:.0f}")

    # Verify each symbol occupies correct number of samples
    # Count transitions
    transitions = np.sum(np.abs(np.diff(upsampled)) > 0.5)
    expected_transitions = len(symbols) - 1
    print(f"  Transitions: {transitions} (expected: {expected_transitions})")

    return transitions == expected_transitions


def test_2_bpsk_modulation():
    """Test 2: Verify BPSK modulation to 57 kHz."""
    print("\n" + "=" * 60)
    print("TEST 2: BPSK Modulation to 57 kHz")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE
    duration = 0.1  # 100 ms
    n_symbols = int(duration * RDS_SYMBOL_RATE)

    # Create random symbols
    rng = np.random.default_rng(42)
    symbols = rng.choice([-1.0, 1.0], size=n_symbols)
    upsampled = _upsample_symbols(symbols, sps)

    # Modulate to 57 kHz
    t = np.arange(len(upsampled)) / sample_rate
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    modulated = upsampled * carrier

    # Check spectrum - should have energy at 57 kHz +/- 1187.5 Hz
    fft = np.fft.rfft(modulated)
    freqs = np.fft.rfftfreq(len(modulated), 1/sample_rate)

    # Find peak near 57 kHz
    mask = (freqs > 54000) & (freqs < 60000)
    peak_idx = np.argmax(np.abs(fft[mask]))
    peak_freq = freqs[mask][peak_idx]
    print(f"  Peak frequency: {peak_freq:.1f} Hz (expected: ~57000 Hz)")

    # Check bandwidth (should be ~2400 Hz centered on 57 kHz)
    total_power = np.sum(np.abs(fft)**2)
    bw_mask = (freqs > 54600) & (freqs < 59400)  # 57 kHz +/- 2400 Hz
    bw_power = np.sum(np.abs(fft[bw_mask])**2)
    power_ratio = bw_power / total_power
    print(f"  Power in 57 kHz +/- 2400 Hz: {power_ratio*100:.1f}%")

    return abs(peak_freq - 57000) < 500 and power_ratio > 0.9


def test_3_pilot_generation():
    """Test 3: Verify pilot extraction and tripling."""
    print("\n" + "=" * 60)
    print("TEST 3: Pilot Extraction and Tripling")
    print("=" * 60)

    sample_rate = 250000
    duration = 0.1
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Generate clean pilot
    pilot_amp = 0.1
    pilot = pilot_amp * np.cos(2 * np.pi * PILOT_FREQ * t)

    # Design pilot BPF (same as decoder)
    nyq = sample_rate / 2.0
    pilot_low = 18500 / nyq
    pilot_high = 19500 / nyq
    pilot_bpf = signal.firwin(201, [pilot_low, pilot_high],
                              pass_zero=False, window=('kaiser', 7.0))

    # Filter pilot
    filtered_pilot = signal.lfilter(pilot_bpf, 1.0, pilot)

    # Normalize and triple
    pilot_rms = np.sqrt(np.mean(filtered_pilot[1000:]**2))  # Skip transient
    print(f"  Pilot RMS: {pilot_rms:.6f} (input amp: {pilot_amp})")

    if pilot_rms > 1e-10:
        pilot_norm = filtered_pilot / (pilot_rms * np.sqrt(2))
        carrier_57k = 4 * pilot_norm**3 - 3 * pilot_norm

        # Check 57 kHz carrier quality
        fft = np.fft.rfft(carrier_57k[2000:])  # Skip transient
        freqs = np.fft.rfftfreq(len(carrier_57k) - 2000, 1/sample_rate)

        mask = (freqs > 50000) & (freqs < 65000)
        peak_idx = np.argmax(np.abs(fft[mask]))
        peak_freq = freqs[mask][peak_idx]
        print(f"  Tripled carrier peak: {peak_freq:.1f} Hz (expected: 57000 Hz)")

        # Check for spurious harmonics (19 kHz should be suppressed)
        mask_19k = (freqs > 18000) & (freqs < 20000)
        mask_57k = (freqs > 56000) & (freqs < 58000)
        power_19k = np.max(np.abs(fft[mask_19k]))
        power_57k = np.max(np.abs(fft[mask_57k]))
        rejection = 20 * np.log10(power_57k / max(power_19k, 1e-10))
        print(f"  57 kHz / 19 kHz ratio: {rejection:.1f} dB")

        return abs(peak_freq - 57000) < 100 and rejection > 10
    return False


def test_4_coherent_demod_clean():
    """Test 4: Coherent demodulation with clean signal (no noise)."""
    print("\n" + "=" * 60)
    print("TEST 4: Coherent Demodulation (Clean Signal)")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE

    # Generate 20 groups worth of bits
    bits = _build_groups(20)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    # Generate baseband with pilot
    n_samples = len(upsampled)
    t = np.arange(n_samples) / sample_rate
    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * upsampled * carrier

    baseband = pilot + rds
    print(f"  Generated {len(bits)} bits ({len(bits)//104} groups)")

    # Run decoder
    decoder = RDSDecoder(sample_rate=sample_rate)
    result = decoder.process(baseband, use_coherent=True)

    print(f"  Synced: {result['synced']}")
    print(f"  Blocks received: {result['blocks_received']}")
    print(f"  Blocks expected: {result['blocks_expected']}")
    print(f"  Block rate: {result['block_rate']*100:.1f}%")
    print(f"  Groups received: {result['groups_received']}")
    print(f"  CRC errors: {result['crc_errors']}")

    # With clean signal we should get >95% block rate
    success = result['block_rate'] > 0.95
    if not success:
        print(f"  FAIL: Block rate {result['block_rate']*100:.1f}% < 95%")
        print(f"  Block errors by position: {result['block_errors']}")
    return success


def test_5_delay_multiply_clean():
    """Test 5: Delay-multiply demodulation with clean signal."""
    print("\n" + "=" * 60)
    print("TEST 5: Delay-Multiply Demodulation (Clean Signal)")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE

    # Generate signal without pilot
    bits = _build_groups(20)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    n_samples = len(upsampled)
    t = np.arange(n_samples) / sample_rate
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * upsampled * carrier

    baseband = rds  # No pilot
    print(f"  Generated {len(bits)} bits ({len(bits)//104} groups)")

    # Run decoder
    decoder = RDSDecoder(sample_rate=sample_rate)
    result = decoder.process(baseband, use_coherent=False)

    print(f"  Synced: {result['synced']}")
    print(f"  Blocks received: {result['blocks_received']}")
    print(f"  Blocks expected: {result['blocks_expected']}")
    print(f"  Block rate: {result['block_rate']*100:.1f}%")
    print(f"  Groups received: {result['groups_received']}")

    success = result['block_rate'] > 0.95
    if not success:
        print(f"  FAIL: Block rate {result['block_rate']*100:.1f}% < 95%")
    return success


def test_6_symbol_timing_only():
    """Test 6: Symbol timing recovery in isolation."""
    print("\n" + "=" * 60)
    print("TEST 6: Symbol Timing Recovery (Baseband Only)")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE

    # Generate known symbol pattern
    bits = _build_groups(10)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    print(f"  Symbols: {len(symbols)}")
    print(f"  Upsampled: {len(upsampled)}")
    print(f"  Samples per symbol: {sps:.2f}")

    # Apply simple LPF to shape
    nyq = sample_rate / 2.0
    lpf_b, lpf_a = signal.butter(2, 800 / nyq, btype='low')
    shaped = signal.lfilter(lpf_b, lpf_a, upsampled)

    # Manually run timing recovery (simplified Gardner TED)
    mu = 0.5
    freq = 0.0
    freq_offset = 0.0
    gain_p = 0.015
    gain_i = 0.0008
    sps_int = int(sps)
    sps_frac = sps - sps_int

    recovered_symbols = []
    buffer = list(shaped)
    symbol_count = 0

    while True:
        half_sps = sps / 2.0
        pos_prev = mu
        pos_mid = mu + half_sps
        pos_next = mu + sps

        if len(buffer) < int(pos_next) + 2:
            break

        def interp(pos):
            idx = int(pos)
            frac = pos - idx
            return (1 - frac) * buffer[idx] + frac * buffer[idx + 1]

        prev_sample = interp(pos_prev)
        mid_sample = interp(pos_mid)
        next_sample = interp(pos_next)

        recovered_symbols.append(next_sample)
        symbol_count += 1

        # Gardner TED
        timing_error_raw = -(next_sample - prev_sample) * mid_sample
        norm = abs(next_sample) + abs(prev_sample) + abs(mid_sample) + 1e-10
        timing_error = np.clip(timing_error_raw / norm, -0.5, 0.5)

        mu += gain_p * timing_error
        freq_offset += gain_i * timing_error
        freq_offset = np.clip(freq_offset, -1.0, 1.0)

        freq += sps_frac + freq_offset
        advance = sps_int
        while freq >= 1.0:
            advance += 1
            freq -= 1.0
        while freq < 0.0:
            advance -= 1
            freq += 1.0

        while mu >= 1.0:
            mu -= 1.0
            advance += 1
        while mu < 0.0:
            mu += 1.0
            advance -= 1

        advance = max(1, advance)
        buffer = buffer[advance:]

    print(f"  Recovered {symbol_count} symbols (expected: ~{len(symbols)})")

    # Check correlation with original symbols
    if symbol_count > 100:
        recovered = np.array(recovered_symbols[:len(symbols)])
        # Hard decision
        recovered_bits = (recovered < 0).astype(int)
        original_bits = bits[:len(recovered_bits)]

        # Find best alignment (allow small offset)
        best_match = 0
        best_offset = 0
        for offset in range(-5, 6):
            if offset >= 0:
                match = np.sum(recovered_bits[offset:] == original_bits[:len(recovered_bits)-offset])
            else:
                match = np.sum(recovered_bits[:len(recovered_bits)+offset] == original_bits[-offset:])
            if match > best_match:
                best_match = match
                best_offset = offset

        match_rate = best_match / (len(recovered_bits) - abs(best_offset))
        print(f"  Best bit match rate: {match_rate*100:.1f}% at offset {best_offset}")

        # Also check inverted (differential encoding)
        best_match_inv = 0
        for offset in range(-5, 6):
            if offset >= 0:
                match = np.sum((1 - recovered_bits[offset:]) == original_bits[:len(recovered_bits)-offset])
            else:
                match = np.sum((1 - recovered_bits[:len(recovered_bits)+offset]) == original_bits[-offset:])
            if match > best_match_inv:
                best_match_inv = match

        match_rate_inv = best_match_inv / (len(recovered_bits) - abs(best_offset))
        print(f"  Inverted match rate: {match_rate_inv*100:.1f}%")

        return max(match_rate, match_rate_inv) > 0.95
    return False


def test_7_fm_roundtrip():
    """Test 7: FM modulation/demodulation roundtrip."""
    print("\n" + "=" * 60)
    print("TEST 7: FM Modulation Roundtrip")
    print("=" * 60)

    sample_rate = 250000
    duration = 0.1
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Create test baseband: pilot + tone + RDS carrier
    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    audio = 0.7 * np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone
    rds_carrier = 0.06 * np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    baseband = pilot + audio + rds_carrier

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

    # Check correlation
    correlation = np.corrcoef(baseband[1000:], recovered[1000:])[0, 1]
    print(f"  Input/output correlation: {correlation:.6f}")

    # Check pilot recovery
    nyq = sample_rate / 2.0
    pilot_bpf = signal.firwin(201, [18500/nyq, 19500/nyq],
                              pass_zero=False, window=('kaiser', 7.0))
    pilot_recovered = signal.lfilter(pilot_bpf, 1.0, recovered)

    pilot_rms_in = np.sqrt(np.mean(pilot[2000:]**2))
    pilot_rms_out = np.sqrt(np.mean(pilot_recovered[2000:]**2))
    print(f"  Pilot RMS in: {pilot_rms_in:.6f}")
    print(f"  Pilot RMS out: {pilot_rms_out:.6f}")
    print(f"  Pilot recovery ratio: {pilot_rms_out/pilot_rms_in:.3f}")

    # Check 57 kHz carrier recovery
    rds_bpf = signal.firwin(201, [54600/nyq, 59400/nyq],
                            pass_zero=False, window=('kaiser', 7.0))
    rds_recovered = signal.lfilter(rds_bpf, 1.0, recovered)
    rds_rms_out = np.sqrt(np.mean(rds_recovered[2000:]**2))
    print(f"  RDS carrier RMS out: {rds_rms_out:.6f}")

    return correlation > 0.99


def test_8_end_to_end_clean():
    """Test 8: Full end-to-end with FM modulation (no noise)."""
    print("\n" + "=" * 60)
    print("TEST 8: End-to-End with FM (Clean Signal)")
    print("=" * 60)

    sample_rate = 250000
    sps = sample_rate / RDS_SYMBOL_RATE

    # Generate signal
    bits = _build_groups(20)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    # Shape symbols (201 taps to match receiver BPF group delay)
    nyq = sample_rate / 2.0
    cutoff = 2400 / nyq
    taps = signal.firwin(201, cutoff, window=('kaiser', 7.0))
    shaped = signal.lfilter(taps, 1.0, upsampled)

    n_samples = len(shaped)
    t = np.arange(n_samples) / sample_rate

    # Generate baseband
    pilot = 0.1 * np.cos(2 * np.pi * PILOT_FREQ * t)
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = 0.06 * shaped * carrier
    baseband = pilot + audio + rds

    print(f"  Generated {len(bits)} bits ({len(bits)//104} groups)")

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

    # Run decoder
    decoder = RDSDecoder(sample_rate=sample_rate)
    block_size = 8192
    result = None
    for i in range(0, len(recovered), block_size):
        result = decoder.process(recovered[i:i + block_size], use_coherent=True)

    print(f"  Synced: {result['synced']}")
    print(f"  Blocks received: {result['blocks_received']}")
    print(f"  Blocks expected: {result['blocks_expected']}")
    print(f"  Block rate: {result['block_rate']*100:.1f}%")
    print(f"  Groups received: {result['groups_received']}")
    print(f"  CRC errors: {result['crc_errors']}")
    print(f"  Block errors: {result['block_errors']}")

    success = result['block_rate'] > 0.90
    if not success:
        print(f"  FAIL: Block rate {result['block_rate']*100:.1f}% < 90%")
    return success


def run_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("RDS DECODER DIAGNOSTIC TESTS")
    print("=" * 70)
    print("\nThese tests isolate different parts of the RDS decoder")
    print("to identify where performance degrades.\n")

    results = []

    tests = [
        ("Symbol Upsampling", test_1_symbol_upsample),
        ("BPSK Modulation", test_2_bpsk_modulation),
        ("Pilot Generation", test_3_pilot_generation),
        ("Coherent Demod (Clean)", test_4_coherent_demod_clean),
        ("Delay-Multiply (Clean)", test_5_delay_multiply_clean),
        ("Symbol Timing Only", test_6_symbol_timing_only),
        ("FM Roundtrip", test_7_fm_roundtrip),
        ("End-to-End (Clean)", test_8_end_to_end_clean),
    ]

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s} {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n  {passed_count}/{len(results)} tests passed")

    return passed_count == len(results)


if __name__ == "__main__":
    run_diagnostics()
