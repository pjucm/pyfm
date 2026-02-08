#!/usr/bin/env python3
"""
RDS Baseband SNR Test

Tests RDS decoder performance with noise added at baseband (after FM demod),
which provides a more accurate and controllable SNR measurement for the RDS
subcarrier specifically.

When noise is added to the FM carrier (RF domain), the FM demodulation process
creates a "noise triangle" where high frequencies see more noise than low
frequencies. At 57 kHz, RDS sees ~9 dB more noise than the 19 kHz pilot.

This test adds noise directly to the baseband to avoid this effect.
"""

import numpy as np
from scipy import signal

from rds_decoder import (
    RDSDecoder,
    RDS_CARRIER_FREQ,
    RDS_SYMBOL_RATE,
    OFFSET_WORDS,
    PTY_NAMES,
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


def generate_clean_baseband(sample_rate, num_groups, rds_amp=0.06, pilot_amp=0.1,
                            audio_amp=0.5, pad_s=0.2):
    """Generate clean FM baseband signal."""
    sps = sample_rate / RDS_SYMBOL_RATE

    bits = _build_groups(num_groups)
    symbols = _differential_encode(bits)
    upsampled = _upsample_symbols(symbols, sps)

    # Shape symbols (201 taps to match receiver BPF)
    nyq = sample_rate / 2.0
    cutoff = 2400 / nyq
    taps = signal.firwin(201, cutoff, window=('kaiser', 7.0))
    shaped = signal.lfilter(taps, 1.0, upsampled)

    n_samples = len(shaped)
    t = np.arange(n_samples) / sample_rate

    pilot = pilot_amp * np.cos(2 * np.pi * PILOT_FREQ * t)
    audio = audio_amp * np.sin(2 * np.pi * 1000 * t)
    carrier = np.cos(2 * np.pi * RDS_CARRIER_FREQ * t)
    rds = rds_amp * shaped * carrier

    baseband = pilot + audio + rds

    # Add padding
    if pad_s > 0:
        pad_samples = int(pad_s * sample_rate)
        baseband = np.concatenate([
            np.zeros(pad_samples),
            baseband,
            np.zeros(pad_samples),
        ])

    return baseband


def add_baseband_noise(baseband, snr_db, rng, sample_rate):
    """
    Add bandlimited noise to baseband (simulates receiver noise).

    This adds noise in a band that affects the RDS signal (roughly 50-65 kHz)
    rather than the entire spectrum.
    """
    n = len(baseband)

    # Generate white noise
    noise = rng.normal(0, 1, n)

    # Bandpass filter noise to RDS region (50-65 kHz)
    nyq = sample_rate / 2.0
    low = 50000 / nyq
    high = min(65000 / nyq, 0.99)
    noise_bpf = signal.firwin(201, [low, high], pass_zero=False, window=('kaiser', 7.0))
    noise_filtered = signal.lfilter(noise_bpf, 1.0, noise)

    # Calculate noise power to achieve target SNR
    # Measure signal power in RDS band
    signal_bpf = signal.firwin(201, [low, high], pass_zero=False, window=('kaiser', 7.0))
    signal_filtered = signal.lfilter(signal_bpf, 1.0, baseband)
    signal_power = np.mean(signal_filtered**2)

    # Target noise power
    snr_linear = 10 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear

    # Scale noise to target power
    current_noise_power = np.mean(noise_filtered**2)
    if current_noise_power > 1e-10:
        noise_scaled = noise_filtered * np.sqrt(target_noise_power / current_noise_power)
    else:
        noise_scaled = noise_filtered

    return baseband + noise_scaled


def run_baseband_snr_test(snr_values, trials=10, sample_rate=250000, num_groups=30):
    """Run SNR sweep with baseband noise."""
    print("\n" + "=" * 60)
    print("BASEBAND SNR TEST (noise added after FM demod equivalent)")
    print("=" * 60)
    print(f"\nConfig: {trials} trials per SNR, {num_groups} groups per trial")
    print("Noise bandlimited to 50-65 kHz (RDS region)")
    print()

    results = []
    for snr_db in snr_values:
        block_rates = []
        decode_ok = 0

        for seed in range(trials):
            rng = np.random.default_rng(seed)

            # Generate clean baseband
            baseband = generate_clean_baseband(sample_rate, num_groups)

            # Add baseband noise
            noisy = add_baseband_noise(baseband, snr_db, rng, sample_rate)

            # Run decoder
            decoder = RDSDecoder(sample_rate=sample_rate)
            block_size = 8192
            result = None
            for i in range(0, len(noisy), block_size):
                result = decoder.process(noisy[i:i + block_size], use_coherent=True)

            block_rates.append(result['block_rate'])

            # Check if decode was successful
            if (result['station_name'] == "TESTPJFM" and
                result['block_rate'] > 0.8):
                decode_ok += 1

        avg_rate = np.mean(block_rates)
        decode_rate = decode_ok / trials
        results.append({
            'snr_db': snr_db,
            'block_rate': avg_rate,
            'decode_rate': decode_rate,
        })
        print(f"  SNR {snr_db:3d} dB: {avg_rate*100:5.1f}% block rate, "
              f"{decode_rate*100:.0f}% full decode")

    return results


def compare_rf_vs_baseband_noise():
    """Compare RF noise (FM carrier) vs baseband noise."""
    print("\n" + "=" * 60)
    print("RF vs BASEBAND NOISE COMPARISON")
    print("=" * 60)

    sample_rate = 250000
    num_groups = 30
    trials = 5

    # SNR values to test
    snr_values = [30, 25, 20, 15, 10]

    print("\n  RF NOISE (added to FM carrier, then demodulated):")
    for snr_db in snr_values:
        block_rates = []
        for seed in range(trials):
            rng = np.random.default_rng(seed)

            baseband = generate_clean_baseband(sample_rate, num_groups)

            # FM modulate
            deviation = 75000
            dt = 1.0 / sample_rate
            phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt
            iq = np.cos(phase) + 1j * np.sin(phase)

            # Add RF noise (AWGN on IQ)
            signal_power = np.mean(np.abs(iq)**2)
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            sigma = np.sqrt(noise_power / 2.0)
            noise = rng.normal(scale=sigma, size=iq.shape) + 1j * rng.normal(scale=sigma, size=iq.shape)
            iq_noisy = iq + noise

            # FM demodulate
            product = iq_noisy[1:] * np.conj(iq_noisy[:-1])
            phase_diff = np.angle(product)
            demod = phase_diff * sample_rate / (2 * np.pi * deviation)
            demod = np.concatenate([demod, demod[-1:]])

            # Decode
            decoder = RDSDecoder(sample_rate=sample_rate)
            for i in range(0, len(demod), 8192):
                result = decoder.process(demod[i:i + 8192], use_coherent=True)
            block_rates.append(result['block_rate'])

        avg_rate = np.mean(block_rates)
        print(f"    SNR {snr_db:3d} dB: {avg_rate*100:5.1f}%")

    print("\n  BASEBAND NOISE (added after FM demod, bandlimited to RDS):")
    for snr_db in snr_values:
        block_rates = []
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            baseband = generate_clean_baseband(sample_rate, num_groups)
            noisy = add_baseband_noise(baseband, snr_db, rng, sample_rate)

            decoder = RDSDecoder(sample_rate=sample_rate)
            for i in range(0, len(noisy), 8192):
                result = decoder.process(noisy[i:i + 8192], use_coherent=True)
            block_rates.append(result['block_rate'])

        avg_rate = np.mean(block_rates)
        print(f"    SNR {snr_db:3d} dB: {avg_rate*100:5.1f}%")


def main():
    compare_rf_vs_baseband_noise()

    print("\n" + "-" * 60)

    snr_values = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6]
    run_baseband_snr_test(snr_values, trials=10)


if __name__ == "__main__":
    main()
