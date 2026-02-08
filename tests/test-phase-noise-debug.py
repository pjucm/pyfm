#!/usr/bin/env python3
"""
Debug phase noise test to understand PLL behavior.
"""

import numpy as np
from scipy import signal

from pll_stereo_decoder import PLLStereoDecoder
from demodulator import FMStereoDecoder


def generate_composite_with_phase_noise(test_freq, duration, fs_baseband, phase_noise_std):
    """
    Generate FM stereo composite with pilot phase noise.

    Returns: composite, pilot_clean, pilot_noisy, carrier_tx
    """
    n_samples = int(duration * fs_baseband)
    t = np.arange(n_samples) / fs_baseband

    # Generate test tone
    test_tone = np.sin(2 * np.pi * test_freq * t) * 0.4
    nyq = fs_baseband / 2
    lpf = signal.firwin(201, 15000 / nyq, window='hamming')
    test_tone = signal.filtfilt(lpf, 1, test_tone)

    # Left channel only
    left = test_tone
    right = np.zeros_like(test_tone)
    lr_sum = left + right
    lr_diff = left - right

    # Generate pilot with phase noise
    if phase_noise_std > 0:
        # REALISTIC phase noise: band-limited to simulate oscillator jitter
        # Use low-frequency random walk (< 1 Hz) to simulate slow drift
        # Plus small high-frequency jitter (< 10 Hz) to simulate thermal noise

        # Low-frequency drift (< 1 Hz)
        drift_noise = np.random.randn(n_samples) * phase_noise_std
        # Lowpass filter to < 1 Hz
        b, a = signal.butter(2, 1.0 / (fs_baseband / 2), 'low')
        phase_drift = signal.lfilter(b, a, drift_noise)

        # Integrate to get phase (makes it a random walk in phase)
        phase_drift = np.cumsum(phase_drift) / fs_baseband * 100  # Scale factor

        # High-frequency jitter (< 10 Hz)
        jitter_noise = np.random.randn(n_samples) * phase_noise_std * 0.1
        b_jit, a_jit = signal.butter(2, 10.0 / (fs_baseband / 2), 'low')
        phase_jitter = signal.lfilter(b_jit, a_jit, jitter_noise)

        # Combined phase noise
        phase_noise = phase_drift + phase_jitter

        pilot_noisy = np.cos(2 * np.pi * 19000 * t + phase_noise)
        # Carrier derived from noisy pilot
        carrier_tx = 2 * np.cos(2 * np.pi * 19000 * t + phase_noise) ** 2 - 1
        pilot_clean = np.cos(2 * np.pi * 19000 * t)
    else:
        pilot_noisy = np.cos(2 * np.pi * 19000 * t)
        carrier_tx = 2 * pilot_noisy ** 2 - 1
        pilot_clean = pilot_noisy
        phase_noise = np.zeros(n_samples)

    # FM stereo composite
    composite = lr_sum + 0.09 * pilot_noisy + lr_diff * carrier_tx

    return composite, pilot_clean, pilot_noisy, carrier_tx, phase_noise


def fm_modulate(baseband, fs, deviation=75000):
    """FM modulate to IQ."""
    phase = 2 * np.pi * deviation * np.cumsum(baseband) / fs
    return np.exp(1j * phase).astype(np.complex64)


def analyze_phase_noise(phase_noise_std_values, test_freq=1000, duration=2.0):
    """Analyze phase noise effects on both decoders."""

    fs_baseband = 480000
    fs_audio = 48000

    print("=" * 80)
    print("PHASE NOISE DEBUG ANALYSIS")
    print("=" * 80)
    print(f"\nTest parameters:")
    print(f"  Duration: {duration} s")
    print(f"  Test frequency: {test_freq} Hz (left channel only)")
    print(f"  Baseband rate: {fs_baseband} Hz")
    print(f"  Audio rate: {fs_audio} Hz")

    for phase_noise_std in phase_noise_std_values:
        print(f"\n{'='*80}")
        print(f"PHASE NOISE STD: {phase_noise_std:.3f} rad")
        print(f"{'='*80}")

        # Generate composite
        composite, pilot_clean, pilot_noisy, carrier_tx, phase_noise = \
            generate_composite_with_phase_noise(test_freq, duration, fs_baseband, phase_noise_std)

        # Analyze phase noise characteristics
        if phase_noise_std > 0:
            phase_noise_rms = np.sqrt(np.mean(phase_noise ** 2))
            phase_noise_pk = np.max(np.abs(phase_noise))

            # Compute instantaneous frequency deviation from phase noise
            # f_dev = (1/2π) * dφ/dt
            phase_noise_diff = np.diff(phase_noise)
            freq_deviation = phase_noise_diff * fs_baseband / (2 * np.pi)
            freq_dev_rms = np.sqrt(np.mean(freq_deviation ** 2))
            freq_dev_pk = np.max(np.abs(freq_deviation))

            print(f"\nPhase noise statistics:")
            print(f"  RMS: {phase_noise_rms:.6f} rad")
            print(f"  Peak: {phase_noise_pk:.6f} rad")
            print(f"  Frequency deviation RMS: {freq_dev_rms:.3f} Hz")
            print(f"  Frequency deviation peak: {freq_dev_pk:.3f} Hz")
            print(f"  (PLL loop bandwidth: 30 Hz)")

            # Measure pilot phase error
            pilot_error = np.sqrt(np.mean((pilot_noisy - pilot_clean) ** 2))
            print(f"  Pilot amplitude error RMS: {pilot_error:.6f}")

        # FM modulate
        iq_samples = fm_modulate(composite, fs_baseband)

        # Test PLL decoder
        print(f"\n--- PLLStereoDecoder ---")
        decoder_pll = PLLStereoDecoder(iq_sample_rate=fs_baseband, audio_sample_rate=fs_audio)

        # Decode in blocks
        block_size = 8192
        warmup_blocks = 5
        left_chunks = []
        right_chunks = []
        pll_locked_history = []
        phase_error_history = []
        freq_offset_history = []

        for i in range(0, len(iq_samples), block_size):
            block = iq_samples[i:i+block_size]
            if len(block) < block_size // 2:
                break

            audio = decoder_pll.demodulate(block)

            if i >= warmup_blocks * block_size:
                left_chunks.append(audio[:, 0])
                right_chunks.append(audio[:, 1])
                pll_locked_history.append(decoder_pll.pll_locked)
                phase_error_history.append(decoder_pll.pll_phase_error_deg)
                freq_offset_history.append(decoder_pll.pll_frequency_offset)

        left_pll = np.concatenate(left_chunks)
        right_pll = np.concatenate(right_chunks)

        # PLL statistics
        lock_rate = np.mean(pll_locked_history) * 100
        avg_phase_error = np.mean(phase_error_history)
        max_phase_error = np.max(phase_error_history)
        avg_freq_offset = np.mean(freq_offset_history)

        print(f"  PLL lock rate: {lock_rate:.1f}%")
        print(f"  PLL phase error: avg={avg_phase_error:.2f}°, max={max_phase_error:.2f}°")
        print(f"  PLL frequency offset: avg={avg_freq_offset:.3f} Hz")
        print(f"  SNR: {decoder_pll.snr_db:.1f} dB")
        print(f"  Blend factor: {decoder_pll.stereo_blend_factor:.3f}")

        # Audio levels
        left_rms = np.sqrt(np.mean(left_pll ** 2))
        right_rms = np.sqrt(np.mean(right_pll ** 2))
        print(f"  Audio: L_rms={left_rms:.6f}, R_rms={right_rms:.6f}")

        # Measure separation
        if right_rms > 0:
            sep_pll = 20 * np.log10(left_rms / right_rms)
        else:
            sep_pll = float('inf')
        print(f"  Separation (RMS method): {sep_pll:.1f} dB")

        # Test pilot-squaring decoder
        print(f"\n--- FMStereoDecoder (pilot-squaring) ---")
        decoder_ps = FMStereoDecoder(iq_sample_rate=fs_baseband, audio_sample_rate=fs_audio)

        left_chunks = []
        right_chunks = []

        for i in range(0, len(iq_samples), block_size):
            block = iq_samples[i:i+block_size]
            if len(block) < block_size // 2:
                break

            audio = decoder_ps.demodulate(block)

            if i >= warmup_blocks * block_size:
                left_chunks.append(audio[:, 0])
                right_chunks.append(audio[:, 1])

        left_ps = np.concatenate(left_chunks)
        right_ps = np.concatenate(right_chunks)

        print(f"  Pilot detected: {decoder_ps.pilot_detected}")
        print(f"  SNR: {decoder_ps.snr_db:.1f} dB")
        print(f"  Blend factor: {decoder_ps.stereo_blend_factor:.3f}")

        # Audio levels
        left_rms = np.sqrt(np.mean(left_ps ** 2))
        right_rms = np.sqrt(np.mean(right_ps ** 2))
        print(f"  Audio: L_rms={left_rms:.6f}, R_rms={right_rms:.6f}")

        # Measure separation
        if right_rms > 0:
            sep_ps = 20 * np.log10(left_rms / right_rms)
        else:
            sep_ps = float('inf')
        print(f"  Separation (RMS method): {sep_ps:.1f} dB")

        # Summary
        print(f"\n>>> SEPARATION SUMMARY:")
        print(f"    FMStereoDecoder:  {sep_ps:>6.1f} dB")
        print(f"    PLLStereoDecoder: {sep_pll:>6.1f} dB")
        print(f"    Difference: {sep_pll - sep_ps:>+6.1f} dB (PLL vs pilot-squaring)")


def main():
    np.random.seed(42)

    # Test increasing levels of phase noise
    phase_noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]

    analyze_phase_noise(phase_noise_levels)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
