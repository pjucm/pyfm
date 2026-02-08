#!/usr/bin/env python3
"""
Clean-room SNR and Stereo Separation Test for FM Stereo Decoders

Tests two decoder implementations:
- FMStereoDecoder (pilot-squaring)
- PLLStereoDecoder (PLL-based carrier recovery)

Methodology:
1. Generate synthetic FM stereo composite baseband signals
2. Frequency-modulate onto IQ carrier
3. Add controlled AWGN at various SNR levels
4. Demodulate with both decoders
5. Measure stereo separation and SNR performance
"""

import numpy as np
from scipy import signal
import sys

# Import the two decoders under test
from demodulator import FMStereoDecoder
from pll_stereo_decoder import PLLStereoDecoder


def _generate_ar1_phase_noise(n_samples, phase_rms_rad, pole=0.9995, rng=None):
    """
    Generate stationary correlated phase noise using an AR(1) process.

    Args:
        n_samples: Number of samples
        phase_rms_rad: Target RMS phase jitter in radians
        pole: AR(1) pole coefficient (close to 1 = low-frequency dominant)
        rng: numpy random generator (optional)

    Returns:
        phase_noise: Stationary correlated phase noise in radians
    """
    if n_samples <= 0 or phase_rms_rad <= 0:
        return np.zeros(max(0, n_samples), dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()

    # Choose drive gain so stationary variance is ~1 before final RMS normalization.
    drive_gain = np.sqrt(max(1.0 - pole * pole, 1e-12))
    white = rng.standard_normal(n_samples)
    phase_noise = np.zeros(n_samples, dtype=np.float64)
    for i in range(1, n_samples):
        phase_noise[i] = pole * phase_noise[i - 1] + drive_gain * white[i]

    # Normalize to requested RMS after short transient removal.
    rms_meas = np.sqrt(np.mean(phase_noise[int(0.1 * n_samples):] ** 2))
    if rms_meas > 0:
        phase_noise *= (phase_rms_rad / rms_meas)

    return phase_noise


def generate_pilot_and_carrier(
    n_samples,
    fs,
    phase_noise_rms=0.0,
    phase_noise_pole=0.9995,
    rng=None,
):
    """
    Generate 19 kHz pilot tone and 38 kHz subcarrier with correct phase relationship.

    CRITICAL: The 38 kHz carrier must be derived from frequency-doubling the 19 kHz
    pilot to maintain the correct phase relationship for stereo decoding.

    Args:
        n_samples: Number of samples to generate
        fs: Sample rate in Hz
        phase_noise_rms: RMS phase noise in radians (default 0.0)
        phase_noise_pole: AR(1) phase-noise pole (default 0.9995)
        rng: numpy random generator (optional)

    Returns:
        pilot_19k, carrier_38k: Pilot and carrier signals (phase-locked)
    """
    t = np.arange(n_samples) / fs

    # Generate 19 kHz pilot with optional phase noise
    phase_noise = _generate_ar1_phase_noise(
        n_samples,
        phase_rms_rad=phase_noise_rms,
        pole=phase_noise_pole,
        rng=rng,
    )
    theta = 2 * np.pi * 19000 * t + phase_noise
    pilot_19k = np.cos(theta)
    # Carrier is derived by frequency doubling.
    carrier_38k = np.cos(2 * theta)

    return pilot_19k, carrier_38k


def generate_stereo_composite(
    left,
    right,
    fs,
    pilot_level=0.09,
    phase_noise_rms=0.0,
    phase_noise_pole=0.9995,
    sum_gain=0.5,
    diff_gain=0.5,
    rng=None,
):
    """
    Generate FM stereo composite baseband signal.

    FM Stereo multiplex format:
    - 0-15 kHz: L+R (mono sum)
    - 19 kHz: Pilot tone
    - 23-53 kHz: (L-R) modulated on 38 kHz DSB-SC carrier

    Args:
        left, right: Audio signals (numpy arrays, normalized to ±1)
        fs: Sample rate of baseband (must be >100 kHz for stereo)
        pilot_level: Pilot amplitude relative to mono (default 0.09 = 9%)
        phase_noise_rms: RMS phase noise in radians
        phase_noise_pole: AR(1) phase-noise pole
        sum_gain: Gain for (L+R) path in composite
        diff_gain: Gain for (L-R) path in composite
        rng: numpy random generator (optional)

    Returns:
        composite: FM stereo composite baseband signal
    """
    n = len(left)

    # Generate L+R and L-R
    lr_sum = left + right
    lr_diff = left - right

    # Generate pilot and carrier
    pilot_19k, carrier_38k = generate_pilot_and_carrier(
        n,
        fs,
        phase_noise_rms=phase_noise_rms,
        phase_noise_pole=phase_noise_pole,
        rng=rng,
    )

    # Composite signal
    # L+R is baseband (0-15 kHz) - already filtered by audio generation
    # Pilot at 19 kHz with 9% modulation (standard)
    # L-R modulated on 38 kHz carrier (DSB-SC)
    # Use balanced sum/difference scaling to avoid heavy over-modulation.
    composite = sum_gain * lr_sum + pilot_level * pilot_19k + diff_gain * lr_diff * carrier_38k

    return composite


def fm_modulate_to_iq(
    baseband,
    fs_baseband,
    fs_rf,
    fc,
    deviation=75000,
    rf_phase_noise_rms=0.0,
    rf_phase_noise_pole=0.9995,
    rng=None,
):
    """
    Frequency modulate baseband signal onto IQ carrier.

    Args:
        baseband: Baseband signal (FM composite)
        fs_baseband: Sample rate of baseband signal
        fs_rf: Sample rate of RF (IQ) signal
        fc: Carrier frequency (Hz) - can be 0 for baseband IQ
        deviation: FM deviation (Hz)
        rf_phase_noise_rms: RMS RF phase jitter added to IQ carrier phase (radians)
        rf_phase_noise_pole: AR(1) pole for RF phase jitter
        rng: numpy random generator (optional)

    Returns:
        iq_samples: Complex IQ samples
    """
    # Resample baseband to RF rate if needed
    if fs_rf != fs_baseband:
        baseband = signal.resample_poly(baseband, fs_rf, fs_baseband)

    # FM modulation: frequency is integral of baseband
    # Instantaneous phase = 2π * deviation * integral(baseband)
    phase = 2 * np.pi * deviation * np.cumsum(baseband) / fs_rf
    if rf_phase_noise_rms > 0:
        phase += _generate_ar1_phase_noise(
            len(phase),
            phase_rms_rad=rf_phase_noise_rms,
            pole=rf_phase_noise_pole,
            rng=rng,
        )

    # Generate complex IQ samples
    if fc == 0:
        # Baseband IQ (no carrier offset)
        iq_samples = np.exp(1j * phase)
    else:
        # Modulate onto carrier
        t = np.arange(len(phase)) / fs_rf
        carrier_phase = 2 * np.pi * fc * t
        iq_samples = np.exp(1j * (carrier_phase + phase))

    return iq_samples.astype(np.complex64)


def add_awgn_at_snr(signal, snr_db, rng=None):
    """
    Add Additive White Gaussian Noise to achieve target SNR.

    Args:
        signal: Input signal (complex or real)
        snr_db: Target SNR in dB

    Returns:
        noisy_signal: Signal with added noise
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    if np.iscomplexobj(signal):
        # Complex noise: divide power between I and Q
        noise = (
            rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
        ) / np.sqrt(2)
    else:
        noise = rng.standard_normal(len(signal))

    noise = noise * np.sqrt(noise_power)
    return signal + noise


def measure_stereo_separation(left, right, test_freq, fs_audio, channel='L', debug=False):
    """
    Measure stereo separation for a test tone in one channel.

    Args:
        left, right: Decoded audio channels
        test_freq: Test tone frequency (Hz)
        fs_audio: Audio sample rate
        channel: 'L' or 'R' - which channel contains the test tone
        debug: Print diagnostic information

    Returns:
        separation_db: Stereo separation in dB (positive = good isolation)
    """
    # FFT to find test tone power in each channel
    n_fft = len(left)
    window = np.hanning(n_fft)

    left_fft = np.fft.rfft(left * window)
    right_fft = np.fft.rfft(right * window)

    freqs = np.fft.rfftfreq(n_fft, 1/fs_audio)

    # Find bin closest to test frequency
    test_bin = np.argmin(np.abs(freqs - test_freq))

    # Measure power in test bin ±1 bin
    left_power = np.sum(np.abs(left_fft[test_bin-1:test_bin+2]) ** 2)
    right_power = np.sum(np.abs(right_fft[test_bin-1:test_bin+2]) ** 2)

    if debug:
        print(f"\n  DEBUG: Separation measurement for {test_freq} Hz in {channel} channel:")
        print(f"    Audio samples: {len(left)}")
        print(f"    Left RMS: {np.sqrt(np.mean(left**2)):.6f}")
        print(f"    Right RMS: {np.sqrt(np.mean(right**2)):.6f}")
        print(f"    Left peak: {np.max(np.abs(left)):.6f}")
        print(f"    Right peak: {np.max(np.abs(right)):.6f}")
        print(f"    FFT bin: {test_bin}, freq: {freqs[test_bin]:.1f} Hz")
        print(f"    Left FFT power: {left_power:.3e}")
        print(f"    Right FFT power: {right_power:.3e}")

    if channel == 'L':
        # Test tone is in left channel
        wanted = left_power
        unwanted = right_power
    else:
        # Test tone is in right channel
        wanted = right_power
        unwanted = left_power

    # Separation in dB
    if unwanted > 0 and wanted > 0:
        separation_db = 10 * np.log10(wanted / unwanted)
    else:
        separation_db = float('inf')

    if debug:
        print(f"    Wanted power: {wanted:.3e}")
        print(f"    Unwanted power: {unwanted:.3e}")
        print(f"    Separation: {separation_db:.1f} dB")

    return separation_db


def generate_test_signal(duration, test_freq, fs_audio, fs_baseband, channel='L'):
    """
    Generate stereo test signal with tone in one channel.

    Args:
        duration: Duration in seconds
        test_freq: Test tone frequency (Hz)
        fs_audio: Audio sample rate
        fs_baseband: FM baseband sample rate
        channel: 'L' or 'R' - which channel gets the tone

    Returns:
        left, right: Audio signals at baseband sample rate
    """
    n_audio = int(duration * fs_audio)
    t_audio = np.arange(n_audio) / fs_audio

    # Generate test tone
    tone = 0.5 * np.sin(2 * np.pi * test_freq * t_audio)

    # Apply lowpass filter (15 kHz) to simulate broadcast audio
    nyq_audio = fs_audio / 2
    audio_lpf = signal.firwin(101, 15000 / nyq_audio)
    tone = signal.lfilter(audio_lpf, 1, tone)

    # Assign to channels
    if channel == 'L':
        left = tone
        right = np.zeros_like(tone)
    else:
        left = np.zeros_like(tone)
        right = tone

    # Resample to baseband rate
    if fs_baseband != fs_audio:
        left = signal.resample_poly(left, fs_baseband, fs_audio)
        right = signal.resample_poly(right, fs_baseband, fs_audio)

    return left, right


def run_decoder_test(
    decoder_class,
    iq_samples,
    iq_fs,
    audio_fs,
    decoder_name,
    warmup_seconds=1.0,
    disable_processing_features=True,
):
    """
    Run a decoder on test IQ samples.

    Args:
        decoder_class: Decoder class (FMStereoDecoder or PLLStereoDecoder)
        iq_samples: IQ samples to decode
        iq_fs: IQ sample rate
        audio_fs: Audio sample rate
        decoder_name: Name for display
        warmup_seconds: Seconds of decoded audio to discard for warmup
        disable_processing_features: Disable blend/tone controls for raw bench measurement

    Returns:
        left, right: Decoded audio channels (after warmup)
        snr_measured: SNR measured by decoder
        blend_factor: Stereo blend factor
    """
    # Create decoder instance
    decoder = decoder_class(iq_sample_rate=iq_fs, audio_sample_rate=audio_fs)

    if disable_processing_features:
        # Disable processing features that mask raw stereo separation behavior.
        decoder.bass_boost_enabled = False
        decoder.treble_boost_enabled = False
        decoder.stereo_blend_enabled = False
        decoder.stereo_blend_low = -100.0

    # Process in blocks
    block_size = 4096
    left_out = []
    right_out = []
    snr_history = []
    blend_history = []

    for i in range(0, len(iq_samples), block_size):
        block = iq_samples[i:i+block_size]
        if len(block) < block_size // 2:
            break  # Skip partial blocks at end

        audio = decoder.demodulate(block)

        left_out.append(audio[:, 0])
        right_out.append(audio[:, 1])
        snr_history.append(decoder.snr_db)
        blend_history.append(decoder.stereo_blend_factor)

    left_out = np.concatenate(left_out) if left_out else np.array([], dtype=np.float32)
    right_out = np.concatenate(right_out) if right_out else np.array([], dtype=np.float32)

    warmup_samples = int(warmup_seconds * audio_fs)
    if warmup_samples > 0:
        left_out = left_out[warmup_samples:]
        right_out = right_out[warmup_samples:]

    # Average measurements from stable portion
    stable_start = len(snr_history) // 2
    stable_snr = snr_history[stable_start:] if snr_history else []
    stable_blend = blend_history[stable_start:] if blend_history else []
    snr_measured = np.mean(stable_snr) if len(stable_snr) > 0 else 0.0
    blend_factor = np.mean(stable_blend) if len(stable_blend) > 0 else 0.0

    return left_out, right_out, snr_measured, blend_factor


def diagnostic_test():
    """Run a single test with full debug output."""
    print("=" * 80)
    print("DIAGNOSTIC TEST - Single 1 kHz Left Channel Test")
    print("=" * 80)
    print()

    fs_audio = 48000
    fs_baseband = 480000
    fs_iq = 480000
    rf_snr = 40
    duration = 2.0
    test_freq = 1000

    print(f"Test parameters:")
    print(f"  Audio sample rate: {fs_audio} Hz")
    print(f"  Baseband sample rate: {fs_baseband} Hz")
    print(f"  RF SNR: {rf_snr} dB")
    print(f"  Test frequency: {test_freq} Hz (left channel only)")
    print()

    # Generate test signal
    print("1. Generating test signal...")
    left, right = generate_test_signal(duration, test_freq, fs_audio, fs_baseband, 'L')
    print(f"   Left channel - samples: {len(left)}, RMS: {np.sqrt(np.mean(left**2)):.6f}, peak: {np.max(np.abs(left)):.6f}")
    print(f"   Right channel - samples: {len(right)}, RMS: {np.sqrt(np.mean(right**2)):.6f}, peak: {np.max(np.abs(right)):.6f}")

    # Create composite
    print("\n2. Creating FM stereo composite...")

    # Manual composite generation with diagnostics
    lr_sum = left + right
    lr_diff = left - right
    print(f"   L+R (mono) - RMS: {np.sqrt(np.mean(lr_sum**2)):.6f}, peak: {np.max(np.abs(lr_sum)):.6f}")
    print(f"   L-R (diff) - RMS: {np.sqrt(np.mean(lr_diff**2)):.6f}, peak: {np.max(np.abs(lr_diff)):.6f}")

    pilot_19k, carrier_38k = generate_pilot_and_carrier(
        len(left),
        fs_baseband,
        phase_noise_rms=0.0,
    )
    print(f"   Pilot 19kHz - RMS: {np.sqrt(np.mean(pilot_19k**2)):.6f}")
    print(f"   Carrier 38kHz - RMS: {np.sqrt(np.mean(carrier_38k**2)):.6f}")

    composite = lr_sum + 0.09 * pilot_19k + lr_diff * carrier_38k
    print(f"   Composite - samples: {len(composite)}, RMS: {np.sqrt(np.mean(composite**2)):.6f}, peak: {np.max(np.abs(composite)):.6f}")

    # Modulate to IQ
    print("\n3. FM modulating to IQ...")
    iq_samples = fm_modulate_to_iq(composite, fs_baseband, fs_iq, fc=0)
    print(f"   IQ samples: {len(iq_samples)}, RMS: {np.sqrt(np.mean(np.abs(iq_samples)**2)):.6f}")

    # Add noise
    print("\n4. Adding noise...")
    iq_noisy = add_awgn_at_snr(iq_samples, rf_snr, rng=np.random.default_rng(1234))
    actual_snr = 10 * np.log10(np.mean(np.abs(iq_samples)**2) / np.mean(np.abs(iq_noisy - iq_samples)**2))
    print(f"   Actual SNR: {actual_snr:.1f} dB")

    # Test with PLL decoder
    print("\n5. Decoding with PLLStereoDecoder...")
    decoder = PLLStereoDecoder(iq_sample_rate=fs_iq, audio_sample_rate=fs_audio)
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False
    decoder.stereo_blend_low = -100.0

    # Process in blocks
    block_size = 4096
    warmup_blocks = 20
    left_out = []
    right_out = []

    for i in range(0, len(iq_noisy), block_size):
        block = iq_noisy[i:i+block_size]
        if len(block) < block_size // 2:
            break

        audio = decoder.demodulate(block)

        if i >= warmup_blocks * block_size:
            left_out.append(audio[:, 0])
            right_out.append(audio[:, 1])
            if i == warmup_blocks * block_size:
                print(f"   After warmup - SNR: {decoder.snr_db:.1f} dB, blend: {decoder.stereo_blend_factor:.3f}, PLL locked: {decoder.pll_locked}")

    left_decoded = np.concatenate(left_out)
    right_decoded = np.concatenate(right_out)

    print(f"   Decoded audio - samples: {len(left_decoded)}")
    print(f"   Final decoder state - SNR: {decoder.snr_db:.1f} dB, blend: {decoder.stereo_blend_factor:.3f}")

    # Measure separation
    print("\n6. Measuring stereo separation...")
    separation = measure_stereo_separation(left_decoded, right_decoded, test_freq, fs_audio, 'L', debug=True)

    print(f"\n{'='*60}")
    print(f"RESULT: Stereo separation = {separation:.1f} dB")
    print(f"{'='*60}")
    print()


def test_stereo_separation():
    """Test stereo separation at various test frequencies."""
    print("=" * 80)
    print("STEREO SEPARATION TEST")
    print("=" * 80)
    print()

    # Test parameters
    fs_audio = 48000
    fs_baseband = 480000  # High rate for good stereo performance
    fs_iq = 480000
    rf_snr = 40  # High SNR for separation test
    duration = 2.0

    test_freqs = [100, 1000, 5000, 10000, 15000]  # Hz

    decoders = [
        (FMStereoDecoder, "FMStereoDecoder (pilot-squaring)"),
        (PLLStereoDecoder, "PLLStereoDecoder (PLL)")
    ]

    # Precompute identical noisy IQ inputs for all decoder comparisons.
    iq_cases = {}
    for test_freq in test_freqs:
        left, right = generate_test_signal(duration, test_freq, fs_audio, fs_baseband, 'L')
        composite = generate_stereo_composite(left, right, fs_baseband)
        iq_samples = fm_modulate_to_iq(composite, fs_baseband, fs_iq, fc=0)
        iq_cases[(test_freq, 'L')] = add_awgn_at_snr(
            iq_samples,
            rf_snr,
            rng=np.random.default_rng(10_000 + test_freq),
        )

        left, right = generate_test_signal(duration, test_freq, fs_audio, fs_baseband, 'R')
        composite = generate_stereo_composite(left, right, fs_baseband)
        iq_samples = fm_modulate_to_iq(composite, fs_baseband, fs_iq, fc=0)
        iq_cases[(test_freq, 'R')] = add_awgn_at_snr(
            iq_samples,
            rf_snr,
            rng=np.random.default_rng(20_000 + test_freq),
        )

    for decoder_class, decoder_name in decoders:
        print(f"\n{decoder_name}")
        print("-" * 60)
        print(f"{'Freq (Hz)':<12} {'L-only Sep (dB)':<18} {'R-only Sep (dB)':<18}")
        print("-" * 60)

        for test_freq in test_freqs:
            left_dec, right_dec, _, _ = run_decoder_test(
                decoder_class,
                iq_cases[(test_freq, 'L')],
                fs_iq,
                fs_audio,
                decoder_name,
                warmup_seconds=1.0,
            )

            sep_l = measure_stereo_separation(left_dec, right_dec, test_freq, fs_audio, 'L')

            left_dec, right_dec, _, _ = run_decoder_test(
                decoder_class,
                iq_cases[(test_freq, 'R')],
                fs_iq,
                fs_audio,
                decoder_name,
                warmup_seconds=1.0,
            )

            sep_r = measure_stereo_separation(left_dec, right_dec, test_freq, fs_audio, 'R')

            print(f"{test_freq:<12} {sep_l:>10.1f} dB        {sep_r:>10.1f} dB")

        print()


def test_snr_performance():
    """Test SNR measurement and stereo blend at various noise levels."""
    print("=" * 80)
    print("SNR PERFORMANCE TEST")
    print("=" * 80)
    print()

    # Test parameters
    fs_audio = 48000
    fs_baseband = 480000
    fs_iq = 480000
    duration = 2.0
    test_freq = 1000  # Hz test tone

    # RF SNR levels to test
    rf_snr_levels = [10, 15, 20, 25, 30, 35, 40]

    decoders = [
        (FMStereoDecoder, "FMStereoDecoder (pilot-squaring)"),
        (PLLStereoDecoder, "PLLStereoDecoder (PLL)")
    ]

    print(f"Test signal: {test_freq} Hz stereo tone")
    print()

    # Precompute identical noisy IQ inputs for fair decoder comparison.
    iq_cases = {}
    for rf_snr in rf_snr_levels:
        left, right = generate_test_signal(duration, test_freq, fs_audio, fs_baseband, 'L')
        right = generate_test_signal(duration, test_freq, fs_audio, fs_baseband, 'R')[1]
        left = left * 0.7
        right = right * 0.7
        composite = generate_stereo_composite(left, right, fs_baseband)
        iq_samples = fm_modulate_to_iq(composite, fs_baseband, fs_iq, fc=0)
        iq_cases[rf_snr] = add_awgn_at_snr(
            iq_samples,
            rf_snr,
            rng=np.random.default_rng(30_000 + int(rf_snr * 10)),
        )

    for decoder_class, decoder_name in decoders:
        print(f"\n{decoder_name}")
        print("-" * 70)
        print(f"{'RF SNR (dB)':<12} {'Measured SNR (dB)':<20} {'Blend Factor':<15} {'Status':<12}")
        print("-" * 70)

        for rf_snr in rf_snr_levels:
            # Decode
            left_dec, right_dec, snr_measured, blend_factor = run_decoder_test(
                decoder_class,
                iq_cases[rf_snr],
                fs_iq,
                fs_audio,
                decoder_name,
                warmup_seconds=0.5,
                disable_processing_features=False,
            )

            # Determine status
            if blend_factor > 0.9:
                status = "Full Stereo"
            elif blend_factor > 0.1:
                status = "Blending"
            else:
                status = "Mono"

            print(f"{rf_snr:<12} {snr_measured:>12.1f}         {blend_factor:>8.2f}        {status:<12}")

        print()


def test_phase_noise_robustness():
    """Test decoder robustness to pilot phase noise."""
    print("=" * 80)
    print("PHASE NOISE ROBUSTNESS TEST")
    print("=" * 80)
    print()

    # Test parameters
    fs_audio = 48000
    fs_baseband = 480000
    fs_iq = 480000
    duration = 2.0
    test_freq = 1000
    rf_snr = 35  # Good SNR baseline

    # Phase jitter levels (radians RMS)
    phase_noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    phase_noise_pole = 0.9995

    decoders = [
        (FMStereoDecoder, "FMStereoDecoder (pilot-squaring)"),
        (PLLStereoDecoder, "PLLStereoDecoder (PLL)")
    ]

    print(f"RF SNR: {rf_snr} dB, Test frequency: {test_freq} Hz")
    print("Warmup: first 1.0 s of audio is discarded for all measurements.")
    print()

    impairment_modes = [
        ("rf_jitter", "RF carrier phase jitter (after FM modulation)"),
        ("tx_common", "TX common pilot/subcarrier phase jitter"),
    ]

    for mode_key, mode_label in impairment_modes:
        print(f"\nMode: {mode_label}")
        print("-" * 80)

        # Build one deterministic IQ case per phase-noise level and reuse it for both decoders.
        iq_cases = {}
        for idx, phase_noise in enumerate(phase_noise_levels):
            left, right = generate_test_signal(duration, test_freq, fs_audio, fs_baseband, 'L')
            seed_base = 40_000 + idx * 100 + (0 if mode_key == "rf_jitter" else 10_000)

            if mode_key == "tx_common":
                composite = generate_stereo_composite(
                    left,
                    right,
                    fs_baseband,
                    phase_noise_rms=phase_noise,
                    phase_noise_pole=phase_noise_pole,
                    rng=np.random.default_rng(seed_base + 1),
                )
                iq_samples = fm_modulate_to_iq(
                    composite,
                    fs_baseband,
                    fs_iq,
                    fc=0,
                    deviation=75000,
                )
            else:
                composite = generate_stereo_composite(left, right, fs_baseband)
                iq_samples = fm_modulate_to_iq(
                    composite,
                    fs_baseband,
                    fs_iq,
                    fc=0,
                    deviation=75000,
                    rf_phase_noise_rms=phase_noise,
                    rf_phase_noise_pole=phase_noise_pole,
                    rng=np.random.default_rng(seed_base + 1),
                )

            iq_cases[phase_noise] = add_awgn_at_snr(
                iq_samples,
                rf_snr,
                rng=np.random.default_rng(seed_base + 2),
            )

        for decoder_class, decoder_name in decoders:
            print(f"\n{decoder_name}")
            print("-" * 72)
            print(f"{'Phase Noise (rad RMS)':<22} {'Separation (dB)':<18} {'Meas SNR (dB)':<14}")
            print("-" * 72)

            for phase_noise in phase_noise_levels:
                left_dec, right_dec, snr_measured, _ = run_decoder_test(
                    decoder_class,
                    iq_cases[phase_noise],
                    fs_iq,
                    fs_audio,
                    decoder_name,
                    warmup_seconds=1.0,
                )

                separation = measure_stereo_separation(left_dec, right_dec, test_freq, fs_audio, 'L')

                print(f"{phase_noise:<22.3f} {separation:>12.1f} dB    {snr_measured:>9.1f}")

            print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "FM STEREO DECODER COMPARISON TEST" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    np.random.seed(42)  # Reproducible results

    # Check for --debug flag
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv

    try:
        if debug_mode:
            # Run diagnostic test only
            diagnostic_test()
        else:
            # Run full test suite
            test_stereo_separation()
            print("\n")
            test_snr_performance()
            print("\n")
            test_phase_noise_robustness()

            print("\n" + "=" * 80)
            print("TEST COMPLETE")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
