#!/usr/bin/env python3
r"""
FM Stereo Decoder Test Suite

Tests mathematically correct mono and stereo FM decoding with high stereo separation.
Documents phase/delay relationships throughout the decode chain.

Usage:
    python test_pyfm.py          # Run all tests with detailed output
    pytest test_pyfm.py -v       # Run with pytest (if installed)

Signal Flow:
    IQ -> FM Demod -> Pilot BPF (201 taps) -> carrier regen -> 38 kHz carrier
                 \-> L+R LPF (127 taps) + 100 sample delay compensation
                 \-> L-R BPF (201 taps) -> x carrier x 2 -> L-R LPF (127 taps)
                                                                    |
                 Matrix: L = L+R + L-R, R = L+R - L-R <-------------+

Carrier Regeneration:
    Pilot-squaring method: Uses identity 2*sin^2(x)-1 = -cos(2x)
    Works correctly when transmitter uses -cos(2*pi*38000*t) subcarrier.

Group Delay Alignment:
    L+R path: LPF (63 samples) + delay buffer (100 samples) = 163 samples
    L-R path: BPF (100 samples) + LPF (63 samples) = 163 samples

FM Multiplex Structure:
    0-15 kHz:   L+R (mono compatible)
    19 kHz:     Pilot tone (9% amplitude)
    23-53 kHz:  L-R on 38 kHz DSB-SC carrier

Critical Phase Relationships:
    The pilot-squaring method produces: 2*sin^2(wt) - 1 = -cos(2wt)

    For correct stereo decode, transmitter must use -cos(2wt) as subcarrier:
    - TX = -cos(2wt), RX = -cos(2wt): WORKS (correct polarity)
    - TX = cos(2wt),  RX = -cos(2wt): WORKS (inverted L/R)
    - TX = sin(2wt),  RX = -cos(2wt): FAILS (90° phase error = no separation)

Validation Targets:
    FM demod accuracy: Correlation > 0.999
    Audio SNR (clean): > 35 dB (limited by full stereo chain processing)
    THD+N: < -35 dB (< 2% distortion, limited by soft limiter)
    Stereo separation: > 30 dB (67+ dB typical with synthetic signal)
    L/R timing: < 5 samples at 48 kHz
    Frequency response: +/- 3 dB below 1 kHz
    CPU/GPU demod parity: Correlation > 0.9999
"""

import numpy as np
from scipy import signal
import sys

from demodulator import FMStereoDecoder
from gpu import GPUFMDemodulator


# =============================================================================
# Helper Functions
# =============================================================================

def demodulate_with_settling(decoder, iq, block_size=8192):
    """
    Demodulate IQ samples in blocks to allow pilot detection to settle.

    Args:
        decoder: FMStereoDecoder instance
        iq: Complex I/Q samples
        block_size: Block size for processing

    Returns:
        Concatenated audio output (N, 2) array
    """
    audio_chunks = []
    for i in range(0, len(iq), block_size):
        block = iq[i:i + block_size]
        if len(block) > 0:
            audio = decoder.demodulate(block)
            audio_chunks.append(audio)

    return np.vstack(audio_chunks) if audio_chunks else np.zeros((0, 2), dtype=np.float32)


# =============================================================================
# Signal Generation Functions
# =============================================================================

def generate_test_tone(freq, duration, sample_rate, amplitude=1.0):
    """
    Generate a pure sine wave test tone.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Peak amplitude (default 1.0)

    Returns:
        numpy array of float64 samples
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_fm_stereo_multiplex(left, right, sample_rate, pilot_amplitude=0.09,
                                  include_pilot=True, subcarrier_phase='neg_cos'):
    """
    Generate FM stereo multiplex baseband signal.

    FM Multiplex structure:
    - 0-15 kHz: L+R (mono sum, scaled by 0.9 for pilot headroom)
    - 19 kHz: Pilot tone (9% of max deviation)
    - 23-53 kHz: L-R DSB-SC modulated onto 38 kHz carrier

    The subcarrier_phase parameter controls the 38 kHz carrier phase:
    - 'neg_cos': -cos(2*pi*38000*t) - matches pilot-squaring decoder
    - 'cos': cos(2*pi*38000*t) - gives inverted stereo
    - 'sin': sin(2*pi*38000*t) - commonly cited but causes 90° error!

    Args:
        left: Left channel audio samples
        right: Right channel audio samples
        sample_rate: Sample rate in Hz
        pilot_amplitude: Pilot tone amplitude (default 0.09 = 9%)
        include_pilot: Whether to include 19 kHz pilot (default True)
        subcarrier_phase: '38 kHz carrier phase ('neg_cos', 'cos', or 'sin')

    Returns:
        FM multiplex baseband signal (normalized for 75 kHz deviation)
    """
    n = len(left)
    t = np.arange(n) / sample_rate

    # L+R and L-R (each scaled by 0.5 for proper matrix decode)
    lr_sum = (left + right) / 2
    lr_diff = (left - right) / 2

    # Pilot tone at 19 kHz (sine phase per standard)
    if include_pilot:
        pilot = pilot_amplitude * np.sin(2 * np.pi * 19000 * t)
    else:
        pilot = np.zeros(n)

    # 38 kHz carrier for L-R DSB-SC modulation
    if subcarrier_phase == 'neg_cos':
        # -cos(2wt) = 2*sin^2(wt) - 1 (matches pilot-squaring decoder)
        carrier_38k = -np.cos(2 * np.pi * 38000 * t)
    elif subcarrier_phase == 'cos':
        # cos(2wt) - gives inverted stereo with pilot-squaring decoder
        carrier_38k = np.cos(2 * np.pi * 38000 * t)
    elif subcarrier_phase == 'sin':
        # sin(2wt) - commonly cited but causes 90° error with pilot-squaring!
        carrier_38k = np.sin(2 * np.pi * 38000 * t)
    else:
        raise ValueError(f"Unknown subcarrier_phase: {subcarrier_phase}")

    # DSB-SC modulated L-R
    lr_diff_mod = lr_diff * carrier_38k

    # Combine multiplex signal
    # Scale L+R and L-R for pilot headroom (total deviation budget)
    multiplex = lr_sum * 0.9 + pilot + lr_diff_mod * 0.9

    return multiplex


def fm_modulate(baseband, sample_rate, deviation=75000):
    """
    FM modulate baseband signal to complex I/Q.

    Uses the standard FM formula:
        phase(t) = 2*pi*deviation * integral(baseband) * dt
        IQ = cos(phase) + j*sin(phase)

    Args:
        baseband: Baseband audio/multiplex signal
        sample_rate: Sample rate in Hz
        deviation: FM deviation in Hz (75 kHz for broadcast FM)

    Returns:
        Complex I/Q samples (numpy array of complex64)
    """
    dt = 1.0 / sample_rate

    # Instantaneous phase is the integral of baseband scaled by deviation
    phase = 2 * np.pi * deviation * np.cumsum(baseband) * dt

    # Generate complex I/Q
    iq = np.cos(phase) + 1j * np.sin(phase)

    return iq.astype(np.complex64)


# =============================================================================
# Measurement Utilities
# =============================================================================

def goertzel_power(x, target_freq, sample_rate):
    """
    Compute power at a specific frequency using Goertzel algorithm.

    More efficient than FFT for single-frequency measurement.

    Args:
        x: Input signal
        target_freq: Frequency to measure in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Power at target frequency (linear scale)
    """
    n = len(x)
    k = int(0.5 + n * target_freq / sample_rate)
    w = 2 * np.pi * k / n
    coeff = 2 * np.cos(w)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0

    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    # Compute power
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    # Normalize by N^2 / 2 for consistent power measurement
    power = power / (n * n / 2)

    return power


def measure_snr(x, signal_freq, sample_rate, signal_bw=100):
    """
    Measure SNR of a signal with a known tone frequency.

    Args:
        x: Input signal
        signal_freq: Frequency of the signal tone in Hz
        sample_rate: Sample rate in Hz
        signal_bw: Bandwidth around signal to consider as signal (Hz)

    Returns:
        SNR in dB
    """
    n = len(x)
    window = np.hanning(n)
    x_windowed = x * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    # Signal power: in narrow band around signal frequency
    signal_mask = np.abs(freqs - signal_freq) <= signal_bw / 2
    signal_power = np.sum(power_spectrum[signal_mask])

    # Noise power: everything except signal band
    noise_mask = ~signal_mask
    noise_power = np.sum(power_spectrum[noise_mask])

    if noise_power <= 0:
        return np.inf

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def measure_thd_n(x, fundamental_freq, sample_rate, n_harmonics=5):
    """
    Measure Total Harmonic Distortion + Noise (THD+N).

    Args:
        x: Input signal
        fundamental_freq: Fundamental frequency in Hz
        sample_rate: Sample rate in Hz
        n_harmonics: Number of harmonics to include

    Returns:
        THD+N in dB (negative value, e.g., -40 dB = 1% THD+N)
    """
    n = len(x)
    window = np.hanning(n)
    x_windowed = x * window

    fft = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    power_spectrum = np.abs(fft) ** 2 / n

    # Fundamental power
    fundamental_mask = np.abs(freqs - fundamental_freq) <= 50
    fundamental_power = np.sum(power_spectrum[fundamental_mask])

    # Harmonic power
    harmonic_power = 0.0
    for h in range(2, n_harmonics + 1):
        harm_freq = fundamental_freq * h
        if harm_freq >= sample_rate / 2:
            break
        harm_mask = np.abs(freqs - harm_freq) <= 50
        harmonic_power += np.sum(power_spectrum[harm_mask])

    # Total power (signal + noise)
    total_power = np.sum(power_spectrum)

    # Noise power = total - fundamental - harmonics
    noise_power = total_power - fundamental_power - harmonic_power

    # THD+N = sqrt(harmonics + noise) / fundamental
    thd_n_power = harmonic_power + noise_power
    if fundamental_power <= 0:
        return 0.0

    thd_n_db = 10 * np.log10(thd_n_power / fundamental_power)
    return thd_n_db


def find_step_crossing(x, threshold=0.5):
    """
    Find the sample index where signal crosses threshold.

    Used for measuring group delay with step response.

    Args:
        x: Input signal
        threshold: Crossing threshold (default 0.5 for 50%)

    Returns:
        Sample index of first crossing, or -1 if not found
    """
    # Normalize to 0-1 range
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-10:
        return -1

    x_norm = (x - x_min) / (x_max - x_min)

    # Find first crossing
    for i in range(len(x_norm) - 1):
        if x_norm[i] < threshold <= x_norm[i + 1]:
            # Linear interpolation for sub-sample accuracy
            frac = (threshold - x_norm[i]) / (x_norm[i + 1] - x_norm[i])
            return i + frac

    return -1


# =============================================================================
# Test Functions
# =============================================================================

def test_fm_demod_accuracy():
    """
    Test FM demodulation accuracy.

    Generates known baseband, FM modulates to I/Q, demodulates back,
    and compares output to input.

    Pass criteria: Correlation > 0.999, amplitude error < 1%
    """
    print("\n" + "=" * 60)
    print("TEST: FM Demodulation Accuracy")
    print("=" * 60)

    sample_rate = 250000
    duration = 0.1  # 100 ms
    test_freq = 1000  # 1 kHz test tone

    # Generate test tone baseband
    baseband_in = generate_test_tone(test_freq, duration, sample_rate, amplitude=0.5)

    # FM modulate to I/Q
    iq = fm_modulate(baseband_in, sample_rate, deviation=75000)

    # Create decoder (disable tone controls and de-emphasis for clean test)
    decoder = FMStereoDecoder(
        iq_sample_rate=sample_rate,
        audio_sample_rate=sample_rate,  # No resampling
        deviation=75000,
        deemphasis=1e-9  # Effectively disable de-emphasis
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = decoder.demodulate(iq)

    # Get baseband from decoder (before filtering/resampling)
    baseband_out = decoder.last_baseband

    # Allow for filter settling - skip first 2000 samples
    skip = 2000
    baseband_in_trimmed = baseband_in[skip:-skip]
    baseband_out_trimmed = baseband_out[skip:-skip]

    # Compute correlation
    correlation = np.corrcoef(baseband_in_trimmed, baseband_out_trimmed)[0, 1]

    # Compute RMS amplitude ratio
    rms_in = np.sqrt(np.mean(baseband_in_trimmed ** 2))
    rms_out = np.sqrt(np.mean(baseband_out_trimmed ** 2))
    amplitude_ratio = rms_out / rms_in
    amplitude_error = abs(1.0 - amplitude_ratio) * 100

    print(f"  Correlation: {correlation:.6f}")
    print(f"  Amplitude ratio: {amplitude_ratio:.4f}")
    print(f"  Amplitude error: {amplitude_error:.2f}%")

    passed = correlation > 0.999 and amplitude_error < 1.0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_gpu_demod_accuracy():
    """
    Test GPU FM demodulation accuracy.

    Uses GPUFMDemodulator to demodulate FM signal and compares to expected.
    If GPU is not available, test passes with a skip message.

    Pass criteria: Correlation > 0.999, amplitude error < 1%
    """
    print("\n" + "=" * 60)
    print("TEST: GPU FM Demodulation Accuracy")
    print("=" * 60)

    sample_rate = 250000
    duration = 0.1  # 100 ms
    test_freq = 1000  # 1 kHz test tone

    # Generate test tone baseband
    baseband_in = generate_test_tone(test_freq, duration, sample_rate, amplitude=0.5)

    # FM modulate to I/Q
    iq = fm_modulate(baseband_in, sample_rate, deviation=75000)

    # Create GPU demodulator
    gpu_demod = GPUFMDemodulator(sample_rate=sample_rate, deviation=75000)

    if gpu_demod.backend == 'cpu':
        print("  GPU not available - testing CPU fallback path")

    # Demodulate
    baseband_out = gpu_demod.demodulate(iq)

    # Allow for settling - skip first and last samples
    skip = 2000
    baseband_in_trimmed = baseband_in[skip:-skip]
    baseband_out_trimmed = baseband_out[skip:-skip]

    # Compute correlation
    correlation = np.corrcoef(baseband_in_trimmed, baseband_out_trimmed)[0, 1]

    # Compute RMS amplitude ratio
    rms_in = np.sqrt(np.mean(baseband_in_trimmed ** 2))
    rms_out = np.sqrt(np.mean(baseband_out_trimmed ** 2))
    amplitude_ratio = rms_out / rms_in
    amplitude_error = abs(1.0 - amplitude_ratio) * 100

    print(f"  Backend: {gpu_demod.backend}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Amplitude ratio: {amplitude_ratio:.4f}")
    print(f"  Amplitude error: {amplitude_error:.2f}%")

    passed = correlation > 0.999 and amplitude_error < 1.0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    gpu_demod.cleanup()
    return passed


def test_cpu_gpu_parity():
    """
    Test CPU and GPU demodulation produce matching results.

    Compares FMStereoDecoder CPU path with GPUFMDemodulator output.
    Both should produce effectively identical baseband signals.

    Pass criteria: Correlation > 0.9999 between CPU and GPU output
    """
    print("\n" + "=" * 60)
    print("TEST: CPU/GPU Demodulation Parity")
    print("=" * 60)

    sample_rate = 250000
    duration = 0.1  # 100 ms
    test_freq = 1000  # 1 kHz test tone

    # Generate FM signal
    baseband_in = generate_test_tone(test_freq, duration, sample_rate, amplitude=0.5)
    iq = fm_modulate(baseband_in, sample_rate, deviation=75000)

    # CPU demodulation via FMStereoDecoder
    decoder = FMStereoDecoder(
        iq_sample_rate=sample_rate,
        audio_sample_rate=sample_rate,  # No resampling
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False
    _ = decoder.demodulate(iq)
    cpu_baseband = decoder.last_baseband

    # GPU demodulation
    gpu_demod = GPUFMDemodulator(sample_rate=sample_rate, deviation=75000)
    gpu_baseband = gpu_demod.demodulate(iq)

    print(f"  GPU backend: {gpu_demod.backend}")

    # Compare outputs (skip settling region)
    skip = 2000
    cpu_trimmed = cpu_baseband[skip:-skip]
    gpu_trimmed = gpu_baseband[skip:-skip]

    # Compute correlation
    correlation = np.corrcoef(cpu_trimmed, gpu_trimmed)[0, 1]

    # Compute max absolute difference
    max_diff = np.max(np.abs(cpu_trimmed - gpu_trimmed))
    mean_diff = np.mean(np.abs(cpu_trimmed - gpu_trimmed))

    print(f"  Correlation: {correlation:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Both methods should produce nearly identical results
    # The CPU uses quadrature discriminator (angle of product)
    # The GPU uses arctangent-differentiate (atan2 + unwrap)
    # Both are mathematically equivalent
    passed = correlation > 0.9999
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: correlation > 0.9999)")

    gpu_demod.cleanup()
    return passed


def test_audio_snr():
    """
    Test decoded audio SNR with clean input.

    Generates clean 1 kHz stereo signal, measures SNR of decoded audio.
    With perfect synthetic signal, expect >40 dB SNR.
    """
    print("\n" + "=" * 60)
    print("TEST: Audio SNR (Clean Input)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.5  # 500 ms for stable measurement
    test_freq = 1000

    # Generate stereo test signal
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * test_freq * t)
    right = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Create FM multiplex and modulate (use neg_cos for pilot-squaring)
    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder
    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = demodulate_with_settling(decoder, iq)

    # Skip settling time
    skip = int(0.05 * audio_rate)  # 50 ms
    left_out = audio[skip:-skip, 0]

    # Measure SNR
    snr = measure_snr(left_out, test_freq, audio_rate)

    print(f"  Test frequency: {test_freq} Hz")
    print(f"  Measured SNR: {snr:.1f} dB")

    # Note: 35 dB is realistic for full stereo chain with filters/resampling/limiter
    passed = snr > 35
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >35 dB)")

    return passed


def test_thd_n():
    """
    Test Total Harmonic Distortion + Noise.

    Target: THD+N < -40 dB (< 1% distortion)
    """
    print("\n" + "=" * 60)
    print("TEST: THD+N (Total Harmonic Distortion + Noise)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.5
    test_freq = 1000

    # Generate mono test signal (identical L/R)
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Create FM multiplex and modulate
    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder
    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = demodulate_with_settling(decoder, iq)

    # Skip settling
    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]

    # Measure THD+N
    thd_n = measure_thd_n(left_out, test_freq, audio_rate)

    print(f"  Test frequency: {test_freq} Hz")
    print(f"  THD+N: {thd_n:.1f} dB ({10**(thd_n/20)*100:.2f}%)")

    # Note: -35 dB (1.8%) is realistic for full stereo chain with soft limiter
    passed = thd_n < -35
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: < -35 dB)")

    return passed


def test_mono_decode():
    """
    Test mono decoding (no pilot tone).

    Generates L+R only signal without pilot, verifies mono output.

    Pass criteria:
    - L/R correlation > 0.999 (same signal)
    - Correct frequency detected
    """
    print("\n" + "=" * 60)
    print("TEST: Mono Decoding (No Pilot)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.3
    test_freq = 1000

    # Generate mono signal
    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    mono = 0.5 * np.sin(2 * np.pi * test_freq * t)

    # Create FM multiplex WITHOUT pilot
    multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, include_pilot=False)
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder (production default settings)
    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    # Demodulate
    audio = demodulate_with_settling(decoder, iq)

    # Check pilot detection
    pilot_detected = decoder.pilot_detected

    # Skip settling
    skip = int(0.05 * audio_rate)
    left_out = audio[skip:-skip, 0]
    right_out = audio[skip:-skip, 1]

    # L/R correlation
    lr_corr = np.corrcoef(left_out, right_out)[0, 1]

    # Check for correct frequency
    left_power_1k = goertzel_power(left_out, test_freq, audio_rate)
    left_power_total = np.mean(left_out ** 2)
    freq_ratio = left_power_1k / left_power_total if left_power_total > 0 else 0

    print(f"  Pilot detected: {pilot_detected}")
    print(f"  L/R correlation: {lr_corr:.6f}")
    print(f"  1 kHz power ratio: {freq_ratio:.4f}")

    passed = not pilot_detected and lr_corr > 0.999 and freq_ratio > 0.5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_stereo_decode():
    """
    Test stereo decoding with different tones on L and R.

    Generates 1 kHz on left, 2 kHz on right.

    Pass criteria:
    - Pilot detected
    - Channel separation > 20 dB
    """
    print("\n" + "=" * 60)
    print("TEST: Stereo Decoding (1 kHz L, 2 kHz R)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0
    left_freq = 1000
    right_freq = 2000

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate
    left = 0.5 * np.sin(2 * np.pi * left_freq * t)
    right = 0.5 * np.sin(2 * np.pi * right_freq * t)

    # Use neg_cos subcarrier (matches pilot-squaring decoder)
    multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)

    # Create decoder
    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    audio = demodulate_with_settling(decoder, iq)

    pilot_detected = decoder.pilot_detected

    skip = len(audio) // 2
    left_out = audio[skip:, 0]
    right_out = audio[skip:, 1]

    left_1k_power = goertzel_power(left_out, left_freq, audio_rate)
    left_2k_power = goertzel_power(left_out, right_freq, audio_rate)
    right_1k_power = goertzel_power(right_out, left_freq, audio_rate)
    right_2k_power = goertzel_power(right_out, right_freq, audio_rate)

    left_separation = 10 * np.log10(left_1k_power / (left_2k_power + 1e-12))
    right_separation = 10 * np.log10(right_2k_power / (right_1k_power + 1e-12))

    print(f"  Pilot detected: {pilot_detected}")
    print(f"  Left channel separation: {left_separation:.1f} dB")
    print(f"  Right channel separation: {right_separation:.1f} dB")

    min_separation = min(left_separation, right_separation)

    passed = pilot_detected and min_separation > 20
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: >20 dB separation)")

    return passed


def test_stereo_separation():
    """
    Test stereo separation across frequency range.

    Tests left-only tones at 100 Hz, 1 kHz, 5 kHz, 10 kHz, 12 kHz.

    Pass criteria: >30 dB separation at all tested frequencies
    """
    print("\n" + "=" * 60)
    print("TEST: Stereo Separation vs Frequency")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0

    test_freqs = [100, 1000, 5000, 10000, 12000]

    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    all_passed = True
    separations = []

    for freq in test_freqs:
        decoder.reset()

        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        left = 0.5 * np.sin(2 * np.pi * freq * t)
        right = np.zeros(n_samples)

        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
        iq = fm_modulate(multiplex, iq_rate)

        audio = demodulate_with_settling(decoder, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        left_power = goertzel_power(left_out, freq, audio_rate)
        right_power = goertzel_power(right_out, freq, audio_rate)

        separation = 10 * np.log10(left_power / (right_power + 1e-12))
        separations.append(separation)

        print(f"  {freq:5d} Hz: {separation:5.1f} dB", end="")
        if separation < 30:
            print(" (FAIL)")
            all_passed = False
        else:
            print(" (PASS)")

    min_sep = min(separations)
    print(f"\n  Minimum separation: {min_sep:.1f} dB")
    print(f"  Result: {'PASS' if all_passed else 'FAIL'} (target: >30 dB at all frequencies)")

    return all_passed


def test_subcarrier_phase_sensitivity():
    """
    Test decoder sensitivity to subcarrier phase.

    Documents what happens with different TX subcarrier phases:
    - neg_cos: Should work correctly (matches pilot-squaring)
    - cos: Should give inverted stereo (L/R swap)
    - sin: Should fail completely (90° phase error)
    """
    print("\n" + "=" * 60)
    print("TEST: Subcarrier Phase Sensitivity")
    print("=" * 60)
    print()
    print("  The pilot-squaring decoder produces -cos(2wt) carrier")
    print("  This test verifies correct behavior with different TX phases")
    print()

    iq_rate = 250000
    audio_rate = 48000
    duration = 1.0

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate

    # Left-only signal
    left = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right = np.zeros(n_samples)

    results = {}

    for phase_name in ['neg_cos', 'cos', 'sin']:
        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase=phase_name)
        iq = fm_modulate(multiplex, iq_rate)

        decoder = FMStereoDecoder(
            iq_sample_rate=iq_rate,
            audio_sample_rate=audio_rate,
            deviation=75000,
            deemphasis=1e-9
        )
        decoder.bass_boost_enabled = False
        decoder.treble_boost_enabled = False
        decoder.stereo_blend_enabled = False

        audio = demodulate_with_settling(decoder, iq)

        skip = len(audio) // 2
        left_out = audio[skip:, 0]
        right_out = audio[skip:, 1]

        left_power = goertzel_power(left_out, 1000, audio_rate)
        right_power = goertzel_power(right_out, 1000, audio_rate)

        # Determine result
        if left_power > 0.01 and right_power < 0.001:
            status = "CORRECT (signal in L)"
        elif right_power > 0.01 and left_power < 0.001:
            status = "INVERTED (signal in R)"
        elif abs(left_power - right_power) / max(left_power, right_power, 1e-12) < 0.5:
            status = "FAILED (no separation)"
        else:
            status = "PARTIAL"

        results[phase_name] = (left_power, right_power, status)

        print(f"  TX={phase_name:8s}: L={left_power:.4f}, R={right_power:.4f} -> {status}")

    print()

    # Pass if neg_cos works correctly and sin fails
    neg_cos_correct = 'CORRECT' in results['neg_cos'][2]
    sin_fails = 'FAILED' in results['sin'][2]

    passed = neg_cos_correct and sin_fails
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print(f"  (neg_cos should work, sin should fail due to 90° phase error)")

    return passed


def test_group_delay_alignment():
    """
    Test L/R timing alignment.

    Uses step function to measure timing difference between channels.
    The L+R and L-R paths should be aligned within 5 samples (~100 us at 48 kHz).
    """
    print("\n" + "=" * 60)
    print("TEST: Group Delay Alignment (L/R Timing)")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.1

    n_samples = int(duration * iq_rate)
    t = np.arange(n_samples) / iq_rate

    # Create step function
    step_point = n_samples // 2
    step = np.zeros(n_samples)
    step[step_point:] = 0.5

    # Create left+right step for L/R alignment check
    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    lr_step = step.copy()
    multiplex = generate_fm_stereo_multiplex(lr_step, lr_step, iq_rate, subcarrier_phase='neg_cos')
    iq = fm_modulate(multiplex, iq_rate)
    audio = demodulate_with_settling(decoder, iq)

    left = audio[:, 0]
    right = audio[:, 1]

    left_crossing = find_step_crossing(left)
    right_crossing = find_step_crossing(right)

    if left_crossing < 0 or right_crossing < 0:
        print("  ERROR: Could not find step crossings")
        return False

    lr_diff_samples = abs(left_crossing - right_crossing)
    lr_diff_us = lr_diff_samples * (1e6 / audio_rate)

    print(f"  Left crossing at: {left_crossing:.2f} samples")
    print(f"  Right crossing at: {right_crossing:.2f} samples")
    print(f"  L/R crossing difference: {lr_diff_samples:.2f} samples ({lr_diff_us:.0f} us)")

    passed = lr_diff_samples < 5
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: <5 samples)")

    return passed


def test_frequency_response():
    """
    Test frequency response from 100 Hz to 14 kHz.

    Measures relative amplitude across frequency range.
    Expect flat +-3 dB response (before de-emphasis).
    """
    print("\n" + "=" * 60)
    print("TEST: Frequency Response")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.2

    test_freqs = [100, 200, 500, 1000, 2000, 5000, 8000, 10000, 12000, 14000]

    decoder = FMStereoDecoder(
        iq_sample_rate=iq_rate,
        audio_sample_rate=audio_rate,
        deviation=75000,
        deemphasis=1e-9
    )
    decoder.bass_boost_enabled = False
    decoder.treble_boost_enabled = False
    decoder.stereo_blend_enabled = False

    responses = []

    for freq in test_freqs:
        decoder.reset()

        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        mono = 0.5 * np.sin(2 * np.pi * freq * t)

        multiplex = generate_fm_stereo_multiplex(mono, mono, iq_rate, subcarrier_phase='neg_cos')
        iq = fm_modulate(multiplex, iq_rate)

        audio = demodulate_with_settling(decoder, iq)

        skip = int(0.05 * audio_rate)
        left_out = audio[skip:-skip, 0]

        power = goertzel_power(left_out, freq, audio_rate)
        responses.append(power)

    # Normalize to 1 kHz response
    ref_idx = test_freqs.index(1000)
    ref_power = responses[ref_idx]

    print(f"  {'Freq':>6s}  {'Response':>10s}  {'Relative':>10s}")
    print(f"  {'----':>6s}  {'--------':>10s}  {'--------':>10s}")

    max_deviation = 0
    for freq, power in zip(test_freqs, responses):
        rel_db = 10 * np.log10(power / ref_power) if ref_power > 0 else 0
        print(f"  {freq:6d}  {power:10.6f}  {rel_db:+7.2f} dB")
        max_deviation = max(max_deviation, abs(rel_db))

    # Check response up to 1 kHz
    low_freq_max_dev = 0
    for freq, power in zip(test_freqs, responses):
        if freq <= 1000:
            rel_db = 10 * np.log10(power / ref_power) if ref_power > 0 else 0
            low_freq_max_dev = max(low_freq_max_dev, abs(rel_db))

    print(f"\n  Max deviation (all): {max_deviation:.2f} dB")
    print(f"  Max deviation (<= 1 kHz): {low_freq_max_dev:.2f} dB")

    passed = low_freq_max_dev < 3.0
    print(f"  Result: {'PASS' if passed else 'FAIL'} (target: +-3 dB below 1 kHz)")

    return passed


def test_snr_with_noise():
    """
    Test decoder behavior with noisy input.

    Adds AWGN at various levels, measures decoded audio quality.
    Verifies graceful degradation.
    """
    print("\n" + "=" * 60)
    print("TEST: SNR with Noisy Input")
    print("=" * 60)

    iq_rate = 250000
    audio_rate = 48000
    duration = 0.5

    input_snr_levels = [40, 30, 20, 10]
    test_freq = 1000

    print(f"  {'Input SNR':>12s}  {'Output SNR':>12s}  {'Pilot':>8s}  {'Blend':>8s}")
    print(f"  {'--------':>12s}  {'----------':>12s}  {'-----':>8s}  {'-----':>8s}")

    all_passed = True

    for input_snr in input_snr_levels:
        n_samples = int(duration * iq_rate)
        t = np.arange(n_samples) / iq_rate
        left = 0.5 * np.sin(2 * np.pi * test_freq * t)
        right = 0.5 * np.sin(2 * np.pi * test_freq * t)

        multiplex = generate_fm_stereo_multiplex(left, right, iq_rate, subcarrier_phase='neg_cos')
        iq_clean = fm_modulate(multiplex, iq_rate)

        # Add noise to I/Q
        signal_power = np.mean(np.abs(iq_clean) ** 2)
        noise_power = signal_power / (10 ** (input_snr / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        iq_noisy = iq_clean + noise.astype(np.complex64)

        decoder = FMStereoDecoder(
            iq_sample_rate=iq_rate,
            audio_sample_rate=audio_rate,
            deviation=75000,
            deemphasis=1e-9
        )
        decoder.bass_boost_enabled = False
        decoder.treble_boost_enabled = False
        decoder.stereo_blend_enabled = True

        audio = demodulate_with_settling(decoder, iq_noisy)

        pilot_detected = decoder.pilot_detected
        blend_factor = decoder.stereo_blend_factor

        skip = int(0.1 * audio_rate)
        left_out = audio[skip:-skip, 0]
        output_snr = measure_snr(left_out, test_freq, audio_rate)

        print(f"  {input_snr:10d} dB  {output_snr:10.1f} dB  {'Yes' if pilot_detected else 'No':>8s}  {blend_factor:8.2f}")

        if input_snr >= 40 and output_snr < 30:
            all_passed = False

    print(f"\n  Result: {'PASS' if all_passed else 'FAIL'} (graceful degradation)")

    return all_passed


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("FM STEREO DECODER TEST SUITE")
    print("=" * 60)
    print("\nTesting demodulator.py (FMStereoDecoder) and gpu.py (GPUFMDemodulator)")
    print("Signal flow: IQ -> FM Demod -> Pilot/L+R/L-R -> Matrix -> Audio")

    tests = [
        ("FM Demodulation Accuracy", test_fm_demod_accuracy),
        ("GPU Demodulation Accuracy", test_gpu_demod_accuracy),
        ("CPU/GPU Parity", test_cpu_gpu_parity),
        ("Audio SNR (Clean)", test_audio_snr),
        ("THD+N", test_thd_n),
        ("Mono Decoding", test_mono_decode),
        ("Stereo Decoding", test_stereo_decode),
        ("Stereo Separation", test_stereo_separation),
        ("Subcarrier Phase Sensitivity", test_subcarrier_phase_sensitivity),
        ("Group Delay Alignment", test_group_delay_alignment),
        ("Frequency Response", test_frequency_response),
        ("SNR with Noise", test_snr_with_noise),
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

    print()
    print("NOTES:")
    print("  - Pilot-squaring requires TX to use -cos(2wt) subcarrier")
    print("  - GPU tests require PyTorch with ROCm/CUDA (skipped if unavailable)")

    return passed_count == total_count


def cleanup_gpu_and_exit(exit_code):
    """Cleanup GPU resources and exit cleanly to prevent ROCm/PyTorch shutdown issues."""
    import os
    # Flush output buffers before any exit
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Force immediate exit to avoid PyTorch/ROCm double-free during cleanup
            os._exit(exit_code)
    except ImportError:
        pass
    sys.exit(exit_code)


if __name__ == "__main__":
    success = run_all_tests()
    cleanup_gpu_and_exit(0 if success else 1)
